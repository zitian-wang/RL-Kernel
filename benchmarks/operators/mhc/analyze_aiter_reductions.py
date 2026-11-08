# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Which reduction in aiter's mHC kernels makes a launch parameter change the bits, and why.

For each launch config of ``mhc_pre_dsv4`` and ``mhc_head_dsv4`` (one parameter
changed from a reference config of the same bit class), this script

1. runs the op on four kinds of input and compares every output bit for bit with
   the reference config at M = 17 and 1024:
   - ``random``: random residual and fn (both reductions round);
   - ``exact sum of squares``: residual entries in {-2, -1, 1, 2}, so every sum
     of squares is an integer below 2^24 and exact in any order; a difference
     comes from the projection or the stream sum;
   - ``exact projection``: one 1 per fn row, so the projection is one input
     element, exact in any order; a difference comes from the sum of squares or
     the stream sum;
   - ``both exact``: both of the above; the mixing weights are then the same
     for every config, so a difference comes from the stream sum alone;
2. records the TTGIR of every kernel the launch compiled and reports the layout
   of the operand of each ``tt.reduce`` (sum of squares, stream sum), and the
   ``kWidth`` of the dot operands.

Nothing is asserted: the script prints a markdown report. ``--layouts-only``
skips step 1 (one launch per config at M = 17), e.g. to compare Triton versions.
``--digest FILE`` instead writes SHA-256 digests of every output of the three
class references for every input kind and M; with ``--compare OTHER`` it prints
which of them differ from OTHER (e.g. another Triton version).

Usage (GPU; aiter with the DeepSeek-V4 mHC ops and AITER_TRITON_ONLY=1):
    python benchmarks/operators/mhc/analyze_aiter_reductions.py --output report.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import sys
from pathlib import Path

import torch
import triton

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.tuning.tune_mhc_head_config import aiter_head_config, launch_mhc_head  # noqa: E402

DEV = "cuda"
N, C = 4, 4096
K = N * C
PRE_ROWS = 2 * N + N * N
M_VALUES = [17, 1024]
KINDS = ("random", "exact sum of squares", "exact projection", "both exact")
PRE_OUTPUTS = ("post_mix", "comb_mix", "layer_input")

PRE_SPLIT1_REF = {
    "NUM_KSPLIT": 1,
    "BLOCK_M": 32,
    "BLOCK_K": 128,
    "BLOCK_C": 128,
    "num_warps": 4,
    "num_stages": 1,
    "waves_per_eu": 2,
}
PRE_SPLIT1_VARIANTS = [
    {"BLOCK_M": 16},
    {"BLOCK_M": 64},
    {"BLOCK_M": 128},
    {"num_warps": 1},
    {"num_warps": 2},
    {"num_warps": 8},
    {"BLOCK_M": 16, "num_warps": 2},
    {"BLOCK_M": 16, "num_warps": 8},
    {"BLOCK_M": 64, "num_warps": 8},
    {"num_stages": 2},
    {"BLOCK_K": 32},
    {"BLOCK_K": 64},
    {"BLOCK_K": 256},
    {"BLOCK_C": 64},
    {"BLOCK_C": 256},
    {"kpack": 2},
    {"matrix_instr_nonkdim": 32},
]
# The gfx942 table's M_LEQ_1 entry, and every other entry of the table.
PRE_SPLIT4_REF = {
    "NUM_KSPLIT": 4,
    "BLOCK_M": 16,
    "BLOCK_K": 128,
    "BLOCK_C": 64,
    "num_warps": 4,
    "num_stages": 1,
    "waves_per_eu": 0,
}
PRE_SPLIT4_VARIANTS = [
    {"BLOCK_M": 64, "num_stages": 2, "waves_per_eu": 1},
    {"BLOCK_M": 64, "BLOCK_C": 128, "num_warps": 2, "num_stages": 2, "waves_per_eu": 1},
    {"num_warps": 2, "waves_per_eu": 2},
    {"BLOCK_C": 128, "num_warps": 2, "waves_per_eu": 2},
    {"BLOCK_M": 128, "num_stages": 2},
    {"BLOCK_M": 64, "num_warps": 2},
    {"BLOCK_M": 128, "waves_per_eu": 2},
    {"num_warps": 8},
    {"BLOCK_C": 256},
    {"BLOCK_K": 64},
    {"kpack": 2},
]
HEAD_VARIANTS = [
    {"BLOCK_M": 32},
    {"num_warps": 2},
    {"num_warps": 8},
    {"BLOCK_M": 32, "num_warps": 8},
    {"num_stages": 2},
    {"waves_per_eu": 2},
    {"BLOCK_K": 32},
    {"BLOCK_K": 128},
    {"BLOCK_C": 64},
    {"BLOCK_C": 256},
    {"kpack": 2},
]


# --- TTGIR capture and parsing ---------------------------------------------------------------


class Capture:
    """Records the CompiledKernel of every launch of the given Triton kernels."""

    def __init__(self, kernels):
        self.kernels = kernels
        self.compiled = []

    def __enter__(self):
        for kernel in self.kernels:
            run = kernel.run

            def wrapped(*args, _run=run, _name=kernel.fn.__name__, **kwargs):
                compiled = _run(*args, **kwargs)
                self.compiled.append((_name, compiled))
                return compiled

            kernel.run = wrapped
        return self

    def __exit__(self, *exc):
        for kernel in self.kernels:
            del kernel.run


_ALIAS = re.compile(r"^(#[\w]+) = (.*)$", re.M)
_REDUCE_TYPE = re.compile(r"\}\) : \((tensor<[^\n]*?>)\) ->")
_TENSOR = re.compile(r"tensor<([\dx]+)x(\w+), (.*)>$")


def _resolve(encoding, aliases):
    return aliases.get(encoding, encoding)


def _blocked_params(layout):
    """sizePerThread, threadsPerWarp, warpsPerCTA, order of a #ttg.blocked layout, else None."""
    if not layout.startswith("#ttg.blocked"):
        return None
    fields = {}
    for key in ("sizePerThread", "threadsPerWarp", "warpsPerCTA", "order"):
        match = re.search(key + r" = \[([\d, ]+)\]", layout)
        fields[key] = [int(v) for v in match.group(1).split(",")] if match else None
    return fields


def reductions(ttgir):
    """Every tt.reduce: (shape, axis, layout of its operand)."""
    aliases = dict(_ALIAS.findall(ttgir))
    found = []
    for match in re.finditer(r'"tt\.reduce"\(', ttgir):
        axis = int(re.search(r"axis = (\d+)", ttgir[match.end() : match.end() + 80]).group(1))
        typed = _REDUCE_TYPE.search(ttgir, match.end())
        tensor = _TENSOR.match(typed.group(1))
        shape = [int(v) for v in tensor.group(1).split("x")]
        found.append((shape, axis, _resolve(tensor.group(3), aliases)))
    return found


def dot_operands(ttgir):
    """Distinct (opIdx, kWidth) of dot operands and the MFMA layouts."""
    aliases = dict(_ALIAS.findall(ttgir))
    ops = sorted(set(re.findall(r"#ttg\.dot_op<\{opIdx = (\d), parent = [^,]+, kWidth = (\d+)\}>", ttgir)))
    mfma = sorted({v for v in aliases.values() if "amd_mfma" in v})
    return ops, mfma


def _axis_signature(shape, axis, layout):
    """What fixes a row's reduction order: the operand layout along the reduced axis."""
    params = _blocked_params(layout)
    if params is None:
        return f"non-blocked: {layout}"
    return (
        f"len {shape[axis]}: spt {params['sizePerThread'][axis]}, "
        f"lanes {params['threadsPerWarp'][axis]}, warps {params['warpsPerCTA'][axis]}"
    )


def _role(kernel, shape, axis, config):
    if len(shape) == 2 and axis == 1 and kernel != "_mhc_asymmetric_sinkhorn_kernel":
        return "sum of squares"
    if len(shape) == 3 and axis == 1 and shape[2] == config.get("BLOCK_C"):
        return "stream sum"
    return None


def summarize_layouts(compiled, config):
    """{role: axis signature} and dot-operand info over the captured kernels."""
    roles, dots = {}, set()
    for name, kernel in compiled:
        ttgir = kernel.asm["ttgir"]
        for shape, axis, layout in reductions(ttgir):
            role = _role(name, shape, axis, config)
            if role:
                roles.setdefault(role, set()).add(_axis_signature(shape, axis, layout))
        ops, mfma = dot_operands(ttgir)
        for idx, width in ops:
            dots.add(f"op{idx} kWidth {width}")
        for layout in mfma:
            shape = re.search(r"instrShape = \[([\d, ]+)\]", layout)
            dots.add(f"mfma {shape.group(1) if shape else layout}")
    return {k: "; ".join(sorted(v)) for k, v in roles.items()}, "; ".join(sorted(dots))


# --- inputs and launches ---------------------------------------------------------------------


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def residual(kind, m, seed=0):
    gen = _gen(seed)
    if kind in ("exact sum of squares", "both exact"):
        magnitude = torch.randint(1, 3, (m, N, C), generator=gen, device=DEV)
        sign = 2 * torch.randint(0, 2, (m, N, C), generator=gen, device=DEV) - 1
        return (magnitude * sign).to(torch.bfloat16)
    return torch.randn(m, N, C, generator=gen, device=DEV).to(torch.bfloat16)


def fn_matrix(kind, rows, seed=1):
    if kind in ("exact projection", "both exact"):
        fn = torch.zeros(rows, K, device=DEV)
        fn[torch.arange(rows), (torch.arange(rows) * 997) % K] = 1.0
        return fn
    return torch.randn(rows, K, generator=_gen(seed), device=DEV) / K**0.5


def scale_base(count, rows, seed=2):
    gen = _gen(seed)
    return 0.5 + torch.rand(count, generator=gen, device=DEV), 0.1 * torch.randn(
        rows, generator=gen, device=DEV
    )


def run_pre(config, inputs):
    from aiter.ops.triton.fusions import mhc_pre_dsv4

    with torch.no_grad():
        out = mhc_pre_dsv4(*inputs, sinkhorn_repeat=20, config=dict(config))
    torch.cuda.synchronize()
    return out


def run_head(config, inputs):
    with torch.no_grad():
        out = launch_mhc_head(*inputs, config)
    torch.cuda.synchronize()
    return (out,)


def pre_kernels():
    from aiter.ops.triton._triton_kernels import fusions

    return [
        fusions._mhc_fused_kernel,
        fusions._mhc_fused_split_kernel,
        fusions._mhc_reduce_apply_kernel,
        fusions._mhc_asymmetric_sinkhorn_kernel,
    ]


def head_kernels():
    from aiter.ops.triton._triton_kernels import fusions

    return [fusions._mhc_head_kernel]


def _differing(a, b):
    bits = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[a.dtype]
    return int((a.contiguous().view(bits) != b.contiguous().view(bits)).sum())


def _label(config, reference):
    changed = {k: v for k, v in config.items() if reference.get(k) != v}
    return " ".join(f"{k}={v}" for k, v in changed.items()) or "reference"


def study(name, reference, variants, run, kernels, output_names, rows, scale_count, layouts_only):
    """Rows of the report for one bit class: config, bits per input kind, layouts."""
    m_values = M_VALUES[:1] if layouts_only else M_VALUES
    kinds = KINDS[:1] if layouts_only else KINDS
    inputs = {}
    for kind in kinds:
        scale, base = scale_base(scale_count, rows)
        fn = fn_matrix(kind, rows)
        inputs[kind] = {m: (residual(kind, m), fn, scale, base) for m in m_values}

    def launch(config):
        outs, compiled = {}, None
        for kind in kinds:
            for m in m_values:
                with Capture(kernels) as cap:
                    outs[kind, m] = run(config, inputs[kind][m])
                compiled = compiled or cap.compiled
        return outs, compiled

    ref_out, ref_compiled = launch(reference)
    ref_layouts, ref_dots = summarize_layouts(ref_compiled, reference)
    rows_out = [(_label(reference, reference), {k: "reference" for k in kinds}, ref_layouts, ref_dots)]
    for change in variants:
        config = dict(reference, **change)
        try:
            outs, compiled = launch(config)
        except Exception as exc:  # a config can exceed LDS or registers
            rows_out.append((_label(config, reference), f"cannot launch: {type(exc).__name__}", {}, ""))
            continue
        cells = {}
        for kind in kinds:
            diffs = []
            for name_i, idx in zip(output_names, range(len(output_names))):
                counts = [
                    f"{_differing(outs[kind, m][idx], ref_out[kind, m][idx])} at M={m}"
                    for m in m_values
                    if _differing(outs[kind, m][idx], ref_out[kind, m][idx])
                ]
                if counts:
                    diffs.append(f"{name_i} ({', '.join(counts)})")
            cells[kind] = "differs: " + "; ".join(diffs) if diffs else "same"
        layouts, dots = summarize_layouts(compiled, config)
        rows_out.append((_label(config, reference), cells, layouts, dots))
    return name, reference, kinds, rows_out


def to_markdown(studies, layouts_only):
    lines = [
        "# aiter mHC reductions: bits and layouts per launch config",
        "",
        f"Host `{socket.gethostname()}`, {torch.cuda.get_device_name()}, torch {torch.__version__}, "
        f"triton {triton.__version__}. n={N}, C={C}, K={K}, M in "
        f"{M_VALUES[:1] if layouts_only else M_VALUES}. Bits are compared with the class "
        "reference (first row) for the same inputs; layout cells give, for the operand of "
        "each reduction, the length of the reduced axis and how it is spread: elements per "
        "thread (spt), lanes and warps.",
        "",
    ]
    for name, reference, kinds, rows in studies:
        lines += [f"## {name}", "", f"Reference: `{reference}`", ""]
        if not layouts_only:
            lines += [
                "| config | " + " | ".join(kinds) + " |",
                "|---|" + "---|" * len(kinds),
            ]
            for label, cells, _, _ in rows:
                if isinstance(cells, str):
                    lines.append(f"| {label} | {cells} |" + " |" * (len(kinds) - 1))
                else:
                    lines.append(f"| {label} | " + " | ".join(cells[k] for k in kinds) + " |")
            lines.append("")
        lines += [
            "| config | sum of squares | stream sum | dot operands |",
            "|---|---|---|---|",
        ]
        for label, cells, layouts, dots in rows:
            if isinstance(cells, str):
                lines.append(f"| {label} | {cells} | | |")
            else:
                lines.append(
                    f"| {label} | {layouts.get('sum of squares', '-')} | "
                    f"{layouts.get('stream sum', '-')} | {dots} |"
                )
        lines.append("")
    return "\n".join(lines)


def digests(head_ref):
    """{(class, kind, M, output): sha256} for the three class references."""
    classes = [
        ("pre NUM_KSPLIT=1", PRE_SPLIT1_REF, run_pre, PRE_OUTPUTS, PRE_ROWS, 3),
        ("pre NUM_KSPLIT=4", PRE_SPLIT4_REF, run_pre, PRE_OUTPUTS, PRE_ROWS, 3),
        ("head", head_ref, run_head, ("output",), N, 1),
    ]
    result = {}
    for name, config, run, outputs, rows, scale_count in classes:
        for kind in KINDS:
            scale, base = scale_base(scale_count, rows)
            fn = fn_matrix(kind, rows)
            for m in M_VALUES:
                outs = run(config, (residual(kind, m), fn, scale, base))
                for out_name, out in zip(outputs, outs):
                    bits = out.contiguous().view(
                        {torch.float32: torch.int32, torch.bfloat16: torch.int16}[out.dtype]
                    )
                    key = f"{name} | {kind} | M={m} | {out_name}"
                    result[key] = hashlib.sha256(bits.cpu().numpy().tobytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", help="also write the markdown report to this file")
    parser.add_argument("--layouts-only", action="store_true")
    parser.add_argument("--digest", help="write output digests (JSON) to this file instead")
    parser.add_argument("--compare", help="with --digest: compare with this digest file")
    args = parser.parse_args()

    head_ref = aiter_head_config()
    if args.digest:
        result = digests(head_ref)
        Path(args.digest).write_text(
            json.dumps({"triton": triton.__version__, "digests": result}, indent=1)
        )
        if args.compare:
            other = json.loads(Path(args.compare).read_text())
            lines = [
                f"| class | input | M | output | triton {other['triton']} vs {triton.__version__} |",
                "|---|---|---|---|---|",
            ]
            for key, value in result.items():
                same = "same" if other["digests"].get(key) == value else "differs"
                lines.append("| " + " | ".join(key.split(" | ")) + f" | {same} |")
            report = "\n".join(lines)
            print(report)
            if args.output:
                Path(args.output).write_text(report + "\n")
        return
    studies = [
        study(
            "mhc_pre, NUM_KSPLIT=1 (_mhc_fused_kernel)",
            PRE_SPLIT1_REF,
            PRE_SPLIT1_VARIANTS,
            run_pre,
            pre_kernels(),
            PRE_OUTPUTS,
            PRE_ROWS,
            3,
            args.layouts_only,
        ),
        study(
            "mhc_pre, NUM_KSPLIT=4 (_mhc_fused_split_kernel + _mhc_reduce_apply_kernel)",
            PRE_SPLIT4_REF,
            PRE_SPLIT4_VARIANTS,
            run_pre,
            pre_kernels(),
            PRE_OUTPUTS,
            PRE_ROWS,
            3,
            args.layouts_only,
        ),
        study(
            "mhc_head (_mhc_head_kernel)",
            head_ref,
            HEAD_VARIANTS,
            run_head,
            head_kernels(),
            ("output",),
            N,
            1,
            args.layouts_only,
        ),
    ]
    report = to_markdown(studies, args.layouts_only)
    print(report)
    if args.output:
        Path(args.output).write_text(report + "\n")


if __name__ == "__main__":
    main()
