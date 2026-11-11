# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Evaluate RL-Kernel's batch-invariant Triton mHC pre, post and head on ROCm.

- baseline: the aiter ops with the per-M config tables (``AiterMHCPreOp``,
  ``AiterMHCPostOp``, ``AiterMHCHeadOp``), the ops before this change;
- ours: ``TritonMHCPreOp``, ``TritonMHCPostOp``, ``TritonMHCHeadOp``
  (rl_engine/backends/rocm/mhc/triton_mhc.py) with their one default config.

Sections, each written to the report as soon as it is done:

1. launch-parameter invariance of ours: every launch parameter changed against
   the default config, all outputs compared bit for bit at several M;
2. batch invariance of ours with rl_engine/validation/common/batch_invariance.py
   (a row alone vs in batches of M up to 65536, chunking/reordering/padding,
   strides, repeated runs);
3. dynamic shape: compiled variants of every kernel of ours after a warm-up and
   after many unseen M;
4. error of baseline and ours against the fp32 PyTorch reference (``forward_fp32``);
5. device time per M from GPU events around a CUDA graph; ours with its default
   config and the fastest of a set of launch configs (same bits, section 1),
   ratios against the baseline;
6. ``--digest``/``--compare``: SHA-256 of the outputs of ours, to compare two
   Triton versions.

Usage (GPU; aiter with the DeepSeek-V4 mHC ops and AITER_TRITON_ONLY=1, for the baseline):
    python benchmarks/operators/mhc/evaluate_triton_mhc.py --output report.md
    python benchmarks/operators/mhc/evaluate_triton_mhc.py --digest d.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import socket
import sys
import traceback
from pathlib import Path

import torch
import triton

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.tuning.tune_mhc_head_config import time_ms  # noqa: E402
from rl_engine.backends.rocm.mhc import triton_mhc  # noqa: E402
from rl_engine.validation.common import batch_invariance as bi  # noqa: E402

DEV = "cuda"
N, C = 4, 4096
K = N * C
PRE_ROWS = 2 * N + N * N
INVARIANCE_M = [1, 17, 1024, 4096]
BI_M = [1, 8, 16, 17, 59, 64, 128, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
PERF_M = [1, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
UNSEEN_M = [2, 3, 7, 31, 33, 100, 255, 257, 1000, 1023, 1025, 3000, 4097, 10000]

PRE_LAUNCH_VARIANTS = [
    {"BLOCK_M": 16},
    {"BLOCK_M": 64},
    {"BLOCK_M": 128},
    {"num_warps": 2},
    {"num_warps": 8},
    {"BLOCK_M": 16, "num_warps": 8},
    {"BLOCK_M": 128, "num_warps": 8},
    {"num_stages": 2},
    {"waves_per_eu": 2},
    {"APPLY_BLOCK_M": 16},
    {"APPLY_BLOCK_M": 64},
    {"APPLY_BLOCK_C": 64},
    {"APPLY_BLOCK_C": 256},
    {"apply_num_warps": 2},
    {"apply_num_warps": 8},
    {"SINKHORN_BLOCK_M": 1},
    {"SINKHORN_BLOCK_M": 64},
]
HEAD_LAUNCH_VARIANTS = [v for v in PRE_LAUNCH_VARIANTS if "SINKHORN_BLOCK_M" not in v]
POST_LAUNCH_VARIANTS = [
    {"BLOCK_M": 16},
    {"BLOCK_M": 32},
    {"BLOCK_M": 128},
    {"BLOCK_C": 64},
    {"BLOCK_C": 128},
    {"num_warps": 2},
    {"num_warps": 8},
    {"num_stages": 2},
    {"waves_per_eu": 0},
]
# Candidates for the fastest config per M (section 5).
PERF_SPLIT = [
    {"BLOCK_M": bm, "num_warps": w} for bm, w in itertools.product([16, 32, 64], [2, 4, 8])
]
PERF_APPLY = [
    {"APPLY_BLOCK_M": 32, "APPLY_BLOCK_C": 128},
    {"APPLY_BLOCK_M": 64, "APPLY_BLOCK_C": 64},
]
PERF_PRE_CONFIGS = [dict(s, **a) for s, a in itertools.product(PERF_SPLIT, PERF_APPLY)]
PERF_POST_CONFIGS = [
    {"BLOCK_M": bm, "BLOCK_C": bc, "num_warps": w}
    for bm, bc, w in itertools.product([16, 32, 64], [128, 256], [4, 8])
]


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def pre_weights(seed=0):
    gen = _gen(seed)
    fn = torch.randn(PRE_ROWS, K, generator=gen, device=DEV) / K**0.5
    scale = 0.5 + torch.rand(3, generator=gen, device=DEV)
    base = 0.1 * torch.randn(PRE_ROWS, generator=gen, device=DEV)
    return fn, scale, base


def head_weights(seed=0):
    gen = _gen(seed + 1)
    fn = torch.randn(N, K, generator=gen, device=DEV) / K**0.5
    scale = 0.5 + torch.rand(1, generator=gen, device=DEV)
    base = 0.1 * torch.randn(N, generator=gen, device=DEV)
    return fn, scale, base


def residual(m, seed=0):
    return torch.randn(m, N, C, generator=_gen(seed), device=DEV).to(torch.bfloat16)


def post_inputs(m, seed=0):
    """y, residual, post_mix [M, n], comb_mix [M, n, n]."""
    gen = _gen(seed)
    y = torch.randn(m, C, generator=gen, device=DEV).to(torch.bfloat16)
    res = torch.randn(m, N, C, generator=gen, device=DEV).to(torch.bfloat16)
    post_mix = 2.0 * torch.sigmoid(torch.randn(m, N, generator=gen, device=DEV))
    comb_mix = torch.softmax(torch.randn(m, N, N, generator=gen, device=DEV), dim=-1)
    return y, res, post_mix, comb_mix


def _flat_pre(out):
    post, comb, li = out
    return [post.reshape(post.shape[0], N), comb.reshape(comb.shape[0], N, N), li]


def ours(pre_w, head_w):
    """{op: callable(inputs, config=None) -> outputs}, inputs as made by make_inputs.

    The launchers need a contiguous residual (the ops make it contiguous themselves)."""
    return {
        "pre": lambda r, config=None: _flat_pre(
            triton_mhc.mhc_pre_dsv4(r.contiguous(), *pre_w, config=config)
        ),
        "post": lambda inputs, config=None: [triton_mhc.mhc_post_dsv4(*inputs, config=config)],
        "head": lambda r, config=None: [
            triton_mhc.mhc_head_dsv4(r.contiguous(), *head_w, config=config)
        ],
    }


def baseline(pre_w, head_w):
    from rl_engine.backends.rocm.mhc.aiter_mhc import AiterMHCHeadOp, AiterMHCPostOp, AiterMHCPreOp

    a_pre, a_post, a_head = AiterMHCPreOp(), AiterMHCPostOp(), AiterMHCHeadOp()
    return {
        "pre": lambda r: _flat_pre(a_pre(r, *pre_w)),
        "post": lambda inputs: [a_post(*inputs)],
        "head": lambda r: [a_head(r, *head_w)],
    }


def make_inputs(op, m, seed=0):
    return post_inputs(m, seed) if op == "post" else residual(m, seed)


OUTPUT_NAMES = {"pre": ["post_mix", "comb_mix", "layer_input"], "post": ["out"], "head": ["out"]}


def _bits(t):
    view = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[t.dtype]
    return t.contiguous().view(view)


def _ndiff(a, b):
    return int((_bits(a) != _bits(b)).sum())


# --- 1. launch-parameter invariance ---------------------------------------------------------------


def section_launch_invariance(pre_w, head_w):
    lines = [
        "## 1. Launch-parameter invariance of ours",
        "",
        "Each row changes the listed launch parameters of the default config (pre "
        f"`{triton_mhc.PRE_CONFIG}`, head `{triton_mhc.HEAD_CONFIG}`, post "
        f"`{triton_mhc.POST_CONFIG}`) and compares every output bit for bit with the default "
        f"config, random inputs, M = {INVARIANCE_M}.",
        "",
        "| op | changed | result |",
        "|---|---|---|",
    ]
    run = ours(pre_w, head_w)
    for op, launch_variants in (
        ("pre", PRE_LAUNCH_VARIANTS),
        ("head", HEAD_LAUNCH_VARIANTS),
        ("post", POST_LAUNCH_VARIANTS),
    ):
        inputs = {m: make_inputs(op, m, seed=10 + m) for m in INVARIANCE_M}
        ref = {m: run[op](inputs[m]) for m in INVARIANCE_M}
        for change in launch_variants:
            try:
                diffs = []
                for m in INVARIANCE_M:
                    out = run[op](inputs[m], change)
                    for name, a, b in zip(OUTPUT_NAMES[op], out, ref[m]):
                        if _ndiff(a, b):
                            diffs.append(f"{name}: {_ndiff(a, b)} elements at M={m}")
                result = "differs: " + "; ".join(diffs) if diffs else "same"
            except Exception as exc:
                result = f"cannot launch: {type(exc).__name__}: {exc}"[:200]
            label = " ".join(f"{k}={v}" for k, v in change.items())
            lines.append(f"| {op} | {label} | {result} |")
    return lines


# --- 2. batch invariance --------------------------------------------------------------------------


def section_batch_invariance(pre_w, head_w):
    lines = [
        "## 2. Batch invariance of ours",
        "",
        "Cells list the outputs whose bits differ (none: no difference). Checks of "
        "rl_engine/validation/common/batch_invariance.py: a row alone vs at the first and "
        "last position of a batch of M; chunking, reordering and padding; other strides; "
        f"two runs. M = {BI_M}; the other checks at M = 17 and 1024.",
        "",
        "| op | batch (M with a difference) | chunk/reorder/pad | strides | repeated runs |",
        "|---|---|---|---|---|",
    ]
    run = ours(pre_w, head_w)
    for op in ("pre", "post", "head"):
        if op == "post":
            case = bi.RowwiseCase(
                op,
                OUTPUT_NAMES[op],
                lambda rows: run["post"](tuple(rows)),
                lambda m, seed: list(post_inputs(m, 1000 + seed)),
            )
        else:
            case = bi.RowwiseCase(
                op,
                OUTPUT_NAMES[op],
                lambda rows, op=op: run[op](rows[0]),
                lambda m, seed: [residual(m, 1000 + seed)],
            )
        try:
            batch = {}
            for m in BI_M:
                batch[m] = bi.differing_in_batch(case, m)
                torch.cuda.empty_cache()
            batch_cell = "; ".join(f"M={m}: {','.join(d)}" for m, d in batch.items() if d) or "none"
            chunk = sorted(
                {n for m in (17, 1024) for n in bi.differing_after_chunk_reorder_pad(case, m)}
            )
            stride = sorted({n for m in (17, 1024) for n in bi.differing_after_restride(case, m)})
            runs = sorted({n for m in (17, 1024) for n in bi.differing_between_runs(case, m)})
            lines.append(
                f"| {op} | {batch_cell} | {','.join(chunk) or 'none'} | "
                f"{','.join(stride) or 'none'} | {','.join(runs) or 'none'} |"
            )
        except Exception as exc:
            lines.append(f"| {op} | error: {type(exc).__name__}: {str(exc)[:150]} | | | |")
    return lines


# --- 3. dynamic shape -----------------------------------------------------------------------------


def _variants():
    return {
        k.fn.__name__: sum(len(c[0]) for c in k.device_caches.values()) for k in triton_mhc.KERNELS
    }


def section_dynamic_shape(pre_w, head_w):
    run = ours(pre_w, head_w)

    def run_all(m):
        for op in ("pre", "post", "head"):
            run[op](make_inputs(op, m, seed=m))

    before = _variants()
    run_all(5)
    warm = _variants()
    for m in UNSEEN_M:
        run_all(m)
    torch.cuda.synchronize()
    after = _variants()
    lines = [
        "## 3. Dynamic shape",
        "",
        "Compiled variants in each kernel's Triton cache: before any call, after one call of "
        f"pre, post and head at M = 5, and after calls at M = {UNSEEN_M}.",
        "",
        "| kernel | before (variants) | after M = 5 (variants) | after the unseen M (variants) |",
        "|---|---|---|---|",
    ]
    for name in warm:
        lines.append(f"| `{name}` | {before[name]} | {warm[name]} | {after[name]} |")
    return lines


# --- 4. error against the fp32 reference ----------------------------------------------------------


def section_accuracy(pre_w, head_w):
    from rl_engine.reference.mhc.mhc import NativeMHCHeadOp, NativeMHCPostOp, NativeMHCPreOp

    m = 1024
    r = residual(m, seed=7)
    p_in = post_inputs(m, seed=8)
    pre_ref = NativeMHCPreOp().forward_fp32(r, *pre_w)
    refs = {
        "pre": [t.reshape(s) for t, s in zip(pre_ref, ((m, N), (m, N, N), (m, C)))],
        "post": [NativeMHCPostOp().forward_fp32(*p_in)],
        "head": [NativeMHCHeadOp().forward_fp32(r, *head_w)],
    }
    lines = [
        "## 4. Error against the fp32 reference",
        "",
        f"M = {m}, random inputs. Reference: `NativeMHC*Op.forward_fp32` (fp32 projection with "
        "fp32 fn, fp32 outputs). Largest / mean absolute error per output.",
        "",
        "| variant | pre post_mix | pre comb_mix | pre layer_input | post out | head out |",
        "|---|---|---|---|---|---|",
    ]
    variants = {"baseline: aiter ops with config tables": None, "ours": ours(pre_w, head_w)}
    try:
        variants["baseline: aiter ops with config tables"] = baseline(pre_w, head_w)
    except Exception as exc:
        lines.append(f"baseline unavailable: {type(exc).__name__}: {exc}")
        del variants["baseline: aiter ops with config tables"]
    for name, run in variants.items():
        cells = []
        for op, inputs in (("pre", r), ("post", p_in), ("head", r)):
            try:
                for out, ref in zip(run[op](inputs), refs[op]):
                    err = (out.float() - ref.float()).abs()
                    cells.append(f"{err.max().item():.3e} / {err.mean().item():.3e}")
            except Exception as exc:
                cells += [f"error: {type(exc).__name__}"] * len(refs[op])
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


# --- 5. device time -------------------------------------------------------------------------------


def _measure(call):
    try:
        return time_ms(call)
    except Exception as exc:
        return f"cannot launch: {type(exc).__name__}"


def _best(make_call, configs):
    """(timing, config) of the fastest config; configs that cannot launch are skipped."""
    best = None
    for cfg in configs:
        result = _measure(make_call(cfg))
        if isinstance(result, str):
            continue
        if best is None or result[0] < best[0][0]:
            best = (result, cfg)
    return best or ("cannot launch", None)


def _cell(result):
    if isinstance(result, str):
        return result
    return f"{result[0]:.4f} ({result[1]:.4f}-{result[2]:.4f})"


def _ratio(result, base):
    if isinstance(result, str) or isinstance(base, str):
        return "n/a"
    return f"{result[0] / base[0]:.2f}"


def section_perf(pre_w, head_w):
    run = ours(pre_w, head_w)
    base_run = baseline(pre_w, head_w)
    candidates = {"pre": PERF_PRE_CONFIGS, "head": PERF_PRE_CONFIGS, "post": PERF_POST_CONFIGS}
    lines = []
    for op in ("pre", "post", "head"):
        lines += [
            f"## 5{'abc'[('pre', 'post', 'head').index(op)]}. Device time, mHC {op}",
            "",
            f"Forward of mHC {op}, DeepSeek-V4-Flash shapes (n=4, C=4096, bf16 residual, fp32 "
            "weights, 20 Sinkhorn rounds), ms per call from GPU events around a CUDA graph; median "
            "(min-max) over replays. Baseline: the aiter op with its config table. Ours: the "
            f"default config, and the fastest of {len(candidates[op])} launch configs "
            "(same bits, section 1). Ratios: ours / baseline.",
            "",
            "| M | baseline (ms) | ours, default config (ms) | ours, fastest config (ms) "
            "| fastest config | ours default / baseline (ratio) "
            "| ours fastest / baseline (ratio) |",
            "|---|---|---|---|---|---|---|",
        ]
        for m in PERF_M:
            inputs = make_inputs(op, m, seed=3)
            base = _measure(lambda: base_run[op](inputs))
            default = _measure(lambda: run[op](inputs))
            fastest, cfg = _best(lambda c: lambda: run[op](inputs, c), candidates[op])
            label = " ".join(f"{k}={v}" for k, v in (cfg or {}).items())
            lines.append(
                f"| {m} | {_cell(base)} | {_cell(default)} | {_cell(fastest)} | {label} | "
                f"{_ratio(default, base)} | {_ratio(fastest, base)} |"
            )
            del inputs
            torch.cuda.empty_cache()
        lines.append("")
    return lines


# --- 6. digests for comparing Triton versions ----------------------------------------------------


def digests(pre_w, head_w):
    run = ours(pre_w, head_w)
    result = {}
    for op in ("pre", "post", "head"):
        for m in (17, 1024):
            for name, out in zip(OUTPUT_NAMES[op], run[op](make_inputs(op, m, seed=5))):
                result[f"{op} {name}, M={m}"] = hashlib.sha256(
                    _bits(out).cpu().numpy().tobytes()
                ).hexdigest()
    return result


def header():
    return [
        "# Batch-invariant Triton mHC on ROCm",
        "",
        f"Host `{socket.gethostname()}`, {torch.cuda.get_device_name()}, "
        f"torch {torch.__version__}, triton {triton.__version__}. "
        f"n={N}, C={C}, K={K}, 20 Sinkhorn rounds.",
        "",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", help="markdown report, rewritten after every section")
    parser.add_argument("--sections", default="1,2,3,4,5", help="comma-separated section numbers")
    parser.add_argument("--digest", help="write output digests (JSON) to this file and exit")
    parser.add_argument("--compare", help="with --digest: also compare with this digest file")
    args = parser.parse_args()

    pre_w, head_w = pre_weights(), head_weights()
    if args.digest:
        result = digests(pre_w, head_w)
        Path(args.digest).write_text(
            json.dumps({"triton": triton.__version__, "digests": result}, indent=1)
        )
        if args.compare:
            other = json.loads(Path(args.compare).read_text())
            print(f"| output | triton {other['triton']} vs {triton.__version__} |\n|---|---|")
            for key, value in result.items():
                print(f"| {key} | {'same' if other['digests'].get(key) == value else 'differs'} |")
        return

    lines = header()

    def flush():
        if args.output:
            Path(args.output).write_text("\n".join(lines) + "\n")

    sections = {
        "1": lambda: section_launch_invariance(pre_w, head_w),
        "2": lambda: section_batch_invariance(pre_w, head_w),
        "3": lambda: section_dynamic_shape(pre_w, head_w),
        "4": lambda: section_accuracy(pre_w, head_w),
        "5": lambda: section_perf(pre_w, head_w),
    }
    for key in args.sections.split(","):
        try:
            lines += sections[key]() + [""]
        except Exception:
            lines += [f"## section {key} failed", "", "```", traceback.format_exc(), "```", ""]
        print("\n".join(lines[-3:]), flush=True)
        flush()
    print("\n".join(lines))
    flush()


if __name__ == "__main__":
    main()
