# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Evaluate the candidates for batch-invariant mHC pre and head on ROCm.

Candidates (docs/operators/mhc-reduction-order.md):

- A: the current ops, aiter's kernels with launch configs whose bits are equal
  for every M (``AiterMHCPreOp`` / ``AiterMHCHeadOp``); baseline of every table;
- B: ``fixed_order_mhc_pre`` / ``fixed_order_mhc_head``, Triton kernels with every
  reduction order written in the kernel; ``S`` is the number of K splits, a
  constant of the kernel (K = 16384: the default rule gives S = 4);
- D: ``fixed_order_mhc_pre(projection="aiter_linear")``: projection by aiter's
  gemm_a16w16 with the batch-invariant linear's config, sum of squares by B's
  kernel (pre only);
- aiter default: the aiter ops with their own config choice (not batch
  invariant), for reference.

Sections, each written to the report as soon as it is done:

1. launch-parameter invariance of B: every launch parameter changed one at a
   time (and a few together) against B's default config, all outputs compared
   bit for bit at several M;
2. batch invariance of A, B and D with rl_engine/validation/common/batch_invariance.py
   (a row alone vs in batches, chunking/reordering/padding, strides, repeated runs);
   B also with a launch config that changes with M;
3. error against the fp32 PyTorch reference (``forward_fp32``), same inputs for all;
4. device time per M from GPU events around a CUDA graph of ITERS calls; median
   and min-max over TRIALS replays; for B and D the fastest of a set of launch
   configs that section 1 shows give the same bits;
5. ``--digest``/``--compare``: SHA-256 of the outputs of fixed configs, to compare
   two Triton versions.

Usage (GPU; aiter with the DeepSeek-V4 mHC ops and AITER_TRITON_ONLY=1):
    python benchmarks/operators/mhc/evaluate_mhc_candidates.py --output report.md
    python benchmarks/operators/mhc/evaluate_mhc_candidates.py --digest d.json
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
from rl_engine.backends.rocm.mhc.fixed_order_mhc import (  # noqa: E402
    DEFAULT_HEAD_CONFIG,
    DEFAULT_PRE_CONFIG,
    fixed_order_mhc_head,
    fixed_order_mhc_pre,
)
from rl_engine.validation.common import batch_invariance as bi  # noqa: E402

DEV = "cuda"
N, C = 4, 4096
K = N * C
PRE_ROWS = 2 * N + N * N
EPS = 1e-6
INVARIANCE_M = [1, 17, 1024, 4096]
BI_M = [1, 8, 16, 17, 59, 64, 128, 256, 1024, 2048, 4096, 8192, 16384]
PERF_M = [1, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
SPLITS = [1, 4, 16]

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
PERF_PRENORM = [
    {"BLOCK_M": bm, "num_warps": w} for bm, w in itertools.product([16, 32, 64], [2, 4, 8])
]
PERF_APPLY = [{"APPLY_BLOCK_M": 32, "APPLY_BLOCK_C": 128}, {"APPLY_BLOCK_M": 64, "APPLY_BLOCK_C": 64}]


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


# --- the variants as functions of residual -----------------------------------------------------


def variants(pre_w, head_w):
    """{name: (pre(residual) -> outputs, head(residual) -> outputs or None)}."""
    from aiter.ops.triton.fusions import mhc_head_dsv4, mhc_pre_dsv4

    from rl_engine.backends.rocm.mhc.aiter_mhc import AiterMHCHeadOp, AiterMHCPreOp

    a_pre, a_head = AiterMHCPreOp(), AiterMHCHeadOp()

    def flat_pre(out):
        post, comb, li = out
        return [post.reshape(post.shape[0], N), comb.reshape(comb.shape[0], N, N), li]

    table = {
        "aiter default": (
            lambda r: flat_pre(mhc_pre_dsv4(r, *pre_w, sinkhorn_repeat=20)),
            lambda r: [mhc_head_dsv4(r, *head_w)],
        ),
        "A": (lambda r: flat_pre(a_pre(r, *pre_w)), lambda r: [a_head(r, *head_w)]),
    }
    for s in SPLITS:
        table[f"B S={s}"] = (
            lambda r, s=s, cfg=None: flat_pre(fixed_order_mhc_pre(r, *pre_w, k_splits=s, config=cfg)),
            lambda r, s=s, cfg=None: [fixed_order_mhc_head(r, *head_w, k_splits=s, config=cfg)],
        )
    table["D S=4"] = (
        lambda r, cfg=None: flat_pre(
            fixed_order_mhc_pre(r, *pre_w, k_splits=4, config=cfg, projection="aiter_linear")
        ),
        None,
    )
    return table


def _bits(t):
    view = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[t.dtype]
    return t.contiguous().view(view)


def _ndiff(a, b):
    return int((_bits(a) != _bits(b)).sum())


# --- 1. launch-parameter invariance of B --------------------------------------------------------


def section_launch_invariance(pre_w, head_w):
    lines = [
        "## 1. Launch-parameter invariance of B",
        "",
        "Each row changes the listed launch parameters of B's default config "
        f"(pre `{DEFAULT_PRE_CONFIG}`, head `{DEFAULT_HEAD_CONFIG}`) and compares every "
        f"output bit for bit with the default config, random inputs, M = {INVARIANCE_M}.",
        "",
        "| op | S | changed | result |",
        "|---|---|---|---|",
    ]
    inputs = {m: residual(m, seed=10 + m) for m in INVARIANCE_M}
    for op, run, launch_variants in (
        ("pre", lambda r, s, cfg: list(fixed_order_mhc_pre(r, *pre_w, k_splits=s, config=cfg)), PRE_LAUNCH_VARIANTS),
        ("head", lambda r, s, cfg: [fixed_order_mhc_head(r, *head_w, k_splits=s, config=cfg)], HEAD_LAUNCH_VARIANTS),
    ):
        for s in SPLITS:
            ref = {m: run(inputs[m], s, None) for m in INVARIANCE_M}
            for change in launch_variants:
                try:
                    diffs = []
                    for m in INVARIANCE_M:
                        out = run(inputs[m], s, change)
                        for idx, (a, b) in enumerate(zip(out, ref[m])):
                            if _ndiff(a, b):
                                diffs.append(f"output {idx}: {_ndiff(a, b)} at M={m}")
                    result = "differs: " + "; ".join(diffs) if diffs else "same"
                except Exception as exc:
                    result = f"cannot launch: {type(exc).__name__}: {exc}"[:200]
                label = " ".join(f"{k}={v}" for k, v in change.items())
                lines.append(f"| {op} | {s} | {label} | {result} |")
    return lines


# --- 2. batch invariance --------------------------------------------------------------------------


def _case(name, outputs, fn):
    return bi.RowwiseCase(
        name=name,
        output_names=outputs,
        run=lambda rows: fn(rows[0]),
        make_rows=lambda m, seed: [residual(m, seed=1000 + seed)],
    )


def _m_dependent_config(m):
    """A launch config that changes with M, to show that B does not depend on it."""
    if m < 64:
        return {"BLOCK_M": 16, "num_warps": 8, "APPLY_BLOCK_C": 256, "SINKHORN_BLOCK_M": 1}
    return {"BLOCK_M": 64, "num_warps": 4, "APPLY_BLOCK_C": 64, "SINKHORN_BLOCK_M": 64}


def section_batch_invariance(table, pre_w, head_w):
    lines = [
        "## 2. Batch invariance",
        "",
        "Cells list the outputs whose bits differ (empty: none). Checks of "
        "rl_engine/validation/common/batch_invariance.py: a row alone vs at the first and "
        "last position of a batch of M; chunking, reordering and padding; other strides; "
        f"two runs. M = {BI_M}.",
        "",
        "| variant | op | batch (M with a difference) | chunk/reorder/pad | strides | repeated runs |",
        "|---|---|---|---|---|---|",
    ]
    pre_out, head_out = ["post_mix", "comb_mix", "layer_input"], ["output"]
    runners = []
    for name in ("A", "B S=1", "B S=4", "B S=16", "D S=4"):
        pre, head = table[name]
        runners.append((name, "pre", pre_out, pre))
        if head is not None:
            runners.append((name, "head", head_out, head))
    runners.append(
        (
            "B S=4, config by M",
            "pre",
            pre_out,
            lambda r: [
                t.reshape(t.shape[0], -1) if t.dim() == 3 and t.shape[-1] == 1 else t
                for t in fixed_order_mhc_pre(r, *pre_w, k_splits=4, config=_m_dependent_config(r.shape[0]))
            ],
        )
    )
    runners.append(
        (
            "B S=4, config by M",
            "head",
            head_out,
            lambda r: [
                fixed_order_mhc_head(
                    r,
                    *head_w,
                    k_splits=4,
                    config={k: v for k, v in _m_dependent_config(r.shape[0]).items() if k != "SINKHORN_BLOCK_M"},
                )
            ],
        )
    )
    for name, op, outputs, fn in runners:
        case = _case(f"{name} {op}", outputs, fn)
        try:
            batch = {m: bi.differing_in_batch(case, m) for m in BI_M}
            batch_cell = "; ".join(f"M={m}: {','.join(d)}" for m, d in batch.items() if d) or "none"
            chunk = sorted({n for m in (17, 1024) for n in bi.differing_after_chunk_reorder_pad(case, m)})
            stride = sorted({n for m in (17, 1024) for n in bi.differing_after_restride(case, m)})
            runs = sorted({n for m in (17, 1024) for n in bi.differing_between_runs(case, m)})
            lines.append(
                f"| {name} | {op} | {batch_cell} | {','.join(chunk) or 'none'} | "
                f"{','.join(stride) or 'none'} | {','.join(runs) or 'none'} |"
            )
        except Exception as exc:
            lines.append(f"| {name} | {op} | error: {type(exc).__name__}: {str(exc)[:150]} | | | |")
    return lines


# --- 3. error against the fp32 reference ---------------------------------------------------------


def section_accuracy(table, pre_w, head_w):
    from rl_engine.reference.mhc.mhc import NativeMHCHeadOp, NativeMHCPreOp

    m = 1024
    r = residual(m, seed=7)
    post, comb, li = NativeMHCPreOp().forward_fp32(r, *pre_w)
    ref_pre = [post, comb, li]
    ref_head = [NativeMHCHeadOp().forward_fp32(r, *head_w)]
    lines = [
        "## 3. Error against the fp32 reference",
        "",
        f"M = {m}, random inputs. Reference: `NativeMHCPreOp.forward_fp32` / "
        "`NativeMHCHeadOp.forward_fp32` (fp32 projection with fp32 fn). Max and mean absolute "
        "error per output.",
        "",
        "| variant | post_mix max / mean | comb_mix max / mean | layer_input max / mean | head output max / mean |",
        "|---|---|---|---|---|",
    ]
    for name, (pre, head) in table.items():
        cells = []
        try:
            for out, ref in zip(pre(r), ref_pre):
                err = (out.float() - ref.reshape(out.shape).float()).abs()
                cells.append(f"{err.max().item():.3e} / {err.mean().item():.3e}")
        except Exception as exc:
            cells += [f"error: {type(exc).__name__}"] * (3 - len(cells))
        if head is None:
            cells.append("-")
        else:
            try:
                err = (head(r)[0].float() - ref_head[0].float()).abs()
                cells.append(f"{err.max().item():.3e} / {err.mean().item():.3e}")
            except Exception as exc:
                cells.append(f"error: {type(exc).__name__}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


# --- 4. device time --------------------------------------------------------------------------------


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


def _ratio(result, baseline):
    if isinstance(result, str) or isinstance(baseline, str):
        return "n/a"
    return f"{result[0] / baseline[0]:.2f}"


def section_perf(pre_w, head_w):
    from aiter.ops.triton.fusions import mhc_head_dsv4, mhc_pre_dsv4

    from rl_engine.backends.rocm.mhc.aiter_mhc import AiterMHCHeadOp, AiterMHCPreOp

    a_pre, a_head = AiterMHCPreOp(), AiterMHCHeadOp()
    b_configs = [dict(p, **a) for p, a in itertools.product(PERF_PRENORM, PERF_APPLY)]
    lines = []
    for op in ("pre", "head"):
        names = ["aiter default", "A"] + [f"B S={s}" for s in SPLITS] + (["D S=4"] if op == "pre" else [])
        lines += [
            f"## 4{'a' if op == 'pre' else 'b'}. Device time, mhc_{op}",
            "",
            f"Forward of mhc_{op}, DeepSeek-V4-Flash shapes (n=4, C=4096, bf16 residual), "
            "ms per call from GPU events around a CUDA graph; median (min-max) over replays. "
            "Baseline: A (the current op). Ratio columns: variant time / A time. For B and D "
            f"the fastest of {len(b_configs)} launch configs (BLOCK_M x num_warps x apply tile), "
            "which give the same bits (section 1).",
            "",
            "| M | " + " | ".join(f"{n} (ms)" for n in names) + " | "
            + " | ".join(f"{n} / A (ratio)" for n in names if n != "A") + " |",
            "|---|" + "---|" * (2 * len(names) - 1),
        ]
        for m in PERF_M:
            r = residual(m, seed=3)
            results = {}
            if op == "pre":
                results["aiter default"] = _measure(lambda: mhc_pre_dsv4(r, *pre_w, sinkhorn_repeat=20))
                results["A"] = _measure(lambda: a_pre(r, *pre_w))
                for s in SPLITS:
                    results[f"B S={s}"], _ = _best(
                        lambda cfg, s=s: lambda: fixed_order_mhc_pre(r, *pre_w, k_splits=s, config=cfg),
                        b_configs,
                    )
                results["D S=4"], _ = _best(
                    lambda cfg: lambda: fixed_order_mhc_pre(
                        r, *pre_w, k_splits=4, config=cfg, projection="aiter_linear"
                    ),
                    b_configs,
                )
            else:
                results["aiter default"] = _measure(lambda: mhc_head_dsv4(r, *head_w))
                results["A"] = _measure(lambda: a_head(r, *head_w))
                for s in SPLITS:
                    results[f"B S={s}"], _ = _best(
                        lambda cfg, s=s: lambda: fixed_order_mhc_head(r, *head_w, k_splits=s, config=cfg),
                        b_configs,
                    )
            base = results["A"]
            lines.append(
                f"| {m} | "
                + " | ".join(_cell(results[n]) for n in names)
                + " | "
                + " | ".join(_ratio(results[n], base) for n in names if n != "A")
                + " |"
            )
            del r
            torch.cuda.empty_cache()
        lines.append("")
    return lines


# --- 5. digests for comparing Triton versions -------------------------------------------------


def digests(pre_w, head_w):
    from aiter.ops.triton.fusions import mhc_pre_dsv4

    from benchmarks.tuning.tune_mhc_head_config import aiter_head_config, launch_mhc_head

    split4 = {"NUM_KSPLIT": 4, "BLOCK_M": 16, "BLOCK_K": 128, "BLOCK_C": 64, "num_warps": 4,
              "num_stages": 1, "waves_per_eu": 0}
    runs = {
        "A pre (table class, NUM_KSPLIT=4)": lambda r: list(
            mhc_pre_dsv4(r, *pre_w, sinkhorn_repeat=20, config=dict(split4))
        ),
        "A head (aiter config)": lambda r: [launch_mhc_head(r, *head_w, aiter_head_config())],
        "B pre S=4": lambda r: list(fixed_order_mhc_pre(r, *pre_w, k_splits=4)),
        "B head S=4": lambda r: [fixed_order_mhc_head(r, *head_w, k_splits=4)],
    }
    result = {}
    for name, run in runs.items():
        for m in (17, 1024):
            outs = run(residual(m, seed=5))
            for idx, out in enumerate(outs):
                result[f"{name}, M={m}, output {idx}"] = hashlib.sha256(
                    _bits(out).cpu().numpy().tobytes()
                ).hexdigest()
    return result


def header():
    return [
        "# Batch-invariant mHC candidates on ROCm",
        "",
        f"Host `{socket.gethostname()}`, {torch.cuda.get_device_name()}, torch {torch.__version__}, "
        f"triton {triton.__version__}. n={N}, C={C}, K={K}, 20 Sinkhorn rounds.",
        "",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", help="markdown report, rewritten after every section")
    parser.add_argument("--sections", default="1,2,3,4", help="comma-separated section numbers")
    parser.add_argument("--digest", help="write output digests (JSON) to this file and exit")
    parser.add_argument("--compare", help="with --digest: also compare with this digest file")
    args = parser.parse_args()

    pre_w, head_w = pre_weights(), head_weights()
    if args.digest:
        result = digests(pre_w, head_w)
        Path(args.digest).write_text(json.dumps({"triton": triton.__version__, "digests": result}, indent=1))
        if args.compare:
            other = json.loads(Path(args.compare).read_text())
            print(f"| output | triton {other['triton']} vs {triton.__version__} |\n|---|---|")
            for key, value in result.items():
                print(f"| {key} | {'same' if other['digests'].get(key) == value else 'differs'} |")
        return

    lines = header()

    def flush():
        report = "\n".join(lines)
        if args.output:
            Path(args.output).write_text(report + "\n")

    table = variants(pre_w, head_w)
    sections = {
        "1": lambda: section_launch_invariance(pre_w, head_w),
        "2": lambda: section_batch_invariance(table, pre_w, head_w),
        "3": lambda: section_accuracy(table, pre_w, head_w),
        "4": lambda: section_perf(pre_w, head_w),
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
