# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""aiter's HIP mHC pre (``aiter.ops.mhc.mhc_pre``), the op vLLM's ROCm DeepSeek-V4 model
calls for mHC pre and, with a zero-padded fn and no Sinkhorn, for mHC head
(vllm/_aiter_ops.py ``rocm_aiter_ops.mhc_pre`` / ``hc_head``): candidate E of
docs/operators/mhc-reduction-order.md.

``mhc_pre`` picks the number of K splits and tile_k per M (``get_mhc_pre_splitk``)
and then runs ``mhc_pre_gemm_sqrsum`` and ``mhc_pre_big_fuse``. "fixed" variants
call the two kernels with one (split_k, tile_k) for every M. Sections:

1. the (split_k, tile_k) that ``get_mhc_pre_splitk`` picks per M on this GPU;
2. batch invariance (rl_engine/validation/common/batch_invariance.py) of pre and head;
3. error against the fp32 PyTorch reference, same inputs as
   evaluate_mhc_candidates.py section 3;
4. device time per M, same method and inputs as evaluate_mhc_candidates.py section 4.

Nothing is asserted: the script prints a markdown report. Needs an aiter whose
``module_mhc`` is built or can be JIT-built (no AITER_TRITON_ONLY). With
AITER_AOT_IMPORT=1, ``import aiter`` loads only its JIT core (the full import
needs flydsl >= 0.2.4, which the image lacks); ``aiter.dtypes`` is then bound
here before ``aiter.ops.mhc`` is imported.

Usage:
    python benchmarks/operators/mhc/probe_aiter_hip_mhc_pre.py --output report.md
"""

from __future__ import annotations

import argparse
import socket
import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.operators.mhc.evaluate_mhc_candidates import (  # noqa: E402
    BI_M,
    PERF_M,
    head_weights,
    pre_weights,
    residual,
)
from benchmarks.tuning.tune_mhc_head_config import time_ms  # noqa: E402
from rl_engine.validation.common import batch_invariance as bi  # noqa: E402

DEV = "cuda"
N, C = 4, 4096
K = N * C
ROWS = 2 * N + N * N
SPLIT_M = [1, 16, 17, 64, 65, 128, 256, 1024, 2048, 4096, 8192, 16384]
FIXED = [(1, 128), (4, 128), (8, 64), (64, 128)]
EPS = 1e-6


def _import_mhc():
    import aiter

    if not hasattr(aiter, "dtypes"):
        from aiter.utility import dtypes

        aiter.dtypes = dtypes
    from aiter.ops import mhc

    return aiter, mhc


def pre_fixed(mhc, r, fn, scale, base, split_k, tile_k, sinkhorn=20, post_mult=2.0):
    m = r.shape[0]
    out_pad = torch.empty(split_k, m, (ROWS + 31) // 32 * 32, dtype=torch.float32, device=DEV)
    out = out_pad[:, :, :ROWS]
    sqrsum = torch.empty(split_k, m, dtype=torch.float32, device=DEV)
    mhc.mhc_pre_gemm_sqrsum(out, sqrsum, r, fn, tile_k)
    post = torch.empty(m, N, 1, dtype=torch.float32, device=DEV)
    comb = torch.empty(m, N, N, dtype=torch.float32, device=DEV)
    li = torch.empty(m, C, dtype=torch.bfloat16, device=DEV)
    mhc.mhc_pre_big_fuse(
        post, comb, li, out, sqrsum, scale, base, r, EPS, EPS, EPS, post_mult, sinkhorn
    )
    return post, comb, li


def head_padded(fn, scale, base):
    """hc_head as vLLM runs it through mhc_pre: fn and base padded to 24 rows, scale to 3."""
    full_fn = torch.zeros(ROWS, K, dtype=fn.dtype, device=DEV)
    full_fn[:N] = fn
    full_base = torch.zeros(ROWS, dtype=base.dtype, device=DEV)
    full_base[:N] = base
    full_scale = torch.zeros(3, dtype=scale.dtype, device=DEV)
    full_scale[0] = scale[0]
    return full_fn, full_scale, full_base


def variants(mhc, pre_w, head_w):
    """{name: (pre(r) -> [post, comb, li], head(r) -> [out])}."""
    head_full = head_padded(*head_w)

    def flat(out):
        post, comb, li = out
        return [post.reshape(post.shape[0], N), comb.reshape(comb.shape[0], N, N), li]

    table = {
        "E aiter split": (
            lambda r: flat(mhc.mhc_pre(r, *pre_w, hc_post_mult_value=2.0, sinkhorn_repeat=20)),
            lambda r: [
                mhc.mhc_pre(r, *head_full, EPS, EPS, 0.0, 1.0, 0)[2],
            ],
        )
    }
    for s, t in FIXED:
        table[f"E fixed {s}x{t}"] = (
            lambda r, s=s, t=t: flat(pre_fixed(mhc, r, *pre_w, s, t)),
            lambda r, s=s, t=t: [pre_fixed(mhc, r, *head_full, s, t, 0, 1.0)[2]],
        )
    return table


def section_splits(mhc):
    lines = ["## 1. Split per M", "", "| M | get_mhc_pre_splitk (split_k, tile_k) |", "|---|---|"]
    lines += [f"| {m} | {mhc.get_mhc_pre_splitk(m, K)} |" for m in SPLIT_M]
    return lines


def section_bi(table):
    lines = [
        "## 2. Batch invariance",
        "",
        f"Outputs whose bits differ. M = {BI_M}; chunk/reorder/pad, strides and runs at "
        "M = 17 and 1024.",
        "",
        "| variant | op | batch (M with a difference) | chunk/reorder/pad | strides "
        "| repeated runs |",
        "|---|---|---|---|---|---|",
    ]
    for name, (pre, head) in table.items():
        for op, outputs, fn in (
            ("pre", ["post_mix", "comb_mix", "layer_input"], pre),
            ("head", ["output"], head),
        ):
            case = bi.RowwiseCase(
                name=f"{name} {op}",
                output_names=outputs,
                run=lambda rows, fn=fn: fn(rows[0]),
                make_rows=lambda m, seed: [residual(m, seed=1000 + seed)],
            )
            cells = []
            for check, ms in (
                (bi.differing_in_batch, BI_M),
                (bi.differing_after_chunk_reorder_pad, (17, 1024)),
                (bi.differing_after_restride, (17, 1024)),
                (bi.differing_between_runs, (17, 1024)),
            ):
                try:
                    found = {m: check(case, m) for m in ms}
                    if check is bi.differing_in_batch:
                        cell = "; ".join(f"M={m}: {','.join(d)}" for m, d in found.items() if d)
                    else:
                        cell = ",".join(sorted({n for d in found.values() for n in d}))
                    cells.append(cell or "none")
                except Exception as exc:
                    cells.append(f"error: {type(exc).__name__}: {str(exc)[:100]}")
            lines.append(f"| {name} | {op} | " + " | ".join(cells) + " |")
    return lines


def section_accuracy(table, pre_w, head_w):
    from rl_engine.reference.mhc.mhc import NativeMHCHeadOp, NativeMHCPreOp

    r = residual(1024, seed=7)
    post, comb, li = NativeMHCPreOp().forward_fp32(r, *pre_w)
    ref_pre = [post, comb, li]
    ref_head = NativeMHCHeadOp().forward_fp32(r, *head_w)
    lines = [
        "## 3. Error against the fp32 reference",
        "",
        "M = 1024, the inputs of evaluate_mhc_candidates.py section 3. Max / mean absolute error.",
        "",
        "| variant | post_mix | comb_mix | layer_input | head output |",
        "|---|---|---|---|---|",
    ]
    for name, (pre, head) in table.items():
        cells = []
        for out, ref in zip(pre(r), ref_pre):
            err = (out.float() - ref.reshape(out.shape).float()).abs()
            cells.append(f"{err.max().item():.3e} / {err.mean().item():.3e}")
        err = (head(r)[0].float() - ref_head.float()).abs()
        cells.append(f"{err.max().item():.3e} / {err.mean().item():.3e}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def _cell(fn):
    try:
        med, low, high = time_ms(fn)
        return f"{med:.4f} ({low:.4f}-{high:.4f})"
    except Exception as exc:
        return f"cannot launch: {type(exc).__name__}"


def section_perf(table):
    lines = []
    for op, idx in (("pre", 0), ("head", 1)):
        names = list(table)
        lines += [
            f"## 4{'a' if op == 'pre' else 'b'}. Device time, mhc_{op}",
            "",
            "ms per call from GPU events around a CUDA graph; median (min-max) over replays.",
            "",
            "| M | " + " | ".join(f"{n} (ms)" for n in names) + " |",
            "|---|" + "---|" * len(names),
        ]
        for m in PERF_M:
            r = residual(m, seed=3)
            cells = [_cell(lambda r=r, f=table[n][idx]: f(r)) for n in names]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
            del r
            torch.cuda.empty_cache()
        lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output")
    args = parser.parse_args()
    lines = [
        "# aiter HIP mhc_pre (candidate E)",
        "",
        f"Host `{socket.gethostname()}`, {torch.cuda.get_device_name()}, "
        f"torch {torch.__version__}.",
        "",
    ]

    def flush():
        if args.output:
            Path(args.output).write_text("\n".join(lines) + "\n")

    try:
        aiter, mhc = _import_mhc()
        lines += [f"aiter: `{Path(aiter.__file__).parent}`", ""]
        pre_w, head_w = pre_weights(), head_weights()
        table = variants(mhc, pre_w, head_w)
        for section in (
            lambda: section_splits(mhc),
            lambda: section_bi(table),
            lambda: section_accuracy(table, pre_w, head_w),
            lambda: section_perf(table),
        ):
            try:
                lines += section() + [""]
            except Exception:
                lines += ["```", traceback.format_exc(), "```", ""]
            flush()
    except Exception:
        lines += ["```", traceback.format_exc(), "```"]
    print("\n".join(lines))
    flush()


if __name__ == "__main__":
    main()
