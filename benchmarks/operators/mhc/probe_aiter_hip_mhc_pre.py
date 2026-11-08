# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch invariance of aiter's HIP mHC pre (``aiter.ops.mhc.mhc_pre``), the op vLLM's ROCm
DeepSeek-V4 model calls (vllm/_aiter_ops.py ``rocm_aiter_ops.mhc_pre``).

``mhc_pre`` picks the number of K splits and tile_k per M (``get_mhc_pre_splitk``)
and then runs ``mhc_pre_gemm_sqrsum`` and ``mhc_pre_big_fuse``. This script

1. prints the (split_k, tile_k) that ``get_mhc_pre_splitk`` picks per M on this GPU;
2. compares two rows computed alone (M=1) with the same rows at the first and
   last position of a batch of M, bit for bit, for ``mhc_pre`` as is and for the
   two kernels called with one fixed (split_k, tile_k) for every M.

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
import traceback
from pathlib import Path

import torch

DEV = "cuda"
N, C = 4, 4096
K = N * C
ROWS = 2 * N + N * N
M_VALUES = [1, 16, 17, 64, 65, 128, 256, 1024, 2048, 4096, 8192, 16384]
FIXED = [(1, 128), (4, 128), (8, 64)]


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def residual(m, seed):
    return torch.randn(m, N, C, generator=_gen(seed), device=DEV).to(torch.bfloat16)


def weights():
    gen = _gen(0)
    fn = torch.randn(ROWS, K, generator=gen, device=DEV) / K**0.5
    scale = 0.5 + torch.rand(3, generator=gen, device=DEV)
    base = 0.1 * torch.randn(ROWS, generator=gen, device=DEV)
    return fn, scale, base


def run_fixed(mhc, r, fn, scale, base, split_k, tile_k):
    m = r.shape[0]
    out_pad = torch.empty(split_k, m, (ROWS + 31) // 32 * 32, dtype=torch.float32, device=DEV)
    out = out_pad[:, :, :ROWS]
    sqrsum = torch.empty(split_k, m, dtype=torch.float32, device=DEV)
    mhc.mhc_pre_gemm_sqrsum(out, sqrsum, r, fn, tile_k)
    post = torch.empty(m, N, 1, dtype=torch.float32, device=DEV)
    comb = torch.empty(m, N, N, dtype=torch.float32, device=DEV)
    li = torch.empty(m, C, dtype=torch.bfloat16, device=DEV)
    mhc.mhc_pre_big_fuse(post, comb, li, out, sqrsum, scale, base, r, 1e-6, 1e-6, 1e-6, 2.0, 20)
    return post, comb, li


def _bits(t):
    return t.contiguous().view({torch.float32: torch.int32, torch.bfloat16: torch.int16}[t.dtype])


def differing(run, m):
    checked = residual(2, 1)
    alone = [run(checked[i : i + 1]) for i in range(2)]
    batch = residual(m, 100 + m)
    positions = [0] if m == 1 else [0, m - 1]
    for i, p in enumerate(positions):
        batch[p] = checked[i]
    outs = run(batch)
    names = []
    for i, p in enumerate(positions):
        for name, a, b in zip(("post_mix", "comb_mix", "layer_input"), outs, alone[i]):
            if not torch.equal(_bits(a[p : p + 1]), _bits(b)) and name not in names:
                names.append(name)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output")
    args = parser.parse_args()
    lines = [
        "# aiter HIP mhc_pre: batch invariance",
        "",
        f"Host `{socket.gethostname()}`, {torch.cuda.get_device_name()}, "
        f"torch {torch.__version__}.",
        "",
    ]
    try:
        import aiter

        if not hasattr(aiter, "dtypes"):
            from aiter.utility import dtypes

            aiter.dtypes = dtypes
        from aiter.ops import mhc

        lines += [f"aiter: `{Path(aiter.__file__).parent}`", ""]
        fn, scale, base = weights()
        lines += ["| M | get_mhc_pre_splitk (split_k, tile_k) |", "|---|---|"]
        for m in M_VALUES:
            lines.append(f"| {m} | {mhc.get_mhc_pre_splitk(m, K)} |")
        fixed_cols = " | ".join(f"fixed split_k={s}, tile_k={t}" for s, t in FIXED)
        lines += ["", f"| M | mhc_pre as is | {fixed_cols} |", "|---|---|" + "---|" * len(FIXED)]
        runs = [
            lambda r: mhc.mhc_pre(r, fn, scale, base, hc_post_mult_value=2.0, sinkhorn_repeat=20)
        ]
        runs += [lambda r, s=s, t=t: run_fixed(mhc, r, fn, scale, base, s, t) for s, t in FIXED]
        for m in M_VALUES:
            cells = []
            for run in runs:
                try:
                    cells.append(", ".join(differing(run, m)) or "same")
                except Exception as exc:
                    cells.append(f"error: {type(exc).__name__}: {str(exc)[:120]}")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    except Exception:
        lines += ["```", traceback.format_exc(), "```"]
    report = "\n".join(lines)
    print(report)
    if args.output:
        Path(args.output).write_text(report + "\n")


if __name__ == "__main__":
    main()
