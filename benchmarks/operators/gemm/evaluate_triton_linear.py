# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Evaluate RL-Kernel's batch-invariant Triton linear (aiter's gemm_a16w16, copied) on ROCm.

- baseline: ``AiterBatchInvariantLinearOp``, aiter's gemm_a16w16 with the per-M
  config table, the op before this change;
- ours: ``TritonBatchInvariantLinearOp`` (rl_engine/backends/rocm/gemm/triton_gemm.py),
  with the launch config of ``CONFIGS`` for M.

Sections, each written to the report as soon as it is done:

1. launch-config invariance of ours: every entry of CONFIGS and a few other
   configs against the first entry, bit for bit, on several shapes and M;
2. batch invariance of ours with rl_engine/validation/common/batch_invariance.py
   on the lm_head and the WS1 shapes;
3. dynamic shape: compiled variants of the kernel after ``warmup`` and after many unseen M;
4. error of baseline and ours against an fp32 PyTorch linear;
5. lm_head device time per M from GPU events: baseline, ours, and every
   candidate config, to choose the CONFIGS buckets; ratios against the baseline;
6. ``--digest``/``--compare``: SHA-256 of the outputs of ours, to compare two
   Triton versions.

Usage (GPU; aiter with the Triton gemm_a16w16 and AITER_TRITON_ONLY=1, for the baseline):
    python benchmarks/operators/gemm/evaluate_triton_linear.py --output report.md
    python benchmarks/operators/gemm/evaluate_triton_linear.py --digest d.json
"""

from __future__ import annotations

import argparse
import hashlib
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

from benchmarks.operators.gemm.benchmark_lm_head_linear import _cell, _measure  # noqa: E402
from rl_engine.backends.rocm.gemm import triton_gemm  # noqa: E402
from rl_engine.validation.common import batch_invariance as bi  # noqa: E402

DEV = "cuda"
# name -> (N, K, output dtype, with bias)
SHAPES = {
    "lm_head": (129280, 4096, torch.float32, False),
    "mhc_pre_projection": (24, 16384, torch.bfloat16, False),
    "ws1_projection": (4096, 4096, torch.bfloat16, False),
    "ws1_gate_up": (12288, 4096, torch.bfloat16, False),
    "tail_with_bias": (1000, 4100, torch.float32, True),
}
INVARIANCE_M = [1, 17, 1024, 4096]
BI_M = [1, 8, 16, 17, 59, 64, 128, 129, 256, 1024, 1025, 2048, 4096, 8192, 16384]
# lm_head logits are 4.2 GB at M = 8192.
MAX_M = {"lm_head": 8192}
PERF_M = [1, 4, 16, 17, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
UNSEEN_M = [2, 3, 7, 31, 33, 100, 255, 257, 1000, 1023, 1025, 3000, 4097]

_BASE = {"GROUP_SIZE_M": 1, "num_stages": 2, "waves_per_eu": 0, "cache_modifier": ".cg"}
EXTRA_CONFIGS = [
    dict(_BASE, BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, num_warps=4),
    dict(_BASE, BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, num_warps=4),
    dict(_BASE, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, num_warps=4, num_stages=1),
    dict(_BASE, BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, num_warps=8, GROUP_SIZE_M=4),
    dict(_BASE, BLOCK_SIZE_M=256, BLOCK_SIZE_N=128, num_warps=8, GROUP_SIZE_M=8),
]
CANDIDATES = [config for _, config in triton_gemm.CONFIGS] + EXTRA_CONFIGS


def _label(config):
    keys = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "GROUP_SIZE_M", "num_warps", "num_stages")
    return "/".join(str(config[k]) for k in keys)


def weights(shape, seed=0):
    n, k, out_dtype, with_bias = SHAPES[shape]
    gen = torch.Generator(device=DEV).manual_seed(seed)
    weight = (torch.randn(n, k, generator=gen, device=DEV) / k**0.5).to(torch.bfloat16)
    bias = torch.randn(n, generator=gen, device=DEV).to(out_dtype) if with_bias else None
    return weight, bias, out_dtype


def rows(m, k, seed=0):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(m, k, generator=gen, device=DEV).to(torch.bfloat16)


def _bits(t):
    view = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[t.dtype]
    return t.contiguous().view(view)


def _ndiff(a, b):
    return int((_bits(a) != _bits(b)).sum())


def section_launch_invariance():
    lines = [
        "## 1. Launch-config invariance of ours",
        "",
        "Each config (BLOCK_SIZE_M/BLOCK_SIZE_N/GROUP_SIZE_M/num_warps/num_stages) against "
        f"the first entry of CONFIGS, every output element bit for bit, M = {INVARIANCE_M}. "
        f"BLOCK_K = {triton_gemm.BLOCK_K}, {triton_gemm.DOT_OPTIONS} in every config.",
        "",
        "| shape | config | result |",
        "|---|---|---|",
    ]
    for shape in SHAPES:
        weight, bias, out_dtype = weights(shape)
        inputs = {m: rows(m, weight.shape[1], seed=m) for m in INVARIANCE_M}
        ref = {
            m: triton_gemm.gemm_a16w16(inputs[m], weight, bias, out_dtype, CANDIDATES[0])
            for m in INVARIANCE_M
        }
        for config in CANDIDATES[1:]:
            try:
                diffs = []
                for m in INVARIANCE_M:
                    y = triton_gemm.gemm_a16w16(inputs[m], weight, bias, out_dtype, config)
                    if _ndiff(y, ref[m]):
                        diffs.append(f"{_ndiff(y, ref[m])} elements at M={m}")
                result = "differs: " + "; ".join(diffs) if diffs else "same"
            except Exception as exc:
                result = f"cannot launch: {type(exc).__name__}: {exc}"[:200]
            lines.append(f"| {shape} | {_label(config)} | {result} |")
        del ref
        torch.cuda.empty_cache()
    return lines


def section_batch_invariance():
    op = triton_gemm.TritonBatchInvariantLinearOp()
    lines = [
        "## 2. Batch invariance of ours",
        "",
        "Cells list where the bits differ (none: no difference). A row alone vs at the first "
        f"and last position of a batch of M = {BI_M} (lm_head up to 8192); chunking, "
        "reordering and padding; other strides; two runs, at M = 17 and 1024.",
        "",
        "| shape | batch (M with a difference) | chunk/reorder/pad | strides | repeated runs |",
        "|---|---|---|---|---|",
    ]
    for shape in SHAPES:
        weight, bias, out_dtype = weights(shape)
        case = bi.RowwiseCase(
            shape,
            ("y",),
            lambda r, w=weight, b=bias, d=out_dtype: [op(r[0], w, b, out_dtype=d)],
            lambda m, seed, k=weight.shape[1]: [rows(m, k, 1000 + seed)],
        )
        try:
            bad = []
            for m in BI_M:
                if m <= MAX_M.get(shape, m) and bi.differing_in_batch(case, m):
                    bad.append(m)
                torch.cuda.empty_cache()
            chunk = sorted(
                {n for m in (17, 1024) for n in bi.differing_after_chunk_reorder_pad(case, m)}
            )
            stride = sorted({n for m in (17, 1024) for n in bi.differing_after_restride(case, m)})
            runs = sorted({n for m in (17, 1024) for n in bi.differing_between_runs(case, m)})
            lines.append(
                f"| {shape} | {', '.join(map(str, bad)) or 'none'} | {','.join(chunk) or 'none'} "
                f"| {','.join(stride) or 'none'} | {','.join(runs) or 'none'} |"
            )
        except Exception as exc:
            lines.append(f"| {shape} | error: {type(exc).__name__}: {str(exc)[:150]} | | | |")
    return lines


def _variants():
    kernel = triton_gemm._gemm_a16_w16_kernel
    return sum(len(c[0]) for c in kernel.device_caches.values())


def section_dynamic_shape():
    op = triton_gemm.TritonBatchInvariantLinearOp()
    weight, _, out_dtype = weights("lm_head")
    before = _variants()
    op.warmup(weight, out_dtype=out_dtype)
    warm = _variants()
    for m in UNSEEN_M:
        op(rows(m, weight.shape[1], seed=m), weight, out_dtype=out_dtype)
    torch.cuda.synchronize()
    return [
        "## 3. Dynamic shape",
        "",
        "Compiled variants of `_gemm_a16_w16_kernel` for the lm_head shape: before any call, "
        f"after `warmup` ({len(triton_gemm.CONFIGS)} configs in CONFIGS), and after calls at "
        f"M = {UNSEEN_M}.",
        "",
        "| before (variants) | after warmup (variants) | after the unseen M (variants) |",
        "|---|---|---|",
        f"| {before} | {warm} | {_variants()} |",
    ]


def section_accuracy():
    from rl_engine.backends.rocm.gemm.aiter_gemm import AiterBatchInvariantLinearOp

    variants = {
        "baseline: aiter gemm_a16w16 with config table": AiterBatchInvariantLinearOp(),
        "ours": triton_gemm.TritonBatchInvariantLinearOp(),
    }
    lines = [
        "## 4. Error against an fp32 linear",
        "",
        "M = 1024 (lm_head: 256), random inputs; reference: x.float() @ w.float().T (+ bias). "
        "Largest / mean absolute error.",
        "",
        "| variant | " + " | ".join(SHAPES) + " |",
        "|---|" + "---|" * len(SHAPES),
    ]
    for name, op in variants.items():
        cells = []
        for shape in SHAPES:
            weight, bias, out_dtype = weights(shape)
            x = rows(256 if shape == "lm_head" else 1024, weight.shape[1], seed=7)
            ref = x.float() @ weight.float().t() + (0 if bias is None else bias.float())
            try:
                err = (op(x, weight, bias, out_dtype=out_dtype).float() - ref).abs()
                cells.append(f"{err.max().item():.3e} / {err.mean().item():.3e}")
            except Exception as exc:
                cells.append(f"error: {type(exc).__name__}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def section_perf():
    from rl_engine.backends.rocm.gemm.aiter_gemm import AiterBatchInvariantLinearOp

    baseline_op = AiterBatchInvariantLinearOp()
    ours_op = triton_gemm.TritonBatchInvariantLinearOp()
    weight, _, out_dtype = weights("lm_head")
    labels = [_label(c) for c in CANDIDATES]
    lines = [
        "## 5. Device time, lm_head",
        "",
        "x [M, 4096] bf16 times weight [129280, 4096] bf16, fp32 logits; ms per call from GPU "
        "events, median (min-max) over trials. Baseline: aiter gemm_a16w16 with the config "
        "table. Ours: the CONFIGS entry for M. Candidate columns: ours with that config "
        "(BLOCK_SIZE_M/BLOCK_SIZE_N/GROUP_SIZE_M/num_warps/num_stages), all with the same bits "
        "(section 1). Ratios: ours / baseline.",
        "",
        "| M | baseline (ms) | ours (ms) | ours / baseline (ratio) | fastest candidate "
        "| fastest / baseline (ratio) | "
        + " | ".join(f"{label} (ms)" for label in labels)
        + " |",
        "|---|---|---|---|---|---|" + "---|" * len(labels),
    ]
    for m in PERF_M:
        x = rows(m, weight.shape[1], seed=3)
        with torch.no_grad():
            base = _measure(lambda: baseline_op(x, weight, out_dtype=out_dtype))
            ours = _measure(lambda: ours_op(x, weight, out_dtype=out_dtype))
            cands = [
                _measure(lambda c=c: triton_gemm.gemm_a16w16(x, weight, None, out_dtype, c))
                for c in CANDIDATES
            ]
        timed = [(r[0], label) for r, label in zip(cands, labels) if not isinstance(r, str)]
        best = min(timed) if timed else None

        def ratio(t):
            return "n/a" if isinstance(base, str) or t is None else f"{t / base[0]:.2f}"

        lines.append(
            f"| {m} | {_cell(base)} | {_cell(ours)} | "
            f"{ratio(None if isinstance(ours, str) else ours[0])} | "
            f"{best[1] if best else 'n/a'} | {ratio(best[0] if best else None)} | "
            + " | ".join(_cell(r) for r in cands)
            + " |"
        )
        print(lines[-1], flush=True)
        del x
        torch.cuda.empty_cache()
    return lines


def digests():
    op = triton_gemm.TritonBatchInvariantLinearOp()
    result = {}
    for shape in SHAPES:
        weight, bias, out_dtype = weights(shape)
        for m in (17, 1024):
            y = op(rows(m, weight.shape[1], seed=5), weight, bias, out_dtype=out_dtype)
            digest = hashlib.sha256(_bits(y).cpu().numpy().tobytes()).hexdigest()
            result[f"{shape}, M={m}"] = digest
    return result


def header():
    return [
        "# Batch-invariant Triton linear on ROCm",
        "",
        f"Host `{socket.gethostname()}`, {torch.cuda.get_device_name()}, "
        f"torch {torch.__version__}, triton {triton.__version__}.",
        "",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", help="markdown report, rewritten after every section")
    parser.add_argument("--sections", default="3,1,2,4,5", help="comma-separated section numbers")
    parser.add_argument("--digest", help="write output digests (JSON) to this file and exit")
    parser.add_argument("--compare", help="with --digest: also compare with this digest file")
    args = parser.parse_args()

    if args.digest:
        result = digests()
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
        "1": section_launch_invariance,
        "2": section_batch_invariance,
        "3": section_dynamic_shape,
        "4": section_accuracy,
        "5": section_perf,
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
