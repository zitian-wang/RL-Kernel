# Batch-Invariant Linear (ROCm)

## Summary

`y = x @ weightᵀ + bias` whose output row does not depend on the other rows in the
batch, on ROCm with aiter's `gemm_a16w16` Triton kernel copied into RL-Kernel and
made batch invariant by construction (aiter cannot guarantee batch invariance).
DSV4 RFC task 9
([#510](https://github.com/RL-Align/RL-Kernel/issues/510)): a shared GEMM
with a fixed K accumulation order, bias and output conversion, with fp32 output
for DeepSeek-V4's lm_head.

## Entry Point

```python
from rl_engine.runtime.registry import kernel_registry

linear = kernel_registry.get_op("batch_invariant_linear", device="cuda")
logits = linear(x, weight, out_dtype=torch.float32)  # x [..., K], weight [N, K]
```

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA | — | — | Not implemented |
| ROCm | `TritonBatchInvariantLinearOp` in `rl_engine/backends/rocm/gemm/triton_gemm.py` | `_gemm_a16_w16_kernel` (Triton), copied from aiter `326cffa` | Forward batch invariant; backward through det_gemm's fixed-order Triton GEMM in fp32 |
| PyTorch fallback | — | — | None |

`AiterBatchInvariantLinearOp` (`rl_engine/backends/rocm/gemm/aiter_gemm.py`, aiter
`gemm_a16w16` with a per-M config table) is no longer registered; it stays as the
baseline of the benchmarks until it is removed.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[..., K]` | bf16 | other dtypes raise |
| `weight` | `[N, K]` | same as `x` | |
| `bias` | `[N]` | `out_dtype` | optional |
| output | `[..., N]` | `out_dtype`: bf16 (default) or fp32 | |

## Dispatch Behavior

The registry hands out `TritonBatchInvariantLinearOp` for `batch_invariant_linear`
on ROCm only, without fallback. CUDA, MUSA, CPU and NPU have no backend, and
`get_op` raises there. Inputs outside the tensor contract raise; an
empty batch returns an empty output without a launch. The semantic catalog
declares it for bf16 on ROCm with `batch_invariant: True`.

What the copy changes in aiter's kernel (module docstring of `triton_gemm.py`):

- **Batch invariant by construction.** K is never split: aiter's split-K path,
  which some of aiter's configs choose by M, is removed. The K tile is the module
  constant `BLOCK_K = 64` and the MFMA shape is fixed (`matrix_instr_nonkdim=16`,
  `kpack=1`), so an output element is accumulated in the same order for every M
  and every launch config. Inside a tile there is only `tl.dot`, no `tl.sum`.
- **Dynamic shape.** M and N are runtime arguments that Triton does not
  specialize; aiter's compile-time `EVEN_MN` switch is removed (always the masked
  path), and `EVEN_K` depends on K only. A new M compiles nothing.
- **Launch config.** `CONFIGS` holds one launch config per range of M
  (M ≤ 16, 32, 64, 128, 256, 512, larger; 6 different configs), and the op uses
  the one for its M. The
  configs only group output tiles into workgroups; they do not change the bits. Each compiles once per process;
  `TritonBatchInvariantLinearOp.warmup(weight)` compiles all of them.
- Activations, `skip_reduce` and aiter's persistent and gluon paths are removed.

## Accuracy

bf16 inputs, fp32 accumulation, one rounding to `out_dtype`. DeepSeek-V4's lm_head
multiplies in fp32, but `head.weight` is BF16 in the checkpoint and its input holds
bf16 values; the product of two bf16 values is exact in fp32, so only the
accumulation order differs from the reference. Output and input gradient match an
fp32 PyTorch linear within the WS1 contract tolerance
(`rl_engine/contracts/profiles/precision/ws1.json`, bfloat16 `reduction` cells of
`forward_accuracy` and `gradient_accuracy`).

## Verification and Performance

All numbers: AMD Instinct MI325X (gfx942), torch 2.12 with ROCm 7.14, Triton 3.7.0,
commit `de1ec324`, Slurm jobs 68343361, 68343364–68343366 (2026-11-10),
`benchmarks/operators/gemm/evaluate_triton_linear.py`. Baseline:
`AiterBatchInvariantLinearOp`, aiter `gemm_a16w16` with the per-M config table, the op
before the copy. Ours: `TritonBatchInvariantLinearOp`.

**Correctness checks of ours.**

| Check | Result |
| --- | --- |
| Tests listed under Tests below (with the mHC tests) | 505 passed, 80 xfailed, 4 xpassed (all xfail/xpass are mHC backward) |
| 9 launch configs against the first, 5 shapes (lm_head, N = 24 with K = 16384, the WS1 shapes, N = 1000 with K = 4100 and bias), M = 1, 17, 1024, 4096 | every output element bit for bit the same |
| Batch invariance, M = 1 to 16384 (lm_head to 8192), chunking, reordering, padding, strides, repeated runs, same 5 shapes | no difference |
| Compiled variants for the lm_head shape: after `warmup`, then 13 unseen M from 2 to 4097 | 4 after `warmup` (the 4 configs of that run), unchanged after the unseen M |
| Output digests, Triton 3.7.0 against 3.8.0, 5 shapes, M = 17 and 1024 | the same |

**Error against an fp32 linear**: the same as the baseline on all 5 shapes (for
example lm_head, M = 256: largest 1.335e-05, mean 4.553e-07).

**Device time of the DeepSeek-V4-Flash lm_head**: x [M, 4096] bf16 times weight
[129280, 4096] bf16, fp32 logits; ms per call from GPU events, median over trials.
"Ours, configs of this run" is the first choice of `CONFIGS` (4 configs taken from
aiter's per-M table); "fastest candidate" is the fastest of 9 configs (tile M × tile
N / `GROUP_SIZE_M` / `num_warps` / `num_stages`), all with `BLOCK_K = 64` and the
same bits. After this run `CONFIGS` was set to the fastest candidate of each M
range, to be measured again. Ratio columns: ours / baseline.

| M | baseline (ms) | ours, configs of this run (ms) | fastest candidate (ms) | fastest candidate | ours / baseline (ratio) | fastest / baseline (ratio) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.2170 | 0.2277 | 0.2241 | 16×32/1/2/2 | 1.05 | 1.03 |
| 4 | 0.2204 | 0.2275 | 0.2271 | 16×32/1/2/2 | 1.03 | 1.03 |
| 16 | 0.2253 | 0.2381 | 0.2387 | 16×32/1/2/2 | 1.06 | 1.06 |
| 17 | 0.2408 | 0.2880 | 0.2570 | 32×64/1/4/2 | 1.20 | 1.07 |
| 32 | 0.2473 | 0.2970 | 0.2702 | 32×64/1/4/2 | 1.20 | 1.09 |
| 64 | 0.2712 | 0.3257 | 0.3255 | 64×128/1/4/2 | 1.20 | 1.20 |
| 128 | 0.3309 | 0.6767 | 0.3934 | 128×256/4/8/2 | 2.04 | 1.19 |
| 256 | 0.5807 | 0.6234 | 0.5827 | 256×256/4/8/2 | 1.07 | 1.00 |
| 512 | 0.9457 | 1.0205 | 1.0254 | 128×128/4/4/2 | 1.08 | 1.08 |
| 1024 | 1.7011 | 1.8547 | 1.8315 | 256×256/4/8/2 | 1.09 | 1.08 |
| 2048 | 3.1623 | 3.4007 | 3.4132 | 256×256/4/8/2 | 1.08 | 1.08 |
| 4096 | 6.2212 | 6.7213 | 6.6940 | 256×256/4/8/2 | 1.08 | 1.08 |
| 8192 | 12.2279 | 13.1098 | 13.1333 | 256×256/4/8/2 | 1.07 | 1.07 |

Even the fastest candidate is 0–20% slower than the baseline. The baseline's table
uses a K tile of 128 or 256 at small M and 64 at large M; that the fixed K tile of
64 causes the difference is not measured. Full report:
`/home/AMD/zitwang/dev/logs/rlk-triton-mhc-bi-68343365-linear-time.md`.

```bash
python benchmarks/operators/gemm/evaluate_triton_linear.py --sections 3,1,2,4 --output checks.md
python benchmarks/operators/gemm/evaluate_triton_linear.py --sections 5 --output time.md
```

## Tests

```bash
python -m pytest tests/backends/rocm/test_rocm_batch_invariant_linear.py \
    tests/backends/rocm/test_rocm_dynamic_shape.py -v
```

Covers the DeepSeek-V4 mHC projection shapes (N = 24 and 4, K = 16384), the WS1
manifest's GEMM shapes (K = 4096, N = 4096 and 12288), a shape whose N and K are
not tile multiples, the lm_head shape, bf16 and fp32 output, and bias: a row alone
and in batches of M = 1 to 16384 (8192 for lm_head, including the WS1 manifest's
8 and 59), chunking, reordering and padding, other input strides, repeated runs,
the fp32 reference, the input gradient, empty batches and unsupported inputs.
M includes the largest M of every range in `CONFIGS` and the M after it, and every
entry of `CONFIGS` is also compared with the first one directly.
`test_rocm_dynamic_shape.py` checks that unseen M values compile nothing after
`warmup`.

## Known Limitations

- The configs in `CONFIGS` and their M ranges are chosen for the lm_head shape on
  MI325X; other shapes use the same ranges.
- No CUDA kernel.
