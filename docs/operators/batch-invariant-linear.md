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
- **Launch config.** `CONFIGS` holds four launch configs, one per range of M
  (M ≤ 16, M ≤ 128, M ≤ 1024, larger), and the op uses the one for its M. The
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

## Performance Notes

Not measured yet for `TritonBatchInvariantLinearOp`; the measurement is
`benchmarks/operators/gemm/evaluate_triton_linear.py`, with the aiter op below as
the baseline. The table below is the aiter op before the copy.

Device time of the DeepSeek-V4-Flash lm_head GEMM: x [M, 4096] bf16 × weight
[129280, 4096] bf16, fp32 logits; ms per call from GPU events, median over 5
trials, one run. Baseline: aiter's own config (no batch-invariance guarantee). Ours: the
gfx942 table. The output bits of ours equal `FIXED_LINEAR_CONFIG`'s at every M.

| M | baseline: aiter's own config (ms) | ours: tuned table (ms) | ours / baseline (ratio) | reference: `FIXED_LINEAR_CONFIG` (ms) | reference: `torch.matmul`, hipBLASLt, fp32 output, not batch invariant (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.2715 | 0.2155 | 0.79 | 0.2725 | 0.2412 |
| 16 | 0.2865 | 0.2242 | 0.78 | 0.2866 | 0.2536 |
| 17 | 0.2881 | 0.2446 | 0.85 | 0.2895 | 0.2578 |
| 64 | 0.2959 | 0.2755 | 0.93 | 0.2987 | 0.2919 |
| 256 | 0.6756 | 0.5810 | 0.86 | 1.1509 | 0.5335 |
| 1024 | 1.7177 | 1.6855 | 0.98 | 4.4109 | 1.5802 |
| 3000 | 5.2874 | 5.2716 | 1.00 | 13.0877 | 4.7584 |
| 4096 | 6.2061 | 6.1617 | 0.99 | 17.8398 | 6.4660 |
| 8192 | 12.2308 | 12.2341 | 1.00 | 35.7319 | 12.7593 |

Measured before the port, on the KLR branch with the same aiter kernels (aiter
`326cffa`, `AITER_TRITON_ONLY=1`), the same launch-config tables and the same
benchmark code, on AMD Instinct MI325X (gfx942), Triton 3.7, torch 2.12, ROCm 7.14.
Rerun the commands below to measure this branch.

```bash
python benchmarks/operators/gemm/benchmark_lm_head_linear.py --config-table \
    rl_engine/backends/rocm/configs/gfx942/gemm_a16w16/GEMM-A16W16-N=129280-K=4096.json
```

`benchmarks/tuning/tune_batch_invariant_linear.py` regenerates the table.

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

- The four configs in `CONFIGS` and their M ranges are not measured yet: they
  were taken from aiter's per-M table for the lm_head on gfx942, with the K tile
  set to 64. They are to be set from the device-time measurement.
- No CUDA kernel.
