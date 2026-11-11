# DeepSeek-V4 mHC (pre, post, head) and Layer End

## Summary

Batch-invariant DeepSeek-V4 manifold-constrained hyper-connections (mHC) on ROCm,
DSV4 RFC tasks 1–4 and 8 ([#508](https://github.com/RL-Align/RL-Kernel/issues/508)).
DeepSeek-V4 keeps `n = 4` residual streams of hidden size `C` per token. Before each sublayer, mHC pre mixes them into one input; after
it, mHC post mixes the output back; after the last layer, mHC head contracts the
streams for the final norm and lm_head. A token's result must not depend on the
other tokens in the batch, so that rollout and training produce the same bits.

## Entry Point

```python
from rl_engine.runtime.registry import kernel_registry

pre = kernel_registry.get_op("mhc_pre", device="cuda")
post_mix, comb_mix, layer_input = pre(residual, fn, scale, base)

post = kernel_registry.get_op("mhc_post", device="cuda")
new_residual = post(y, residual, post_mix, comb_mix)

head = kernel_registry.get_op("mhc_head", device="cuda")
hidden = head(residual, head_fn, head_scale, head_base)

from rl_engine.models.deepseek_v4.layer_end import mhc_layer_end

stages = mhc_layer_end(residual, head_fn, head_scale, head_base,
                       norm_weight, lm_head_weight, target_ids)
# {"head", "final_norm", "logits", "logprob"}
```

`rl_engine.ops.mhc` exposes `mhc_pre`, `mhc_post` and `mhc_head` as functions
that look the op up in `kernel_registry` for the input's device. The arithmetic
each backend follows is recorded in `rl_engine/contracts/operators/mhc.py`
(`describe("mhc_pre")` and so on).

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA | — | — | Not implemented; the registry raises |
| ROCm | `TritonMHCPreOp`, `TritonMHCPostOp`, `TritonMHCHeadOp` in `rl_engine/backends/rocm/mhc/triton_mhc.py` | Triton kernels copied from aiter `326cffa` (`mhc_pre_dsv4`, `mhc_post_dsv4`, `mhc_head_dsv4`) | Forward batch invariant by construction; backward recomputes in PyTorch, as aiter's |
| PyTorch fallback | `NativeMHCPreOp`, `NativeMHCPostOp`, `NativeMHCHeadOp` in `rl_engine/reference/mhc/` | — | fp32 reference following DeepSeek-V4-Flash-0731 `inference/model.py`; CPU only |

The ROCm ops do not need aiter. aiter cannot support batch invariance officially,
so its Triton kernels were copied into RL-Kernel and changed so that batch
invariance holds by construction. The aiter ops with per-M config tables
(`AiterMHCPreOp`, `AiterMHCPostOp`, `AiterMHCHeadOp` in
`rl_engine/backends/rocm/mhc/aiter_mhc.py`) are no longer registered; they stay
as the baseline of the benchmarks until they are removed.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `residual` | `[..., n, C]` | bf16 | |
| `fn` | `[2n + n², n·C]` | fp32 | `hc_attn_fn` / `hc_ffn_fn` |
| `scale`, `base` | `[3]`, `[2n + n²]` | fp32 | |
| `post_mix`, `comb_mix` | `[..., n]`, `[..., n, n]` | fp32 | `post_mix` may also be `[..., n, 1]` |
| `layer_input`, `y`, `hidden` | `[..., C]` | bf16 | |
| `head_fn`, `head_scale`, `head_base` | `[n, n·C]`, `[1]`, `[n]` | fp32 | |
| `norm_weight` | `[C]` | fp32 | |
| `lm_head_weight` | `[V, C]` | bf16 | as `head.weight` in the checkpoint; other dtypes are rounded to bf16 |
| `target_ids` | `[T]` | int64 | |

`comb_mix` is indexed [source stream, destination stream]. Leading dimensions are
flattened into the token count M.

## Dispatch Behavior

On ROCm the registry hands out only the Triton ops, without fallback to an op
whose bits change with the batch. Inputs outside the tensor contract (other
dtypes, n ≠ 4, mismatched shapes) raise. CPU gets the PyTorch references; CUDA,
MUSA and NPU have no mHC backend, and `get_op` raises there. The semantic catalog
declares the Triton ops for bf16 on ROCm with `batch_invariant: True` and
`backward_batch_invariant: False`.

The copy keeps aiter's arithmetic except for three changes, so its results
differ from aiter's only by the order of the sums and by the fp32 projection
(module docstring of `triton_mhc.py`; why aiter's bits change with launch
parameters: [mHC Reduction Order and Batch Invariance on ROCm](mhc-reduction-order.md)):

- **Batch invariant by construction.** mHC pre and mHC head always run a split
  kernel over 4 segments of K (`NUM_KSPLIT`, for K = 16384), then an apply kernel
  that adds the segments in segment order. aiter switches between a fused
  kernel and this split pair, and picks the number of segments by M. Every sum
  has its order written in the source instead of leaving it to `tl.sum`, whose
  tree follows the tile layout:
  - sum of squares: accumulated element-wise with `tl.fma` over the K tiles of
    128 (`BLOCK_K`) of a segment, then the 128 columns are added pairwise;
  - projection: `tl.dot` over the same K tiles;
  - weighted sum of the 4 streams (`layer_input`, mHC head output): one `tl.fma`
    per stream in stream order;
  - Sinkhorn row and column sums: pairwise, `(p0 + p1) + (p2 + p3)`;
  - mHC post: aiter's kernel, which already adds the source streams in a loop.

  `NUM_KSPLIT`, `BLOCK_K` and the MFMA shape (`matrix_instr_nonkdim=16`,
  `kpack=1`) are module constants. The other launch parameters only group tokens
  and hidden positions into workgroups and do not change the bits; each op uses
  one config (`PRE_CONFIG`, `HEAD_CONFIG`, `POST_CONFIG`) for every M.
- **Dynamic shape.** M is a runtime argument of every kernel and Triton does
  not specialize it (`do_not_specialize`); the partial results are stored
  `[M, segment, column]`, so no stride depends on M. A new token count compiles
  nothing (aiter declares M `tl.constexpr` in its fused, split and head kernels).
- **Projection precision of the DeepSeek-V4 model.** `fn` and `head_fn` stay fp32
  and the projection is an fp32 `tl.dot` with `input_precision="ieee"` (aiter
  rounds them to bf16). The PyTorch-recompute backward projects with fp32 `fn` too.

mHC head uses the split and apply structure of mHC pre; aiter's head kernel runs
one workgroup per 16 tokens over all of K and C.

## Accuracy

- The PyTorch references compute in fp32 and cast `layer_input`, the post output
  and the head output to bf16, as DeepSeek-V4 does; `forward_fp32` keeps fp32.
  The ROCm ops match `forward_fp32`, including the residual gradient, within
  the WS1 contract tolerance (`rl_engine/contracts/profiles/precision/ws1.json`,
  bfloat16 `reduction` cells of `forward_accuracy` and `gradient_accuracy`).
- The RMS factor is `rsqrt(mean(x²) + eps)`, as in DeepSeek-V4 and aiter.
- The projection uses the fp32 `fn` and `head_fn`, as DeepSeek-V4 does (decided
  2026-11-10); aiter rounds them to bf16, which dominated its error against the
  reference (largest `post_mix` error 3.6e-3).
- The four streams of `layer_input` and of the mHC head output are added as
  `fma(w3, x3, fma(w2, x2, fma(w1, x1, w0·x0)))` in fp32 and rounded to bf16 once.
- `mhc_layer_end`'s lm_head uses `batch_invariant_linear` with bf16 inputs and
  fp32 output (see [Batch-Invariant Linear](batch-invariant-linear.md)).

## Verification and Performance

All numbers: AMD Instinct MI325X (gfx942), torch 2.12 with ROCm 7.14, Triton 3.7.0,
n = 4, C = 4096, bf16 residual, fp32 `fn`, 20 Sinkhorn rounds, commit `de1ec324`,
Slurm jobs 68343361–68343366 (2026-11-10),
`benchmarks/operators/mhc/evaluate_triton_mhc.py`. Baseline: the aiter ops with
their per-M config tables (`AiterMHCPreOp`, `AiterMHCPostOp`, `AiterMHCHeadOp`),
the ops before the copy. Ours: the Triton ops.

**Correctness checks of ours.**

| Check | Result |
| --- | --- |
| Tests listed under Tests below | 505 passed, 80 xfailed (backward batch invariance), 4 xpassed (the same non-strict xfail at single M values) |
| Launch parameters: 17 changes for mHC pre, 15 for mHC head, 9 for mHC post, at M = 1, 17, 1024, 4096 | every output bit for bit the same as with the default config |
| Compiled variants after one call and then calls at 14 unseen M from 2 to 10000 | unchanged: 1 per kernel (2 for the split kernel: one for mHC pre, one for mHC head) |
| Batch invariance, M = 1 to 65536, chunking, reordering, padding, strides, repeated runs | mHC post: no difference. mHC pre and mHC head: the script section failed on non-contiguous inputs (a script bug, fixed; to be rerun); the tests above cover the same checks at M ≤ 16384 and passed |
| Output digests, Triton 3.7.0 against 3.8.0, M = 17 and 1024 | the same for every output |

**Error against the fp32 reference** (`NativeMHC*Op.forward_fp32`), M = 1024,
random inputs; largest / mean absolute error.

| Variant | `post_mix` | `comb_mix` | `layer_input` | mHC post output | mHC head output |
| --- | --- | --- | --- | --- | --- |
| baseline | 3.601e-03 / 6.416e-04 | 1.227e-03 / 2.053e-04 | 2.182e-02 / 1.502e-03 | 3.121e-02 / 1.334e-03 | 1.893e-02 / 1.393e-03 |
| ours | 4.172e-06 / 4.408e-07 | 1.073e-06 / 1.437e-07 | 1.562e-02 / 1.268e-03 | 3.121e-02 / 1.334e-03 | 1.562e-02 / 1.216e-03 |

**Forward device time**, ms per call from GPU events around a CUDA graph, median
over replays. Ratio columns: ours / baseline (below 1: ours is faster). "Ours,
fastest config" is the fastest of 18 launch configs for mHC pre and mHC head and
12 for mHC post, all with the same bits as the default.

mHC pre:

| M | baseline (ms) | ours, default config (ms) | ours, fastest config (ms) | ours default / baseline (ratio) | ours fastest / baseline (ratio) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0556 | 0.0721 | 0.0657 | 1.30 | 1.18 |
| 16 | 0.0505 | 0.0730 | 0.0662 | 1.45 | 1.31 |
| 64 | 0.0660 | 0.0738 | 0.0667 | 1.12 | 1.01 |
| 256 | 0.0589 | 0.0744 | 0.0675 | 1.26 | 1.15 |
| 1024 | 0.0846 | 0.0864 | 0.0811 | 1.02 | 0.96 |
| 2048 | 0.0835 | 0.1004 | 0.0961 | 1.20 | 1.15 |
| 4096 | 0.1231 | 0.1501 | 0.1435 | 1.22 | 1.17 |
| 8192 | 0.2106 | 0.2717 | 0.2592 | 1.29 | 1.23 |
| 16384 | 0.4381 | 0.5376 | 0.5345 | 1.23 | 1.22 |
| 32768 | 0.8201 | 1.0523 | 1.0261 | 1.28 | 1.25 |
| 65536 | 1.5149 | 2.0569 | 2.0676 | 1.36 | 1.36 |

Ours is slower than the baseline at every M. The `fixed_order` prototype, the same
structure with `fn` rounded to bf16, took 0.68–0.96 of the baseline
([reduction-order page, 7.6](mhc-reduction-order.md)); that the fp32 projection
causes the difference is not measured yet.

mHC head:

| M | baseline (ms) | ours, default config (ms) | ours, fastest config (ms) | ours default / baseline (ratio) | ours fastest / baseline (ratio) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.7444 | 0.0602 | 0.0527 | 0.08 | 0.07 |
| 16 | 0.7459 | 0.0609 | 0.0529 | 0.08 | 0.07 |
| 64 | 0.7479 | 0.0611 | 0.0532 | 0.08 | 0.07 |
| 256 | 0.7429 | 0.0617 | 0.0537 | 0.08 | 0.07 |
| 1024 | 0.7767 | 0.0709 | 0.0679 | 0.09 | 0.09 |
| 2048 | 0.7831 | 0.0858 | 0.0832 | 0.11 | 0.11 |
| 4096 | 0.7880 | 0.1308 | 0.1126 | 0.17 | 0.14 |
| 8192 | 1.4327 | 0.2436 | 0.1915 | 0.17 | 0.13 |
| 16384 | 2.8810 | 0.4824 | 0.4236 | 0.17 | 0.15 |
| 32768 | 5.1805 | 0.9499 | 0.8023 | 0.18 | 0.15 |
| 65536 | 10.7338 | 1.8118 | 1.5258 | 0.17 | 0.14 |

mHC post. The default config of this run was aiter's (`BLOCK_M=64`,
`BLOCK_C=256`); after this run `POST_CONFIG` was changed to `BLOCK_M=16`,
`BLOCK_C=256`, `num_warps=4`, to be measured again.

| M | baseline (ms) | ours, aiter's config (ms) | ours, fastest config (ms) | fastest config | ours aiter's config / baseline (ratio) | ours fastest / baseline (ratio) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.7296 | 1.7359 | 0.0464 | BLOCK_M=16 BLOCK_C=256 num_warps=8 | 1.00 | 0.03 |
| 16 | 1.7371 | 1.7350 | 0.0474 | BLOCK_M=16 BLOCK_C=256 num_warps=8 | 1.00 | 0.03 |
| 64 | 1.8153 | 1.8109 | 0.0476 | BLOCK_M=16 BLOCK_C=256 num_warps=8 | 1.00 | 0.03 |
| 256 | 1.8340 | 1.8298 | 0.0542 | BLOCK_M=16 BLOCK_C=256 num_warps=8 | 1.00 | 0.03 |
| 1024 | 1.8525 | 1.8469 | 0.0657 | BLOCK_M=16 BLOCK_C=256 num_warps=8 | 1.00 | 0.04 |
| 2048 | 2.0988 | 2.0966 | 0.0725 | BLOCK_M=16 BLOCK_C=256 num_warps=8 | 1.00 | 0.03 |
| 4096 | 2.8015 | 2.8006 | 0.1007 | BLOCK_M=16 BLOCK_C=256 num_warps=8 | 1.00 | 0.04 |
| 8192 | 3.0418 | 3.0376 | 0.1892 | BLOCK_M=16 BLOCK_C=256 num_warps=4 | 1.00 | 0.06 |
| 16384 | 3.7749 | 3.7789 | 0.3490 | BLOCK_M=16 BLOCK_C=256 num_warps=4 | 1.00 | 0.09 |
| 32768 | 6.3778 | 6.3823 | 0.6408 | BLOCK_M=16 BLOCK_C=256 num_warps=4 | 1.00 | 0.10 |
| 65536 | 6.9636 | 6.9372 | 1.4190 | BLOCK_M=16 BLOCK_C=128 num_warps=4 | 1.00 | 0.20 |

Full reports: `/home/AMD/zitwang/dev/logs/rlk-triton-mhc-bi-6834336[1-6]-*`.

```bash
python benchmarks/operators/mhc/evaluate_triton_mhc.py --sections 3,1,2,4 --output checks.md
python benchmarks/operators/mhc/evaluate_triton_mhc.py --sections 5 --output time.md
```

## Tests

```bash
python -m pytest tests/reference/test_mhc_reference.py tests/ops/mhc \
    tests/contracts/operators/test_mhc_contract.py \
    tests/backends/rocm/test_rocm_mhc_pre.py tests/backends/rocm/test_rocm_mhc_post.py \
    tests/backends/rocm/test_rocm_mhc_head.py \
    tests/backends/rocm/test_rocm_mhc_batch_invariance.py \
    tests/backends/rocm/test_rocm_dynamic_shape.py tests/models/deepseek_v4 -v
```

`tests/backends/rocm/test_rocm_mhc_batch_invariance.py` checks mHC pre, mHC post
and mHC head, and `tests/models/deepseek_v4/test_layer_end_batch_invariance.py`
the layer end, with `rl_engine/validation/common/batch_invariance.py`: a row alone
and in batches of M = 1 to 16384 (including the WS1 manifest's 8 and 59),
chunking, reordering and padding, other input strides, repeated runs, and forward
with and without grad, at C = 4096 and at C = 7168, which is not a power of two.
`test_rocm_mhc_pre.py` isolates the projection, RMS, Sinkhorn and stream
aggregation with exact inputs, and checks the projection and RMS under chunking,
reordering, padding and other strides as well. The pre, post and head tests
check that other launch configs give the same bits, and
`test_rocm_dynamic_shape.py` that unseen M values compile nothing. Accuracy against the fp32
references uses the WS1 contract tolerance. Unsupported inputs and the layer end
against a PyTorch reference are tested as well, and empty batches for mHC pre,
mHC post, mHC head and the batch-invariant linear.

## Known Limitations

- Backward is not batch invariant: it recomputes the formula in PyTorch, as
  aiter's does. Those tests are marked xfail.
- The number of K segments (4) and the launch configs are chosen for
  DeepSeek-V4-Flash's n = 4, C = 4096 on gfx942; other hidden sizes run with the
  same constants (K positions past the end of K are zeros).
- No CUDA kernel. The registry has no mhc_pre, mhc_post or mhc_head backend on
  CUDA, MUSA or NPU, so `get_op` raises there.
- `mhc_layer_end` does not support an empty batch: with zero tokens the PyTorch
  RMSNorm reference it uses for the final norm raises
  (`rl_engine/reference/norm/rms_norm.py` reshapes the empty tensor with
  `reshape(*shape[:-1], -1, 32)`, whose `-1` cannot be inferred).
- CUDA/HIP graph capture is not covered here; it belongs to the framework
  integration (DSV4 RFC task 7, #510).
- Open for the RFC author: the RFC asks for a controller RMS of
  `1 / (sqrt(mean) + eps)`, while aiter and the DeepSeek-V4 reference use
  `rsqrt(mean + eps)`. The tests record the latter.
