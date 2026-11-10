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

## Performance Notes

Not measured yet for the Triton ops; the measurement is
`benchmarks/operators/mhc/evaluate_triton_mhc.py`, with the aiter ops and their
config tables as the baseline. The tables below are the aiter ops before the copy.

Forward device time of mHC pre, n = 4, C = 4096, bf16 residual, fp32 `fn`, 20
Sinkhorn rounds; ms per call from GPU events, median over 5 trials of 50 calls.
Baseline: aiter's own config (not batch invariant). Ours: the `mhc_pre` op with the
gfx942 table. Three runs on two nodes; each cell gives the range of the three
medians, and the ratio column the range of ours / baseline per run.

| M | baseline: aiter's own config (ms) | ours: `mhc_pre` op (ms) | ours / baseline (ratio) |
| --- | --- | --- | --- |
| 1 | 0.1985–0.1986 | 0.0965–0.1176 | 0.49–0.59 |
| 16 | 0.2431–0.2432 | 0.0982–0.1183 | 0.40–0.49 |
| 17 | 0.2482–0.2484 | 0.0991–0.1181 | 0.40–0.48 |
| 64 | 0.2503–0.2514 | 0.0998–0.1192 | 0.40–0.47 |
| 100 | 0.2546–0.2553 | 0.0987–0.1187 | 0.39–0.46 |
| 256 | 0.2549–0.2560 | 0.0986–0.1181 | 0.39–0.46 |
| 1024 | 0.0910–0.0962 | 0.0996–0.1196 | 1.09–1.24 |
| 2048 | 0.0910–0.0961 | 0.0991–0.1184 | 1.07–1.23 |
| 3000 | 0.1315–0.1330 | 0.1000–0.1179 | 0.76–0.89 |
| 4096 | 0.1395–0.1424 | 0.1158–0.1224 | 0.82–0.86 |
| 8192 | cannot launch: needs 128 KB LDS, gfx942 has 64 KB | 0.1978–0.2076 | n/a: baseline cannot launch |
| 16384 | 0.4458–0.5202 | 0.4047–0.5137 | 0.91–0.99 |
| 32768 | 0.8665–0.8963 | 0.7747–0.8114 | 0.89–0.91 |
| 65536 | 1.6500–1.7960 | 1.4913–1.6645 | 0.90–0.93 |

Forward device time of mHC head, n = 4, C = 4096, bf16 residual, fp32 `head_fn`;
ms per call from GPU events around a CUDA graph of 20 calls, median over 5
replays, one run. Baseline: aiter's own head config, which is itself batch
invariant. Ours: the `mhc_head` op with the gfx942 table. The output bits of ours
and the baseline were the same at every M.

| M | baseline: aiter's head config (ms) | ours: `mhc_head` op (ms) | ours / baseline (ratio) |
| --- | --- | --- | --- |
| 1 | 0.8195 | 0.7440 | 0.91 |
| 16 | 0.8319 | 0.7454 | 0.90 |
| 17 | 0.8311 | 0.7476 | 0.90 |
| 64 | 0.8374 | 0.7497 | 0.90 |
| 100 | 0.8399 | 0.7458 | 0.89 |
| 256 | 0.8396 | 0.7446 | 0.89 |
| 1024 | 0.9042 | 0.7780 | 0.86 |
| 2048 | 0.9101 | 0.7815 | 0.86 |
| 3000 | 0.9121 | 0.7848 | 0.86 |
| 4096 | 0.9153 | 0.7879 | 0.86 |
| 5000 | 1.7825 | 1.3547 | 0.76 |
| 8192 | 1.8948 | 1.4265 | 0.75 |
| 16384 | 3.8658 | 2.8688 | 0.74 |
| 32768 | 6.7690 | 5.1442 | 0.76 |
| 65536 | 12.0018 | 10.7097 | 0.89 |

Measured before the port, on the KLR branch with the same aiter kernels (aiter
`326cffa`, `AITER_TRITON_ONLY=1`), the same launch-config tables and the same
benchmark code, on AMD Instinct MI325X (gfx942), Triton 3.7, torch 2.12, ROCm 7.14.
Rerun the commands below to measure this branch.

```bash
python benchmarks/operators/mhc/benchmark_mhc_pre.py --output mhc_pre.md
python benchmarks/operators/mhc/benchmark_mhc_head.py --output mhc_head.md
```

`benchmarks/tuning/tune_mhc_pre_config.py` and
`benchmarks/tuning/tune_mhc_head_config.py` regenerate the tables.

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
