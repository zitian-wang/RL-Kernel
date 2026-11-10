# mHC Reduction Order and Batch Invariance on ROCm

This page explains why the DeepSeek-V4 mHC pre and mHC head kernels that
RL-Kernel takes from aiter give a token different result bits when launch
parameters change, how other projects deal with the same problem, and which
approaches we measured to fix the order of every sum. It belongs to DSV4 RFC
tasks 1 and 8 ([#508](https://github.com/RL-Align/RL-Kernel/issues/508)) and to
the branch `feat/issue-81-mhc-bi-reduction`. Which approach to adopt is not
decided yet.

Each statement is either read in the source file cited next to it or measured
in the Slurm job cited next to it. The few statements that are neither are
marked "not verified".

## Terms used on this page

**Shapes.** One token holds `n = 4` residual streams of hidden size `C = 4096`
(DeepSeek-V4-Flash). `K = n × C = 16384` is the length of one token's flattened
streams. `M` is the number of tokens in the batch.

**Batch invariant.** A kernel is batch invariant when the bits of a token's
outputs do not depend on which other tokens are in the batch, and therefore not
on `M`. RL-Kernel checks this with `rl_engine/validation/common/batch_invariance.py`:
a token computed alone is compared bit for bit with the same token placed in
batches of many sizes, after chunking, reordering and padding, and with
non-contiguous inputs.

**Reduction order.** Floating-point addition is not associative, so a sum of
many numbers can round differently when the numbers are added in a different
grouping. The grouping is the reduction order. A kernel is batch invariant when
the reduction order of every sum depends only on the token's own data.

**Launch parameters.** The Triton kernels discussed here take tile sizes as
compile-time constants: `BLOCK_M` (tokens per workgroup), `BLOCK_K` (step of the
loop over K), `BLOCK_C` (hidden positions per workgroup in the final weighted
sum), `NUM_KSPLIT` (number of K segments computed by different workgroups and
added afterwards), plus Triton's own options `num_warps`, `num_stages`,
`waves_per_eu`, `kpack` and `matrix_instr_nonkdim`.

**Layout of a tensor in Triton.** When a Triton kernel holds a tile such as
`[BLOCK_M, BLOCK_K]` in registers, the compiler decides how the tile is spread
over the GPU threads. Along one axis this is three numbers: how many
consecutive elements one thread holds ("elements per thread"), how many lanes
of a wavefront the axis is spread over ("lanes"; a wavefront has 64 lanes on
gfx942), and how many wavefronts the axis is spread over ("warps").

**The approaches compared** (section 7). Each name says what the approach does:

| Name | What it is |
|---|---|
| `aiter_own_config` | aiter's Triton mHC ops with the launch config aiter chooses for each `M`. Not batch invariant; shown for reference. |
| `aiter_config_table` | The current RL-Kernel ops on the PR branch: aiter's Triton mHC ops with a launch config per `M` from RL-Kernel's tables, chosen so that the result bits are the same for every `M`. The baseline of every comparison. |
| `fixed_order` | New Triton kernels in RL-Kernel (`rl_engine/backends/rocm/mhc/fixed_order_mhc.py`) whose every sum is written in the kernel source, so no launch parameter can change its order. |
| `fixed_order_in_aiter_fork` | The same kernels as `fixed_order`, but kept in a fork of aiter that RL-Kernel pins. |
| `separate_projection` | mHC pre with the projection computed by aiter's `gemm_a16w16` (RL-Kernel's batch-invariant linear) and the sum of squares by the `fixed_order` kernel. |
| `aiter_hip` | aiter's HIP mHC pre kernels (`aiter.ops.mhc`), the ones vLLM uses on ROCm, called with one fixed number of K segments for every `M`. |

## 1. Summary

- mHC pre and mHC head compute, per token, three sums: the projection
  `x @ fn^T` and the RMS sum of squares `Σ x²`, both over the `K = 16384`
  values of the token, and the weighted sum of the 4 residual streams.
- In aiter's Triton kernels the projection is `tl.dot`. Its bits change only
  when `NUM_KSPLIT` or `kpack` changes. `BLOCK_M`, `BLOCK_K`, `num_warps` and the
  MFMA instruction shape do not change them.
- The sum of squares and the weighted stream sum are `tl.sum`. Triton decides
  the order of a `tl.sum` from the layout of its input, and the layout follows
  `BLOCK_M`, `BLOCK_K`, `BLOCK_C` and `num_warps`. For each of the 39 launch
  configs we measured, the bits changed if and only if the layout of one of the
  sums (or the `kpack` packing of the projection) changed.
- aiter chooses these parameters per `M`, so its own configs are not batch
  invariant. `aiter_config_table` avoids that by giving every `M` a config with
  the same layouts. This holds within one Triton version. Between Triton 3.7.0
  and 3.8.0 the layouts stay the same, but 3.8.0 adds a thread's own elements
  as a tree instead of left to right, and the outputs of `aiter_config_table`
  change.
- On NVIDIA, vLLM and SGLang have batch-invariant mHC paths. On ROCm neither
  has one for gfx942: vLLM calls aiter's HIP mHC pre, which chooses its K split
  per `M`; SGLang runs plain PyTorch on gfx942 and its batch-invariant Triton
  kernel only on gfx950. Only DeepGEMM (NVIDIA only) writes the sum-of-squares
  order by hand instead of relying on fixed compile-time constants.
- `fixed_order` was measured on MI325X: its bits did not change under any of
  96 launch configs nor between Triton 3.7.0 and 3.8.0; it is batch invariant
  for `M` from 1 to 16384; its error against the fp32 reference equals that of
  `aiter_config_table`; and it takes 0.68–0.96 of the baseline's time for mHC
  pre and 0.04–0.15 for mHC head.
- `aiter_hip` with 1, 4 or 8 K segments is batch invariant and about as fast as
  `fixed_order`. It projects with fp32 weights, so its mixing weights are about
  1000 times closer to the fp32 reference than those of the other approaches,
  which round the weights to bf16. It needs aiter's HIP build and calls
  internal aiter functions.

## 2. The three sums in mHC pre and mHC head

The DeepSeek-V4-Flash reference (Hugging Face revision `60d8d70`,
`inference/model.py:676-680` for mHC pre, `:730-734` for mHC head):

```python
x = x.flatten(2).float()                                   # [tokens, K], K = 16384
rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
mixes = F.linear(x, hc_fn) * rsqrt                         # 24 mixing values (pre) or 4 (head)
...
y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)    # weighted sum of the 4 streams
```

| Sum | Number of terms per output | In aiter's Triton kernels |
|---|---|---|
| projection `x @ fn^T` | 16384 | `tl.dot` inside the loop over K |
| sum of squares `Σ x²` | 16384 | `tl.sum(x * x, axis=1)` once per `BLOCK_K` tile, inside the same loop |
| weighted stream sum `Σ_i weight_i · x_i` | 4 | `tl.sum(weight[:, :, None] * x, axis=1)` once per `BLOCK_C` tile |
| Sinkhorn row and column sums (mHC pre only) | 4 | separate kernel, one token per workgroup |

## 3. How aiter's Triton kernels compute them

Source: aiter commit `326cffa`, `aiter/ops/triton/fusions/mhc.py` (the Python
functions that launch the kernels) and `aiter/ops/triton/_triton_kernels/fusions/mhc.py`
(the kernels).

### 3.1 mHC pre

The launcher `mhc()` (`fusions/mhc.py:251-369`) takes one of two paths and
passes the same config dictionary to every kernel it launches:

- With `NUM_KSPLIT = 1` it launches `_mhc_fused_kernel`. Every workgroup loops
  over the whole K (`:294-314`):

  ```python
  for k in range(0, K, BLOCK_K):
      x_tile = tl.load(...)                              # [BLOCK_M, BLOCK_K], bf16
      phi_tile = tl.load(...).to(x_ptr.dtype.element_ty) # the fp32 weights rounded to bf16
      acc = tl.dot(x_tile, phi_tile, acc=acc)            # projection
      acc_sq += tl.sum(x_tile_f32 * x_tile_f32, axis=1)  # sum of squares
  ```

  The workgroup that owns the pre weights then computes the weighted stream
  sum, one `BLOCK_C` tile of hidden positions at a time
  (`_mhc_apply_pre_mix_tile`, `:44`).
- With `NUM_KSPLIT > 1` it launches `_mhc_fused_split_kernel`, which runs the
  same loop over one of `NUM_KSPLIT` segments of K and stores partial results
  (`:476-493`), and then `_mhc_reduce_apply_kernel`, which adds the partial
  results in segment order (`:674-686`) and computes the RMS factor, the
  activations and the weighted stream sum.
- Sinkhorn always runs afterwards in `_mhc_asymmetric_sinkhorn_kernel`, one
  token per workgroup, with a fixed `num_warps` (`fusions/mhc.py:1044-1060`).

When no config is passed, aiter chooses one per `M` from
`configs/gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=4096.json`:

| Range of M | NUM_KSPLIT | BLOCK_K | BLOCK_C | BLOCK_M | num_warps |
|---|---|---|---|---|---|
| up to 256 | 1 | 128 | 128 | 32 | 4 or 8 |
| up to 1024 | 8 | 32 | 128 | 32 | 4 |
| up to 2048 | 8 | 128 | 64 | 64 | 8 |
| up to 4096 | 8 | 256 | 128 | 64 | 4 |
| up to 8192 | 4 | 256 | 256 | 64 | 4 |
| up to 16384, and above | 2 | 128 | 128 | 64 | 4 |

Every column except `BLOCK_M` changes the order of some sum (section 5), so a
token's bits change when `M` crosses a range boundary.

### 3.2 mHC head

`_mhc_head_dsv4_forward` (`fusions/mhc.py:1188-1242`) launches a single kernel,
`_mhc_head_kernel` (`:108-186`). It runs the same loop with `tl.dot` and
`tl.sum(x * x, axis=1)` (`:142-156`), with the 4 output columns padded to 16
because `tl.dot` needs at least 16, and then the weighted stream sum. Its
config comes from aiter's tuned entry for this kernel and does not depend on
`M`. The public `config` argument cannot set `num_warps` or `num_stages`.

### 3.3 mHC post

`_mhc_post_kernel` adds the four streams in explicit loops over the stream
index (`:914-955`). It has no `tl.sum`, and its config does not depend on `M`,
so it is not part of this problem.

## 4. Why a launch parameter changes the result bits

### 4.1 The projection (`tl.dot`)

Triton v3.7.0 (source at `/home/AMD/zitwang/dev/triton-3.7.0`, commit `5f3f125`)
replaces `tl.dot` by MFMA matrix instructions in
`third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp`. Each MFMA
instruction multiplies a fixed number of K values. A `tl.dot` over a
`[BLOCK_M, BLOCK_K]` tile becomes a sequence of such instructions that add into
the same accumulator in K order, and the next loop iteration continues on the
same accumulator. `BLOCK_K` therefore only decides where this sequence is cut
into loop iterations, not the order of the additions.

`kpack` is a compile option of Triton's AMD backend
(`third_party/amd/backend/compiler.py:56`, passed to the MFMA pass at `:225`).
The comment at `AccelerateAMDMatmul.cpp:683-717` explains it: with `kpack = 1`
each thread loads exactly the K values one MFMA instruction needs; with
`kpack = 2` it loads twice as many consecutive values and splits them over two
instructions, so each instruction sums a different set of K values. On gfx950
and later Triton forces `kpack` back to 1 (`compiler.py:90-94`).

### 4.2 The sums written as `tl.sum`

Triton turns a `tl.sum` over one axis into three steps
(`lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp:38-74` in v3.7.0):

1. each thread adds the elements it holds, left to right (`:120-151`);
2. the lanes that share the axis exchange values with shuffles at distances
   `lanes/2, lanes/4, …, 1` (`:155-171`);
3. if the axis is also spread over several warps, the warp results go through
   shared memory and are added once more (`:56-74`).

The order of a row's sum is therefore fixed by the layout of the input along
the summed axis: elements per thread, lanes and warps. Triton derives that
layout from the tile shape and `num_warps`. For the sum of squares the input is
the loaded `x_tile`, whose layout comes from the coalescing pass
(`lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp:63-91`) and the layout
builder (`include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:836-850`):

- elements per thread = `min(8, BLOCK_M × BLOCK_K / (64 × num_warps))` for bf16
  input (8 bf16 values fill one 128-bit load);
- lanes along K = `BLOCK_K / elements per thread`, at most 64; the remaining
  lanes and all warps go to the token axis.

Consequences, confirmed by the compiled kernels in section 5:

| Kernel and config | Elements per thread × lanes along K | Same order as aiter's config for this kernel? |
|---|---|---|
| mHC pre, `BLOCK_M = 32, BLOCK_K = 128, num_warps = 4` | 8 × 16 | (reference) |
| mHC pre, `BLOCK_M` from 16 to 128 with `num_warps` 1 to 4, or `BLOCK_M = 64` with `num_warps = 8` | 8 × 16 | yes |
| mHC pre, `BLOCK_M = 16, num_warps = 8` | 4 × 32 | no |
| mHC head, `BLOCK_M = 16, BLOCK_K = 64, num_warps = 4` (aiter's) | 4 × 16 | (reference) |
| mHC head, `BLOCK_M = 32`, or `num_warps = 2` | 8 × 8 | no |
| mHC head, `num_warps = 8` | 2 × 32 | no |
| mHC head, `BLOCK_M = 32, num_warps = 8` | 4 × 16 | yes |

This is why `BLOCK_M` and `num_warps` look free for mHC pre: with
`BLOCK_K = 128` the number of elements per thread almost always reaches its cap
of 8. For mHC head, with `BLOCK_K = 64`, it is 4, below the cap, so they are not
free.

The weighted stream sum works the same way, over the 4 stream values of a
`[BLOCK_M, 4, BLOCK_C]` tile (16 in mHC head because of the padding). Larger
`BLOCK_C` leaves fewer lanes for the stream axis and changes its order.

Not verified: in mHC head with `BLOCK_C = 64`, the stream-sum layout changes but
the first step of the new order only adds padding zeros; the bits change only
when the products `weight × x` round. A possible reason is that the compiler
fuses a product with the first addition into one FMA instruction in one layout
and not in the other.

### 4.3 Triton version

Triton v3.8.0 rewrote the first step. Instead of adding a thread's own
elements left to right, it adds them as a binary tree (`treeReduce`,
`ReduceOpToLLVM.cpp:116-142` and `:286-303` at tag `v3.8.0`; the tree has two
inputs per node for additions, `third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp:689-697`).
With 4 or 8 elements per thread the two orders differ, so the same kernel and
the same launch config give other bits under Triton 3.8.0 (section 7.4).

## 5. Measurements that confirm it

Job 68321538, MI325X, Triton 3.7.0 from the container image, aiter `326cffa`,
script `benchmarks/operators/mhc/analyze_aiter_reductions.py`. Full tables:
`/home/AMD/zitwang/dev/logs/mhc-bi-analysis-68321538-triton-image.md`.

Each launch config differs from a reference config in the listed parameters.
Both run at `M = 17` and `M = 1024` on four kinds of input, chosen so that a
difference can only come from certain sums:

- random input: every sum rounds;
- "exact sum of squares": residual values in {−2, −1, 1, 2}, so every sum of
  squares is an integer below 2^24 and exact in any order; a difference must
  come from the projection or the stream sum;
- "exact projection": each weight row has a single 1, so each projection is a
  single input value; a difference must come from the sum of squares or the
  stream sum;
- "both exact": the mixing weights are then equal for every config; a
  difference must come from the stream sum.

The script also reads the compiled Triton IR of every kernel and reports, for
the input of each `tl.sum`, the elements per thread, lanes and warps along the
summed axis, and the `kWidth` (consecutive K values per thread) of the dot
inputs.

| Kernel | Parameters changed from the reference | Bits changed for input kind | Sum whose layout changed | Conclusion |
|---|---|---|---|---|
| mHC pre, no K split | `BLOCK_M` 16/64/128; `num_warps` 1/2/8; `BLOCK_M = 16` with `num_warps = 2`; `BLOCK_M = 64` with `num_warps = 8`; `num_stages = 2`; `BLOCK_C = 64` | none | none | free |
| mHC pre, no K split | `BLOCK_M = 16` with `num_warps = 8` | random, exact projection | sum of squares: 8 × 16 → 4 × 32 | sum of squares |
| mHC pre, no K split | `BLOCK_K` 32/64/256 | random, exact projection | sum of squares | sum of squares |
| mHC pre, no K split | `BLOCK_C = 256` | all four, only `layer_input` | stream sum: 4 lanes → 2 lanes × 2 warps | stream sum |
| mHC pre, no K split | `kpack = 2` | random, exact sum of squares | dot input `kWidth` 4 → 8 | projection |
| mHC pre, no K split | `matrix_instr_nonkdim = 32` | none | MFMA 16×16×16 → 32×32×8 only | free |
| mHC pre, 4 K segments | the other 7 entries of RL-Kernel's table (various `BLOCK_M`, `BLOCK_C ≤ 128`, `num_warps` 2/4, `num_stages`, `waves_per_eu`) | none | none (some select MFMA 32×32×8) | free |
| mHC pre, 4 K segments | `num_warps = 8` with `BLOCK_M = 16` | random, exact projection | sum of squares: 8 × 16 → 4 × 32 | sum of squares |
| mHC pre, 4 K segments | `BLOCK_C = 256`; `BLOCK_K = 64`; `kpack = 2` | as in the rows above | as above | as above |
| mHC head | `BLOCK_M = 32`; `num_warps` 2 or 8 | random, exact projection | sum of squares: 4 × 16 → 8 × 8 or 2 × 32 | sum of squares |
| mHC head | `BLOCK_M = 32` with `num_warps = 8` | none | none (4 × 16 again) | free |
| mHC head | `num_stages = 2`; `waves_per_eu = 2` | none | none | free |
| mHC head | `BLOCK_K` 32/128 | random, exact projection | sum of squares | sum of squares |
| mHC head | `BLOCK_C` 64/256 | random, exact projection (256: all four) | stream sum | stream sum |
| mHC head | `kpack = 2` | random, exact sum of squares | dot input `kWidth` 4 → 8 | projection |

In every one of the 39 configs, the bits changed exactly when one of these
layouts changed. Compiling the same configs with Triton 3.8.0
(`3.8.0+amd.rocm7.2.0.git7cb7b059`, the version aiter pins) gives the same
layouts in every row (same job, `--layouts-only`).

## 6. How other implementations handle it

The code below was fetched at the commits the RFC links to (tasks 1, 8 and 9)
and read; it was not run, except aiter's HIP kernels in 6.5.

### 6.1 DeepSeek-V4 official reference

Hugging Face `deepseek-ai/DeepSeek-V4-Flash`, revision `60d8d70`,
`inference/model.py`.

- `hc_pre` (`:676-680`) and `hc_head` (`:730-734`) are plain PyTorch:
  `x.square().mean(-1)` and `F.linear(x, hc_fn)` in fp32 with fp32 weights; the
  stream sum is `torch.sum(..., dim=2)`.
- The only mHC kernel in `inference/kernel.py` is the Sinkhorn kernel
  `hc_split_sinkhorn` (`:372-427`, TileLang, one block of 64 threads per token).
- The order of the sums is whatever PyTorch's reduction and cuBLAS choose for
  the tensor shape. The reference does not aim for batch invariance.

### 6.2 DeepGEMM `tf32_hc_prenorm_gemm`

`ISEEKYAN/DeepGEMM` commit `dfad223`; used by vLLM and SGLang on NVIDIA.

- One kernel computes the projection (TF32 matrix instructions, fp32 weights)
  and the sum of squares in the same loop over K.
- The sum of squares is written by hand
  (`deep_gemm/include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh:163-234`).
  Every thread owns fixed columns, adds their squares over all K blocks, and a
  fixed 4-lane shuffle combines the threads of a row:

  ```cpp
  uint32_t row = warp_idx * 16 + lane_idx / 4;           // fixed row of each thread
  ...
  sqr_sum_acc_0 += a_float2_0.x * a_float2_0.x + a_float2_1.x * a_float2_1.x;
  ...
  const auto& reduced_sum_0 = math::warp_reduce_sum<4>(sqr_sum_acc_0);
  ```

  The sm100 and sm120 versions follow the same pattern.
- Every tile size is a constant
  (`csrc/jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp:72-82`): 64 tokens and
  64 K values per tile, 128 math threads. `M` only sets the number of
  workgroups.
- The number of K segments is an argument of the API
  (`csrc/apis/hyperconnection.hpp:18-57`, default 1); the kernel stores one
  partial result per segment and leaves the addition to the caller.
- The result is batch invariant as long as the caller passes the same number
  of segments for every `M`.

### 6.3 vLLM on NVIDIA, and vLLM PR #59929

`inaniloquentee/vllm` commit `e1107b01` (PR [#59929](https://github.com/vllm-project/vllm/pull/59929))
and the earlier `ISEEKYAN/vllm` commit `4b5c627`, directory
`vllm/model_executor/kernels/mhc/`.

- `_hc_prenorm_gemm_outputs` (`tilelang.py:25-75`) calls DeepGEMM when it is
  available. The number of K segments is
  `compute_num_split(64, K, cdiv(split_tokens, 64))` with
  `split_tokens = 1 if envs.VLLM_BATCH_INVARIANT else num_tokens` (`:42-44`). In
  batch-invariant mode it is therefore `min(number of SMs, K / 256)`
  (`tilelang_kernels.py:45-54`): fixed for a given GPU and K.
- The partial results are added in segment order by a serial loop in the
  TileLang kernel that follows.
- PR #59929 changed no kernel. It restored `split_tokens = 1` in two helpers
  where a refactor had dropped it, and in batch-invariant mode it turns off the
  fused post-and-pre kernel (`models/deepseek_v4/nvidia/model.py:1265`,
  `use_fused_mhc = not envs.VLLM_BATCH_INVARIANT`), whose small-`M` path squares
  the fp32 residual instead of the bf16 one.

### 6.4 vLLM's TileLang kernel for GPUs without DeepGEMM

Same files. `hc_prenorm_gemm_tilelang` (`tilelang_kernels.py:879-919`) also
writes its order by hand: each thread adds its K values serially, then
`T.warp_reduce_sum`, then a serial loop over the warps. But it chooses its
thread count from `M` (`tilelang_kernels.py:1147-1163`):

```python
use_block_m = use_default_config and num_tokens >= 1024
use_small_tile = (use_default_config and not use_block_m
                  and num_tokens < 128 and hc_hidden_size % 1024 == 0)
effective_n_thr = 1024 if use_small_tile else n_thr     # n_thr is 512 by default
```

Thread `t` owns K values `t, t + n_thr, …`, so the order changes between
`M < 128` and `M ≥ 128`. Batch-invariant mode does not change this, and the
test added by #59929 only covers the DeepGEMM path.

### 6.5 vLLM on ROCm: aiter's HIP mHC pre

`vllm/models/deepseek_v4/amd/model.py` uses `MHCPreOp`, which calls
`rocm_aiter_ops.mhc_pre` (`vllm/_aiter_ops.py:4111`) and through it
`aiter.ops.mhc.mhc_pre`, aiter's HIP kernels (not the Triton ones of section 3).
mHC head runs through the same function with the weights padded to 24 rows and
Sinkhorn turned off (`vllm/_aiter_ops.py:4433-4480`). The ROCm model file has no
batch-invariant switch.

- `mhc_pre` (`aiter/ops/mhc.py:610-700`) chooses the number of K segments and
  the K tile with `get_mhc_pre_splitk(m, hc_hidden_size)` (`:292-332`), from the
  number of token tiles and the number of compute units, so the choice depends
  on `M`.
- Inside, `mhc_pre_gemm_sqrsum` (`csrc/kernels/mhc_kernels.cu:880-975`) also
  chooses a tile width of 16 or 32 outputs from the number of workgroups, and
  `mhc_pre_big_fuse` chooses its block size from `m <= cu_num * 12`
  (`:1387-1392`).
- Within one launch, each lane adds its own squares and a fixed two-step
  exchange combines the lanes (`cross_row_sum_4`, `:133-147`); the K segments
  are added with a wavefront reduction (`:1060-1140`).
- The projection uses the fp32 weights by default; an optional mode splits each
  weight into two bf16 parts (`mhc_shuffle_fn`, `aiter/ops/mhc.py:44-70`).

Measured on MI325X (job 68322040, `benchmarks/operators/mhc/probe_aiter_hip_mhc_pre.py`):

| M | 1 to 256 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|
| K segments and K tile chosen by `get_mhc_pre_splitk` | 64, 128 | 32, 128 | 16, 128 | 8, 128 | 4, 128 | 2, 128 |

With this choice, a token's bits change from `M = 1024` on, for mHC pre and for
mHC head (section 7.2).

### 6.6 vLLM's batch-invariant RMSNorm and matmul

`inaniloquentee/vllm` commit `76aacf3`,
`vllm/model_executor/determinism/batch_invariant.py`; PR
[#59952](https://github.com/vllm-project/vllm/pull/59952).

- RMSNorm (`:1132-1181`, launched at `:1225-1241`): one workgroup per token,
  a loop over chunks of 1024 values, one `tl.sum` per chunk, chunks added in
  order, `num_warps` left at Triton's default. The order inside a chunk is the
  `tl.sum` order for that fixed layout; it stays fixed because the constants
  never change, the same mechanism as `aiter_config_table`.
- The CUDA `fused_add_rms_norm` launcher fixes its block size to
  `min(hidden, 1024)` in batch-invariant mode
  (`csrc/libtorch_stable/layernorm_kernels.cu:329-372`).
- Matmul (`:46-54`): "What is summed is a function of (K, N, dtype) only --
  never of M". The K tile is fixed per dtype (32 for fp32, 64 otherwise);
  `_split_k_plan(K, N)` (`:66-83`) uses one K segment when N > 64 or K < 1024,
  otherwise `K // 512` segments (at most 16), added in order. `BLOCK_M`,
  `BLOCK_N` and `num_warps` follow `M`. On ROCm this matmul replaces hipBLASLt
  ("Neither cuBLASLt on SM80 (Ampere) nor hipBLASLt can be made batch
  invariant", PR head `3fd5277`, `batch_invariant.py:1534`) and
  `VLLM_ROCM_USE_SKINNY_GEMM` is set to 0 (`:1613`).
- PR [#59925](https://github.com/vllm-project/vllm/pull/59925) is about the
  DeepSeek-V4 MoE path, not GEMM or RMSNorm.

### 6.7 Megatron PR #7050 and mLite

`ISEEKYAN/Megatron-LM` commit `327a238`, which is also the head of PR
[#7050](https://github.com/NVIDIA/Megatron-LM/pull/7050),
`experimental/lite/megatron/lite/model/deepseek_v4/vllm/primitive/dense.py`.

- The forward pass calls vLLM's functions (`_MHC_ENTRIES`, `:24-30`), so it
  behaves as in 6.3 when `VLLM_BATCH_INVARIANT=1`.
- The backward pass (`:512-845`) recomputes in PyTorch: the pre backward
  recomputes the norm as `1 / (norm / sqrt(K) + eps)` in bf16, and the weight
  and bias gradients are matmuls and sums over the tokens. It is not batch
  invariant, and mLite does not claim it is.

### 6.8 SGLang

`sgl-project/sglang` main, commit `4003bf1` (2026-11-08).
`_mix_stats_impl` in `python/sglang/srt/models/deepseek_v4_mhc.py:361-468`
chooses the path that computes the projection and the sum of squares:

- NVIDIA sm90 and sm100 with `SGLANG_OPT_DEEPGEMM_HC_PRENORM`: DeepGEMM; turned
  off in batch-invariant mode (`:384`).
- Blackwell, some Hopper ranges of `M`, and gfx950: `hc_mix_stats_sinkhorn`,
  the kernel below fused with Sinkhorn ("The default split-K/Sinkhorn fusion
  preserves batch invariance", `:400`).
- Other CUDA GPUs: `hc_mix_stats` (`kernels/ops/layernorm/mhc.py:2237-2420`).
- Everything else, which includes gfx942: plain PyTorch
  `x.square().mean()` and `F.linear` (`:446-449`).

`hc_mix_stats` is the closest reference for an AMD Triton kernel:

- the number of K segments depends on K only (`_num_slices_for`), and the
  segments are added in order; the K tile (64) and `num_warps` (4) are fixed;
- the sum of squares is still `tl.sum(x_tile * x_tile, axis=1)` per K tile, so
  its order rests on fixed constants, as in `aiter_config_table`;
- `BLOCK_M` follows `M` on CUDA but is fixed to 16 on HIP, with the comment
  "fp32 MFMA tile heights do not share a per-row reduction order, so one height
  serves every M" (`_block_m_for`);
- the projection runs in fp32 on HIP (`input_precision="ieee"`, "Triton on AMD
  has no tf32x3").

SGLang's aiter mHC path (`_use_aiter_mhc`, `kernels/ops/layernorm/mhc.py:37-43`)
is used only on gfx950 with `SGLANG_USE_AITER`.

### 6.9 Overview

| Implementation | Projection and sum of squares in one kernel | How the order of the sum of squares is fixed | Depends on M | gfx942 |
|---|---|---|---|---|
| DeepSeek-V4 reference | no | by PyTorch | yes | not batch invariant |
| DeepGEMM | yes | written in the kernel; tile sizes constant; caller chooses the K segments | no | NVIDIA only |
| vLLM, NVIDIA | yes (DeepGEMM) | as DeepGEMM, K segments computed as for M = 1 | no | — |
| vLLM, TileLang without DeepGEMM | yes | written in the kernel, but thread count chosen from M | yes, at M = 128 | — |
| vLLM, ROCm (aiter HIP) | yes | written in the kernel; K segments, K tile and output tile chosen from M | yes | not batch invariant |
| vLLM batch-invariant RMSNorm | — | `tl.sum` on 1024-value chunks with fixed constants | no | depends on the Triton version |
| SGLang `hc_mix_stats` | yes | `tl.sum` per 64-value tile with fixed constants; `BLOCK_M` fixed on HIP | no | not used; gfx942 runs plain PyTorch |
| aiter Triton with RL-Kernel's tables (`aiter_config_table`) | yes | `tl.sum` per tile; configs chosen so that layouts are equal | no (measured) | depends on the Triton version |

### 6.10 The principles behind these implementations

The implementations above use four principles, usually several at once.

| Principle | What it means | Used by |
|---|---|---|
| The K split depends only on K | The number of K segments is computed from K, N, the dtype or the GPU, never from M, and the segments are added in a fixed order. | DeepGEMM (the caller passes the number of segments); vLLM on NVIDIA (computed as for M = 1); vLLM's batch-invariant matmul (`_split_k_plan(K, N)`); SGLang `hc_mix_stats` (`_num_slices_for(K)`); `aiter_config_table` (`NUM_KSPLIT = 4` for every M); `fixed_order` (`num_k_splits(K)`); `aiter_hip` with a fixed split |
| The order inside a tile is fixed by fixed compile-time constants | The kernel still calls `tl.sum`; the parameters that decide its layout (`BLOCK_*`, `num_warps`) are held constant, so the compiler produces the same order every time. | `aiter_config_table`; vLLM's batch-invariant RMSNorm (`BLOCK_SIZE = 1024`, default `num_warps`); SGLang `hc_mix_stats` (`BLOCK_K = 64`, `num_warps = 4`, `BLOCK_M = 16` on HIP) |
| The order inside a tile is written in the kernel source | Each thread adds its values serially and a fixed exchange or pairwise addition combines threads; the tile sizes are constants. | DeepGEMM (serial sums and `warp_reduce_sum<4>`); `fixed_order` (pairwise rounds); aiter's HIP `mhc_pre_gemm_sqrsum` (`cross_row_sum_4`); vLLM's TileLang kernel (serial sum, warp reduction, serial sum over warps). In the last two the thread count or the split still follows M, so they are not batch invariant as a whole. |
| One code path for every M | Fused kernels, library calls and dispatch decisions that switch with M are turned off. | vLLM #59929 (no fused post-and-pre kernel in batch-invariant mode); vLLM on ROCm in batch-invariant mode (Triton matmul instead of hipBLASLt, no skinny GEMM); vLLM's CUDA layernorm (block size locked); SGLang in batch-invariant mode (no DeepGEMM, no bf16x3 projection) |
| One launch per token | Every token runs as M = 1, so the batch cannot matter; the cost is M launches. | vLLM before #59929 (ISEEKYAN `4b5c627`): the fused post-and-pre kernel looped over tokens in Python in batch-invariant mode; #59929 replaced this with one code path for every M. |

The second and third principles differ in what they depend on. Fixing the
constants relies on the compiler turning the same constants into the same
order. Between Triton 3.7.0 and 3.8.0 the layouts stayed the same but the order
inside a thread changed, and the bits of `aiter_config_table` changed with it
(section 7.4). vLLM's RMSNorm and SGLang's `hc_mix_stats` rely on the same
mechanism; that was read in their code, not measured. Writing the order in the
source does not depend on the layout or on the Triton version: `fixed_order`
kept its bits between 3.7.0 and 3.8.0.

How the projection and the sum of squares of mHC are handled:

| Implementation | K split depends only on K | Order inside a tile | One code path for every M | gfx942 |
|---|---|---|---|---|
| `aiter_config_table` (RL-Kernel PR) | yes | fixed constants | yes | batch invariant; bits change between Triton versions |
| `fixed_order` (RL-Kernel) | yes | written in the source | yes | batch invariant; same bits with Triton 3.7.0 and 3.8.0 |
| `aiter_hip` with a fixed split | yes | written in the source | partly: the output tile and the big-fuse launch still follow M; no effect measured with up to 8 segments | batch invariant |
| DeepGEMM, vLLM on NVIDIA | yes | written in the source | yes | NVIDIA only |
| SGLang `hc_mix_stats` | yes | fixed constants | yes | used on gfx950 only; gfx942 runs plain PyTorch |
| vLLM on ROCm (aiter HIP as is) | no, the split follows M | written in the source | no | not batch invariant from M = 1024 on (measured) |

## 7. Approaches we compared

All measurements: MI325X (gfx942), torch 2.12.0 with ROCm 7.14, Triton 3.7.0,
aiter `326cffa`, `n = 4`, `C = 4096`, bf16 residual, fp32 weights, 20 Sinkhorn
iterations. Jobs 68321539 and 68321590, script
`benchmarks/operators/mhc/evaluate_mhc_candidates.py`, unless another job is
named. Job 68321590 repeated everything after the `fixed_order` source was
reformatted; its outputs had the same SHA-256 digests as before.

### 7.1 What each approach does

**`aiter_config_table` (current, baseline).** `rl_engine/backends/rocm/mhc/aiter_mhc.py`
passes aiter a config per `M` from `configs/gfx942/mhc_pre/*.json` and
`configs/gfx942/mhc_head/*.json`. When a table is loaded, these parameters must
be equal in every entry: `NUM_KSPLIT`, `BLOCK_K` and `kpack` for mHC pre;
`BLOCK_M`, `BLOCK_K`, `BLOCK_C`, `num_warps` and `kpack` for mHC head. The
freedom left for mHC pre (`BLOCK_M`, `BLOCK_C` up to 128, `num_warps`) is safe
only because the layouts happen to stay equal (section 4.2). Two combinations
that would break it (`BLOCK_C = 256`, and `BLOCK_M = 16` with `num_warps = 8`)
are kept out by the tuner and the tests, not by the table loader.

**`fixed_order`.** `rl_engine/backends/rocm/mhc/fixed_order_mhc.py`, not
registered as an op. Four Triton kernels:

1. `_prenorm_kernel` runs one K segment per workgroup. For each tile of 128 K
   values (`PRENORM_BLOCK_K`) it does `acc = tl.dot(x, fn_bf16, acc)` and
   `sq = tl.fma(x, x, sq)`, where `sq` has the shape of the tile, so no sum
   happens inside the loop. After the loop, `_pairwise_sum_last` adds
   neighbouring pairs of the 128 columns in 7 rounds:

   ```python
   for i in tl.static_range(LOG2_WIDTH):
       v = tl.sum(tl.reshape(v, (ROWS, WIDTH >> (i + 1), 2)), axis=2)
   ```

   Each `tl.sum` there adds exactly two numbers, which rounds the same way
   whatever lanes hold them, so the order is set by the source.
2. `_pre_apply_kernel` and `_head_apply_kernel` add the K segments in segment
   order, compute the RMS factor and the activations with the same expressions
   as aiter, and compute the weighted stream sum one stream at a time:
   `out = w0·x0`, then `out = fma(w1, x1, out)`, and so on.
3. `_sinkhorn_kernel` processes several tokens per workgroup and computes the
   row and column sums of each 4×4 matrix with the same pairwise rounds.

The number of K segments and the K tile of 128 are part of the arithmetic and
must be the same for every `M`. `num_k_splits(K) = clamp(K // 4096, 1, 16)`
gives 4 for `K = 16384`; 1 and 16 were measured as well. All other launch
parameters only regroup tokens and hidden positions.

**`fixed_order_in_aiter_fork`.** The same source as `fixed_order`, so it was not
measured separately. It differs only in where the code lives. RL-Kernel does
not pin aiter today (`pyproject.toml:34` declares `rocm = ["aiter"]` without a
version, and neither the Dockerfiles nor the requirements install it), so this
approach also needs a way to pin a patched aiter.

**`separate_projection`.** `fixed_order_mhc_pre(..., projection="aiter_linear")`.
The projection runs as `AiterBatchInvariantLinearOp` (aiter `gemm_a16w16`
without K split), and the sum of squares as `fixed_order`'s `_prenorm_kernel`
without the dot product. The residual is read twice. mHC pre only.

**`aiter_hip`.** aiter's `mhc_pre_gemm_sqrsum` and `mhc_pre_big_fuse` called
directly with one number of K segments and one K tile for every `M`; mHC head
as vLLM runs it (weights and bias padded to 24 rows, scale to 3, no Sinkhorn).
The projection uses the fp32 weights. Points that matter for adoption:

- the kernels still choose internal sizes from `M` (section 6.5). With 1, 4 or
  8 K segments the bits did not change with `M`; with 64 segments they changed
  from `M = 4096` on, which is the first measured `M` above the big-fuse
  threshold `cu_num × 12 = 3648`;
- `mhc_pre_gemm_sqrsum` reads only the token stride of the residual
  (`x.stride(0)`, `mhc_kernels.cu:952`), so the streams of a token must be
  contiguous; a wrapper must pass a contiguous residual;
- it needs aiter's HIP build (hipcc, the composable_kernel headers, aiter's JIT).
  In the current image `import aiter` fails unless `AITER_TRITON_ONLY=1`,
  because aiter needs flydsl ≥ 0.2.4 and the image has 0.1.6; the measurement
  imported only aiter's JIT core (`AITER_AOT_IMPORT=1`). The two functions are
  internal, not aiter's public `mhc_pre`.

### 7.2 Batch invariance

Checks of `rl_engine/validation/common/batch_invariance.py`: a token alone
compared with the same token at the first and last position of batches with
`M` = 1, 8, 16, 17, 59, 64, 128, 256, 1024, 2048, 4096, 8192 and 16384;
chunking, reordering and padding at `M` = 17 and 1024; non-contiguous inputs;
two runs of the same batch. Cells name the outputs whose bits differ.

| Approach | mHC pre | mHC head |
|---|---|---|
| `aiter_config_table` | no difference | no difference |
| `fixed_order` with 1, 4 or 16 K segments | no difference | no difference |
| `fixed_order` with 4 K segments and a launch config that changes with M (`BLOCK_M = 16`, `num_warps = 8`, `BLOCK_C = 256`, 1 token per Sinkhorn workgroup below M = 64; 64, 4, 64, 64 from M = 64 on) | no difference | no difference |
| `separate_projection` with 4 K segments | no difference | — |
| aiter HIP with the split chosen per M (as vLLM calls it) | all outputs differ from M = 1024 on; also after chunking and with non-contiguous inputs | output differs from M = 1024 on; also after chunking and with non-contiguous inputs |
| `aiter_hip` with 1, 4 or 8 K segments | differs only with non-contiguous inputs | differs only with non-contiguous inputs |
| `aiter_hip` with 64 K segments | `comb_mix` and `layer_input` differ at M = 4096, 8192, 16384; non-contiguous inputs differ | output differs at M = 4096, 8192, 16384; non-contiguous inputs differ |

The aiter HIP rows come from job 68322040.

### 7.3 Independence from launch parameters (`fixed_order`)

Every launch parameter of `fixed_order` was changed, alone and in a few
combinations, against its default config, with 1, 4 and 16 K segments, at
`M` = 1, 17, 1024 and 4096. Every output was compared bit for bit.

| Kernel | Parameters changed | Configs | Result |
|---|---|---|---|
| mHC pre | `BLOCK_M` 16/64/128; `num_warps` 2/8; `BLOCK_M = 16` with `num_warps = 8`; `BLOCK_M = 128` with `num_warps = 8`; `num_stages = 2`; `waves_per_eu = 2`; tile of the apply kernel 16/64 tokens and 64/256 hidden positions; `num_warps` of the apply kernel 2/8; 1 or 64 tokens per Sinkhorn workgroup | 17 for each of the 3 segment counts | all identical |
| mHC head | the same without Sinkhorn | 15 for each of the 3 segment counts | all identical |

For comparison, aiter's kernels change bits with `BLOCK_K`, `BLOCK_C = 256`,
`BLOCK_M = 16` together with `num_warps = 8`, and `kpack` (mHC pre), and with
`BLOCK_M`, `BLOCK_K`, `BLOCK_C`, `num_warps` and `kpack` (mHC head), section 5.

### 7.4 Triton 3.7.0 against Triton 3.8.0

SHA-256 digests of every output for the same inputs, computed with the image's
Triton 3.7.0 and with Triton 3.8.0 (`3.8.0+amd.rocm7.2.0.git7cb7b059`) on the
same GPU (jobs 68321539, 68321590 and 68321583):

| Outputs | 3.7.0 and 3.8.0 |
|---|---|
| `aiter_config_table` mHC pre (4 K segments), M = 17 | equal |
| `aiter_config_table` mHC pre (4 K segments), M = 1024 | all three outputs differ |
| `aiter_config_table` mHC head (aiter's config), M = 17 and 1024 | differ |
| `fixed_order` mHC pre and mHC head (4 K segments), M = 17 and 1024 | equal |

Which sum moves (job 68321583, the three aiter configs of section 5 on the
four kinds of input):

| aiter config | random input | exact sum of squares | exact projection | both exact |
|---|---|---|---|---|
| mHC pre, no K split | M = 17 equal; M = 1024 `post_mix`, `comb_mix` differ | equal | M = 17 equal; M = 1024 all three differ | equal |
| mHC pre, 4 K segments | M = 17 equal; M = 1024 all three differ | equal | M = 17 equal; M = 1024 all three differ | equal |
| mHC head | differs at M = 17 and 1024 | equal | differs at M = 17 and 1024 | equal |

The outputs differ only when the sum of squares rounds, so the sum that moves
is the sum of squares, whose layout is the same in both versions; the cause is
the new order inside a thread (section 4.3). Why mHC pre keeps its bits at
`M = 17` was not investigated (`M` is a compile-time constant of these kernels,
so `M = 17` and `M = 1024` are compiled separately).

### 7.5 Error against the fp32 reference

`M = 1024`, random input. Reference: `NativeMHCPreOp.forward_fp32` and
`NativeMHCHeadOp.forward_fp32`, which project with the fp32 weights. Each cell:
largest / mean absolute error.

| Approach | `post_mix` | `comb_mix` | `layer_input` | mHC head output |
|---|---|---|---|---|
| `aiter_own_config` | 3.600e-03 / 6.416e-04 | 1.227e-03 / 2.053e-04 | 2.182e-02 / 1.502e-03 | 1.893e-02 / 1.393e-03 |
| `aiter_config_table` (baseline) | 3.601e-03 / 6.416e-04 | 1.227e-03 / 2.053e-04 | 2.182e-02 / 1.502e-03 | 1.893e-02 / 1.393e-03 |
| `fixed_order`, 1 / 4 / 16 K segments | 3.600e-03 to 3.601e-03 / 6.416e-04 | 1.227e-03 / 2.053e-04 | 2.182e-02 / 1.502e-03 | 1.893e-02 / 1.393e-03 |
| `separate_projection`, 4 K segments | 3.600e-03 / 6.416e-04 | 1.227e-03 / 2.053e-04 | 2.182e-02 / 1.502e-03 | — |
| aiter HIP, split chosen per M | 1.311e-06 / 2.162e-07 | 5.066e-07 / 7.387e-08 | 3.125e-02 / 2.532e-03 | 3.125e-02 / 2.426e-03 |
| `aiter_hip`, 1 K segment | 6.437e-06 / 8.687e-07 | 2.503e-06 / 3.232e-07 | 3.124e-02 / 2.532e-03 | 3.125e-02 / 2.426e-03 |
| `aiter_hip`, 4 K segments | 3.934e-06 / 4.871e-07 | 9.835e-07 / 1.582e-07 | 3.124e-02 / 2.532e-03 | 3.125e-02 / 2.426e-03 |
| `aiter_hip`, 8 K segments | 2.384e-06 / 3.673e-07 | 7.153e-07 / 1.202e-07 | 3.124e-02 / 2.532e-03 | 3.125e-02 / 2.426e-03 |

The Triton approaches round the weights to bf16 before the projection, as
aiter's Triton kernels do, and that rounding dominates their error; the order
of the sums does not show at this scale. `aiter_hip` projects with the fp32
weights, so its mixing weights are about 1000 times closer to the reference.
Its `layer_input` and head output have a 1.7 times larger mean error than the
baseline; the cause was not investigated.

### 7.6 Device time

Forward pass, milliseconds per call, measured with GPU events around a CUDA
graph of 20 calls, median over 5 replays. Baseline: `aiter_config_table`. Every
ratio column is time of the approach divided by time of the baseline in the
same run; below 1 means faster than the baseline. For `fixed_order` and
`separate_projection` each cell is the fastest of 18 launch configs
(`BLOCK_M` 16/32/64 × `num_warps` 2/4/8 × two apply tiles), which all give the
same bits (7.3). `aiter_own_config` is not batch invariant and is shown only for
reference; it cannot launch at `M = 8192` on gfx942 because it needs 128 KB of
LDS.

The first two tables come from two runs on the same node (jobs 68321539 and
68321590); a cell with two numbers gives the lowest and highest of the two run
medians.

#### mHC pre

| M | `aiter_config_table` (ms) | `fixed_order`, 4 K segments (ms) | `fixed_order`, 16 K segments (ms) | `fixed_order`, 1 K segment (ms) | `separate_projection`, 4 K segments (ms) | `aiter_own_config` (ms) | `fixed_order` 4 segments / baseline (ratio) | `fixed_order` 16 segments / baseline (ratio) | `fixed_order` 1 segment / baseline (ratio) | `separate_projection` / baseline (ratio) | `aiter_own_config` / baseline (ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0558–0.0559 | 0.0451–0.0457 | 0.0236–0.0238 | 0.1379–0.1386 | 0.1976–0.1981 | 0.1983–0.1987 | 0.81–0.82 | 0.42–0.43 | 2.47–2.48 | 3.54 | 3.55 |
| 16 | 0.0497–0.0498 | 0.0442–0.0443 | 0.0245–0.0246 | 0.1350–0.1357 | 0.1974–0.1979 | 0.2433–0.2437 | 0.89 | 0.49 | 2.72 | 3.97 | 4.89–4.90 |
| 64 | 0.0652–0.0654 | 0.0446 | 0.0238 | 0.1362 | 0.1974–0.1977 | 0.2507–0.2508 | 0.68 | 0.36–0.37 | 2.08–2.09 | 3.02–3.03 | 3.83–3.85 |
| 256 | 0.0589 | 0.0452–0.0453 | 0.0253 | 0.1368 | 0.2026–0.2035 | 0.2554–0.2555 | 0.77 | 0.43 | 2.32 | 3.44–3.46 | 4.34 |
| 1024 | 0.0841–0.0850 | 0.0586–0.0588 | 0.0466–0.0468 | 0.1642 | 0.2183–0.2184 | 0.0511–0.0522 | 0.69–0.70 | 0.55 | 1.93–1.95 | 2.57–2.60 | 0.61 |
| 2048 | 0.0803–0.0820 | 0.0738–0.0744 | 0.0684–0.0687 | 0.1747–0.1753 | 0.2363–0.2372 | 0.0885–0.0890 | 0.90–0.93 | 0.84–0.85 | 2.14–2.18 | 2.89–2.94 | 1.08–1.11 |
| 4096 | 0.1215–0.1239 | 0.1154–0.1174 | 0.1033–0.1063 | 0.1940–0.1947 | 0.2665–0.2679 | 0.1437–0.1465 | 0.95 | 0.83–0.87 | 1.57–1.60 | 2.15–2.20 | 1.16–1.21 |
| 8192 | 0.2190–0.2255 | 0.1998–0.2006 | 0.2166–0.2206 | 0.2701–0.2703 | 0.3578–0.3581 | cannot launch (LDS) | 0.89–0.91 | 0.98–0.99 | 1.20–1.23 | 1.59–1.64 | — |
| 16384 | 0.4127–0.4644 | 0.3828–0.4242 | 0.4860–0.4979 | 0.4648–0.4899 | 0.5748–0.6097 | 0.4666–0.4827 | 0.91–0.93 | 1.07–1.18 | 1.05–1.13 | 1.31–1.39 | 1.04–1.13 |
| 32768 | 0.8072–0.8273 | 0.7564–0.7920 | 1.0888–1.1070 | 0.8400–0.8833 | 1.1640–1.2162 | 0.8925–0.9091 | 0.94–0.96 | 1.32–1.37 | 1.04–1.07 | 1.44–1.47 | 1.10–1.11 |
| 65536 | 1.5191–1.6493 | 1.4373–1.5711 | 2.2426–2.2810 | 1.4096–1.5215 | 2.3081–2.4126 | 1.6955–1.7916 | 0.95 | 1.38–1.48 | 0.92–0.93 | 1.46–1.52 | 1.09–1.12 |

#### mHC head

| M | `aiter_config_table` (ms) | `fixed_order`, 4 K segments (ms) | `fixed_order`, 16 K segments (ms) | `fixed_order`, 1 K segment (ms) | `aiter_own_config` (ms) | `fixed_order` 4 segments / baseline (ratio) | `fixed_order` 16 segments / baseline (ratio) | `fixed_order` 1 segment / baseline (ratio) | `aiter_own_config` / baseline (ratio) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.7451–0.7457 | 0.0297–0.0301 | 0.0128–0.0129 | 0.1030 | 0.8209–0.8214 | 0.04 | 0.02 | 0.14 | 1.10 |
| 16 | 0.7464–0.7467 | 0.0296–0.0297 | 0.0128 | 0.1023–0.1025 | 0.8334–0.8335 | 0.04 | 0.02 | 0.14 | 1.12 |
| 64 | 0.7483 | 0.0296 | 0.0129 | 0.1029–0.1030 | 0.8358–0.8360 | 0.04 | 0.02 | 0.14 | 1.12 |
| 256 | 0.7431–0.7432 | 0.0300 | 0.0135 | 0.1034 | 0.8380–0.8382 | 0.04 | 0.02 | 0.14 | 1.13 |
| 1024 | 0.7759–0.7772 | 0.0411 | 0.0249–0.0250 | 0.1251–0.1253 | 0.9005 | 0.05 | 0.03 | 0.16 | 1.16 |
| 2048 | 0.7802–0.7812 | 0.0597–0.0604 | 0.0422–0.0433 | 0.1346–0.1347 | 0.9077–0.9080 | 0.08 | 0.05–0.06 | 0.17 | 1.16 |
| 4096 | 0.7872–0.7886 | 0.0811–0.0816 | 0.0761–0.0772 | 0.1530–0.1531 | 0.9139–0.9153 | 0.10 | 0.10 | 0.19 | 1.16 |
| 8192 | 1.4278–1.4318 | 0.1469–0.1470 | 0.1502–0.1524 | 0.2327–0.2362 | 1.9152–1.9159 | 0.10 | 0.10–0.11 | 0.16–0.17 | 1.34 |
| 16384 | 2.8747–2.8792 | 0.3503–0.4345 | 0.3524–0.4188 | 0.3977–0.4608 | 3.8589–3.8630 | 0.12–0.15 | 0.12–0.15 | 0.14–0.16 | 1.34 |
| 32768 | 5.1495–5.1618 | 0.6799–0.7074 | 0.7172–0.7296 | 0.6797–0.6988 | 6.7778–6.7855 | 0.13–0.14 | 0.14 | 0.13–0.14 | 1.31–1.32 |
| 65536 | 10.7165–10.7307 | 1.2896–1.3927 | 1.4297–1.5369 | 1.2655–1.3696 | 12.0050–12.0124 | 0.12–0.13 | 0.13–0.14 | 0.12–0.13 | 1.12 |

The baseline mHC head is slow because aiter's head kernel lets one workgroup
of 16 tokens run the whole loop over K and over the hidden positions; at
`M = 1` a single workgroup does all the work.

#### `aiter_hip`, with the baseline and `fixed_order` from the same run

Job 68322040, one run on the same node: `evaluate_mhc_candidates.py
--sections 4` first, then `probe_aiter_hip_mhc_pre.py`. `aiter_hip` has no
launch parameters to tune.

| M | baseline, pre (ms) | `fixed_order` 4 segments, pre (ms) | `aiter_hip` 4 segments, pre (ms) | `aiter_hip` 8 segments, pre (ms) | `fixed_order` 4 segments / baseline, pre (ratio) | `aiter_hip` 4 segments / baseline, pre (ratio) | `aiter_hip` 8 segments / baseline, pre (ratio) | baseline, head (ms) | `fixed_order` 4 segments, head (ms) | `aiter_hip` 8 segments, head (ms) | `aiter_hip` 8 segments / baseline, head (ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0561 | 0.0452 | 0.0391 | 0.0242 | 0.81 | 0.70 | 0.43 | 0.7450 | 0.0298 | 0.0231 | 0.03 |
| 16 | 0.0504 | 0.0441 | 0.0405 | 0.0244 | 0.88 | 0.80 | 0.48 | 0.7463 | 0.0298 | 0.0234 | 0.03 |
| 64 | 0.0654 | 0.0447 | 0.0414 | 0.0248 | 0.68 | 0.63 | 0.38 | 0.7481 | 0.0295 | 0.0238 | 0.03 |
| 256 | 0.0590 | 0.0452 | 0.0420 | 0.0267 | 0.77 | 0.71 | 0.45 | 0.7430 | 0.0301 | 0.0264 | 0.04 |
| 1024 | 0.0845 | 0.0587 | 0.0515 | 0.0352 | 0.69 | 0.61 | 0.42 | 0.7776 | 0.0413 | 0.0350 | 0.05 |
| 2048 | 0.0849 | 0.0747 | 0.0645 | 0.0565 | 0.88 | 0.76 | 0.67 | 0.7810 | 0.0603 | 0.0590 | 0.08 |
| 4096 | 0.1227 | 0.1133 | 0.1195 | 0.1174 | 0.92 | 0.97 | 0.96 | 0.7877 | 0.0815 | 0.1140 | 0.14 |
| 8192 | 0.2141 | 0.1988 | 0.2017 | 0.2086 | 0.93 | 0.94 | 0.97 | 1.4302 | 0.1485 | 0.1979 | 0.14 |
| 16384 | 0.4421 | 0.4115 | 0.3875 | 0.3676 | 0.93 | 0.88 | 0.83 | 2.8778 | 0.3579 | 0.3547 | 0.12 |
| 32768 | 0.7796 | 0.7262 | 0.7316 | 0.7280 | 0.93 | 0.94 | 0.93 | 5.1556 | 0.6657 | 0.6916 | 0.13 |
| 65536 | 1.5166 | 1.4563 | 1.4735 | 1.4556 | 0.96 | 0.97 | 0.96 | 10.6996 | 1.2847 | 1.3890 | 0.13 |

The full table, with 1 and 64 K segments and with the split chosen per `M`, is
in `/home/AMD/zitwang/dev/logs/mhc-hip-probe-68322040.md`.

### 7.7 Comparison

| | `aiter_config_table` (current) | `fixed_order` | `fixed_order_in_aiter_fork` | `separate_projection` | `aiter_hip` |
|---|---|---|---|---|---|
| Batch invariant with one software version | yes | yes | same as `fixed_order` | yes | yes with 1, 4 or 8 K segments; no with 64 |
| Bits independent of launch parameters | no; held by the tables (7 parameters forced equal, 2 combinations excluded only by tests) | yes (96 configs) | same as `fixed_order` | yes for the sum of squares; the projection keeps `gemm_a16w16`'s rules (`NUM_KSPLIT`, `kpack` fixed) | no parameters exposed; internal sizes chosen from M |
| Same bits with Triton 3.7.0 and 3.8.0 | no | yes | same as `fixed_order` | not measured | not applicable (HIP); hipcc versions not measured |
| Largest `post_mix` error against the fp32 reference | 3.6e-3 (weights rounded to bf16) | 3.6e-3 | same as `fixed_order` | 3.6e-3 | 2.4e-6 to 6.4e-6 (fp32 weights) |
| Mean `layer_input` error | 1.50e-3 | 1.50e-3 | same as `fixed_order` | 1.50e-3 | 2.53e-3 |
| mHC pre time / baseline | 1 | 4 segments: 0.68–0.96. 16 segments: 0.36–0.55 up to M = 1024, 1.07–1.48 from M = 16384 | same as `fixed_order` | 1.31–3.97 | 8 segments: 0.38–0.97. 4 segments: 0.61–0.97 |
| mHC head time / baseline | 1 | 4 segments: 0.04–0.15 | same as `fixed_order` | — | 8 segments: 0.03–0.14 |
| Where the code lives | aiter's Triton kernels, aiter not pinned by RL-Kernel | RL-Kernel only, Triton; no aiter | an aiter fork plus a pinning mechanism that RL-Kernel does not have yet | aiter `gemm_a16w16` plus the `fixed_order` kernel | aiter's HIP module (hipcc, composable_kernel headers, JIT); internal functions; a normal `import aiter` needs flydsl ≥ 0.2.4 |
| Work still missing | — | backward pass (the baseline uses aiter's PyTorch recomputation), registry entry and tests, other hidden sizes than 4096, gfx950, choosing the number of K segments | as `fixed_order`, plus pinning aiter | — | wrapper that makes the residual contiguous, backward pass, mHC head without the 20 unused projection rows |

Questions that stay open whichever approach is chosen:

- the precision of the projection: the DeepSeek-V4 reference uses fp32 weights,
  the Triton approaches round them to bf16, `aiter_hip` keeps fp32 (an open
  question for the RFC author already);
- whether results must also match across Triton or ROCm versions;
  `aiter_config_table` does not, `fixed_order` did for Triton 3.7.0 and 3.8.0.

## 8. How the problem was found

From the agent transcripts of the KLR work (`~/.claude/projects/-home-AMD-zitwang-dev-KLR/`)
and the KLR analysis document `issue-81-zitian-wang-mhc-tasks-analysis.md`:

| Date | What happened |
|---|---|
| 2026-10-29 and 30 | The RFC evaluation attributes the bit changes of mHC pre from `M = 1024` on to aiter's K split (analysis document lines 254 and 343). The test that isolates each sum ran with aiter's own configs, where `NUM_KSPLIT`, `BLOCK_K` and `BLOCK_C` change together, so it could not tell them apart. |
| 2026-11-02 | First written suspicion that the order of the sum-of-squares `tl.sum` follows `BLOCK_K`; marked as unverified. |
| 2026-11-03 | Tuning per `M` measures that `BLOCK_K` changes the bits of the mHC kernels but not those of `gemm_a16w16`; the analysis document (line 628) still says "probably". |
| 2026-11-04 | The weekly report names the K split as the cause, with the sum of squares changing "along with it"; the script that would separate the two is written but not run. |
| 2026-11-06 | Job 68297426: `BLOCK_K` changes the bits through the sum of squares only. |
| 2026-11-08 | Jobs 68321538, 68321539, 68321583, 68321590 and 68322040 (this page). |

## 9. How to reproduce

| Measurement | Script | Job | Result file under `/home/AMD/zitwang/dev/logs/` |
|---|---|---|---|
| Bits per input kind and Triton layouts of aiter's configs | `benchmarks/operators/mhc/analyze_aiter_reductions.py` | 68321538 | `mhc-bi-analysis-68321538-triton-image.md`, `…-triton-3.8.0-amd.md` |
| Which sum changes between Triton 3.7.0 and 3.8.0 | `analyze_aiter_reductions.py --digest` | 68321583 | `mhc-bi-triton-versions-68321583-compare.md` |
| `fixed_order` and `separate_projection`: launch parameters, batch invariance, error, device time, digests across Triton versions | `benchmarks/operators/mhc/evaluate_mhc_candidates.py` | 68321539, 68321590 | `mhc-bi-candidates-68321539.md`, `mhc-bi-candidates-68321590.md` and their `-digest-*` files |
| `aiter_hip` | `benchmarks/operators/mhc/probe_aiter_hip_mhc_pre.py` | 68322040 | `mhc-hip-probe-68322040.md`, `mhc-hip-probe-68322040-baseline.md` |
| First isolation of `BLOCK_K` | KLR `scripts/measure_mhc_block_k_bit_source.py` | 68297426 | `mhc-block-k-bit-source-68297426.md` |

The job scripts are `run-mhc-bi-analysis.sbatch`, `run-mhc-bi-candidates.sbatch`
and `run-mhc-bi-triton-versions.sbatch` in `/home/AMD/zitwang/dev/`. The
`aiter_hip` measurement ran on a copy of the aiter checkout without `.git`, with
`3rdparty/composable_kernel` fetched at aiter's submodule commit `af9e1d1`, so
that aiter's JIT build did not write into `/home/AMD/zitwang/dev/aiter`; that
copy has been deleted.
