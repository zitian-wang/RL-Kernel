# mHC Reduction Order and Batch Invariance on ROCm

Investigation for DSV4 RFC tasks 1 and 8 ([#508](https://github.com/RL-Align/RL-Kernel/issues/508)),
branch `feat/issue-81-mhc-bi-reduction`. Every statement below is either read
in the cited source (file:line) or measured in the cited Slurm job; the few
inferences are marked **inferred**. Section 7 lists the candidates with their
measured trade-offs; the choice between them is open.

Contents:

1. [Summary](#1-summary)
2. [What mHC pre and head reduce](#2-what-mhc-pre-and-head-reduce)
3. [How aiter computes them and picks launch configs](#3-how-aiter-computes-them-and-picks-launch-configs)
4. [Why a launch parameter changes the bits](#4-why-a-launch-parameter-changes-the-bits)
5. [Measured attribution and layouts](#5-measured-attribution-and-layouts)
6. [Reference implementations](#6-reference-implementations)
7. [Candidates](#7-candidates)
8. [How this was found](#8-how-this-was-found)
9. [Reproduction](#9-reproduction)

## 1. Summary

- mHC pre and head each contain two long reductions per token and one short
  one: the projection `x @ fn^T` and the RMS sum of squares `Σ x²`, both over
  K = n·C = 16384, and the weighted sum over the n = 4 residual streams.
- In aiter's Triton kernels the projection is `tl.dot`; its bits change only
  with how K is split (`NUM_KSPLIT`) and with `kpack`. Tile sizes, `num_warps`
  and the MFMA instruction shape do not change them.
- The sum of squares and the stream sum are `tl.sum`. Triton 3.7 lowers `tl.sum`
  into a per-thread sequential sum, a lane butterfly and, if needed, a
  cross-warp step, and the shape of that tree is the operand's layout, which
  Triton derives from `BLOCK_M`, `BLOCK_K`, `BLOCK_C` and `num_warps`. A launch
  parameter changes the bits exactly when it changes that layout along the
  reduced axis (section 5, every config measured).
- aiter picks those parameters per M, so its own configs are not batch
  invariant. The current RL-Kernel ops (candidate A) avoid this by giving every
  M a config with the same layouts. Within one Triton version this holds
  (measured); across versions it does not: Triton 3.8.0 keeps the layouts but
  adds a thread's own elements as a tree instead of left to right, and A's
  outputs change (section 7.5).
- NVIDIA has batch-invariant mHC paths in vLLM (DeepGEMM with the split taken
  as for M = 1) and SGLang (DeepGEMM off in batch-invariant mode; Triton
  `hc_mix_stats` with K slices from K). On ROCm: vLLM calls aiter's HIP
  `mhc_pre`, which picks its K split per M and has no batch-invariant gate;
  SGLang runs its Triton `hc_mix_stats` path on gfx950 only (with `BLOCK_M`
  fixed on HIP) and plain PyTorch on gfx942. That SGLang kernel still relies on
  fixed constexprs for its `tl.sum`, like A; only DeepGEMM writes the
  sum-of-squares tree by hand.
- Candidate B (`rl_engine/backends/rocm/mhc/fixed_order_mhc.py`) writes every
  reduction order in the kernel. Measured: the bits do not change with any
  launch parameter (96 launch configs) nor between Triton 3.7.0 and 3.8.0, the
  ops are batch invariant for M = 1 to 16384, and the error against the fp32
  reference equals aiter's.

- Device time (MI325X, forward, vs A): B with S = 4 takes 0.68–0.96× of A for
  mhc_pre and 0.04–0.15× for mhc_head (A's head runs the whole K and C loop in
  one workgroup per 16 rows: 0.75 ms at M = 1). D is 1.3–4.0× slower than A.
  E (aiter's HIP mhc_pre, the op vLLM calls on ROCm) is batch invariant only
  with a fixed small split (1, 4 or 8 segments), is as fast as B, and projects
  with fp32 `fn`, which puts its mixing weights 1000× closer to the fp32
  reference than A and B.

## 2. What mHC pre and head reduce

DeepSeek-V4-Flash-0731 (`inference/model.py:676-680`, `:730-734`, Hugging Face
revision `60d8d70`):

```python
x = x.flatten(2).float()                                   # [tokens, n*C], n*C = 16384
rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
mixes = F.linear(x, hc_fn) * rsqrt                         # [tokens, 24] (pre) or [tokens, 4] (head)
...
y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)    # sum over the n = 4 streams
```

| Reduction | Length per output | Where in aiter |
|---|---|---|
| projection `x @ fn^T` | K = 16384 | `tl.dot` in the K loop |
| sum of squares `Σ x²` | K = 16384 | `tl.sum(x*x, axis=1)` per K tile, in the same K loop |
| stream sum `Σ_i pre_i · x_i` | n = 4 | `tl.sum(pre[:, :, None] * x, axis=1)` per C tile |
| Sinkhorn row/column sums (pre) | n = 4 | separate kernel, one row per workgroup |

Batch invariance (BI) here means: the bits of a token's outputs do not depend
on the other tokens in the batch (M), checked by
`rl_engine/validation/common/batch_invariance.py`.

## 3. How aiter computes them and picks launch configs

aiter `326cffa`, `aiter/ops/triton/fusions/mhc.py` (wrappers) and
`aiter/ops/triton/_triton_kernels/fusions/mhc.py` (kernels).

### 3.1 mHC pre

`mhc()` (`fusions/mhc.py:251-369`) takes one of two paths by `NUM_KSPLIT`, and
passes the same config dict (`**config`) to every kernel it launches:

- `NUM_KSPLIT = 1`: `_mhc_fused_kernel`, grid `(cdiv(M, BLOCK_M), 3)`. Each of
  the pre, post and res workgroups runs the whole K loop (`:294-314`):

  ```python
  for k in range(0, K, BLOCK_K):
      x_tile = tl.load(...)                              # [BLOCK_M, BLOCK_K] bf16
      phi_tile = tl.load(...).to(x_ptr.dtype.element_ty) # fn rounded to bf16
      acc = tl.dot(x_tile, phi_tile, acc=acc)            # projection
      acc_sq += tl.sum(x_tile_f32 * x_tile_f32, axis=1)  # sum of squares
  ```

  The pre workgroup then computes the stream sum over C tiles of `BLOCK_C`
  (`_mhc_apply_pre_mix_tile`, `:44`:
  `li_acc = tl.sum(pre_mix_2d[:, :, None] * x_tile, axis=1)`).
- `NUM_KSPLIT > 1`: `_mhc_fused_split_kernel` (grid `(cdiv(M, BLOCK_M),
  NUM_KSPLIT)`, the same loop over one K segment, `:476-493`) writes partial
  `acc` and `acc_sq`; `_mhc_reduce_apply_kernel` (grid `(cdiv(M, BLOCK_M),
  cdiv(C, BLOCK_C))`) adds the partials in segment order (`:674-686`), applies
  RMS, sigmoid and bias, and computes the stream sum for its C tile.
- Sinkhorn always runs afterwards in `_mhc_asymmetric_sinkhorn_kernel`, one row
  per workgroup, with a `num_warps` from aiter's tuned config that does not
  depend on M (`fusions/mhc.py:1044-1060`).

Which parameter acts where:

| Parameter | Acts on |
|---|---|
| `NUM_KSPLIT` | number of K segments for the projection and the sum of squares; merge in segment order |
| `BLOCK_K` | step of the K loop: tile of `tl.dot` and of the sum-of-squares `tl.sum` |
| `BLOCK_M` | rows per workgroup in every kernel |
| `BLOCK_C` | C tile of the stream sum (output dimension, not reduced) |
| `num_warps`, `num_stages`, `waves_per_eu`, `kpack`, `matrix_instr_nonkdim` | Triton launch options of every kernel |

With `config=None`, aiter picks the config per M from
`configs/gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=4096.json`:

| M bucket | NUM_KSPLIT | BLOCK_K | BLOCK_C | BLOCK_M | num_warps |
|---|---|---|---|---|---|
| ≤ 256 | 1 | 128 | 128 | 32 | 4 or 8 |
| ≤ 1024 | 8 | 32 | 128 | 32 | 4 |
| ≤ 2048 | 8 | 128 | 64 | 64 | 8 |
| ≤ 4096 | 8 | 256 | 128 | 64 | 4 |
| ≤ 8192 | 4 | 256 | 256 | 64 | 4 |
| ≤ 16384, any | 2 | 128 | 128 | 64 | 4 |

Every column except `BLOCK_M` changes some reduction order (section 5), so a
token's bits change at the bucket boundaries.

### 3.2 mHC head

`_mhc_head_dsv4_forward` (`fusions/mhc.py:1188-1242`) launches one kernel,
`_mhc_head_kernel` (`:108-186`), grid `(cdiv(M, BLOCK_M),)`: the same K loop
with `tl.dot` and `tl.sum(x*x, axis=1)` (`:142-156`), N padded to
`N_TILE = 16` for `tl.dot`, then the stream sum over C tiles. Its config comes
from aiter's tuned `_mhc_head_kernel` entry and does not depend on M; the public
`config` argument cannot set `num_warps` or `num_stages` (duplicate keyword).

### 3.3 mHC post

`_mhc_post_kernel` sums the four streams with `tl.static_range(n)` loops
(`:914-955`), an order written in the kernel; it has no `tl.sum` and its
config does not depend on M.

## 4. Why a launch parameter changes the bits

### 4.1 The projection (`tl.dot`)

Triton v3.7.0 (`/home/AMD/zitwang/dev/triton-3.7.0`, commit `5f3f125`)
replaces `tl.dot` by MFMA instructions in
`third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp`. Each MFMA
has a fixed k (`kDim`), and `tl.dot(x[BLOCK_M, BLOCK_K], fn[BLOCK_K, N], acc)`
becomes `BLOCK_K / kDim` MFMAs accumulating into the same registers in K
order; the next loop iteration continues on the same `acc`. So `BLOCK_K` only
decides where the MFMA sequence is cut into loop iterations.

`kpack` is a compile option of the AMD backend
(`third_party/amd/backend/compiler.py:56`, passed to
`add_accelerate_matmul` at `:225`). `AccelerateAMDMatmul.cpp:683-717`:

```cpp
// kWidth: the number of **consecutive** elements each thread loads from shared memory
// kBase:  the number of elements each thread holds for a single mfma instruction
// kWidth = kBase,     i.e. kPack = 1: each load from shared memory results in one mfma
// kWidth = 2 * kBase, i.e. kPack = 2: each load results in two mfma instructions
auto kWidth = kBase;
...
  kWidth *= kPack;
```

With `kpack = 2` a thread's consecutive k elements are shared between two
MFMAs, so the set of k each MFMA sums changes. Measured (section 5): `kpack=2`
gives dot operands with `kWidth 8` instead of 4 and changes the bits with an
exact sum of squares, not with an exact projection. The MFMA instruction shape
(`matrix_instr_nonkdim=32`: 32x32x8 instead of 16x16x16) does not change the
bits. Since gfx950, `kpack != 1` is overwritten to 1 (`compiler.py:90-94`).

### 4.2 `tl.sum`

**Layout of the operand.** `x_tile` gets the layout of its load, built by the
coalescing pass (`lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp:63-91`):
elements per thread along the contiguous axis
`spt = min(alignment, 128 bits / element, BLOCK_M·BLOCK_K / (64·num_warps))`,
i.e. `min(8, BLOCK_M·BLOCK_K / (64·num_warps))` for bf16. The blocked layout
builder (`include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:836-850`)
then gives the contiguous axis `BLOCK_K / spt` lanes (at most 64) and puts the
remaining lanes and the warps on the rows.

**Lowering.** `lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp:38-74`:

1. `reduceWithinThreads` (`:120-151`): each thread adds its `spt` values in register order;
2. `reduceWithinWarps` → `warpReduce` (`:155-171`): xor shuffles over the lanes
   on the axis, distances `lanes/2, …, 1`; on gfx942 a full 64-lane reduction
   uses DPP instead (`third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp:372-452`);
3. if warps also lie on the axis: partials go through LDS and are reduced again (`:56-74`).

So the tree of one row is fixed by (`spt`, lanes, warps) along the reduced
axis, and these follow `BLOCK_M`, `BLOCK_K`, `BLOCK_C` and `num_warps`. LLVM
may also contract a product with the first addition into an FMA, which depends
on which values meet in the first step (**inferred** from the mHC head
`BLOCK_C=64` row in section 5, whose stream-sum tree only adds zeros at the
first level and whose difference appears only with inexact products).

**What follows for aiter's sum of squares** (spt rule above, all confirmed by
the TTGIR in section 5):

| Kernel, config | spt × lanes | Same tree as the reference? |
|---|---|---|
| pre, `BLOCK_M=32, BLOCK_K=128, num_warps=4` (reference) | 8 × 16 | — |
| pre, `BLOCK_M` 16–128 with `num_warps` 1–4, or 64 with 8 | 8 × 16 | yes |
| pre, `BLOCK_M=16, num_warps=8` | 4 × 32 | no |
| head, `BLOCK_M=16, BLOCK_K=64, num_warps=4` (aiter's) | 4 × 16 | — |
| head, `BLOCK_M=32` or `num_warps=2` | 8 × 8 | no |
| head, `num_warps=8` | 2 × 32 | no |
| head, `BLOCK_M=32, num_warps=8` | 4 × 16 | yes |

For mhc_pre with `BLOCK_K=128` the element count per thread reaches the cap of
8 for almost every config, which is why `BLOCK_M` and `num_warps` look free;
for mhc_head with `BLOCK_K=64` it is 4, below the cap, so they are not.

## 5. Measured attribution and layouts

Job 68321538, MI325X, Triton 3.7.0 (image), aiter `326cffa`,
`benchmarks/operators/mhc/analyze_aiter_reductions.py`. Each config changes
the listed parameters of a class reference and runs on four inputs at M = 17
and 1024:

- `random`: both long reductions round;
- `exact sum of squares`: residual in {−2, −1, 1, 2}, every Σx² is an integer
  below 2^24 and exact in any order — a difference comes from the projection
  or the stream sum;
- `exact projection`: one 1 per fn row, the projection is one input element —
  a difference comes from the sum of squares or the stream sum;
- `both exact`: the mixing weights are then equal for every config — a
  difference comes from the stream sum.

The layout columns come from the TTGIR of the launched kernels: for the
operand of each `tt.reduce`, the length of the reduced axis and its elements
per thread (spt), lanes and warps; for the dot, the MFMA shape and `kWidth`.
Cells: `same` or the outputs that differ (count at M = 17 / 1024).

### 5.1 mhc_pre, NUM_KSPLIT=1 (_mhc_fused_kernel)

Reference: `{'NUM_KSPLIT': 1, 'BLOCK_M': 32, 'BLOCK_K': 128, 'BLOCK_C': 128, 'num_warps': 4, 'num_stages': 1, 'waves_per_eu': 2}`

| config | random | exact sum of squares | exact projection | both exact |
|---|---|---|---|---|
| reference | reference | reference | reference | reference |
| BLOCK_M=16 | same | same | same | same |
| BLOCK_M=64 | same | same | same | same |
| BLOCK_M=128 | same | same | same | same |
| num_warps=1 | same | same | same | same |
| num_warps=2 | same | same | same | same |
| num_warps=8 | same | same | same | same |
| BLOCK_M=16 num_warps=2 | same | same | same | same |
| BLOCK_M=16 num_warps=8 | differs: post_mix (2 at M=17, 159 at M=1024); comb_mix (32 at M=17, 1267 at M=1024); layer_input (1 at M=17, 26 at M=1024) | same | differs: post_mix (5 at M=17, 162 at M=1024); comb_mix (33 at M=17, 1257 at M=1024); layer_input (1 at M=17, 17 at M=1024) | same |
| BLOCK_M=64 num_warps=8 | same | same | same | same |
| num_stages=2 | same | same | same | same |
| BLOCK_K=32 | differs: post_mix (17 at M=17, 1456 at M=1024); comb_mix (108 at M=17, 8912 at M=1024); layer_input (4 at M=17, 246 at M=1024) | same | differs: post_mix (19 at M=17, 1418 at M=1024); comb_mix (112 at M=17, 8901 at M=1024); layer_input (4 at M=17, 230 at M=1024) | same |
| BLOCK_K=64 | differs: post_mix (18 at M=17, 1231 at M=1024); comb_mix (160 at M=17, 8143 at M=1024); layer_input (3 at M=17, 194 at M=1024) | same | differs: post_mix (20 at M=17, 1239 at M=1024); comb_mix (147 at M=17, 8059 at M=1024); layer_input (7 at M=17, 194 at M=1024) | same |
| BLOCK_K=256 | differs: post_mix (12 at M=17, 920 at M=1024); comb_mix (141 at M=17, 6627 at M=1024); layer_input (4 at M=17, 129 at M=1024) | same | differs: post_mix (18 at M=17, 949 at M=1024); comb_mix (146 at M=17, 6655 at M=1024); layer_input (3 at M=17, 144 at M=1024) | same |
| BLOCK_C=64 | same | same | same | same |
| BLOCK_C=256 | differs: layer_input (1 at M=17, 140 at M=1024) | differs: layer_input (103 at M=1024) | differs: layer_input (3 at M=17, 148 at M=1024) | differs: layer_input (47 at M=17, 166 at M=1024) |
| kpack=2 | differs: post_mix (58 at M=17, 3505 at M=1024); comb_mix (252 at M=17, 15440 at M=1024); layer_input (27 at M=17, 1444 at M=1024) | differs: post_mix (2 at M=17, 190 at M=1024); comb_mix (146 at M=17, 8217 at M=1024) | same | same |
| matrix_instr_nonkdim=32 | same | same | same | same |

| config | sum of squares | stream sum | dot operands |
|---|---|---|---|
| reference | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_M=16 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_M=64 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_M=128 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| num_warps=1 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| num_warps=2 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| num_warps=8 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_M=16 num_warps=2 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_M=16 num_warps=8 | len 128: spt 4, lanes 32, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_M=64 num_warps=8 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| num_stages=2 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_K=32 | len 32: spt 4, lanes 8, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_K=64 | len 64: spt 8, lanes 8, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_K=256 | len 256: spt 8, lanes 32, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_C=64 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_C=256 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 2, warps 2 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| kpack=2 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 8; op1 kWidth 8 |
| matrix_instr_nonkdim=32 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 32, 32, 8; op0 kWidth 4; op1 kWidth 4 |

### 5.2 mhc_pre, NUM_KSPLIT=4 (_mhc_fused_split_kernel + _mhc_reduce_apply_kernel)

Reference: `{'NUM_KSPLIT': 4, 'BLOCK_M': 16, 'BLOCK_K': 128, 'BLOCK_C': 64, 'num_warps': 4, 'num_stages': 1, 'waves_per_eu': 0}`

| config | random | exact sum of squares | exact projection | both exact |
|---|---|---|---|---|
| reference | reference | reference | reference | reference |
| BLOCK_M=64 num_stages=2 waves_per_eu=1 | same | same | same | same |
| BLOCK_M=64 BLOCK_C=128 num_warps=2 num_stages=2 waves_per_eu=1 | same | same | same | same |
| num_warps=2 waves_per_eu=2 | same | same | same | same |
| BLOCK_C=128 num_warps=2 waves_per_eu=2 | same | same | same | same |
| BLOCK_M=128 num_stages=2 | same | same | same | same |
| BLOCK_M=64 num_warps=2 | same | same | same | same |
| BLOCK_M=128 waves_per_eu=2 | same | same | same | same |
| num_warps=8 | differs: post_mix (1 at M=17, 107 at M=1024); comb_mix (9 at M=17, 764 at M=1024); layer_input (16 at M=1024) | same | differs: post_mix (90 at M=1024); comb_mix (14 at M=17, 766 at M=1024); layer_input (17 at M=1024) | same |
| BLOCK_C=256 | differs: layer_input (4 at M=17, 151 at M=1024) | differs: layer_input (64 at M=1024) | differs: layer_input (4 at M=17, 143 at M=1024) | differs: layer_input (47 at M=17, 166 at M=1024) |
| BLOCK_K=64 | differs: post_mix (9 at M=17, 374 at M=1024); comb_mix (70 at M=17, 2747 at M=1024); layer_input (1 at M=17, 58 at M=1024) | same | differs: post_mix (8 at M=17, 364 at M=1024); comb_mix (65 at M=17, 2756 at M=1024); layer_input (2 at M=17, 58 at M=1024) | same |
| kpack=2 | differs: post_mix (45 at M=17, 3120 at M=1024); comb_mix (244 at M=17, 14633 at M=1024); layer_input (8 at M=17, 885 at M=1024) | differs: post_mix (2 at M=17, 114 at M=1024); comb_mix (104 at M=17, 5731 at M=1024); layer_input (94 at M=1024) | same | same |

| config | sum of squares | stream sum | dot operands |
|---|---|---|---|
| reference | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_M=64 num_stages=2 waves_per_eu=1 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 32, 32, 8; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_M=64 BLOCK_C=128 num_warps=2 num_stages=2 waves_per_eu=1 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 32, 32, 8; op0 kWidth 4; op1 kWidth 4 |
| num_warps=2 waves_per_eu=2 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_C=128 num_warps=2 waves_per_eu=2 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_M=128 num_stages=2 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 32, 32, 8; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_M=64 num_warps=2 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 32, 32, 8; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_M=128 waves_per_eu=2 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 32, 32, 8; op0 kWidth 4; op1 kWidth 4 |
| num_warps=8 | len 128: spt 4, lanes 32, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_C=256 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 2, warps 2 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_K=64 | len 64: spt 4, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| kpack=2 | len 128: spt 8, lanes 16, warps 1 | len 4: spt 1, lanes 4, warps 1 | mfma 16, 16, 16; op0 kWidth 8; op1 kWidth 8 |

### 5.3 mhc_head (_mhc_head_kernel)

Reference: `{'BLOCK_M': 16, 'BLOCK_K': 64, 'BLOCK_C': 128, 'num_warps': 4, 'num_stages': 1}`

| config | random | exact sum of squares | exact projection | both exact |
|---|---|---|---|---|
| reference | reference | reference | reference | reference |
| BLOCK_M=32 | differs: output (2 at M=17, 31 at M=1024) | same | differs: output (26 at M=1024) | same |
| num_warps=2 | differs: output (3 at M=17, 131 at M=1024) | same | differs: output (1 at M=17, 131 at M=1024) | same |
| num_warps=8 | differs: output (1 at M=17, 23 at M=1024) | same | differs: output (1 at M=17, 27 at M=1024) | same |
| BLOCK_M=32 num_warps=8 | same | same | same | same |
| num_stages=2 | same | same | same | same |
| waves_per_eu=2 | same | same | same | same |
| BLOCK_K=32 | differs: output (4 at M=17, 265 at M=1024) | same | differs: output (5 at M=17, 272 at M=1024) | same |
| BLOCK_K=128 | differs: output (5 at M=17, 200 at M=1024) | same | differs: output (3 at M=17, 212 at M=1024) | same |
| BLOCK_C=64 | differs: output (1 at M=17, 104 at M=1024) | same | differs: output (1 at M=17, 113 at M=1024) | same |
| BLOCK_C=256 | differs: output (1 at M=17, 154 at M=1024) | differs: output (125 at M=1024) | differs: output (3 at M=17, 156 at M=1024) | differs: output (184 at M=1024) |
| kpack=2 | differs: output (21 at M=17, 1456 at M=1024) | differs: output (61 at M=1024) | same | same |

| config | sum of squares | stream sum | dot operands |
|---|---|---|---|
| reference | len 64: spt 4, lanes 16, warps 1 | len 16: spt 1, lanes 4, warps 4 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_M=32 | len 64: spt 8, lanes 8, warps 1 | len 16: spt 1, lanes 4, warps 4 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| num_warps=2 | len 64: spt 8, lanes 8, warps 1 | len 16: spt 1, lanes 4, warps 2 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| num_warps=8 | len 64: spt 2, lanes 32, warps 1 | len 16: spt 1, lanes 4, warps 4 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_M=32 num_warps=8 | len 64: spt 4, lanes 16, warps 1 | len 16: spt 1, lanes 4, warps 4 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| num_stages=2 | len 64: spt 4, lanes 16, warps 1 | len 16: spt 1, lanes 4, warps 4 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| waves_per_eu=2 | len 64: spt 4, lanes 16, warps 1 | len 16: spt 1, lanes 4, warps 4 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_K=32 | len 32: spt 2, lanes 16, warps 1 | len 16: spt 1, lanes 4, warps 4 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_K=128 | len 128: spt 8, lanes 16, warps 1 | len 16: spt 1, lanes 4, warps 4 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_C=64 | len 64: spt 4, lanes 16, warps 1 | len 16: spt 1, lanes 8, warps 2 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| BLOCK_C=256 | len 64: spt 4, lanes 16, warps 1 | len 16: spt 1, lanes 2, warps 4 | mfma 16, 16, 16; op0 kWidth 4; op1 kWidth 4 |
| kpack=2 | len 64: spt 4, lanes 16, warps 1 | len 16: spt 1, lanes 4, warps 4 | mfma 16, 16, 16; op0 kWidth 8; op1 kWidth 8 |


### 5.4 Reading

In all 39 configs, the bits differ from the reference exactly
when one of the three layouts differs: the sum-of-squares layout (spt × lanes),
the stream-sum layout, or the dot `kWidth`. The MFMA shape alone
(`matrix_instr_nonkdim=32`, or `BLOCK_M ≥ 64` in the split path, which selects
32x32x8) never changes them. The exact inputs place each difference:
`BLOCK_K`, `BLOCK_M=16 + num_warps=8` and head `BLOCK_M`/`num_warps` act through
the sum of squares; `kpack` through the projection; `BLOCK_C` through the stream
sum.

### 5.5 Triton 3.8.0

The same configs compiled with Triton 3.8.0 (`3.8.0+amd.rocm7.2.0.git7cb7b059`,
the version aiter pins) give identical layouts (spt, lanes, warps, `kWidth`) in
every row (same job, `--layouts-only`). The bits still change between the two
versions; see 7.5.

## 6. Reference implementations

Sources fetched to `/home/AMD/zitwang/dev/refs/` at the commits linked from the
RFC (task 1, 8 and 9 rows). For each: where the sum of squares and the
projection are computed, what fixes their order, and whether anything depends
on M.

### 6.1 DeepSeek-V4 official reference

Hugging Face `deepseek-ai/DeepSeek-V4-Flash`, revision `60d8d70`,
`inference/model.py`.

- `hc_pre` (`:676-680`) and `hc_head` (`:730-734`) are PyTorch eager:
  `x.square().mean(-1)` and `F.linear(x, hc_fn)` in fp32 with fp32 `hc_fn`;
  the stream sum is `torch.sum(..., dim=2)`.
- The only mHC kernel in `inference/kernel.py` is `hc_split_sinkhorn`
  (`:372-427`, TileLang, one block of 64 threads per row, `T.reduce_sum` on a
  4×4 fragment; the launch shape does not depend on M).
- The reduction orders are those of PyTorch's reduction and cuBLAS SGEMM, which
  choose their algorithm from the shape; the reference does not claim batch
  invariance.

### 6.2 DeepGEMM `tf32_hc_prenorm_gemm`

`ISEEKYAN/DeepGEMM` `dfad223`; used by vLLM and mLite on NVIDIA.

- Fused: one kernel computes the projection (TF32 MMA, fp32 `fn`) and Σx² in
  the same K loop.
- Sum of squares written by hand (`deep_gemm/include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh:163-234`):

  ```cpp
  uint32_t row = warp_idx * 16 + lane_idx / 4;           // fixed row-to-lane map
  ...
  sqr_sum_acc_0 += a_float2_0.x * a_float2_0.x + a_float2_1.x * a_float2_1.x;
  ...
  const auto& reduced_sum_0 = math::warp_reduce_sum<4>(sqr_sum_acc_0);  // 4-lane shuffle
  ```

  Each thread accumulates its fixed columns over all K blocks, then a fixed
  4-lane butterfly. sm100 (`__ffma2_rn`) and sm120 are the same pattern.
- Every tile constant is fixed (`csrc/jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp:72-82`):
  `block_m = 64`, `block_k = 64`, 128 math threads, `block_n = align(n, 16)`.
  The pipeline depth depends only on n. M only sets the grid.
- `num_splits` is an API argument (`csrc/apis/hyperconnection.hpp:18-57`,
  default 1); partials go to `[num_splits, M, N]` and are not merged by the
  kernel.
- Batch invariant by construction if the caller fixes `num_splits`.

### 6.3 vLLM mHC on NVIDIA, and PR #59929

`inaniloquentee/vllm` `e1107b01` (PR [#59929](https://github.com/vllm-project/vllm/pull/59929)),
original `ISEEKYAN/vllm` `4b5c627`, `vllm/model_executor/kernels/mhc/`.

- `_hc_prenorm_gemm_outputs` (`tilelang.py:25-75`) calls DeepGEMM when
  available. The split count is `compute_num_split(64, K, cdiv(split_tokens, 64))`
  with `split_tokens = 1 if envs.VLLM_BATCH_INVARIANT else num_tokens` (`:42-44`),
  i.e. `min(SM count // 1, K/64 // 4)` in batch-invariant mode
  (`tilelang_kernels.py:45-54`): fixed per GPU and K.
- Partials are merged by `for i_split in T.serial(n_splits): rms[0] += …` in
  the TileLang big-fuse kernel (split order).
- What #59929 changed: it restored `split_tokens = 1` in the two helpers where
  a refactor had lost it, and in batch-invariant mode disables the fused
  post+pre kernel (`models/deepseek_v4/nvidia/model.py`, `use_fused_mhc = not
  VLLM_BATCH_INVARIANT`), whose small-M path computes Σx² on the fp32 residual
  instead of the bf16 one. No kernel changed.

### 6.4 vLLM TileLang fallback (no DeepGEMM)

Same files. `hc_prenorm_gemm_tilelang` (`tilelang_kernels.py:879-919`) writes
its tree by hand (per-thread serial sum, `T.warp_reduce_sum`, then a serial
sum over warps), but its compile key picks the thread count from M
(`tilelang_kernels.py:1147-1163`):

```python
use_block_m = use_default_config and num_tokens >= 1024
use_small_tile = (use_default_config and not use_block_m
                  and num_tokens < 128 and hc_hidden_size % 1024 == 0)
effective_n_thr = 1024 if use_small_tile else n_thr     # 512 otherwise
```

Each thread owns K elements `tid, tid + n_thr, …`, so changing `n_thr`
changes a row's order between M < 128 and M ≥ 128. Not gated by
`VLLM_BATCH_INVARIANT`. The batch-invariance test of #59929 runs the DeepGEMM
path only.

### 6.5 vLLM DeepSeek-V4 on ROCm: aiter's HIP `mhc_pre`

`vllm/models/deepseek_v4/amd/model.py` uses `MHCPreOp` →
`vllm/_aiter_ops.py:4111` `rocm_aiter_ops.mhc_pre` → `aiter.ops.mhc.mhc_pre`
(aiter's HIP kernels, not the Triton ones above). The AMD model file has no
batch-invariant gate.

- `aiter/ops/mhc.py:610-700` picks `(split_k, tile_k)` with
  `get_mhc_pre_splitk(m, hc_hidden_size)` (`:292-332`), a score over
  `num_tg_m = ceil(m / 64)` and the CU count: the split depends on M.
- `mhc_pre_gemm_sqrsum` (`csrc/kernels/mhc_kernels.cu:880-975`) also picks
  `tile_n` 16 or 32 from `m_blocks · split_k` vs the CU count.
- Within one launch the sum of squares is per-lane accumulation plus
  `cross_row_sum_4` (`:133-147`, two fixed `ds_bpermute` steps), and
  `mhc_pre_big_fuse` merges the splits with a wave reduce (`:1060-1140`).
- The projection can run in fp32 through a bf16 hi/lo split of `fn`
  (`mhc_shuffle_fn`, `aiter/ops/mhc.py:44-70`).

Measured (job 68322040, `benchmarks/operators/mhc/probe_aiter_hip_mhc_pre.py`,
aiter `326cffa` HIP module built from a copy of the checkout with CK headers at
its submodule commit `af9e1d1`):

| M | 1–256 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|
| `get_mhc_pre_splitk` (split_k, tile_k) on MI325X | (64, 128) | (32, 128) | (16, 128) | (8, 128) | (4, 128) | (2, 128) |

`mhc_pre` as vLLM calls it changes a token's bits from M = 1024 on, for pre and
for head (section 7.3, row "E aiter split").

### 6.6 vLLM batch-invariant RMSNorm and matmul

`inaniloquentee/vllm` `76aacf3`, `vllm/model_executor/determinism/batch_invariant.py`;
PR [#59952](https://github.com/vllm-project/vllm/pull/59952).

- RMSNorm (`:1132-1181`, launch `:1225-1241`): one workgroup per row, a loop
  over chunks of `BLOCK_SIZE = 1024`, `tl.sum` per chunk, chunks added in
  order; `num_warps` left at Triton's default. The order inside a chunk is
  `tl.sum`'s tree for that fixed layout: invariance comes from never changing
  the constexprs, the same mechanism as candidate A.
- The CUDA `fused_add_rms_norm` launcher locks its block size to
  `min(hidden, 1024)` in batch-invariant mode
  (`csrc/libtorch_stable/layernorm_kernels.cu:329-372`).
- Matmul (`:46-54`): "What is summed is a function of (K, N, dtype) only --
  never of M". `BLOCK_SIZE_K` fixed per dtype (32 fp32, 64 otherwise);
  `_split_k_plan(K, N)` (`:66-83`): one chunk if N > 64 or K < 1024, else
  `clamp(K // 512, 1, 16)` chunks aligned to 64, merged in ascending order.
  `BLOCK_M`, `BLOCK_N` and `num_warps` follow M ("Tiles only regroup rows and
  columns"). On ROCm this matmul replaces hipBLASLt ("Neither cuBLASLt on SM80
  (Ampere) nor hipBLASLt can be made batch invariant", PR head `3fd5277`,
  `batch_invariant.py:1534`) and sets `VLLM_ROCM_USE_SKINNY_GEMM=0` (`:1613`).
- PR [#59925](https://github.com/vllm-project/vllm/pull/59925) concerns the
  DeepSeek-V4 MoE path, not GEMM or RMSNorm.

### 6.7 Megatron PR #7050 and mLite

`ISEEKYAN/Megatron-LM` `327a238` (PR [#7050](https://github.com/NVIDIA/Megatron-LM/pull/7050)
head is the same commit), `experimental/lite/megatron/lite/model/deepseek_v4/vllm/primitive/dense.py`.

- Forward: `_MHC_ENTRIES` (`:24-30`) maps pre, post, post_pre and head to vLLM's
  TileLang/DeepGEMM functions (6.3); batch-invariant mode is vLLM's
  `VLLM_BATCH_INVARIANT=1`.
- Backward (`:512-845`): recomputed in PyTorch (`torch.compile`, eager fallback);
  the pre backward recomputes the norm as `1/(norm/sqrt(K)+eps)` in bf16 and
  reduces dW and dbias over tokens with matmul and `.sum()`. Not batch
  invariant; mLite does not claim it is.

### 6.8 SGLang

`sgl-project/sglang` main `4003bf1` (2026-11-08), read only, not run.
`python/sglang/srt/models/deepseek_v4_mhc.py` `_mix_stats_impl` (`:361-468`)
picks the mix-stats path:

- NVIDIA sm90/sm100 with `SGLANG_OPT_DEEPGEMM_HC_PRENORM` and batch-invariant
  mode off: DeepGEMM / compensated bf16x3 projections; batch-invariant mode
  disables them (`:384`).
- `use_fused_stats` (Blackwell, some Hopper M ranges, and **gfx950**):
  `hc_mix_stats_sinkhorn`, the same split kernel as below fused with Sinkhorn
  ("The default split-K/Sinkhorn fusion preserves batch invariance", `:400`).
- other CUDA: `hc_mix_stats` (`kernels/ops/layernorm/mhc.py:2237-2420`),
  "cuBLAS/torch reductions can change order with num_tokens; this kernel keeps
  the mixing and RMS reductions batch-invariant".
- everything else, which includes **gfx942** (`torch.version.cuda` is None and
  `use_fused_stats` needs gfx950): PyTorch eager `x.square().mean()` and
  `F.linear` (`:446-449`), the DeepSeek reference arithmetic, no batch-invariance claim.

`hc_mix_stats` is the closest reference to candidate A/B on AMD:

- K slices from K only (`_num_slices_for`, choices 80 … 1), merged in slice
  order (`_hc_mix_stats_reduce_kernel`); `BLOCK_K = 64`, `num_warps = 4` fixed;
- the sum of squares is still `sq += tl.sum(x_tile * x_tile, axis=1)` per K
  tile, i.e. its order rests on fixed constexprs, as in A;
- `BLOCK_M` follows M on CUDA (8 / 16 / 32, "BLOCK_M 8/16/32 preserve each
  row's K reduction order; thresholds were measured on GB300") but is fixed to
  16 on HIP: "fp32 MFMA tile heights do not share a per-row reduction order, so
  one height serves every M" (`_block_m_for`);
- the projection is fp32: `input_precision="ieee"` on HIP ("Triton on AMD has
  no tf32x3"), `tf32x3` on CUDA.

The aiter mHC path (`_use_aiter_mhc`, `kernels/ops/layernorm/mhc.py:37-43`) is
gated on gfx950 and `SGLANG_USE_AITER`.

### 6.9 Summary of the references

| Implementation | Σx² fused with projection | Σx² tree | Fixed by | Depends on M | BI on ROCm |
|---|---|---|---|---|---|
| DSV4 official | no | PyTorch reduce | library | yes | no claim |
| DeepGEMM | yes | written: per-thread serial + 4-lane butterfly | compile-time constants; `num_splits` from caller | no | NVIDIA only |
| vLLM NVIDIA (#59929) | yes (DeepGEMM) | as DeepGEMM | split from M = 1 | no | — |
| vLLM TileLang fallback | yes | written, thread count from M | `n_thr` 1024 / 512 | yes (M = 128) | no |
| vLLM ROCm (aiter HIP) | yes | per-lane + fixed bpermute | `split_k`, `tile_k`, `tile_n` from M | yes | no |
| vLLM BI RMSNorm | — | `tl.sum` per 1024 chunk | fixed constexprs | no | compiler-dependent |
| SGLang `hc_mix_stats` | yes | `tl.sum` per 64 tile | slices from K; `BLOCK_K`, `num_warps` fixed; `BLOCK_M` fixed on HIP | no | used on gfx950 only; gfx942 runs eager PyTorch |
| aiter Triton (A) | yes | `tl.sum`, layout from config | per-M table, layouts held equal | no (measured) | compiler-dependent |

## 7. Candidates

Baseline in every table: **A**, the current ops of the PR branch. All
measurements: MI325X (gfx942), torch 2.12.0 + ROCm 7.14, Triton 3.7.0, aiter
`326cffa`, n = 4, C = 4096, bf16 residual, fp32 `fn`, 20 Sinkhorn rounds.
Jobs 68321539 and 68321590 unless noted; 68321590 reran everything after a
line-wrapping refactor of B, whose outputs kept the same SHA-256 digests.

### 7.1 The candidates

**A. aiter's Triton kernels with per-M configs whose layouts are equal (current).**
`rl_engine/backends/rocm/mhc/aiter_mhc.py` passes a config per M from
`configs/gfx942/mhc_{pre,head}/*.json`; the table loader forces `NUM_KSPLIT`,
`BLOCK_K`, `kpack` (pre) and `BLOCK_M`, `BLOCK_K`, `BLOCK_C`, `num_warps`,
`kpack` (head) to be equal in every entry. The remaining freedom (pre
`BLOCK_M`, `BLOCK_C ≤ 128`, `num_warps`) is safe only because, with
`BLOCK_K = 128`, Triton gives the same reduction layouts (section 4.2); two
combinations that would not (`BLOCK_C = 256`, `BLOCK_M = 16` with
`num_warps = 8`) are excluded by the tuner and the tests, not by the loader.

**B. Fixed-order Triton kernels in RL-Kernel.**
`rl_engine/backends/rocm/mhc/fixed_order_mhc.py` (not registered as an op).
Four kernels:

1. `_prenorm_kernel`, grid `(cdiv(M, BLOCK_M), S)`: one K segment of
   `K / S` per workgroup; per K tile of `PRENORM_BLOCK_K = 128`:
   `acc = tl.dot(x, fn_bf16, acc)` and `sq = tl.fma(x, x, sq)` with `sq` of
   shape `[BLOCK_M, 128]` (element-wise, no reduction in the loop); after the
   loop `_pairwise_sum_last` adds adjacent pairs in 7 levels
   (`tl.sum(tl.reshape(v, (BLOCK_M, W/2, 2)), axis=2)`). A two-element sum has
   one rounding whatever lanes hold the elements, so the tree is fixed by the
   source.
2. `_pre_apply_kernel` / `_head_apply_kernel`, grid `(cdiv(M, BLOCK_M), cdiv(C, BLOCK_C))`:
   partials added in segment order (`_merge_splits`), RMS factor and
   activations with the same expressions as aiter, stream sum written as
   `li = pre0·x0; li = fma(pre1, x1, li); …` in stream order.
3. `_sinkhorn_kernel`: `BLOCK_M` rows per workgroup, row and column sums and
   maxima with the same pairwise tree over the 4 entries.

`S`, the number of K segments, is part of the arithmetic and must be one value
per K; `num_k_splits(K) = clamp(K // 4096, 1, 16)` gives 4 for K = 16384.
`PRENORM_BLOCK_K` is part of it too. Every other launch parameter only
regroups rows and columns.

**C. The kernels of B inside a pinned aiter fork.** Same arithmetic as B (same
source), so it was not measured separately. It differs in where the code lives:
RL-Kernel does not pin aiter today (`pyproject.toml:34`: `rocm = ["aiter"]`
without a version; the Dockerfiles and requirements do not install it), so C
needs a pinned aiter version plus a patch, and every consumer must build that
aiter.

**D. Projection by the batch-invariant linear, sum of squares separately.**
`fixed_order_mhc_pre(..., projection="aiter_linear")`: the projection runs as
`AiterBatchInvariantLinearOp` (aiter `gemm_a16w16`, whole-K config), the sum of
squares as B's `_prenorm_kernel` without the dot; x is read twice. Pre only.

**E. aiter's HIP `mhc_pre` with a fixed split.** The op vLLM's ROCm model calls
(section 6.5).

Implementation measured here: `mhc_pre_gemm_sqrsum` and `mhc_pre_big_fuse`
called directly with one `(split_k, tile_k)` for every M; mhc_head as vLLM
runs it (`vllm/_aiter_ops.py:4433-4480`): `fn` and `base` zero-padded to 24
rows, `scale` to 3, no Sinkhorn, `layer_input` is the head output. The
projection uses fp32 `fn` (`w_preshuffle_bf16 = 0`). What the measurements
show beyond the split:

- the kernels still choose internal parameters from M: `tile_n` 16 or 32 from
  `m_blocks · split_k` vs the CU count (`mhc_kernels.cu:906-934`) and the
  big-fuse launch from `m <= cu_num * 12` (`:1387-1392`, 3648 on MI325X).
  With split_k = 1, 4 or 8 the bits did not change with M; with split_k = 64
  they change from M = 4096 on, the first measured M above 3648;
- `mhc_pre_gemm_sqrsum` reads only the row stride of the residual
  (`x.stride(0)`, `:952`), so a residual whose streams are not contiguous gives
  other results (strides column of 7.3); a wrapper must pass a contiguous
  residual;
- building it needs hipcc, the CK headers and aiter's JIT; `import aiter` in
  the current image fails outside `AITER_TRITON_ONLY` (aiter needs flydsl ≥ 0.2.4,
  the image has 0.1.6), so the probe imports only aiter's JIT core
  (`AITER_AOT_IMPORT=1`). `mhc_pre_gemm_sqrsum` / `mhc_pre_big_fuse` are not
  the public `mhc_pre` entry point.

### 7.2 Bits do not depend on launch parameters (B)

Section 1 of the job: every launch parameter of B changed one at a time and in
combinations, against B's default config, for `S` = 1, 4 and 16, at
M = 1, 17, 1024 and 4096, every output compared bit for bit.

| Op | Parameters changed | Configs | Result |
|---|---|---|---|
| pre | `BLOCK_M` 16/64/128, `num_warps` 2/8, `BLOCK_M=16 + num_warps=8`, `BLOCK_M=128 + num_warps=8`, `num_stages=2`, `waves_per_eu=2`, apply `BLOCK_M` 16/64, apply `BLOCK_C` 64/256, apply `num_warps` 2/8, Sinkhorn `BLOCK_M` 1/64 | 17 × 3 values of S | all same |
| head | the same without Sinkhorn | 15 × 3 values of S | all same |

For comparison, aiter's kernels change bits with `BLOCK_K`, `BLOCK_C=256`,
`BLOCK_M=16 + num_warps=8`, `kpack` (pre) and `BLOCK_M`, `BLOCK_K`, `BLOCK_C`,
`num_warps`, `kpack` (head) (section 5).

### 7.3 Batch invariance

`rl_engine/validation/common/batch_invariance.py`: two rows alone (M = 1) vs
at the first and last position of a batch of M, for M = 1, 8, 16, 17, 59, 64,
128, 256, 1024, 2048, 4096, 8192, 16384; chunking, reordering and padding
(M = 17, 1024); other input strides; two runs.

| Variant | mhc_pre | mhc_head |
|---|---|---|
| A | no difference | no difference |
| B, S = 1 / 4 / 16 | no difference | no difference |
| B, S = 4, launch config changing with M (`BLOCK_M=16, num_warps=8, BLOCK_C=256, Sinkhorn BLOCK_M=1` for M < 64, else `64 / 4 / 64 / 64`) | no difference | no difference |
| D, S = 4 | no difference | — |
| E, `mhc_pre` as vLLM calls it (split by M) | M = 1024 … 16384 differ; chunking differs; strides differ | M = 1024 … 16384 differ; chunking differs; strides differ |
| E, fixed split_k × tile_k = 1×128, 4×128, 8×64 | no difference in batch, chunking and runs; strides differ | same |
| E, fixed 64×128 | M = 4096, 8192, 16384 differ (comb_mix, layer_input); strides differ | M = 4096, 8192, 16384 differ; strides differ |

E rows: job 68322040.

### 7.4 Error against the fp32 reference

M = 1024, random inputs; reference `NativeMHCPreOp.forward_fp32` /
`NativeMHCHeadOp.forward_fp32` (fp32 `fn`). Max / mean absolute error.

| Variant | post_mix | comb_mix | layer_input | head output |
|---|---|---|---|---|
| aiter default | 3.600e-03 / 6.416e-04 | 1.227e-03 / 2.053e-04 | 2.182e-02 / 1.502e-03 | 1.893e-02 / 1.393e-03 |
| A (baseline) | 3.601e-03 / 6.416e-04 | 1.227e-03 / 2.053e-04 | 2.182e-02 / 1.502e-03 | 1.893e-02 / 1.393e-03 |
| B, S = 1 | 3.600e-03 / 6.416e-04 | 1.227e-03 / 2.053e-04 | 2.182e-02 / 1.502e-03 | 1.893e-02 / 1.393e-03 |
| B, S = 4 | 3.601e-03 / 6.416e-04 | 1.227e-03 / 2.053e-04 | 2.182e-02 / 1.502e-03 | 1.893e-02 / 1.393e-03 |
| B, S = 16 | 3.600e-03 / 6.416e-04 | 1.227e-03 / 2.053e-04 | 2.182e-02 / 1.502e-03 | 1.893e-02 / 1.393e-03 |
| D, S = 4 | 3.600e-03 / 6.416e-04 | 1.227e-03 / 2.053e-04 | 2.182e-02 / 1.502e-03 | — |
| E, split by M | 1.311e-06 / 2.162e-07 | 5.066e-07 / 7.387e-08 | 3.125e-02 / 2.532e-03 | 3.125e-02 / 2.426e-03 |
| E, fixed 1×128 | 6.437e-06 / 8.687e-07 | 2.503e-06 / 3.232e-07 | 3.124e-02 / 2.532e-03 | 3.125e-02 / 2.426e-03 |
| E, fixed 4×128 | 3.934e-06 / 4.871e-07 | 9.835e-07 / 1.582e-07 | 3.124e-02 / 2.532e-03 | 3.125e-02 / 2.426e-03 |
| E, fixed 8×64 | 2.384e-06 / 3.673e-07 | 7.153e-07 / 1.202e-07 | 3.124e-02 / 2.532e-03 | 3.125e-02 / 2.426e-03 |

A, B and D round `fn` to bf16 as aiter's Triton kernels do; their mixing
weights are three orders of magnitude further from the fp32 reference than
E's, which projects with fp32 `fn`. E's `layer_input` and head output have a
1.7× larger mean error than A's; why was not investigated. The reduction order
of A, B and D does not show at this scale.

### 7.5 Triton version

SHA-256 of every output for fixed inputs (random, M = 17 and 1024), Triton
3.7.0 (image) vs 3.8.0 (`3.8.0+amd.rocm7.2.0.git7cb7b059`), same GPU, same job:

| Outputs | 3.7.0 vs 3.8.0 |
|---|---|
| A pre (table class, `NUM_KSPLIT=4`), M = 17 | same |
| A pre (table class, `NUM_KSPLIT=4`), M = 1024 | all three outputs differ |
| A head (aiter's config), M = 17 and 1024 | differs |
| B pre, S = 4, M = 17 and 1024 | same |
| B head, S = 4, M = 17 and 1024 | same |

Which reduction moves (job 68321583, `analyze_aiter_reductions.py --digest`,
the three class references of section 5 on the four input kinds, M = 17 and
1024, Triton 3.7.0 vs 3.8.0):

| Class | random | exact sum of squares | exact projection | both exact |
|---|---|---|---|---|
| pre `NUM_KSPLIT=1` | M = 17 same; M = 1024 post_mix, comb_mix differ | same | M = 17 same; M = 1024 all three differ | same |
| pre `NUM_KSPLIT=4` | M = 17 same; M = 1024 all three differ | same | M = 17 same; M = 1024 all three differ | same |
| head | differs at M = 17 and 1024 | same | differs at M = 17 and 1024 | same |

Differences appear only when the sum of squares rounds, so the moving
reduction is the sum-of-squares `tl.sum`, with the same layout in both
versions. Triton 3.8.0 changed how a thread adds its own elements:
v3.7.0 `reduceWithinThreads` folds them left to right
(`lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp:142-150`), v3.8.0 adds them
as a binary tree (`treeReduce`, `ReduceOpToLLVM.cpp:116-142` and `:286-303` of
tag `v3.8.0`; arity 2 for additions, `TargetInfo.cpp:689-697`), optionally on
packed fp32 pairs. With 4 or 8 elements per thread the two orders differ.
Why mhc_pre at M = 17 keeps its bits was not investigated (M is a
`tl.constexpr` of these kernels, so M = 17 and 1024 compile separately).
B's pairwise reductions add two elements per `tl.sum`, where the two lowerings
coincide.

### 7.6 Device time

Forward, ms per call from GPU events around a CUDA graph of 20 calls; median
over 5 replays. Two runs on the same node (jobs 68321539 and 68321590); each
cell gives the range of the two medians, each ratio the range of the per-run
ratios. Baseline: A. Ratio columns: variant / A (below 1: faster than A).
B and D: the fastest of 18 launch configs per M (`BLOCK_M` 16/32/64 ×
`num_warps` 2/4/8 × two apply tiles), all with the same bits (7.2).
aiter default is shown for reference; it is not batch invariant and cannot
launch at M = 8192 on gfx942 (needs 128 KB of LDS).

#### mhc_pre

| M | aiter default (ms) | A (ms) | B S=1 (ms) | B S=4 (ms) | B S=16 (ms) | D S=4 (ms) | aiter default / A (ratio) | B S=1 / A (ratio) | B S=4 / A (ratio) | B S=16 / A (ratio) | D S=4 / A (ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.1983–0.1987 | 0.0558–0.0559 | 0.1379–0.1386 | 0.0451–0.0457 | 0.0236–0.0238 | 0.1976–0.1981 | 3.55 | 2.47–2.48 | 0.81–0.82 | 0.42–0.43 | 3.54 |
| 16 | 0.2433–0.2437 | 0.0497–0.0498 | 0.1350–0.1357 | 0.0442–0.0443 | 0.0245–0.0246 | 0.1974–0.1979 | 4.89–4.90 | 2.72 | 0.89 | 0.49 | 3.97 |
| 64 | 0.2507–0.2508 | 0.0652–0.0654 | 0.1362 | 0.0446 | 0.0238 | 0.1974–0.1977 | 3.83–3.85 | 2.08–2.09 | 0.68 | 0.36–0.37 | 3.02–3.03 |
| 256 | 0.2554–0.2555 | 0.0589 | 0.1368 | 0.0452–0.0453 | 0.0253 | 0.2026–0.2035 | 4.34 | 2.32 | 0.77 | 0.43 | 3.44–3.46 |
| 1024 | 0.0511–0.0522 | 0.0841–0.0850 | 0.1642 | 0.0586–0.0588 | 0.0466–0.0468 | 0.2183–0.2184 | 0.61 | 1.93–1.95 | 0.69–0.70 | 0.55 | 2.57–2.60 |
| 2048 | 0.0885–0.0890 | 0.0803–0.0820 | 0.1747–0.1753 | 0.0738–0.0744 | 0.0684–0.0687 | 0.2363–0.2372 | 1.08–1.11 | 2.14–2.18 | 0.90–0.93 | 0.84–0.85 | 2.89–2.94 |
| 4096 | 0.1437–0.1465 | 0.1215–0.1239 | 0.1940–0.1947 | 0.1154–0.1174 | 0.1033–0.1063 | 0.2665–0.2679 | 1.16–1.21 | 1.57–1.60 | 0.95 | 0.83–0.87 | 2.15–2.20 |
| 8192 | cannot launch: OutOfResources | 0.2190–0.2255 | 0.2701–0.2703 | 0.1998–0.2006 | 0.2166–0.2206 | 0.3578–0.3581 | n/a | 1.20–1.23 | 0.89–0.91 | 0.98–0.99 | 1.59–1.64 |
| 16384 | 0.4666–0.4827 | 0.4127–0.4644 | 0.4648–0.4899 | 0.3828–0.4242 | 0.4860–0.4979 | 0.5748–0.6097 | 1.04–1.13 | 1.05–1.13 | 0.91–0.93 | 1.07–1.18 | 1.31–1.39 |
| 32768 | 0.8925–0.9091 | 0.8072–0.8273 | 0.8400–0.8833 | 0.7564–0.7920 | 1.0888–1.1070 | 1.1640–1.2162 | 1.10–1.11 | 1.04–1.07 | 0.94–0.96 | 1.32–1.37 | 1.44–1.47 |
| 65536 | 1.6955–1.7916 | 1.5191–1.6493 | 1.4096–1.5215 | 1.4373–1.5711 | 2.2426–2.2810 | 2.3081–2.4126 | 1.09–1.12 | 0.92–0.93 | 0.95 | 1.38–1.48 | 1.46–1.52 |

#### mhc_head

| M | aiter default (ms) | A (ms) | B S=1 (ms) | B S=4 (ms) | B S=16 (ms) | aiter default / A (ratio) | B S=1 / A (ratio) | B S=4 / A (ratio) | B S=16 / A (ratio) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.8209–0.8214 | 0.7451–0.7457 | 0.1030 | 0.0297–0.0301 | 0.0128–0.0129 | 1.10 | 0.14 | 0.04 | 0.02 |
| 16 | 0.8334–0.8335 | 0.7464–0.7467 | 0.1023–0.1025 | 0.0296–0.0297 | 0.0128 | 1.12 | 0.14 | 0.04 | 0.02 |
| 64 | 0.8358–0.8360 | 0.7483 | 0.1029–0.1030 | 0.0296 | 0.0129 | 1.12 | 0.14 | 0.04 | 0.02 |
| 256 | 0.8380–0.8382 | 0.7431–0.7432 | 0.1034 | 0.0300 | 0.0135 | 1.13 | 0.14 | 0.04 | 0.02 |
| 1024 | 0.9005 | 0.7759–0.7772 | 0.1251–0.1253 | 0.0411 | 0.0249–0.0250 | 1.16 | 0.16 | 0.05 | 0.03 |
| 2048 | 0.9077–0.9080 | 0.7802–0.7812 | 0.1346–0.1347 | 0.0597–0.0604 | 0.0422–0.0433 | 1.16 | 0.17 | 0.08 | 0.05–0.06 |
| 4096 | 0.9139–0.9153 | 0.7872–0.7886 | 0.1530–0.1531 | 0.0811–0.0816 | 0.0761–0.0772 | 1.16 | 0.19 | 0.10 | 0.10 |
| 8192 | 1.9152–1.9159 | 1.4278–1.4318 | 0.2327–0.2362 | 0.1469–0.1470 | 0.1502–0.1524 | 1.34 | 0.16–0.17 | 0.10 | 0.10–0.11 |
| 16384 | 3.8589–3.8630 | 2.8747–2.8792 | 0.3977–0.4608 | 0.3503–0.4345 | 0.3524–0.4188 | 1.34 | 0.14–0.16 | 0.12–0.15 | 0.12–0.15 |
| 32768 | 6.7778–6.7855 | 5.1495–5.1618 | 0.6797–0.6988 | 0.6799–0.7074 | 0.7172–0.7296 | 1.31–1.32 | 0.13–0.14 | 0.13–0.14 | 0.14 |
| 65536 | 12.0050–12.0124 | 10.7165–10.7307 | 1.2655–1.3696 | 1.2896–1.3927 | 1.4297–1.5369 | 1.12 | 0.12–0.13 | 0.12–0.13 | 0.13–0.14 |

#### Candidate E, with A and B from the same job

Job 68322040, same node and method; A and B measured first in the same job
(`evaluate_mhc_candidates.py --sections 4`), E second. One run. E has no
launch parameters to tune.

##### mhc_pre

| M | A (ms) | B S=4 (ms) | B S=16 (ms) | E aiter split (ms) | E fixed 1x128 (ms) | E fixed 4x128 (ms) | E fixed 8x64 (ms) | E fixed 64x128 (ms) | B S=4 / A (ratio) | B S=16 / A (ratio) | E aiter split / A (ratio) | E fixed 1x128 / A (ratio) | E fixed 4x128 / A (ratio) | E fixed 8x64 / A (ratio) | E fixed 64x128 / A (ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0561 | 0.0452 | 0.0236 | 0.0089 | 0.1360 | 0.0391 | 0.0242 | 0.0089 | 0.81 | 0.42 | 0.16 | 2.42 | 0.70 | 0.43 | 0.16 |
| 16 | 0.0504 | 0.0441 | 0.0238 | 0.0093 | 0.1397 | 0.0405 | 0.0244 | 0.0093 | 0.88 | 0.47 | 0.18 | 2.77 | 0.80 | 0.48 | 0.18 |
| 64 | 0.0654 | 0.0447 | 0.0237 | 0.0101 | 0.1418 | 0.0414 | 0.0248 | 0.0101 | 0.68 | 0.36 | 0.15 | 2.17 | 0.63 | 0.38 | 0.15 |
| 256 | 0.0590 | 0.0452 | 0.0254 | 0.0139 | 0.1421 | 0.0420 | 0.0267 | 0.0140 | 0.77 | 0.43 | 0.24 | 2.41 | 0.71 | 0.45 | 0.24 |
| 1024 | 0.0845 | 0.0587 | 0.0464 | 0.0342 | 0.1520 | 0.0515 | 0.0352 | 0.0383 | 0.69 | 0.55 | 0.40 | 1.80 | 0.61 | 0.42 | 0.45 |
| 2048 | 0.0849 | 0.0747 | 0.0690 | 0.0575 | 0.1613 | 0.0645 | 0.0565 | 0.0644 | 0.88 | 0.81 | 0.68 | 1.90 | 0.76 | 0.67 | 0.76 |
| 4096 | 0.1227 | 0.1133 | 0.1061 | 0.1125 | 0.1840 | 0.1195 | 0.1174 | 0.1283 | 0.92 | 0.86 | 0.92 | 1.50 | 0.97 | 0.96 | 1.05 |
| 8192 | 0.2141 | 0.1988 | 0.2179 | 0.2058 | 0.2426 | 0.2017 | 0.2086 | 0.2435 | 0.93 | 1.02 | 0.96 | 1.13 | 0.94 | 0.97 | 1.14 |
| 16384 | 0.4421 | 0.4115 | 0.5062 | 0.3879 | 0.4307 | 0.3875 | 0.3676 | 0.4685 | 0.93 | 1.14 | 0.88 | 0.97 | 0.88 | 0.83 | 1.06 |
| 32768 | 0.7796 | 0.7262 | 1.0758 | 0.7713 | 0.7671 | 0.7316 | 0.7280 | 0.9336 | 0.93 | 1.38 | 0.99 | 0.98 | 0.94 | 0.93 | 1.20 |
| 65536 | 1.5166 | 1.4563 | 2.2401 | 1.5320 | 1.5325 | 1.4735 | 1.4556 | 1.8881 | 0.96 | 1.48 | 1.01 | 1.01 | 0.97 | 0.96 | 1.24 |

##### mhc_head

| M | A (ms) | B S=4 (ms) | B S=16 (ms) | E aiter split (ms) | E fixed 1x128 (ms) | E fixed 4x128 (ms) | E fixed 8x64 (ms) | E fixed 64x128 (ms) | B S=4 / A (ratio) | B S=16 / A (ratio) | E aiter split / A (ratio) | E fixed 1x128 / A (ratio) | E fixed 4x128 / A (ratio) | E fixed 8x64 / A (ratio) | E fixed 64x128 / A (ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.7450 | 0.0298 | 0.0129 | 0.0079 | 0.1351 | 0.0381 | 0.0231 | 0.0078 | 0.04 | 0.02 | 0.01 | 0.18 | 0.05 | 0.03 | 0.01 |
| 16 | 0.7463 | 0.0298 | 0.0129 | 0.0082 | 0.1387 | 0.0395 | 0.0234 | 0.0082 | 0.04 | 0.02 | 0.01 | 0.19 | 0.05 | 0.03 | 0.01 |
| 64 | 0.7481 | 0.0295 | 0.0129 | 0.0091 | 0.1403 | 0.0403 | 0.0238 | 0.0090 | 0.04 | 0.02 | 0.01 | 0.19 | 0.05 | 0.03 | 0.01 |
| 256 | 0.7430 | 0.0301 | 0.0136 | 0.0137 | 0.1421 | 0.0418 | 0.0264 | 0.0137 | 0.04 | 0.02 | 0.02 | 0.19 | 0.06 | 0.04 | 0.02 |
| 1024 | 0.7776 | 0.0413 | 0.0250 | 0.0329 | 0.1518 | 0.0512 | 0.0350 | 0.0378 | 0.05 | 0.03 | 0.04 | 0.20 | 0.07 | 0.05 | 0.05 |
| 2048 | 0.7810 | 0.0603 | 0.0424 | 0.0560 | 0.1610 | 0.0636 | 0.0590 | 0.0627 | 0.08 | 0.05 | 0.07 | 0.21 | 0.08 | 0.08 | 0.08 |
| 4096 | 0.7877 | 0.0815 | 0.0764 | 0.1077 | 0.1829 | 0.1196 | 0.1140 | 0.1233 | 0.10 | 0.10 | 0.14 | 0.23 | 0.15 | 0.14 | 0.16 |
| 8192 | 1.4302 | 0.1485 | 0.1489 | 0.1968 | 0.2370 | 0.1961 | 0.1979 | 0.2359 | 0.10 | 0.10 | 0.14 | 0.17 | 0.14 | 0.14 | 0.16 |
| 16384 | 2.8778 | 0.3579 | 0.3620 | 0.3807 | 0.4254 | 0.3782 | 0.3547 | 0.4527 | 0.12 | 0.13 | 0.13 | 0.15 | 0.13 | 0.12 | 0.16 |
| 32768 | 5.1556 | 0.6657 | 0.7082 | 0.7426 | 0.7406 | 0.6901 | 0.6916 | 0.9060 | 0.13 | 0.14 | 0.14 | 0.14 | 0.13 | 0.13 | 0.18 |
| 65536 | 10.6996 | 1.2847 | 1.4338 | 1.4877 | 1.4782 | 1.4040 | 1.3890 | 1.8281 | 0.12 | 0.13 | 0.14 | 0.14 | 0.13 | 0.13 | 0.17 |

### 7.7 Comparison

All numbers from the sections above; "vs A" is device time / A.

| | A (current) | B (fixed order, RL-Kernel Triton) | C (B in a pinned aiter) | D (linear + separate Σx²) | E (aiter HIP, fixed split) |
|---|---|---|---|---|---|
| Batch invariant, one software version | yes | yes | as B | yes | yes for split 1, 4, 8; no for 64 |
| Bits independent of launch parameters | no; held by the table (7 parameters forced equal, 2 combinations excluded by tests only) | yes, 96 configs | as B | yes for the Σx² kernel; projection as gemm_a16w16 (`NUM_KSPLIT`, `kpack` fixed) | no exposed parameters; internal choices by M (`tile_n`, big-fuse launch) |
| Bits equal across Triton 3.7.0 / 3.8.0 | no | yes | as B | not measured | not applicable (HIP); hipcc versions not measured |
| post_mix max error vs fp32 reference | 3.6e-3 (bf16 `fn`) | 3.6e-3 | as B | 3.6e-3 | 2.4e-6 – 6.4e-6 (fp32 `fn`) |
| layer_input mean error | 1.50e-3 | 1.50e-3 | as B | 1.50e-3 | 2.53e-3 |
| mhc_pre time vs A | 1 | S = 4: 0.68–0.96; S = 16: 0.36–0.55 for M ≤ 1024, 1.07–1.48 for M ≥ 16384 | as B | 1.31–3.97 | 8×64: 0.38–0.97; 4×128: 0.61–0.97 |
| mhc_head time vs A | 1 | S = 4: 0.04–0.15 | as B | — | 8×64: 0.03–0.14 |
| Code and dependencies | aiter Triton kernels, not pinned by RL-Kernel | RL-Kernel only (Triton); no aiter | aiter fork + pin mechanism RL-Kernel does not have | aiter gemm_a16w16 + B's kernel | aiter HIP module (hipcc, CK headers, JIT); internal functions; image needs flydsl ≥ 0.2.4 for a normal `import aiter` |
| Not done yet | — | autograd/backward wrapper (A uses aiter's PyTorch recompute), registry entry and tests, shapes other than C = 4096, gfx950; choosing S (part of the numerics) | as B, plus the pin | — | wrapper enforcing a contiguous residual, backward, head without the 20 wasted projection columns |

Open questions that do not depend on the choice:

- projection precision: DeepSeek-V4's reference uses fp32 `fn`; A, B and D
  round it to bf16, E keeps fp32 (RFC task 1 question, unchanged);
- whether results must also match across Triton or ROCm versions (A does not;
  B did for 3.7.0 vs 3.8.0).


## 8. How this was found

From the local agent transcripts of the KLR work (`~/.claude/projects/-home-AMD-zitwang-dev-KLR/`)
and the KLR analysis document `issue-81-zitian-wang-mhc-tasks-analysis.md`:

| Date | What happened |
|---|---|
| 2026-10-29/30 | RFC evaluation: the mHC pre bit changes from M = 1024 on are attributed to aiter's K split (analysis doc `:254`, `:343`); the exact-input isolation ran with aiter's default config, where `NUM_KSPLIT`, `BLOCK_K` and `BLOCK_C` change together, so it could not separate them |
| 2026-11-02 | first written suspicion that the sum-of-squares `tl.sum` tree follows `BLOCK_K` ("很可能", unverified) |
| 2026-11-03 | per-M tuning measures that `BLOCK_K` changes mHC bits but not gemm_a16w16 bits; analysis doc `:628` keeps "很可能" |
| 2026-11-04 | weekly report states the cause as split-K "连同 RMS"; the isolation script `measure_mhc_block_k_bit_source.py` is written but not run |
| 2026-11-06 | job 68297426: `BLOCK_K` acts through the sum of squares only |
| 2026-11-08 | jobs 68321538 / 68321539 (this document) |

## 9. Reproduction

| What | Script | Job | Log |
|---|---|---|---|
| bits per input kind and TTGIR layouts of aiter's configs | `benchmarks/operators/mhc/analyze_aiter_reductions.py` | 68321538 | `logs/mhc-bi-analysis-68321538-triton-{image,3.8.0-amd}.md` |
| B / D: launch invariance, BI, error, device time, Triton digests | `benchmarks/operators/mhc/evaluate_mhc_candidates.py` | 68321539 | `logs/mhc-bi-candidates-68321539*.md` |
| aiter HIP `mhc_pre` | `benchmarks/operators/mhc/probe_aiter_hip_mhc_pre.py` | 68321569 | `logs/mhc-hip-probe-68321569.md` |
| `BLOCK_K` source (first isolation) | KLR `scripts/measure_mhc_block_k_bit_source.py` | 68297426 | `logs/mhc-block-k-bit-source-68297426.md` |

Logs are under `/home/AMD/zitwang/dev/logs/`; job scripts
`run-mhc-bi-analysis.sbatch`, `run-mhc-bi-candidates.sbatch`,
`run-mhc-hip-probe.sbatch` in `/home/AMD/zitwang/dev/`.
