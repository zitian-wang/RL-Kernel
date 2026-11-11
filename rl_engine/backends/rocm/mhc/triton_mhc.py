# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Modifications copyright (c) 2026 RL-Kernel Contributors
"""Batch-invariant DeepSeek-V4 mHC pre, post and head on ROCm (Triton).

Copied from aiter 326cffa: the kernels from
aiter/ops/triton/_triton_kernels/fusions/mhc.py (``_mhc_fused_split_kernel``,
``_mhc_reduce_apply_kernel``, ``_mhc_asymmetric_sinkhorn_kernel``,
``_mhc_post_kernel``), the launchers and the PyTorch-recompute backward from
aiter/ops/triton/fusions/mhc.py (``mhc``, ``mhc_pre_dsv4``, ``mhc_post_dsv4``,
``mhc_head_dsv4``). aiter's arithmetic is kept except for three changes, so the
results differ from aiter's only by the order of the sums and by the fp32
projection (docs/operators/mhc-reduction-order.md):

1. Batch invariance by construction. Every M runs the same two kernels: a split
   kernel over NUM_KSPLIT = 4 segments of K, then an apply kernel that adds the
   segments in segment order (aiter switches between a fused kernel and the split
   pair, and picks the split count by M). Every sum has its order written in the
   source instead of leaving it to ``tl.sum``, whose tree follows the tile layout:
   the sum of squares is accumulated element-wise with ``tl.fma`` over the K tiles
   of a segment, then the BLOCK_K = 128 columns are added pairwise; the weighted
   sum of the n streams is one ``tl.fma`` per stream in stream order; the Sinkhorn
   row and column sums are pairwise. BLOCK_K and NUM_KSPLIT are module constants;
   the other launch parameters only regroup tokens and hidden positions.
2. Dynamic shape. M is a runtime argument of every kernel and is not specialized
   (``do_not_specialize``), and the partial results are stored [M, split, column],
   so no stride depends on M. A new token count does not compile anything.
3. Projection precision of the DeepSeek-V4 model: ``fn`` stays fp32 and the
   projection is an fp32 ``tl.dot`` with ``input_precision="ieee"`` (aiter rounds
   ``fn`` to bf16). Its MFMA shape is fixed (``matrix_instr_nonkdim=16``,
   ``kpack=1``). The PyTorch-recompute backward projects with the fp32 ``fn`` too.

mHC post is aiter's kernel, which already adds the streams in an explicit loop;
only its launch config is fixed. mHC head uses the split and apply structure of
mHC pre (aiter runs one workgroup per 16 tokens over all of K and C).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rl_engine.utils.logger import logger

# Part of the arithmetic: changing them changes the bits.
NUM_KSPLIT = 4
BLOCK_K = 128
DOT_OPTIONS = {"matrix_instr_nonkdim": 16, "kpack": 1}

# Launch configs. They do not change the bits (only how tokens and hidden
# positions are grouped into workgroups); one config for every M, so each kernel
# compiles once per process.
PRE_CONFIG = {
    "BLOCK_M": 32,
    "num_warps": 4,
    "num_stages": 1,
    "waves_per_eu": 0,
    "APPLY_BLOCK_M": 32,
    "APPLY_BLOCK_C": 128,
    "apply_num_warps": 4,
    "SINKHORN_BLOCK_M": 16,
}
HEAD_CONFIG = {k: v for k, v in PRE_CONFIG.items() if k != "SINKHORN_BLOCK_M"}
# aiter 326cffa's mhc_post config on gfx942 (mhc_post/DEFAULT.json) with BLOCK_M = 16 instead of
# 64: on MI325X (C = 4096) it took 0.03-0.20 of the time of aiter's config from M = 1 to 65536.
POST_CONFIG = {"BLOCK_M": 16, "BLOCK_C": 256, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2}

# DeepSeek-V4's hc_mult; the ops are validated with it only.
SUPPORTED_N = 4


@triton.jit
def _pairwise_sum_last(v, ROWS: tl.constexpr, WIDTH: tl.constexpr, LOG2_WIDTH: tl.constexpr):
    """[ROWS, WIDTH] -> [ROWS]: add adjacent pairs, LOG2_WIDTH levels.

    Each ``tl.sum`` adds exactly two numbers, which rounds the same way whatever
    lanes hold them, so the order is fixed by WIDTH alone.
    """
    for i in tl.static_range(LOG2_WIDTH):
        v = tl.sum(tl.reshape(v, (ROWS, WIDTH >> (i + 1), 2)), axis=2)
    return tl.reshape(v, (ROWS,))


@triton.jit
def _sum_splits(ptr, offsets, mask, stride_split, NUM_SPLITS: tl.constexpr):
    """Partial results of the K segments added in segment order."""
    total = tl.load(ptr + offsets, mask=mask, other=0.0)
    for s in tl.static_range(1, NUM_SPLITS):
        total += tl.load(ptr + s * stride_split + offsets, mask=mask, other=0.0)
    return total


@triton.jit(do_not_specialize=["M"])
def _mhc_fused_split_kernel(
    x_ptr,
    phi_ptr,  # fn in checkpoint layout (N, K), fp32
    acc_ptr,  # (M, NUM_KSPLIT, N) fp32
    acc_sq_ptr,  # (M, NUM_KSPLIT) fp32
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_phi_k,
    stride_phi_n,
    stride_acc_m,
    stride_acc_k,
    stride_acc_n,
    stride_acc_sq_m,
    stride_acc_sq_k,
    BLOCK_M: tl.constexpr,
    N_TOTAL_POW2: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOG2_BLOCK_K: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
):
    """Partial projection x @ fn^T and partial sum of x^2 of one K segment.

    Grid: (cdiv(M, BLOCK_M), NUM_KSPLIT). SPLITK_BLOCK_SIZE is a multiple of
    BLOCK_K; K positions past the end of K are loaded as zeros.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = tl.arange(0, N_TOTAL_POW2)

    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)

    acc = tl.zeros([BLOCK_M, N_TOTAL_POW2], dtype=tl.float32)
    # One accumulator per element of the tile: no sum inside the K loop.
    acc_sq = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    for k_idx in range(SPLITK_BLOCK_SIZE // BLOCK_K):
        rk = split_k_start + k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask=(rm[:, None] < M) & (rk[None, :] < split_k_end),
            other=0.0,
        ).to(tl.float32)
        phi_tile = tl.load(
            phi_ptr + rk[:, None] * stride_phi_k + rn[None, :] * stride_phi_n,
            mask=(rk[:, None] < split_k_end) & (rn[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(x_tile, phi_tile, acc=acc, input_precision="ieee")
        acc_sq = tl.fma(x_tile, x_tile, acc_sq)

    tl.store(
        acc_ptr + rm[:, None] * stride_acc_m + pid_k * stride_acc_k + rn[None, :] * stride_acc_n,
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )
    tl.store(
        acc_sq_ptr + rm * stride_acc_sq_m + pid_k * stride_acc_sq_k,
        _pairwise_sum_last(acc_sq, BLOCK_M, BLOCK_K, LOG2_BLOCK_K),
        mask=rm < M,
    )


@triton.jit(do_not_specialize=["M"])
def _mhc_reduce_apply_kernel(
    acc_ptr,  # (M, NUM_KSPLIT, n + n + n_squared), layout [pre | post | res]
    acc_sq_ptr,  # (M, NUM_KSPLIT)
    alpha_ptr,  # (3,) fp32: alpha_pre, alpha_post, alpha_res
    bias_ptr,  # (n + n + n_squared,) fp32
    x_ptr,  # (M, n*C)
    out_ptr,  # (M, n + n_squared), layout [post | res logits]
    layer_input_ptr,  # (M, C) in x.dtype
    M,
    K: tl.constexpr,
    n: tl.constexpr,
    n_squared: tl.constexpr,
    C: tl.constexpr,
    eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    stride_acc_m,
    stride_acc_k,
    stride_acc_n,
    stride_acc_sq_m,
    stride_acc_sq_k,
    stride_xm,
    stride_xk,
    stride_out_m,
    stride_out_n,
    stride_li_m,
    stride_li_c,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    N_POW2: tl.constexpr,
    N_POW2_RES: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    RES_PID_C: tl.constexpr,
):
    """Add the K segments, RMS factor, the three streams and the apply step.

    Grid: ``(cdiv(M, BLOCK_M), cdiv(C, BLOCK_C))``. Every program computes the pre
    weights and its BLOCK_C slice of ``layer_input``; ``pid_c == 0`` also writes
    the post weights, ``pid_c == RES_PID_C`` the raw comb logits (Sinkhorn runs
    in ``_mhc_sinkhorn_kernel``).
    """
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    alpha_pre_val = tl.load(alpha_ptr)
    alpha_post_val = tl.load(alpha_ptr + 1)
    alpha_res_val = tl.load(alpha_ptr + 2)

    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    row_ok = rm < M

    # --- 1) RMS normalization (Eq 15) ---
    acc_sq = _sum_splits(acc_sq_ptr, rm * stride_acc_sq_m, row_ok, stride_acc_sq_k, NUM_SPLITS)
    rms = tl.sqrt(acc_sq / K + eps)
    rsigma = 1.0 / rms

    # --- 2) Pre stream (Eq 16-17) and the apply step, one stream at a time ---
    #   layer_input = pre_0 * x_0, then fma(pre_i, x_i, layer_input) for i = 1 .. n-1
    tile_ok = row_ok[:, None] & (rc[None, :] < C)
    layer_input = tl.zeros([BLOCK_M, BLOCK_C], dtype=tl.float32)
    for i in tl.static_range(n):
        acc_pre = _sum_splits(
            acc_ptr, rm * stride_acc_m + i * stride_acc_n, row_ok, stride_acc_k, NUM_SPLITS
        )
        h_pre = rsigma * alpha_pre_val * acc_pre + tl.load(bias_ptr + i)
        pre_mix = tl.sigmoid(h_pre) + hc_pre_eps
        x_i = tl.load(
            x_ptr + rm[:, None] * stride_xm + (i * C + rc[None, :]) * stride_xk,
            mask=tile_ok,
            other=0.0,
        ).to(tl.float32)
        if i == 0:
            layer_input = pre_mix[:, None] * x_i
        else:
            layer_input = tl.fma(pre_mix[:, None], x_i, layer_input)
    tl.store(
        layer_input_ptr + rm[:, None] * stride_li_m + rc[None, :] * stride_li_c,
        layer_input.to(layer_input_ptr.dtype.element_ty),
        mask=tile_ok,
    )

    # --- 3) Post stream on pid_c == 0 (Eq 18) ---
    if pid_c == 0:
        rn_post_local = tl.arange(0, N_POW2)
        rn_post_global = n + rn_post_local
        post_mask = row_ok[:, None] & (rn_post_local[None, :] < n)
        acc_post = _sum_splits(
            acc_ptr,
            rm[:, None] * stride_acc_m + rn_post_global[None, :] * stride_acc_n,
            post_mask,
            stride_acc_k,
            NUM_SPLITS,
        )
        bias_post = tl.load(bias_ptr + rn_post_global, mask=rn_post_local < n, other=0.0)
        h_post = rsigma[:, None] * alpha_post_val * acc_post + bias_post[None, :]
        out_post = tl.sigmoid(h_post) * hc_post_mult_value
        tl.store(
            out_ptr + rm[:, None] * stride_out_m + rn_post_local[None, :] * stride_out_n,
            out_post,
            mask=post_mask,
        )

    # --- 4) Res stream logits on pid_c == RES_PID_C (Eq 19 before Sinkhorn) ---
    if pid_c == RES_PID_C:
        rn_res_local = tl.arange(0, N_POW2_RES)
        rn_res_global = 2 * n + rn_res_local
        res_mask = row_ok[:, None] & (rn_res_local[None, :] < n_squared)
        acc_res = _sum_splits(
            acc_ptr,
            rm[:, None] * stride_acc_m + rn_res_global[None, :] * stride_acc_n,
            res_mask,
            stride_acc_k,
            NUM_SPLITS,
        )
        bias_res = tl.load(bias_ptr + rn_res_global, mask=rn_res_local < n_squared, other=0.0)
        h_res = rsigma[:, None] * alpha_res_val * acc_res + bias_res[None, :]
        tl.store(
            out_ptr + rm[:, None] * stride_out_m + (n + rn_res_local[None, :]) * stride_out_n,
            h_res,
            mask=res_mask,
        )


@triton.jit
def _row_sums(p, BLOCK_M: tl.constexpr, n: tl.constexpr, LOG2_N: tl.constexpr):
    """[BLOCK_M, n, n] -> [BLOCK_M, n]: sum over the last axis, pairwise."""
    flat = tl.reshape(p, (BLOCK_M * n, n))
    return tl.reshape(_pairwise_sum_last(flat, BLOCK_M * n, n, LOG2_N), (BLOCK_M, n))


@triton.jit
def _col_sums(p, BLOCK_M: tl.constexpr, n: tl.constexpr, LOG2_N: tl.constexpr):
    """[BLOCK_M, n, n] -> [BLOCK_M, n]: sum over the middle axis, pairwise."""
    return _row_sums(tl.permute(p, (0, 2, 1)), BLOCK_M, n, LOG2_N)


@triton.jit(do_not_specialize=["M"])
def _mhc_sinkhorn_kernel(
    logits_ptr,
    out_ptr,
    M,
    stride_logits_m,
    stride_logits_n,
    stride_out_m,
    stride_out_n,
    n: tl.constexpr,
    LOG2_N: tl.constexpr,
    NUM_SINKHORN_ITERS: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """TileKernels-compatible asymmetric exp-domain Sinkhorn, BLOCK_M tokens per program.

    aiter's ``_mhc_asymmetric_sinkhorn_kernel`` with the row and column sums
    written as pairwise additions; the maxima need no fixed order.
    """
    pid = tl.program_id(0)
    rm = (pid * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    ri = tl.arange(0, n)
    entry = ri[None, :, None] * n + ri[None, None, :]
    mask = (rm[:, None, None] < M) & (entry >= 0)
    values = tl.load(
        logits_ptr + rm[:, None, None] * stride_logits_m + entry * stride_logits_n,
        mask=mask,
        other=0.0,
    )
    if NUM_SINKHORN_ITERS > 0:
        row_max = tl.max(values, axis=2)
        P = tl.exp(values - row_max[:, :, None])
        P = P / _row_sums(P, BLOCK_M, n, LOG2_N)[:, :, None] + eps
        P = P / (_col_sums(P, BLOCK_M, n, LOG2_N)[:, None, :] + eps)
        for _ in range(NUM_SINKHORN_ITERS - 1):
            P = P / (_row_sums(P, BLOCK_M, n, LOG2_N)[:, :, None] + eps)
            P = P / (_col_sums(P, BLOCK_M, n, LOG2_N)[:, None, :] + eps)
        values = P
    tl.store(out_ptr + rm[:, None, None] * stride_out_m + entry * stride_out_n, values, mask=mask)


@triton.jit(do_not_specialize=["M"])
def _mhc_head_apply_kernel(
    acc_ptr,  # (M, NUM_KSPLIT, n)
    acc_sq_ptr,  # (M, NUM_KSPLIT)
    scale_ptr,
    base_ptr,
    x_ptr,
    out_ptr,
    M,
    K: tl.constexpr,
    n: tl.constexpr,
    C: tl.constexpr,
    eps: tl.constexpr,
    pre_eps: tl.constexpr,
    stride_acc_m,
    stride_acc_k,
    stride_acc_n,
    stride_acc_sq_m,
    stride_acc_sq_k,
    stride_xm,
    stride_xk,
    stride_om,
    stride_oc,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """aiter ``_mhc_head_kernel``'s epilogue: RMS factor, sigmoid weights and the
    stream contraction, for one (M tile, C tile); the streams are added one
    ``tl.fma`` at a time in stream order."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    row_ok = rm < M
    acc_sq = _sum_splits(acc_sq_ptr, rm * stride_acc_sq_m, row_ok, stride_acc_sq_k, NUM_SPLITS)
    rsigma = tl.math.rsqrt(acc_sq / K + eps)
    scale = tl.load(scale_ptr).to(tl.float32)
    tile_ok = row_ok[:, None] & (rc[None, :] < C)
    out = tl.zeros([BLOCK_M, BLOCK_C], dtype=tl.float32)
    for i in tl.static_range(n):
        acc = _sum_splits(
            acc_ptr, rm * stride_acc_m + i * stride_acc_n, row_ok, stride_acc_k, NUM_SPLITS
        )
        mix = tl.sigmoid(acc * rsigma * scale + tl.load(base_ptr + i).to(tl.float32)) + pre_eps
        x_i = tl.load(
            x_ptr + rm[:, None] * stride_xm + (i * C + rc[None, :]) * stride_xk,
            mask=tile_ok,
            other=0.0,
        ).to(tl.float32)
        if i == 0:
            out = mix[:, None] * x_i
        else:
            out = tl.fma(mix[:, None], x_i, out)
    tl.store(
        out_ptr + rm[:, None] * stride_om + rc[None, :] * stride_oc,
        out.to(out_ptr.dtype.element_ty),
        mask=tile_ok,
    )


@triton.jit(do_not_specialize=["M"])
def _mhc_post_kernel(
    out_ptr,  # (M, n, C)  bf16 / fp16
    x_ptr,  # (M, C)     bf16 / fp16  (layer_input from mhc())
    residual_ptr,  # (M, n, C)  bf16 / fp16
    post_mix_ptr,  # (M, n)     fp32  (mhc()'s h_post)
    comb_mix_ptr,  # (M, n, n)  fp32  [src, dst]  (mhc()'s h_res)
    M,
    C,
    stride_x_m,
    stride_x_c,
    stride_res_m,
    stride_res_n,
    stride_res_c,
    stride_out_m,
    stride_out_n,
    stride_out_c,
    stride_post_m,
    stride_post_n,
    stride_comb_m,
    stride_comb_src,
    stride_comb_dst,
    n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fused mhc_post kernel: compute one M-tile across all `n` output
    streams and the full hidden dim.

        out[m, j, c] = post_mix[m, j] * x[m, c]
                     + sum_h comb_mix[m, h, j] * residual[m, h, c]

    Grid: ``(cdiv(M, BLOCK_M),)``. Each program loads ``post_mix``
    (BLOCK_M, n) and ``comb_mix`` (BLOCK_M, n, n) once and reuses them
    across the persistent loop over ``BLOCK_C``-sized C-tiles. The
    ``n``-source-head contraction inside each C-tile is unrolled via
    ``tl.static_range``: each iteration loads a 2-D ``residual`` slice and
    a 1-D ``comb_mix`` row and accumulates ``comb_h * res_h`` into
    ``out_tile``, in source-stream order. Requires ``n`` to be a power of 2 so
    ``tl.arange(0, n)`` compiles.
    """
    pid_m = tl.program_id(0)

    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    i_n = tl.arange(0, n)

    m_mask = rm < M

    post_mix_tile = tl.load(
        post_mix_ptr + rm[:, None] * stride_post_m + i_n[None, :] * stride_post_n,
        mask=m_mask[:, None],
        other=0.0,
    )

    # Pre-load each row of the per-token (n_src, n_dst) comb_mix matrix into
    # a tuple of 2-D tiles. Each ``comb_rows[h]`` has shape (BLOCK_M, n_dst)
    # and lives in registers across the C-loop, eliminating per-C-tile
    # reloads of the small per-token coefficients.
    comb_rows = ()
    for h in tl.static_range(n):
        comb_rows = comb_rows + (
            tl.load(
                comb_mix_ptr
                + rm[:, None] * stride_comb_m
                + h * stride_comb_src
                + i_n[None, :] * stride_comb_dst,
                mask=m_mask[:, None],
                other=0.0,
            ),
        )

    for c_start in range(0, C, BLOCK_C):
        rc = c_start + tl.arange(0, BLOCK_C)
        c_mask = rc < C

        x_tile = tl.load(
            x_ptr + rm[:, None] * stride_x_m + rc[None, :] * stride_x_c,
            mask=m_mask[:, None] & c_mask[None, :],
            other=0.0,
            cache_modifier=".cg",
        )

        out_tile = post_mix_tile[:, :, None] * x_tile[:, None, :].to(tl.float32)

        for h in tl.static_range(n):
            res_h = tl.load(
                residual_ptr
                + rm[:, None] * stride_res_m
                + h * stride_res_n
                + rc[None, :] * stride_res_c,
                mask=m_mask[:, None] & c_mask[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
            comb_h = comb_rows[h]  # (BLOCK_M, n_dst), pre-loaded 2-D tile
            out_tile += comb_h[:, :, None] * res_h[:, None, :].to(tl.float32)

        tl.store(
            out_ptr
            + rm[:, None, None] * stride_out_m
            + i_n[None, :, None] * stride_out_n
            + rc[None, None, :] * stride_out_c,
            out_tile.to(out_ptr.dtype.element_ty),
            mask=m_mask[:, None, None] & c_mask[None, None, :],
            cache_modifier=".cs",
        )


# --- launchers -------------------------------------------------------------------------------


def _log2(v: int) -> int:
    if v <= 0 or v & (v - 1):
        raise ValueError(f"{v} is not a power of two")
    return v.bit_length() - 1


def _split(x: torch.Tensor, fn: torch.Tensor, config: dict):
    """Partial projections (M, NUM_KSPLIT, N) and sums of squares (M, NUM_KSPLIT)."""
    M, K = x.shape
    N = fn.shape[0]
    # A multiple of BLOCK_K, from K only.
    splitk_block_size = triton.cdiv(triton.cdiv(K, NUM_KSPLIT), BLOCK_K) * BLOCK_K
    acc = torch.empty((M, NUM_KSPLIT, N), dtype=torch.float32, device=x.device)
    acc_sq = torch.empty((M, NUM_KSPLIT), dtype=torch.float32, device=x.device)
    BLOCK_M = config["BLOCK_M"]
    _mhc_fused_split_kernel[(triton.cdiv(M, BLOCK_M), NUM_KSPLIT)](
        x,
        fn,
        acc,
        acc_sq,
        M,
        K=K,
        N=N,
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_phi_k=fn.stride(1),
        stride_phi_n=fn.stride(0),
        stride_acc_m=acc.stride(0),
        stride_acc_k=acc.stride(1),
        stride_acc_n=acc.stride(2),
        stride_acc_sq_m=acc_sq.stride(0),
        stride_acc_sq_k=acc_sq.stride(1),
        BLOCK_M=BLOCK_M,
        N_TOTAL_POW2=max(16, triton.next_power_of_2(N)),
        BLOCK_K=BLOCK_K,
        LOG2_BLOCK_K=_log2(BLOCK_K),
        SPLITK_BLOCK_SIZE=splitk_block_size,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        waves_per_eu=config["waves_per_eu"],
        **DOT_OPTIONS,
    )
    return acc, acc_sq


def _mhc_pre_dsv4_forward(
    residual, fn, scale, base, eps, pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_iters, config
):
    M, n, C = residual.shape
    if M == 0:
        return (
            torch.empty((0, n, 1), dtype=torch.float32, device=residual.device),
            torch.empty((0, n, n), dtype=torch.float32, device=residual.device),
            torch.empty((0, C), dtype=residual.dtype, device=residual.device),
        )
    cfg = dict(PRE_CONFIG, **(config or {}))
    n_squared = n * n
    x = residual.view(M, n * C)
    acc, acc_sq = _split(x, fn, cfg)

    out = torch.empty((M, n + n_squared), dtype=torch.float32, device=x.device)
    layer_input = torch.empty((M, C), dtype=x.dtype, device=x.device)
    BLOCK_M, BLOCK_C = cfg["APPLY_BLOCK_M"], cfg["APPLY_BLOCK_C"]
    # Post weights on pid_c == 0, comb logits on pid_c == 1 when there are two C tiles.
    RES_PID_C = 0 if triton.cdiv(C, BLOCK_C) == 1 else 1
    _mhc_reduce_apply_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(C, BLOCK_C))](
        acc,
        acc_sq,
        scale,
        base,
        x,
        out,
        layer_input,
        M,
        K=n * C,
        n=n,
        n_squared=n_squared,
        C=C,
        eps=eps,
        hc_pre_eps=pre_eps,
        hc_post_mult_value=post_multiplier,
        stride_acc_m=acc.stride(0),
        stride_acc_k=acc.stride(1),
        stride_acc_n=acc.stride(2),
        stride_acc_sq_m=acc_sq.stride(0),
        stride_acc_sq_k=acc_sq.stride(1),
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_out_m=out.stride(0),
        stride_out_n=out.stride(1),
        stride_li_m=layer_input.stride(0),
        stride_li_c=layer_input.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        N_POW2=triton.next_power_of_2(n),
        N_POW2_RES=triton.next_power_of_2(n_squared),
        NUM_SPLITS=NUM_KSPLIT,
        RES_PID_C=RES_PID_C,
        num_warps=cfg["apply_num_warps"],
    )

    raw_comb = out[:, n:]
    comb = torch.empty((M, n, n), dtype=torch.float32, device=x.device)
    comb_flat = comb.view(M, n_squared)
    SINKHORN_BLOCK_M = cfg["SINKHORN_BLOCK_M"]
    _mhc_sinkhorn_kernel[(triton.cdiv(M, SINKHORN_BLOCK_M),)](
        raw_comb,
        comb_flat,
        M,
        stride_logits_m=raw_comb.stride(0),
        stride_logits_n=raw_comb.stride(1),
        stride_out_m=comb_flat.stride(0),
        stride_out_n=comb_flat.stride(1),
        n=n,
        LOG2_N=_log2(n),
        NUM_SINKHORN_ITERS=sinkhorn_iters,
        eps=sinkhorn_eps,
        BLOCK_M=SINKHORN_BLOCK_M,
        num_warps=1,
    )
    return out[:, :n].unsqueeze(-1), comb, layer_input


def _mhc_head_dsv4_forward(residual, fn, scale, base, eps, pre_eps, config):
    M, n, C = residual.shape
    out = torch.empty((M, C), dtype=residual.dtype, device=residual.device)
    if M == 0:
        return out
    cfg = dict(HEAD_CONFIG, **(config or {}))
    x = residual.view(M, n * C)
    acc, acc_sq = _split(x, fn, cfg)
    BLOCK_M, BLOCK_C = cfg["APPLY_BLOCK_M"], cfg["APPLY_BLOCK_C"]
    _mhc_head_apply_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(C, BLOCK_C))](
        acc,
        acc_sq,
        scale,
        base,
        x,
        out,
        M,
        K=n * C,
        n=n,
        C=C,
        eps=eps,
        pre_eps=pre_eps,
        stride_acc_m=acc.stride(0),
        stride_acc_k=acc.stride(1),
        stride_acc_n=acc.stride(2),
        stride_acc_sq_m=acc_sq.stride(0),
        stride_acc_sq_k=acc_sq.stride(1),
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_om=out.stride(0),
        stride_oc=out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        NUM_SPLITS=NUM_KSPLIT,
        num_warps=cfg["apply_num_warps"],
    )
    return out


def _mhc_post_forward(layer_input, residual, post_mix, comb_mix, config):
    M, n, C = residual.shape
    out = torch.empty((M, n, C), dtype=layer_input.dtype, device=layer_input.device)
    if M == 0:
        return out
    cfg = dict(POST_CONFIG, **(config or {}))
    BLOCK_M = cfg.pop("BLOCK_M")
    BLOCK_C = min(cfg.pop("BLOCK_C"), triton.next_power_of_2(C))
    if post_mix.ndim == 3:
        post_mix = post_mix.squeeze(-1)
    _mhc_post_kernel[(triton.cdiv(M, BLOCK_M),)](
        out,
        layer_input,
        residual,
        post_mix,
        comb_mix,
        M,
        C=C,
        stride_x_m=layer_input.stride(0),
        stride_x_c=layer_input.stride(1),
        stride_res_m=residual.stride(0),
        stride_res_n=residual.stride(1),
        stride_res_c=residual.stride(2),
        stride_out_m=out.stride(0),
        stride_out_n=out.stride(1),
        stride_out_c=out.stride(2),
        stride_post_m=post_mix.stride(0),
        stride_post_n=post_mix.stride(1),
        stride_comb_m=comb_mix.stride(0),
        stride_comb_src=comb_mix.stride(1),
        stride_comb_dst=comb_mix.stride(2),
        n=n,
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        **cfg,
    )
    return out


# --- PyTorch-recompute backward (aiter's, with the fp32 projection) ----------------------------


def _mhc_pre_dsv4_torch(
    residual, fn, scale, base, eps, pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_iters
):
    """Differentiable correctness fallback used only by backward."""
    M, n, C = residual.shape
    x = residual.float().reshape(M, n * C)
    rms = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True) + eps)
    logits = (x / rms) @ fn.T
    stream_scale = torch.cat((scale[0].expand(n), scale[1].expand(n), scale[2].expand(n * n)))
    logits = logits * stream_scale + base.float()
    pre = torch.sigmoid(logits[:, :n]) + pre_eps
    post = (post_multiplier * torch.sigmoid(logits[:, n : 2 * n])).unsqueeze(-1)
    comb = logits[:, 2 * n :].reshape(M, n, n)
    if sinkhorn_iters > 0:
        row_max = comb.amax(dim=-1, keepdim=True)
        comb = torch.exp(comb - row_max)
        comb = comb / comb.sum(dim=-1, keepdim=True) + sinkhorn_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + sinkhorn_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    layer_input = torch.einsum("mn,mnc->mc", pre, residual.float())
    return post, comb, layer_input.to(residual.dtype)


def _mhc_head_dsv4_torch(residual, fn, scale, base, eps, pre_eps):
    """Differentiable correctness fallback used only by backward."""
    M, n, C = residual.shape
    x = residual.float().reshape(M, n * C)
    rms = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True) + eps)
    logits = ((x / rms) @ fn.T) * scale.reshape(1) + base.float()
    mix = torch.sigmoid(logits) + pre_eps
    out = torch.einsum("mn,mnc->mc", mix, residual.float())
    return out.to(residual.dtype)


class _MHCPreDSV4(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        residual,
        fn,
        scale,
        base,
        eps,
        pre_eps,
        sinkhorn_eps,
        post_multiplier,
        sinkhorn_iters,
        config,
    ):
        ctx.save_for_backward(residual, fn, scale, base)
        ctx.params = (eps, pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_iters)
        return _mhc_pre_dsv4_forward(
            residual,
            fn,
            scale,
            base,
            eps,
            pre_eps,
            sinkhorn_eps,
            post_multiplier,
            sinkhorn_iters,
            config,
        )

    @staticmethod
    def backward(ctx, grad_post, grad_comb, grad_layer):
        residual, fn, scale, base = ctx.saved_tensors
        inputs = tuple(t.detach().requires_grad_(True) for t in (residual, fn, scale, base))
        with torch.enable_grad():
            outputs = _mhc_pre_dsv4_torch(*inputs, *ctx.params)
        grad_outputs = tuple(
            torch.zeros_like(output) if grad is None else grad
            for output, grad in zip(outputs, (grad_post, grad_comb, grad_layer))
        )
        grads = torch.autograd.grad(outputs, inputs, grad_outputs)
        return (*grads, None, None, None, None, None, None)


class _MHCPostDSV4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, layer_input, residual, post_mix, comb_mix, config):
        ctx.save_for_backward(layer_input, residual, post_mix, comb_mix)
        return _mhc_post_forward(layer_input, residual, post_mix, comb_mix, config)

    @staticmethod
    def backward(ctx, grad_out):
        layer_input, residual, post_mix, comb_mix = ctx.saved_tensors
        inputs = tuple(
            t.detach().requires_grad_(True) for t in (layer_input, residual, post_mix, comb_mix)
        )
        with torch.enable_grad():
            x, res, post, comb = inputs
            post_2d = post.squeeze(-1) if post.ndim == 3 else post
            out = post_2d[:, :, None] * x.float()[:, None, :]
            out = out + torch.einsum("mhc,mhj->mjc", res.float(), comb.float())
            out = out.to(x.dtype)
        grads = torch.autograd.grad(out, inputs, grad_out)
        return (*grads, None)


class _MHCHeadDSV4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, fn, scale, base, eps, pre_eps, config):
        ctx.save_for_backward(residual, fn, scale, base)
        ctx.params = (eps, pre_eps)
        return _mhc_head_dsv4_forward(residual, fn, scale, base, eps, pre_eps, config)

    @staticmethod
    def backward(ctx, grad_out):
        inputs = tuple(t.detach().requires_grad_(True) for t in ctx.saved_tensors)
        with torch.enable_grad():
            out = _mhc_head_dsv4_torch(*inputs, *ctx.params)
        grads = torch.autograd.grad(out, inputs, grad_out)
        return (*grads, None, None, None)


def mhc_pre_dsv4(
    residual,
    fn,
    scale,
    base,
    *,
    rms_eps=1e-6,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    hc_post_mult_value=2.0,
    sinkhorn_repeat=20,
    config=None,
):
    """residual [M, n, C] bf16 contiguous, fn [2n+n², n*C], scale [3], base [2n+n²] fp32
    -> (post_mix [M, n, 1], comb_mix [M, n, n], layer_input [M, C]).

    ``config`` overrides keys of PRE_CONFIG (launch parameters only; the bits do
    not change)."""
    if sinkhorn_repeat < 0:
        raise ValueError("sinkhorn_repeat must be non-negative")
    return _MHCPreDSV4.apply(
        residual,
        fn,
        scale,
        base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        int(sinkhorn_repeat),
        config,
    )


def mhc_post_dsv4(layer_input, residual, post_mix, comb_mix, *, config=None):
    """layer_input [M, C], residual [M, n, C], post_mix [M, n, 1] or [M, n], comb_mix
    [M, n, n] -> [M, n, C]. ``config`` overrides keys of POST_CONFIG."""
    return _MHCPostDSV4.apply(layer_input, residual, post_mix, comb_mix, config)


def mhc_head_dsv4(residual, fn, scale, base, *, rms_eps=1e-6, hc_pre_eps=1e-6, config=None):
    """residual [M, n, C] bf16 contiguous, fn [n, n*C], scale [1], base [n] fp32 -> [M, C].
    ``config`` overrides keys of HEAD_CONFIG."""
    return _MHCHeadDSV4.apply(residual, fn, scale, base, rms_eps, hc_pre_eps, config)


# --- ops ---------------------------------------------------------------------------------------


def _check(name, tensor, dtype, shape=None):
    """Raise for an input these ops do not support, instead of running it unvalidated."""
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}")


def _check_streams(residual):
    """(n, C) of a bf16 residual [..., n, C] with n = SUPPORTED_N."""
    _check("residual", residual, torch.bfloat16)
    if residual.dim() < 2 or residual.shape[-2] != SUPPORTED_N:
        raise ValueError(
            f"residual must have shape [..., {SUPPORTED_N}, C], got {tuple(residual.shape)}"
        )
    return residual.shape[-2:]


def check_pre_inputs(residual, fn, scale, base):
    """Inputs of mHC pre: bf16 residual [..., n, C]; fp32 fn, scale, base."""
    n, c = _check_streams(residual)
    rows = 2 * n + n * n
    _check("fn", fn, torch.float32, (rows, n * c))
    _check("scale", scale, torch.float32, (3,))
    _check("base", base, torch.float32, (rows,))


def check_post_inputs(y, residual, post_mix, comb_mix):
    """Inputs of mHC post: bf16 y [..., C] and residual [..., n, C]; fp32 post_mix
    [..., n] or [..., n, 1] and comb_mix [..., n, n]."""
    n, c = _check_streams(residual)
    lead = tuple(residual.shape[:-2])
    _check("y", y, torch.bfloat16, (*lead, c))
    _check("post_mix", post_mix, torch.float32)
    if tuple(post_mix.shape) not in ((*lead, n), (*lead, n, 1)):
        raise ValueError(f"post_mix must have shape {(*lead, n)}, got {tuple(post_mix.shape)}")
    _check("comb_mix", comb_mix, torch.float32, (*lead, n, n))


def check_head_inputs(residual, fn, scale, base):
    """Inputs of mHC head: bf16 residual [..., n, C]; fp32 fn, scale, base."""
    n, c = _check_streams(residual)
    _check("fn", fn, torch.float32, (n, n * c))
    _check("scale", scale, torch.float32, (1,))
    _check("base", base, torch.float32, (n,))


def _require_rocm():
    if torch.version.hip is None:
        raise RuntimeError("The Triton mHC ops require a ROCm PyTorch build.")


class TritonMHCPreOp:
    """Batch-invariant mHC pre; see NativeMHCPreOp for the formula."""

    def __init__(self):
        _require_rocm()
        logger.info("Linked RL-Kernel's batch-invariant Triton mHC pre.")

    def __call__(
        self,
        residual,
        fn,
        scale,
        base,
        *,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
    ):
        check_pre_inputs(residual, fn, scale, base)
        n, c = residual.shape[-2:]
        lead = residual.shape[:-2]
        post, comb, layer_input = mhc_pre_dsv4(
            residual.reshape(-1, n, c).contiguous(),
            fn.contiguous(),
            scale.contiguous(),
            base.contiguous(),
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        )
        return post.reshape(*lead, n), comb.reshape(*lead, n, n), layer_input.reshape(*lead, c)


class TritonMHCPostOp:
    """Batch-invariant mHC post (stream mix-back); see NativeMHCPostOp."""

    def __init__(self):
        _require_rocm()
        logger.info("Linked RL-Kernel's batch-invariant Triton mHC post.")

    def __call__(self, y, residual, post_mix, comb_mix):
        check_post_inputs(y, residual, post_mix, comb_mix)
        n, c = residual.shape[-2:]
        lead = residual.shape[:-2]
        out = mhc_post_dsv4(
            y.reshape(-1, c).contiguous(),
            residual.reshape(-1, n, c).contiguous(),
            post_mix.reshape(-1, n).contiguous(),
            comb_mix.reshape(-1, n, n).contiguous(),
        )
        return out.reshape(*lead, n, c)


class TritonMHCHeadOp:
    """Batch-invariant mHC head (stream contraction); see NativeMHCHeadOp."""

    def __init__(self):
        _require_rocm()
        logger.info("Linked RL-Kernel's batch-invariant Triton mHC head.")

    def __call__(self, residual, fn, scale, base, *, rms_eps=1e-6, hc_pre_eps=1e-6):
        check_head_inputs(residual, fn, scale, base)
        n, c = residual.shape[-2:]
        lead = residual.shape[:-2]
        out = mhc_head_dsv4(
            residual.reshape(-1, n, c).contiguous(),
            fn.contiguous(),
            scale.contiguous(),
            base.contiguous(),
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
        )
        return out.reshape(*lead, c)


# Kernels whose compiled variants a dynamic-shape check counts.
KERNELS = (
    _mhc_fused_split_kernel,
    _mhc_reduce_apply_kernel,
    _mhc_sinkhorn_kernel,
    _mhc_head_apply_kernel,
    _mhc_post_kernel,
)
