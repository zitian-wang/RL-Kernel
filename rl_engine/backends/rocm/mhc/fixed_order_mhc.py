# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DeepSeek-V4 mHC pre and head on Triton with every reduction order written in the kernel.

Candidate B of docs/operators/mhc-reduction-order.md; not registered as an op.
aiter's kernels leave two reductions to ``tl.sum``, whose tree follows the tile
layout Triton derives from BLOCK_M, BLOCK_K, BLOCK_C and num_warps: the RMS sum
of squares inside each K tile and the sum over the n residual streams. Here no
reduction depends on a launch parameter:

- sum of squares: each element position of a [BLOCK_M, BLOCK_K] tile is
  accumulated across the K tiles of a split with an FMA, then the BLOCK_K
  positions are added pairwise (adjacent pairs, log2(BLOCK_K) levels). Adding
  two numbers is exact up to the one rounding, whatever lanes hold them, so the
  order is fixed by BLOCK_K alone, which is a module constant;
- projection: ``tl.dot`` over the same K tiles, fn rounded to bf16 as aiter does;
- K splits: the count depends on K only (``num_k_splits``); the partial results
  are added in split order;
- stream sum and Sinkhorn: written per stream and per row/column (FMAs, pairwise
  sums over the n = 4 entries).

BLOCK_M, num_warps, num_stages, waves_per_eu and the tile sizes of the apply and
Sinkhorn kernels only regroup rows and columns.

``fixed_order_mhc_pre(..., projection="aiter_linear")`` is candidate D: the
projection runs as aiter's gemm_a16w16 with the batch-invariant linear's config,
and only the sum of squares runs in the kernel below (x is read once more).
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

# Part of the reduction order: changing it changes the bits.
PRENORM_BLOCK_K = 128

DEFAULT_PRE_CONFIG = {
    "BLOCK_M": 32,
    "num_warps": 4,
    "num_stages": 1,
    "waves_per_eu": 0,
    "APPLY_BLOCK_M": 32,
    "APPLY_BLOCK_C": 128,
    "apply_num_warps": 4,
    "SINKHORN_BLOCK_M": 16,
}

DEFAULT_HEAD_CONFIG = {
    "BLOCK_M": 32,
    "num_warps": 4,
    "num_stages": 1,
    "waves_per_eu": 0,
    "APPLY_BLOCK_M": 32,
    "APPLY_BLOCK_C": 128,
    "apply_num_warps": 4,
}


def num_k_splits(k: int) -> int:
    """Number of K splits, from K only: one split per 4096 elements, at most 16."""
    return max(1, min(16, k // 4096))


@triton.jit
def _pairwise_sum_last(v, ROWS: tl.constexpr, WIDTH: tl.constexpr, LOG2_WIDTH: tl.constexpr):
    """[ROWS, WIDTH] -> [ROWS]: add adjacent pairs, LOG2_WIDTH levels."""
    for i in tl.static_range(LOG2_WIDTH):
        v = tl.sum(tl.reshape(v, (ROWS, WIDTH >> (i + 1), 2)), axis=2)
    return tl.reshape(v, (ROWS,))


@triton.jit
def _pairwise_max_last(v, ROWS: tl.constexpr, WIDTH: tl.constexpr, LOG2_WIDTH: tl.constexpr):
    for i in tl.static_range(LOG2_WIDTH):
        v = tl.max(tl.reshape(v, (ROWS, WIDTH >> (i + 1), 2)), axis=2)
    return tl.reshape(v, (ROWS,))


@triton.jit
def _prenorm_kernel(
    x_ptr,
    fn_ptr,
    acc_ptr,
    sq_ptr,
    M,
    stride_xm,
    stride_xk,
    stride_fn_n,
    stride_fn_k,
    stride_acc_s,
    stride_acc_m,
    stride_acc_n,
    stride_sq_s,
    stride_sq_m,
    N: tl.constexpr,
    N_TILE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOG2_BLOCK_K: tl.constexpr,
    WITH_DOT: tl.constexpr,
):
    """Per (row tile, K split): partial x @ fn^T [BLOCK_M, N] and partial sum of x^2 [BLOCK_M]."""
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = tl.arange(0, N_TILE)
    acc = tl.zeros((BLOCK_M, N_TILE), dtype=tl.float32)
    sq = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    k_start = pid_s * SPLIT_K
    for k in range(0, SPLIT_K, BLOCK_K):
        rk = k_start + k + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask=rm[:, None] < M,
            other=0.0,
        )
        if WITH_DOT:
            fn = tl.load(
                fn_ptr + rn[None, :] * stride_fn_n + rk[:, None] * stride_fn_k,
                mask=rn[None, :] < N,
                other=0.0,
            ).to(x_ptr.dtype.element_ty)
            acc = tl.dot(x, fn, acc=acc)
        xf = x.to(tl.float32)
        sq = tl.fma(xf, xf, sq)
    row_sq = _pairwise_sum_last(sq, BLOCK_M, BLOCK_K, LOG2_BLOCK_K)
    if WITH_DOT:
        tl.store(
            acc_ptr + pid_s * stride_acc_s + rm[:, None] * stride_acc_m + rn[None, :] * stride_acc_n,
            acc,
            mask=(rm[:, None] < M) & (rn[None, :] < N),
        )
    tl.store(sq_ptr + pid_s * stride_sq_s + rm * stride_sq_m, row_sq, mask=rm < M)


@triton.jit
def _merge_splits(ptr, offsets, mask, stride_s, SPLITS: tl.constexpr):
    """Partial results of every split added in split order."""
    total = tl.load(ptr + offsets, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLITS):
        total += tl.load(ptr + s * stride_s + offsets, mask=mask, other=0.0)
    return total


@triton.jit
def _pre_apply_kernel(
    acc_ptr,
    sq_ptr,
    scale_ptr,
    base_ptr,
    x_ptr,
    mix_ptr,
    li_ptr,
    M,
    stride_acc_s,
    stride_acc_m,
    stride_acc_n,
    stride_sq_s,
    stride_sq_m,
    stride_xm,
    stride_xk,
    stride_mix_m,
    stride_mix_n,
    stride_li_m,
    stride_li_c,
    K: tl.constexpr,
    C: tl.constexpr,
    NS: tl.constexpr,
    ACC_SPLITS: tl.constexpr,
    SQ_SPLITS: tl.constexpr,
    MIX_TILE: tl.constexpr,
    eps: tl.constexpr,
    pre_eps: tl.constexpr,
    post_mult: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """RMS factor, pre weights and the layer input for one (row tile, C tile);
    the C tile 0 also writes post weights and the comb logits to ``mix``."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    row_ok = rm < M
    sq = _merge_splits(sq_ptr, rm * stride_sq_m, row_ok, stride_sq_s, SQ_SPLITS)
    rsigma = 1.0 / tl.sqrt(sq / K + eps)
    alpha_pre = tl.load(scale_ptr)

    li = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
    tile_ok = row_ok[:, None] & (rc[None, :] < C)
    for i in tl.static_range(NS):
        h = _merge_splits(acc_ptr, rm * stride_acc_m + i * stride_acc_n, row_ok, stride_acc_s, ACC_SPLITS)
        pre = tl.sigmoid(rsigma * alpha_pre * h + tl.load(base_ptr + i)) + pre_eps
        xi = tl.load(
            x_ptr + rm[:, None] * stride_xm + (i * C + rc[None, :]) * stride_xk,
            mask=tile_ok,
            other=0.0,
        ).to(tl.float32)
        if i == 0:
            li = pre[:, None] * xi
        else:
            li = tl.fma(pre[:, None], xi, li)
    tl.store(
        li_ptr + rm[:, None] * stride_li_m + rc[None, :] * stride_li_c,
        li.to(li_ptr.dtype.element_ty),
        mask=tile_ok,
    )

    if pid_c == 0:
        j = tl.arange(0, MIX_TILE)
        col_ok = j < NS + NS * NS
        mix_mask = row_ok[:, None] & col_ok[None, :]
        acc_mix = _merge_splits(
            acc_ptr,
            rm[:, None] * stride_acc_m + (NS + j[None, :]) * stride_acc_n,
            mix_mask,
            stride_acc_s,
            ACC_SPLITS,
        )
        alpha = tl.where(j < NS, tl.load(scale_ptr + 1), tl.load(scale_ptr + 2))
        bias = tl.load(base_ptr + NS + j, mask=col_ok, other=0.0)
        logits = rsigma[:, None] * alpha[None, :] * acc_mix + bias[None, :]
        mix = tl.where(j[None, :] < NS, tl.sigmoid(logits) * post_mult, logits)
        tl.store(mix_ptr + rm[:, None] * stride_mix_m + j[None, :] * stride_mix_n, mix, mask=mix_mask)


@triton.jit
def _sinkhorn_kernel(
    logits_ptr,
    out_ptr,
    M,
    stride_lm,
    stride_ln,
    stride_om,
    stride_on,
    NS: tl.constexpr,
    LOG2_NS: tl.constexpr,
    ITERS: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """DeepSeek-V4's asymmetric Sinkhorn on [NS, NS] logits per row, BLOCK_M rows per workgroup."""
    pid = tl.program_id(0)
    rm = (pid * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    ri = tl.arange(0, NS)
    entry = ri[None, :, None] * NS + ri[None, None, :]
    mask = (rm[:, None, None] < M) & (entry >= 0)
    a = tl.load(logits_ptr + rm[:, None, None] * stride_lm + entry * stride_ln, mask=mask, other=0.0)
    if ITERS > 0:
        row_max = tl.reshape(
            _pairwise_max_last(tl.reshape(a, (BLOCK_M * NS, NS)), BLOCK_M * NS, NS, LOG2_NS),
            (BLOCK_M, NS),
        )
        p = tl.exp(a - row_max[:, :, None])
        row_sum = tl.reshape(_pairwise_sum_last(tl.reshape(p, (BLOCK_M * NS, NS)), BLOCK_M * NS, NS, LOG2_NS), (BLOCK_M, NS))
        p = p / row_sum[:, :, None] + eps
        col_sum = tl.reshape(
            _pairwise_sum_last(tl.reshape(tl.permute(p, (0, 2, 1)), (BLOCK_M * NS, NS)), BLOCK_M * NS, NS, LOG2_NS),
            (BLOCK_M, NS),
        )
        p = p / (col_sum[:, None, :] + eps)
        for _ in range(ITERS - 1):
            row_sum = tl.reshape(
                _pairwise_sum_last(tl.reshape(p, (BLOCK_M * NS, NS)), BLOCK_M * NS, NS, LOG2_NS), (BLOCK_M, NS)
            )
            p = p / (row_sum[:, :, None] + eps)
            col_sum = tl.reshape(
                _pairwise_sum_last(tl.reshape(tl.permute(p, (0, 2, 1)), (BLOCK_M * NS, NS)), BLOCK_M * NS, NS, LOG2_NS),
                (BLOCK_M, NS),
            )
            p = p / (col_sum[:, None, :] + eps)
        a = p
    tl.store(out_ptr + rm[:, None, None] * stride_om + entry * stride_on, a, mask=mask)


@triton.jit
def _head_apply_kernel(
    acc_ptr,
    sq_ptr,
    scale_ptr,
    base_ptr,
    x_ptr,
    out_ptr,
    M,
    stride_acc_s,
    stride_acc_m,
    stride_acc_n,
    stride_sq_s,
    stride_sq_m,
    stride_xm,
    stride_xk,
    stride_om,
    stride_oc,
    K: tl.constexpr,
    C: tl.constexpr,
    NS: tl.constexpr,
    ACC_SPLITS: tl.constexpr,
    SQ_SPLITS: tl.constexpr,
    eps: tl.constexpr,
    pre_eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    row_ok = rm < M
    sq = _merge_splits(sq_ptr, rm * stride_sq_m, row_ok, stride_sq_s, SQ_SPLITS)
    rsigma = tl.math.rsqrt(sq / K + eps)
    scale = tl.load(scale_ptr)
    out = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
    tile_ok = row_ok[:, None] & (rc[None, :] < C)
    for i in tl.static_range(NS):
        h = _merge_splits(acc_ptr, rm * stride_acc_m + i * stride_acc_n, row_ok, stride_acc_s, ACC_SPLITS)
        mix = tl.sigmoid(h * rsigma * scale + tl.load(base_ptr + i)) + pre_eps
        xi = tl.load(
            x_ptr + rm[:, None] * stride_xm + (i * C + rc[None, :]) * stride_xk,
            mask=tile_ok,
            other=0.0,
        ).to(tl.float32)
        if i == 0:
            out = mix[:, None] * xi
        else:
            out = tl.fma(mix[:, None], xi, out)
    tl.store(
        out_ptr + rm[:, None] * stride_om + rc[None, :] * stride_oc,
        out.to(out_ptr.dtype.element_ty),
        mask=tile_ok,
    )


def _log2(v: int) -> int:
    if v <= 0 or v & (v - 1):
        raise ValueError(f"{v} is not a power of two")
    return v.bit_length() - 1


@functools.lru_cache(maxsize=None)
def _linear_op():
    from rl_engine.backends.rocm.gemm.aiter_gemm import AiterBatchInvariantLinearOp

    return AiterBatchInvariantLinearOp()


def _prenorm(x, fn, k_splits, config, *, with_dot=True):
    """Partial projection [S, M, N] (or None) and partial sum of squares [S, M], S = k_splits."""
    m, k = x.shape
    n = fn.shape[0]
    if k % (k_splits * PRENORM_BLOCK_K):
        raise ValueError(f"K={k} is not a multiple of {k_splits} splits of {PRENORM_BLOCK_K}")
    sq = torch.empty(k_splits, m, dtype=torch.float32, device=x.device)
    acc = torch.empty(k_splits, m, n, dtype=torch.float32, device=x.device) if with_dot else sq
    block_m = config["BLOCK_M"]
    _prenorm_kernel[(triton.cdiv(m, block_m), k_splits)](
        x,
        fn,
        acc,
        sq,
        m,
        x.stride(0),
        x.stride(1),
        fn.stride(0),
        fn.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2) if with_dot else 0,
        sq.stride(0),
        sq.stride(1),
        N=n,
        N_TILE=max(16, triton.next_power_of_2(n)),
        SPLIT_K=k // k_splits,
        BLOCK_M=block_m,
        BLOCK_K=PRENORM_BLOCK_K,
        LOG2_BLOCK_K=_log2(PRENORM_BLOCK_K),
        WITH_DOT=with_dot,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        waves_per_eu=config["waves_per_eu"],
    )
    return (acc if with_dot else None), sq


def fixed_order_mhc_pre(
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
    k_splits=None,
    projection="fused",
):
    """mHC pre: residual [M, n, C] bf16, fn [2n+n^2, n*C] fp32, scale [3], base [2n+n^2].

    Returns (post_mix [M, n, 1] fp32, comb_mix [M, n, n] fp32, layer_input [M, C]).
    ``projection="aiter_linear"`` runs the projection as aiter's gemm_a16w16
    (candidate D) instead of in the K loop of the sum of squares.
    """
    cfg = dict(DEFAULT_PRE_CONFIG, **(config or {}))
    m, n, c = residual.shape
    k = n * c
    mixes = 2 * n + n * n
    s = num_k_splits(k) if k_splits is None else k_splits
    x = residual.reshape(m, k)
    if m == 0:
        return (
            residual.new_empty((0, n, 1), dtype=torch.float32),
            residual.new_empty((0, n, n), dtype=torch.float32),
            residual.new_empty((0, c)),
        )
    if projection == "fused":
        acc, sq = _prenorm(x, fn, s, cfg)
        acc_splits = s
    elif projection == "aiter_linear":
        acc = _linear_op()(x, fn.to(x.dtype), out_dtype=torch.float32)[None]
        _, sq = _prenorm(x, fn, s, cfg, with_dot=False)
        acc_splits = 1
    else:
        raise ValueError(f"unknown projection {projection!r}")

    mix = torch.empty(m, n + n * n, dtype=torch.float32, device=x.device)
    layer_input = torch.empty(m, c, dtype=residual.dtype, device=x.device)
    block_m, block_c = cfg["APPLY_BLOCK_M"], cfg["APPLY_BLOCK_C"]
    _pre_apply_kernel[(triton.cdiv(m, block_m), triton.cdiv(c, block_c))](
        acc,
        sq,
        scale,
        base,
        x,
        mix,
        layer_input,
        m,
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        sq.stride(0),
        sq.stride(1),
        x.stride(0),
        x.stride(1),
        mix.stride(0),
        mix.stride(1),
        layer_input.stride(0),
        layer_input.stride(1),
        K=k,
        C=c,
        NS=n,
        ACC_SPLITS=acc_splits,
        SQ_SPLITS=s,
        MIX_TILE=triton.next_power_of_2(mixes - n),
        eps=rms_eps,
        pre_eps=hc_pre_eps,
        post_mult=hc_post_mult_value,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        num_warps=cfg["apply_num_warps"],
    )
    comb = torch.empty(m, n, n, dtype=torch.float32, device=x.device)
    logits = mix[:, n:]
    sink_m = cfg["SINKHORN_BLOCK_M"]
    _sinkhorn_kernel[(triton.cdiv(m, sink_m),)](
        logits,
        comb,
        m,
        logits.stride(0),
        logits.stride(1),
        comb.stride(0),
        comb.stride(2),
        NS=n,
        LOG2_NS=_log2(n),
        ITERS=sinkhorn_repeat,
        eps=hc_sinkhorn_eps,
        BLOCK_M=sink_m,
        num_warps=1,
    )
    return mix[:, :n].unsqueeze(-1), comb, layer_input


def fixed_order_mhc_head(
    residual, fn, scale, base, *, rms_eps=1e-6, hc_pre_eps=1e-6, config=None, k_splits=None
):
    """mHC head: residual [M, n, C] bf16, fn [n, n*C] fp32, scale [1], base [n] -> [M, C]."""
    cfg = dict(DEFAULT_HEAD_CONFIG, **(config or {}))
    m, n, c = residual.shape
    k = n * c
    s = num_k_splits(k) if k_splits is None else k_splits
    x = residual.reshape(m, k)
    out = torch.empty(m, c, dtype=residual.dtype, device=residual.device)
    if m == 0:
        return out
    acc, sq = _prenorm(x, fn, s, cfg)
    block_m, block_c = cfg["APPLY_BLOCK_M"], cfg["APPLY_BLOCK_C"]
    _head_apply_kernel[(triton.cdiv(m, block_m), triton.cdiv(c, block_c))](
        acc,
        sq,
        scale,
        base,
        x,
        out,
        m,
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        sq.stride(0),
        sq.stride(1),
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        K=k,
        C=c,
        NS=n,
        ACC_SPLITS=s,
        SQ_SPLITS=s,
        eps=rms_eps,
        pre_eps=hc_pre_eps,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        num_warps=cfg["apply_num_warps"],
    )
    return out
