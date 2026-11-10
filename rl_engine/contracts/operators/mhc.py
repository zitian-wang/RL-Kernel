# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DeepSeek-V4 mHC contract: mHC pre, mHC post and mHC head (DSV4 RFC tasks 1-4, 8).

Records the arithmetic of the ROCm backend, RL-Kernel's Triton kernels in
rl_engine/backends/rocm/mhc/triton_mhc.py (copied from aiter ``326cffa``,
``mhc_pre_dsv4``, ``mhc_post_dsv4``, ``mhc_head_dsv4``, and made batch invariant
by construction), as measured in tests/backends/rocm/test_rocm_mhc_*.py, and
where the fp32 PyTorch reference (rl_engine/reference/mhc/mhc.py,
DeepSeek-V4-Flash-0731 ``inference/model.py``) differs. ``n`` residual streams
of hidden size ``C`` per token; leading dimensions are flattened into the token
count M; K = n * C.

mHC pre, per token, x = FP32(flatten(residual)) [K]::

    K is cut into NUM_KSPLIT = 4 segments; per segment s:
      sq[s]  = pairwise sum over 128 columns of the element-wise fma(x, x, .)
               accumulated over the K tiles of 128 of the segment
      acc[s] = x[s] @ fn[:, s]^T                 FP32 x FP32 MFMA (ieee)
    r      = 1 / sqrt((((sq[0] + sq[1]) + sq[2]) + sq[3]) / K + rms_eps)
    acc    = ((acc[0] + acc[1]) + acc[2]) + acc[3]
    pre    = sigmoid(r * scale[0] * acc[:n] + base[:n]) + hc_pre_eps
    post   = hc_post_mult_value * sigmoid(r * scale[1] * acc[n:2n] + base[n:2n])
    comb   = sinkhorn(r * scale[2] * acc[2n:] + base[2n:], sinkhorn_repeat)  [n, n]
    layer_input = BF16(fma(pre3, x3, fma(pre2, x2, fma(pre1, x1, pre0 * x0))))

sinkhorn is DeepSeek-V4's asymmetric form: the first round takes a row softmax,
adds hc_sinkhorn_eps, then divides each column by (column sum + eps); each
further round divides rows, then columns, by (sum + eps). Row and column sums
are pairwise: (p0 + p1) + (p2 + p3).

mHC post, per token and destination stream j::

    out[j] = BF16(post[j] * y + comb[0, j] * residual[0] + ... + comb[n-1, j] * residual[n-1])

added in source-stream order in FP32. ``comb`` is indexed [source stream,
destination stream]; the old residual enters once, through ``comb``.

mHC head, per token, with ``fn`` [n, K] and ``scale`` [1]: the K segments, the
sum of squares and the projection as in mHC pre, then::

    r   = rsqrt(sum of squares / K + rms_eps)
    w   = sigmoid(acc * r * scale + base) + hc_pre_eps
    out = BF16(fma(w3, x3, fma(w2, x2, fma(w1, x1, w0 * x0))))
"""

from __future__ import annotations

import copy

import torch

PROFILE = "dsv4-mhc-rocm-triton-v1"
# The aiter commit the ROCm kernels were copied from.
AITER_COMMIT = "326cffa"

HC_MULT = 4
RESIDUAL_DTYPE = torch.bfloat16
PARAM_DTYPE = torch.float32
MIX_DTYPE = torch.float32
OUTPUT_DTYPE = torch.bfloat16

RMS_EPS = 1.0e-6
HC_PRE_EPS = 1.0e-6
HC_SINKHORN_EPS = 1.0e-6
HC_POST_MULT_VALUE = 2.0
SINKHORN_REPEAT = 20

RMS_FORM = "rsqrt(mean(x^2) + rms_eps) over K = n * C, fp32"
PROJECTION = "fp32 fn; fp32 x fp32 MFMA (ieee), 4 K segments added in segment order"
REFERENCE_PROJECTION = "fp32 x fp32"
SUM_OF_SQUARES = (
    "per K segment: element-wise fma over K tiles of 128, then pairwise over the 128 "
    "columns; 4 K segments added in segment order"
)
SINKHORN = "dsv4_asymmetric, pairwise row and column sums"
STREAM_SUM = "fma(w3, x3, fma(w2, x2, fma(w1, x1, w0 * x0))) in fp32, one cast to bf16"

# The constants that fix the reduction order, equal to those of
# rl_engine/backends/rocm/mhc/triton_mhc.py. No launch parameter changes the bits.
ARITHMETIC_CONSTANTS = {"NUM_KSPLIT": 4, "BLOCK_K": 128, "matrix_instr_nonkdim": 16, "kpack": 1}

BATCH_INVARIANCE = {
    "forward": True,
    "backward": False,
    "validated_on": "gfx942",
    "validated_shape": "DeepSeek-V4-Flash n=4, C=4096",
    "launch_configs": "one config per op for every M; the bits do not depend on it",
    "backward_implementation": "PyTorch recompute (as aiter), fp32 fn",
}

OPEN_ITEMS = (
    "RMS form: the RFC states 1 / (sqrt(mean(x^2)) + eps); aiter and the DeepSeek-V4 "
    "reference use rsqrt(mean(x^2) + eps).",
)

_SHAPES = {
    "mhc_pre": {
        "residual": "[..., n, C] bf16",
        "fn": "[2n + n^2, n*C] fp32",
        "scale": "[3] fp32",
        "base": "[2n + n^2] fp32",
        "post_mix": "[..., n] fp32",
        "comb_mix": "[..., n, n] fp32, [source, destination]",
        "layer_input": "[..., C] bf16",
    },
    "mhc_post": {
        "y": "[..., C] bf16",
        "residual": "[..., n, C] bf16",
        "post_mix": "[..., n] or [..., n, 1] fp32",
        "comb_mix": "[..., n, n] fp32, [source, destination]",
        "out": "[..., n, C] bf16",
    },
    "mhc_head": {
        "residual": "[..., n, C] bf16",
        "fn": "[n, n*C] fp32",
        "scale": "[1] fp32",
        "base": "[n] fp32",
        "out": "[..., C] bf16",
    },
}

_ARITHMETIC = {
    "mhc_pre": {
        "rms_form": RMS_FORM,
        "sum_of_squares": SUM_OF_SQUARES,
        "projection": PROJECTION,
        "reference_projection": REFERENCE_PROJECTION,
        "pre_weights": "sigmoid(.) + hc_pre_eps",
        "post_weights": "hc_post_mult_value * sigmoid(.)",
        "sinkhorn": SINKHORN,
        "stream_sum": STREAM_SUM,
        "downcast_points": ["layer_input"],
        "arithmetic_constants": dict(ARITHMETIC_CONSTANTS),
    },
    "mhc_post": {
        "formula": "out[j] = post[j] * y + sum_h comb[h, j] * residual[h]",
        "accumulation": "fp32, source streams in order",
        "residual_terms": "once, through comb",
        "downcast_points": ["out"],
    },
    "mhc_head": {
        "rms_form": RMS_FORM,
        "sum_of_squares": SUM_OF_SQUARES,
        "projection": PROJECTION,
        "reference_projection": REFERENCE_PROJECTION,
        "weights": "sigmoid(.) + hc_pre_eps",
        "stream_sum": STREAM_SUM,
        "downcast_points": ["out"],
        "arithmetic_constants": dict(ARITHMETIC_CONSTANTS),
    },
}

OPS = tuple(_SHAPES)


class MHCContractError(ValueError):
    """Raised for an mHC operator name outside this contract."""


def pre_fn_rows(n: int = HC_MULT) -> int:
    """Rows of mHC pre's ``fn`` and ``base``: n pre, n post and n * n comb logits."""
    return 2 * n + n * n


def describe(op: str) -> dict[str, object]:
    """Provenance fields of one mHC operator: ``"mhc_pre"``, ``"mhc_post"`` or ``"mhc_head"``."""
    if op not in _SHAPES:
        raise MHCContractError(f"unknown mHC operator {op!r}; want one of {OPS}")
    return {
        "profile": PROFILE,
        "op": op,
        "aiter_commit": AITER_COMMIT,
        "hc_mult": HC_MULT,
        "defaults": {
            "rms_eps": RMS_EPS,
            "hc_pre_eps": HC_PRE_EPS,
            "hc_sinkhorn_eps": HC_SINKHORN_EPS,
            "hc_post_mult_value": HC_POST_MULT_VALUE,
            "sinkhorn_repeat": SINKHORN_REPEAT,
        },
        "shapes": dict(_SHAPES[op]),
        "arithmetic": copy.deepcopy(_ARITHMETIC[op]),
        "batch_invariance": dict(BATCH_INVARIANCE),
        "open_items": list(OPEN_ITEMS),
    }


__all__ = [
    "AITER_COMMIT",
    "ARITHMETIC_CONSTANTS",
    "BATCH_INVARIANCE",
    "HC_MULT",
    "HC_POST_MULT_VALUE",
    "HC_PRE_EPS",
    "HC_SINKHORN_EPS",
    "MIX_DTYPE",
    "MHCContractError",
    "OPEN_ITEMS",
    "OPS",
    "OUTPUT_DTYPE",
    "PARAM_DTYPE",
    "PROFILE",
    "RESIDUAL_DTYPE",
    "RMS_EPS",
    "SINKHORN_REPEAT",
    "describe",
    "pre_fn_rows",
]
