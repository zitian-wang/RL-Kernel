# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch invariance of the ROCm linear: aiter's gemm_a16w16 kernel copied into RL-Kernel.

Main check: y = x @ w^T + bias must give a token the same bits whatever other
tokens share the batch. Shapes:

- the narrow DeepSeek-V4 mHC projections (N = 24 and 4, K = 16384), with bf16
  and fp32 output, with and without bias;
- the WS1 manifest's GEMM shapes (K = 4096, N = 4096 and N = 12288);
- a shape whose N and K are not multiples of the tile size;
- the DeepSeek-V4-Flash-0731 lm_head (N = 129280, K = 4096, fp32 output).

The op picks its launch config from ``triton_gemm.CONFIGS`` by M, so a row
alone and the same row in a batch run different configs; the check covers that
they give the same bits. M covers the WS1 manifest's token counts (8, 59), the
largest M of every CONFIGS entry and the M after it, and powers of two up to
16384. Chunking, reordering, padding, other input strides and repeated runs are
checked too, and every config of CONFIGS directly.

Op-specific checks: the result and the input gradient match an fp32 PyTorch
linear; the registry and the semantic descriptor; and the shipped table of the
aiter op, kept as the benchmark baseline.
"""

import pytest
import torch

from rl_engine.backends.rocm import config_utils
from rl_engine.backends.rocm.gemm import triton_gemm
from rl_engine.backends.rocm.gemm.aiter_gemm import FIXED_LINEAR_CONFIG, LINEAR_BIT_PARAMS
from rl_engine.backends.rocm.gemm.triton_gemm import check_linear_inputs
from rl_engine.runtime.registry import kernel_registry
from rl_engine.runtime.semantic_registry import OperatorRequirements, OperatorResolutionError
from rl_engine.validation.common.batch_invariance import (
    RowwiseCase,
    bits_equal,
    differing_after_chunk_reorder_pad,
    differing_after_restride,
    differing_between_runs,
    differing_in_batch,
)
from tests.helpers.contract_tolerance import assert_matches_fp32_reference

_LM_HEAD_TABLE = "GEMM-A16W16-N=129280-K=4096.json"
_BUCKET_EDGES = {e for m_max, _ in triton_gemm.CONFIGS if m_max for e in (m_max, m_max + 1)}
_M_VALUES = sorted({1, 8, 16, 17, 59, 64, 256, 1024, 2048, 4096, 8192, 16384} | _BUCKET_EDGES)

# name -> (N, K, output dtype, with bias)
_SHAPES = {
    "mhc_pre_projection": (24, 16384, torch.bfloat16, False),
    "mhc_head_projection": (4, 16384, torch.bfloat16, False),
    "ws1_projection": (4096, 4096, torch.bfloat16, False),
    "ws1_gate_up": (12288, 4096, torch.bfloat16, False),
    "tail": (1000, 4100, torch.bfloat16, False),
    "fp32_output": (24, 16384, torch.float32, False),
    "bias": (24, 16384, torch.bfloat16, True),
    "fp32_output_bias": (24, 16384, torch.float32, True),
    "lm_head": (129280, 4096, torch.float32, False),
}
# lm_head logits are 4.2 GB at M = 8192.
_MAX_M = {"lm_head": 8192}
_BACKEND_ID = "rlkernel.rocm.triton_gemm_a16w16.batch_invariant.v1"


def _op():
    if not (torch.cuda.is_available() and torch.version.hip is not None):
        return None
    return triton_gemm.TritonBatchInvariantLinearOp()


_OP = _op()
requires_rocm = pytest.mark.skipif(_OP is None, reason="ROCm GPU required.")


def _case(shape):
    n, k, out_dtype, with_bias = _SHAPES[shape]
    gen = torch.Generator(device="cuda").manual_seed(0)
    weight = (torch.randn(n, k, generator=gen, device="cuda") / k**0.5).to(torch.bfloat16)
    bias = torch.randn(n, generator=gen, device="cuda").to(out_dtype) if with_bias else None
    op = _OP

    def make_rows(m, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return [torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)]

    def run(rows):
        return [op(rows[0], weight, bias, out_dtype=out_dtype)]

    return RowwiseCase(shape, ("y",), run, make_rows), weight, bias


def _shape_m():
    return [
        pytest.param(shape, m, id=f"{shape}-M{m}")
        for shape in _SHAPES
        for m in _M_VALUES
        if m <= _MAX_M.get(shape, m)
    ]


# Main check: a row alone (M=1) versus inside a batch of M.
@requires_rocm
@pytest.mark.parametrize("shape,m", _shape_m())
def test_batch_invariance(shape, m):
    case, _, _ = _case(shape)
    differing = differing_in_batch(case, m)
    assert not differing, f"{shape}, M={m}: differs from M=1"


# Every config of CONFIGS gives the bits of the first one, at M inside one tile and larger.
@requires_rocm
@pytest.mark.parametrize("shape", ["mhc_pre_projection", "tail", "fp32_output_bias", "lm_head"])
def test_every_config_gives_the_same_bits(shape):
    _, weight, bias = _case(shape)
    out_dtype = _SHAPES[shape][2]
    for m in (17, 1024):
        x = torch.randn(m, weight.shape[1], device="cuda").to(torch.bfloat16)
        outputs = [
            triton_gemm.gemm_a16w16(x, weight, bias, dtype=out_dtype, config=config)
            for _, config in triton_gemm.CONFIGS
        ]
        differing = [i for i, y in enumerate(outputs) if not bits_equal(y, outputs[0])]
        assert not differing, f"M={m}: configs {differing} differ from config 0"


@requires_rocm
@pytest.mark.parametrize("shape", list(_SHAPES))
def test_chunk_reorder_pad_restride_and_run_to_run(shape):
    case, _, _ = _case(shape)
    assert not differing_after_chunk_reorder_pad(case, 4096)
    assert not differing_after_restride(case, 1024)
    assert not differing_between_runs(case, 2048)


# Op-specific: the result against an fp32 PyTorch linear.
@requires_rocm
@pytest.mark.parametrize("shape", list(_SHAPES))
def test_matches_fp32_reference(shape):
    case, weight, bias = _case(shape)
    x = case.make_rows(64, 1)[0]
    y = case.run([x])[0]
    reference = x.float() @ weight.float().t()
    if bias is not None:
        reference = reference + bias.float()
    assert y.dtype == _SHAPES[shape][2]
    assert_matches_fp32_reference(y, reference, judgment="forward_accuracy", op_class="reduction")


@requires_rocm
def test_input_grad_matches_fp32_reference():
    case, weight, _ = _case("ws1_projection")
    x = case.make_rows(64, 2)[0].requires_grad_(True)
    grad_out = torch.randn(64, weight.shape[0], device="cuda")
    case.run([x])[0].float().backward(grad_out)
    reference = grad_out @ weight.float()
    assert x.grad.dtype == x.dtype
    assert_matches_fp32_reference(
        x.grad, reference, judgment="gradient_accuracy", op_class="reduction"
    )


@requires_rocm
@pytest.mark.parametrize("with_bias", [False, True])
def test_empty_batch_forward_and_backward(with_bias):
    case, weight, _ = _case("ws1_projection")
    weight = weight.detach().clone().requires_grad_(True)
    bias = torch.zeros(weight.shape[0], device="cuda", requires_grad=True) if with_bias else None
    x = case.make_rows(0, 3)[0].requires_grad_(True)
    y = _OP(x, weight, bias, out_dtype=torch.float32)
    assert y.shape == (0, weight.shape[0]) and y.dtype == torch.float32
    y.sum().backward()
    assert x.grad.shape == x.shape
    assert not weight.grad.any()
    if with_bias:
        assert not bias.grad.any()


@requires_rocm
def test_registry_selects_the_triton_op_on_rocm():
    op = kernel_registry.get_op("batch_invariant_linear", device="cuda")
    assert type(op).__name__ == "TritonBatchInvariantLinearOp"


# The aiter op's table (kept as the benchmark baseline until the aiter path is removed).
def test_shipped_lm_head_table_loads_and_is_used(monkeypatch):
    monkeypatch.setattr(config_utils, "_device_arch", lambda index: "gfx942")
    config_utils.load_table.cache_clear()
    config_utils._bucket_table.cache_clear()
    config_utils._SELECTED.clear()
    try:
        path = config_utils.CONFIG_ROOT / "gfx942" / "gemm_a16w16" / _LM_HEAD_TABLE
        table = config_utils.load_table(path, LINEAR_BIT_PARAMS)
        assert table is not None and "any" in table
        assert {entry["NUM_KSPLIT"] for entry in table.values()} == {1}
        chosen = config_utils.get_config(
            "gemm_a16w16",
            _LM_HEAD_TABLE,
            1,
            FIXED_LINEAR_CONFIG,
            LINEAR_BIT_PARAMS,
            torch.device("cuda", 0),
        )
        assert chosen == table["M_LEQ_1"]
    finally:
        config_utils.load_table.cache_clear()
        config_utils._bucket_table.cache_clear()
        config_utils._SELECTED.clear()


def test_semantic_descriptor_resolves_on_rocm_only():
    session = kernel_registry.semantic.session()
    requirements = OperatorRequirements(
        device="rocm", dtype="bfloat16", alignment_properties={"batch_invariant": True}
    )
    for target in ("rollout", "training"):
        resolution = session.resolve(
            semantic_op="batch_invariant_linear",
            requested_backend=_BACKEND_ID,
            target=target,
            requirements=requirements,
        )
        assert resolution.trace.status == "resolved"
    with pytest.raises(OperatorResolutionError):
        session.resolve(
            semantic_op="batch_invariant_linear",
            requested_backend=_BACKEND_ID,
            target="rollout",
            requirements=OperatorRequirements(device="cuda", dtype="bfloat16"),
        )


# Unsupported inputs raise instead of running unvalidated (checked on CPU).
@pytest.mark.parametrize(
    "change,error",
    [
        ({"x": torch.zeros(2, 64, dtype=torch.float16)}, TypeError),
        ({"weight": torch.zeros(8, 64)}, TypeError),
        ({"out_dtype": torch.float16}, TypeError),
        ({"weight": torch.zeros(8, 32, dtype=torch.bfloat16)}, ValueError),
        ({"bias": torch.zeros(8, dtype=torch.bfloat16)}, ValueError),  # not out_dtype
        ({"bias": torch.zeros(4)}, ValueError),
    ],
    ids=["fp16_x", "fp32_weight", "fp16_out", "k_mismatch", "bias_dtype", "bias_shape"],
)
def test_unsupported_inputs_are_rejected(change, error):
    inputs = {
        "x": torch.zeros(2, 64, dtype=torch.bfloat16),
        "weight": torch.zeros(8, 64, dtype=torch.bfloat16),
        "bias": torch.zeros(8),
        "out_dtype": torch.float32,
    }
    check_linear_inputs(**inputs)
    with pytest.raises(error):
        check_linear_inputs(**dict(inputs, **change))
