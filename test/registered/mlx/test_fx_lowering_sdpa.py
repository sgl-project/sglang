"""SDPA semantic parity through the op lowering and the exported executor."""

import importlib.util

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.mlx.fx_lowering import (
    MlxFxLoweringRegistry,
    _lower_mlx_node,
    build_mlx_fx_plan,
    make_mlx_fx_executor,
)
from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=10, suite="stage-a-unit-test-mlx")

pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and importlib.util.find_spec("mlx")),
    reason="requires MLX on Apple Silicon",
)


def _inputs(query_len=3, key_len=3, heads=(3, 3, 3), dtype=torch.float32):
    generator = torch.Generator().manual_seed(42)
    return tuple(
        torch.randn(2, count, length, 8, generator=generator).to(dtype)
        for count, length in zip(heads, (query_len, key_len, key_len))
    )


def _to_mlx(tensor):
    import mlx.core as mx

    dtype = {
        torch.float32: mx.float32,
        torch.float16: mx.float16,
        torch.bfloat16: mx.bfloat16,
        torch.bool: mx.bool_,
    }[tensor.dtype]
    return mx.array(tensor.float().numpy(), dtype=dtype)


def _check_lowering(inputs, **kwargs):
    import mlx.core as mx

    expected = F.scaled_dot_product_attention(*inputs, **kwargs)
    mlx_kwargs = {
        key: _to_mlx(value) if isinstance(value, torch.Tensor) else value
        for key, value in kwargs.items()
    }
    mlx_inputs = tuple(map(_to_mlx, inputs))
    actual = _lower_mlx_node("sdpa", mlx_inputs, mlx_kwargs)
    mx.eval(actual)
    assert actual.dtype == mlx_inputs[0].dtype
    actual = torch.from_numpy(np.array(actual.astype(mx.float32)))
    torch.testing.assert_close(actual, expected.float(), atol=5e-3, rtol=5e-3)
    return actual


@pytest.mark.parametrize("mask_kind", ["boolean", "additive"])
@pytest.mark.parametrize("layout", ["shared", "batch", "head"])
def test_sdpa_mask_broadcast_and_fully_masked_rows(mask_kind, layout):
    inputs = _inputs()
    mask = torch.tensor(
        [[False, False, False], [True, False, True], [False, True, True]]
    )
    if mask_kind == "additive":
        mask = torch.where(mask, -0.75, -float("inf"))
    if layout == "batch":
        mask = mask.expand(2, 1, 3, 3)
    elif layout == "head":
        mask = mask.expand(1, 3, 3, 3)
    actual = _check_lowering(inputs, attn_mask=mask)
    assert torch.count_nonzero(actual[..., 0, :]) == 0


@pytest.mark.parametrize("query_len,key_len", [(1, 4), (2, 5), (5, 2), (3, 3)])
def test_sdpa_rectangular_causal_is_upper_left(query_len, key_len):
    query = torch.zeros(1, 1, query_len, 4)
    key = torch.zeros(1, 1, key_len, 4)
    value = torch.arange(1, key_len + 1, dtype=torch.float32)
    value = value.reshape(1, 1, key_len, 1).expand(1, 1, key_len, 4)
    _check_lowering((query, key, value), is_causal=True)


@pytest.mark.parametrize(
    "query_len,key_len,mask_kind,causal",
    [
        (3, 3, "boolean", False),
        (3, 3, "additive", False),
        (2, 5, None, True),
        (5, 2, None, True),
    ],
)
def test_sdpa_exported_executor_matches_torch(mask_kind, query_len, key_len, causal):
    class Attention(torch.nn.Module):
        def forward(self, query, key, value, mask):
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, is_causal=causal
            )

    inputs = _inputs(query_len, key_len)
    mask = None
    if mask_kind is not None:
        mask = torch.ones(query_len, key_len, dtype=torch.bool)
        mask[:, -1] = False
        mask[1] = False
        if mask_kind == "additive":
            mask = torch.where(mask, 0.5, -float("inf"))
    expected = Attention()(*inputs, mask)
    args = tuple(
        t.to("mps") if isinstance(t, torch.Tensor) else t for t in (*inputs, mask)
    )
    graph = torch.export.export(Attention(), args, strict=False).module()
    # Match the serving consumer's removal of the side-effect-only export guard.
    for node in tuple(graph.graph.nodes):
        if node.op == "call_module" and str(node.target) == "_guards_fn":
            assert not node.users
            graph.graph.erase_node(node)
    graph.recompile()
    plan = build_mlx_fx_plan(graph, MlxFxLoweringRegistry.standard_export_decoder())
    plan.require_fully_supported()
    executor = make_mlx_fx_executor(plan, list(args))
    (actual,) = executor(*args)
    torch.testing.assert_close(actual.cpu(), expected, atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
