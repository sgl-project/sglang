"""LTX-2's convrot_int8 shared-input Q/K/V(/gate) and GELU-input FFN paths must
be bitwise identical to calling every projection on its own."""

from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sglang.multimodal_gen.runtime.models.dits.ltx_2 as ltx_2
from sglang.kernels.ops.gemm.convrot_int8 import convrot_int8_supported_capabilities
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.linear import LinearBase
from sglang.multimodal_gen.runtime.layers.quantization.configs.convrot_int8_config import (
    ConvRotInt8Config,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in convrot_int8_supported_capabilities()
    ),
    reason="needs a GPU in the convrot_int8 capability table",
)

DIM = 512
HEADS = 8
HEAD_DIM = 64


class _EagerAttention(nn.Module):
    def forward(self, q, k, v, attn_mask=None, **kwargs):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        return out.transpose(1, 2)


def _init_tp1():
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _quantize(module: nn.Module, seed: int) -> nn.Module:
    module = module.cuda().to(torch.bfloat16)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, device="cuda", generator=gen) * 0.05
            )
    for sub in module.modules():
        if isinstance(sub, LinearBase):
            sub.quant_method.process_weights_after_loading(sub)
    return module


def _attention(context_dim: int | None, gated: bool) -> ltx_2.LTX2Attention:
    _init_tp1()
    with ExitStack() as stack:
        for name in ("USPAttention", "LocalAttention"):
            stack.enter_context(
                patch.object(ltx_2, name, side_effect=lambda **_: _EagerAttention())
            )
        attn = ltx_2.LTX2Attention(
            query_dim=DIM,
            context_dim=context_dim,
            heads=HEADS,
            dim_head=HEAD_DIM,
            apply_gated_attention=gated,
            quant_config=ConvRotInt8Config(backend="jit"),
        )
    return _quantize(attn, seed=0)


def _unfused(fn):
    """Run ``fn`` with both LTX-2 eligibility checks forced off."""
    with (
        patch.object(ltx_2, "convrot_int8_shares_input", return_value=False),
        patch.object(ltx_2, "convrot_int8_fuses_gelu_input", return_value=False),
    ):
        return fn()


def _count_shared_calls(fn):
    spy = patch.object(
        ltx_2,
        "apply_convrot_int8_shared_input",
        wraps=ltx_2.apply_convrot_int8_shared_input,
    )
    with spy as mock:
        out = fn()
    return out, [len(call.kwargs["layers"]) for call in mock.call_args_list]


@pytest.mark.parametrize("gated", [False, True])
def test_self_attention_projects_qkv_gate_from_one_quantized_input(gated):
    attn = _attention(context_dim=None, gated=gated)
    x = torch.randn(1, 256, DIM, device="cuda", dtype=torch.bfloat16)

    fused, calls = _count_shared_calls(lambda: attn(x))
    assert calls == [4 if gated else 3]
    assert torch.equal(fused, _unfused(lambda: attn(x)))


@pytest.mark.parametrize("gated", [False, True])
def test_cross_attention_shares_kv_and_q_gate_inputs(gated):
    attn = _attention(context_dim=DIM, gated=gated)
    x = torch.randn(1, 256, DIM, device="cuda", dtype=torch.bfloat16)
    context = torch.randn(1, 64, DIM, device="cuda", dtype=torch.bfloat16)

    fused, calls = _count_shared_calls(lambda: attn(x, context=context))
    assert calls == [2, 2 if gated else 1]
    assert torch.equal(fused, _unfused(lambda: attn(x, context=context)))

    # The cached prompt K/V path projects the context once, then only Q (and gate).
    context_kv = attn.project_context_kv(context)
    for ours, theirs in zip(
        context_kv, _unfused(lambda: attn.project_context_kv(context)), strict=True
    ):
        assert torch.equal(ours, theirs)
    cached = attn(x, context=context, context_kv=context_kv)
    assert torch.equal(cached, fused)


def test_feed_forward_folds_gelu_into_the_down_projection():
    _init_tp1()
    ff = _quantize(
        ltx_2.LTX2FeedForward(DIM, quant_config=ConvRotInt8Config(backend="jit")),
        seed=1,
    )
    x = torch.randn(1, 256, DIM, device="cuda", dtype=torch.bfloat16)

    with patch.object(
        ltx_2,
        "apply_convrot_int8_gelu_input",
        wraps=ltx_2.apply_convrot_int8_gelu_input,
    ) as gelu_input:
        fused = ff(x)
    assert gelu_input.call_count == 1
    assert torch.equal(fused, _unfused(lambda: ff(x)))
