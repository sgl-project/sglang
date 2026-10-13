import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import (
    deepgemm_w8a8_block_fp8_linear_with_fallback,
    transform_scale_ue8m0,
    triton_w8a8_block_fp8_linear,
)
from sglang.srt.model_executor import model_runner
from sglang.srt.model_executor.model_runner import (
    _can_replicate_block_fp8,
)
from sglang.srt.models import deepseek_v2
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class _ShardGroup:
    world_size = 2

    def __init__(self, shards, scales, w_kc=None):
        self.values = [torch.cat(shards).view(torch.uint8), torch.cat(scales)]
        if w_kc is not None:
            self.values.append(torch.cat(w_kc))

    def all_gather(self, value, dim):
        result = self.values.pop(0)
        assert dim == 0 and result.dtype == value.dtype
        return result


def _make_shards(packed):
    method = Fp8LinearMethod(
        Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    )
    method.w8a8_block_fp8_linear = (
        deepgemm_w8a8_block_fp8_linear_with_fallback
        if packed
        else triton_w8a8_block_fp8_linear
    )
    shards, scales = [], []
    for _ in range(2):
        layer = torch.nn.Module()
        layer.quant_method = method
        layer.orig_dtype = torch.bfloat16
        layer.weight = torch.nn.Parameter(
            torch.randn(384, 640, device="cuda").to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        # Five input blocks exercise padding in the packed scale layout.
        scale = torch.pow(2.0, torch.randint(-5, -1, (3, 5), device="cuda")).float()
        scales.append(scale)
        layer.weight_scale_inv = torch.nn.Parameter(
            transform_scale_ue8m0(scale, mn=384) if packed else scale,
            requires_grad=False,
        )
        layer.weight_scale_inv.format_ue8m0 = packed
        method.process_weights_after_loading(layer)
        shards.append(layer)
    return shards, scales


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("tokens", [1, 17, 64])
def test_replicated_fp8_projection_matches_sharded_projection(packed, tokens):
    if packed:
        pytest.importorskip("deep_gemm")
        if torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("packed UE8M0 GEMM requires SM100+")
    torch.manual_seed(0)
    shards, scales = _make_shards(packed)
    w_kc = [
        torch.randn(2, 128, 512, device="cuda", dtype=torch.bfloat16) for _ in shards
    ]
    group = _ShardGroup([s.weight for s in shards], scales, w_kc)
    assert _can_replicate_block_fp8(shards[0])
    attn = deepseek_v2.DeepseekV2AttentionMLA.__new__(
        deepseek_v2.DeepseekV2AttentionMLA
    )
    torch.nn.Module.__init__(attn)
    attn.has_q_b_proj = True
    attn.q_b_proj = shards[0]
    attn.w_kc = w_kc[0]
    attn.w_kc_qrep = attn.q_b_proj_qrep = attn.q_b_proj_qrep_weight = None
    attn.num_local_heads, attn.qk_head_dim = 2, 192
    with patch.object(
        model_runner, "get_parallel", return_value=SimpleNamespace(dcp_group=group)
    ):
        model_runner.ModelRunner._prepare_replicated_q_proj(SimpleNamespace(model=attn))
    replica = attn.q_b_proj_qrep
    assert not group.values
    torch.testing.assert_close(attn.w_kc_qrep, torch.cat(w_kc))
    assert torch.equal(
        replica.weight.view(torch.uint8),
        torch.cat([s.weight for s in shards]).view(torch.uint8),
    )
    assert replica.weight_scale_inv.format_ue8m0 == packed
    x = torch.randn(tokens, 640, device="cuda", dtype=torch.bfloat16)
    expected = torch.cat([s.quant_method.apply(s, x) for s in shards], dim=-1)

    def project():
        return deepseek_v2.DeepseekV2AttentionMLA.q_b_proj_replicated_forward(attn, x)

    with patch.object(
        deepseek_v2, "get_parallel", return_value=SimpleNamespace(attn_dcp_size=2)
    ):
        actual = project()
        torch.testing.assert_close(actual.flatten(1), expected, rtol=0.01, atol=0.01)
        # Warm up compilation before capture and verify replay on new inputs.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                project()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = project()
        x.add_(0.25)
        graph.replay()
        expected = torch.cat([s.quant_method.apply(s, x) for s in shards], dim=-1)
        torch.testing.assert_close(captured.flatten(1), expected, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("attribute", ["use_mxfp8", "use_marlin", "block_fp8_as_mxfp8"])
def test_unsupported_fp8_layouts_keep_runtime_q_gather(attribute):
    shards, _ = _make_shards(packed=False)
    layer = shards[0]
    setattr(layer.quant_method, attribute, True)
    assert not _can_replicate_block_fp8(layer)
    # Startup must not gather w_kc unless a supported Q replica was created.
    attn = deepseek_v2.DeepseekV2AttentionMLA.__new__(
        deepseek_v2.DeepseekV2AttentionMLA
    )
    torch.nn.Module.__init__(attn)
    attn.has_q_b_proj = True
    attn.q_b_proj = layer
    attn.w_kc = torch.empty(2, 128, 512, device="cuda", dtype=torch.bfloat16)
    attn.w_kc_qrep = attn.q_b_proj_qrep = attn.q_b_proj_qrep_weight = None
    group = SimpleNamespace(world_size=2, all_gather=Mock())
    with patch.object(
        model_runner, "get_parallel", return_value=SimpleNamespace(dcp_group=group)
    ):
        model_runner.ModelRunner._prepare_replicated_q_proj(SimpleNamespace(model=attn))
    assert attn.w_kc_qrep is None
    assert attn.q_b_proj_qrep is None
    group.all_gather.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_replicated_unquantized_projection(dtype):
    x = torch.randn(17, 640, device="cuda", dtype=dtype)
    weight = torch.randn(768, 640, device="cuda", dtype=dtype)
    attn = SimpleNamespace(
        q_b_proj_qrep=None,
        q_b_proj_qrep_weight=weight,
        num_local_heads=2,
        qk_head_dim=192,
    )
    with patch.object(
        deepseek_v2, "get_parallel", return_value=SimpleNamespace(attn_dcp_size=2)
    ):
        actual = deepseek_v2.DeepseekV2AttentionMLA.q_b_proj_replicated_forward(attn, x)
    torch.testing.assert_close(actual.flatten(1), torch.nn.functional.linear(x, weight))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
