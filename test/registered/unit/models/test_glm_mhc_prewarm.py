import importlib
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.layernorm import mhc
from sglang.srt.environ import envs
from sglang.srt.layers.utils.common import PPMissingLayer
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    maybe_precompile_model_kernels_after_loading,
)
from sglang.srt.models.glm5_next import (
    Glm5NextDecoderLayer,
    Glm5NextForConditionalGeneration,
)
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.offloader import OffloaderV1
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _make_mhc_model(sinkhorn_iters=20):
    config = SimpleNamespace(
        mhc=True,
        hc_mult=4,
        hidden_size=4096,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=sinkhorn_iters,
    )
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    torch.manual_seed(42)
    layer.hc_attn_fn = torch.nn.Parameter(torch.randn(24, 16384, device="cuda") * 0.001)
    layer.hc_attn_scale = torch.nn.Parameter(
        torch.tensor([0.5, 0.25, 0.25], device="cuda")
    )
    layer.hc_attn_base = torch.nn.Parameter(torch.zeros(24, device="cuda"))
    layer.input_layernorm = torch.nn.Module()
    layer.input_layernorm.weight = torch.nn.Parameter(
        torch.rand(4096, dtype=torch.bfloat16, device="cuda") + 0.5
    )
    layer.input_layernorm.variance_epsilon = 1e-6
    model = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = config
    model.model = SimpleNamespace(layers=[PPMissingLayer(), layer])
    return model, layer


def test_glm_mhc_prewarm_covers_forward_kernels(monkeypatch):
    """A prefill must not compile a new mHC kernel after the load hook warmed it."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for mHC prewarm")

    publish(
        ServerArgs(model_path="dummy", device="cuda", chunked_prefill_size=8192),
        role="test",
    )
    monkeypatch.setattr(mhc, "use_symmetric_memory", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(mhc, "is_allocation_symmetric", lambda: False)
    model, layer = _make_mhc_model()
    jit = importlib.import_module("tilelang.jit")

    try:
        with (
            get_parallel().override(tp_group=None),
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(True),
            patch.object(jit, "compile", wraps=jit.compile) as compile_kernel,
            torch.inference_mode(),
        ):
            maybe_precompile_model_kernels_after_loading(model, "cuda")
            compile_kernel.reset_mock()
            for tokens in (1, 17, 2048, 2049, 4097, 8192):
                residual = torch.zeros(
                    tokens, 4, 4096, dtype=torch.bfloat16, device="cuda"
                )
                if tokens <= 2049:
                    residual.normal_(std=0.1)
                output, comb, post, fused = layer.hc_attn_pre(
                    residual.flatten(1),
                    layer.input_layernorm.weight,
                    layer.input_layernorm.variance_epsilon,
                )
                assert fused
                if tokens <= 2049:
                    ref_post, ref_comb, ref_output = mhc._mhc_pre_torch(
                        residual,
                        layer.hc_attn_fn,
                        layer.hc_attn_scale,
                        layer.hc_attn_base,
                        1e-6,
                        1e-6,
                        1e-6,
                        2.0,
                        20,
                    )
                    ref_output = torch.nn.functional.rms_norm(
                        ref_output,
                        (4096,),
                        layer.input_layernorm.weight,
                        eps=1e-6,
                    )
                    torch.testing.assert_close(
                        output, ref_output, atol=0.025, rtol=0.025
                    )
                    torch.testing.assert_close(
                        comb, ref_comb.flatten(1), atol=0.001, rtol=0.001
                    )
                    torch.testing.assert_close(
                        post, ref_post.flatten(1), atol=0.001, rtol=0.001
                    )
                    del ref_output, ref_comb, ref_post
                else:
                    assert not output.count_nonzero().item()
                    torch.testing.assert_close(comb, torch.full_like(comb, 0.25))
                    torch.testing.assert_close(post, torch.ones_like(post))
                del residual, output, comb, post
            assert compile_kernel.call_count == 0, "A warmed prefill compiled a kernel"
    finally:
        reset_context()


@pytest.mark.parametrize("offload_all", [True, False])
def test_glm_mhc_prewarm_preserves_offloaded_weights(monkeypatch, offload_all):
    """The load hook bypasses OffloaderV1's forward-time onload wrapper."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for mHC prewarm")

    publish(
        ServerArgs(model_path="dummy", device="cuda", chunked_prefill_size=256),
        role="test",
    )
    monkeypatch.setattr(mhc, "use_symmetric_memory", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(mhc, "is_allocation_symmetric", lambda: False)
    model, layer = _make_mhc_model(sinkhorn_iters=8 if offload_all else 9)
    layer.forward = lambda x: layer.hc_attn_pre(
        x, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon
    )
    budget = 1 << 24 if offload_all else layer.hc_attn_fn.numel() * 4
    OffloaderV1(budget).maybe_offload_to_cpu(layer)
    assert layer.hc_attn_fn.device.type == "cpu"
    assert layer.input_layernorm.weight.is_cuda != offload_all
    original = [
        (p, p.device, p.data_ptr(), p.detach().clone()) for p in layer.parameters()
    ]
    copies = []
    tensor_to = torch.Tensor.to

    def track_staging(tensor, *args, **kwargs):
        result = tensor_to(tensor, *args, **kwargs)
        if tensor.device.type == "cpu" and result.is_cuda:
            assert result.device.index == torch.cuda.current_device()
            copies.append(weakref.ref(result))
        return result

    jit = importlib.import_module("tilelang.jit")
    try:
        with (
            get_parallel().override(tp_group=None),
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(True),
            patch.object(jit, "compile", wraps=jit.compile) as compile_kernel,
            torch.inference_mode(),
        ):
            with patch.object(torch.Tensor, "to", track_staging):
                maybe_precompile_model_kernels_after_loading(model, "cuda")
            assert copies and all(ref() is None for ref in copies)
            compile_kernel.reset_mock()
            residual = (
                torch.randn(17, 4, 4096, dtype=torch.bfloat16, device="cuda") * 0.1
            )
            output, comb, post, fused = layer(residual.flatten(1))
            assert fused and output.is_cuda
            ref_post, ref_comb, ref_output = mhc._mhc_pre_torch(
                residual.cpu(),
                layer.hc_attn_fn.cpu(),
                layer.hc_attn_scale.cpu(),
                layer.hc_attn_base.cpu(),
                1e-6,
                1e-6,
                1e-6,
                2.0,
                model.config.hc_sinkhorn_iters,
            )
            ref_output = torch.nn.functional.rms_norm(
                ref_output, (4096,), layer.input_layernorm.weight.cpu(), eps=1e-6
            )
            torch.testing.assert_close(output.cpu(), ref_output, atol=0.025, rtol=0.025)
            torch.testing.assert_close(
                comb.cpu(), ref_comb.flatten(1), atol=0.001, rtol=0.001
            )
            torch.testing.assert_close(
                post.cpu(), ref_post.flatten(1), atol=0.001, rtol=0.001
            )
            assert compile_kernel.call_count == 0, (
                "An offloaded prefill compiled a kernel"
            )
            for parameter, device, pointer, value in original:
                assert parameter.device == device and parameter.data_ptr() == pointer
                torch.testing.assert_close(parameter, value)
    finally:
        reset_context()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
