import importlib
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
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _make_mhc_model():
    config = SimpleNamespace(
        mhc=True,
        hc_mult=4,
        hidden_size=4096,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
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
    """A prefill must not compile a new mHC prenorm kernel after the load hook."""
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
                *_, fused = layer.hc_attn_pre(
                    residual.flatten(1),
                    layer.input_layernorm.weight,
                    layer.input_layernorm.variance_epsilon,
                )
                assert fused
            assert compile_kernel.call_count == 0, "A warmed prefill compiled a kernel"
    finally:
        reset_context()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
