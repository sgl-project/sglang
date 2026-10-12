import importlib.util
import sys
import types
from pathlib import Path

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load_megamoe_module(monkeypatch):
    """Load the adapter with only its small import-time dependencies stubbed."""

    class MoeQuantInfo:
        pass

    class MoeRunnerConfig:
        pass

    def register_fused_func(*_args, **_kwargs):
        return lambda fn: fn

    fake_modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.environ": types.ModuleType("sglang.srt.environ"),
        "sglang.srt.layers": types.ModuleType("sglang.srt.layers"),
        "sglang.srt.layers.moe": types.ModuleType("sglang.srt.layers.moe"),
        "sglang.srt.layers.moe.moe_runner": types.ModuleType(
            "sglang.srt.layers.moe.moe_runner"
        ),
        "sglang.srt.layers.moe.moe_runner.base": types.ModuleType(
            "sglang.srt.layers.moe.moe_runner.base"
        ),
        "sglang.srt.layers.moe.token_dispatcher": types.ModuleType(
            "sglang.srt.layers.moe.token_dispatcher"
        ),
        "sglang.srt.runtime_context": types.ModuleType("sglang.srt.runtime_context"),
        "deep_gemm": types.ModuleType("deep_gemm"),
        "deep_gemm.utils": types.ModuleType("deep_gemm.utils"),
        "deep_gemm.utils.math": types.ModuleType("deep_gemm.utils.math"),
    }
    fake_modules["sglang.srt.environ"].envs = types.SimpleNamespace(
        SGLANG_FLASHINFER_MEGAMOE_MAX_TOKENS_PER_RANK=types.SimpleNamespace(
            get=lambda: 0
        ),
        SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE=types.SimpleNamespace(
            get=lambda: "bf16"
        ),
        SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE=types.SimpleNamespace(
            get=lambda: False
        ),
    )
    runtime_context = fake_modules["sglang.srt.runtime_context"]
    runtime_context.cutedsl_moe_max_num_tokens = lambda: 2048
    base = fake_modules["sglang.srt.layers.moe.moe_runner.base"]
    base.MoeQuantInfo = MoeQuantInfo
    base.MoeRunnerConfig = MoeRunnerConfig
    base.register_fused_func = register_fused_func
    token_dispatcher = fake_modules["sglang.srt.layers.moe.token_dispatcher"]

    class StandardCombineInput:
        def __init__(self, *, hidden_states):
            self.hidden_states = hidden_states

    token_dispatcher.StandardCombineInput = StandardCombineInput
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/srt/layers/moe/flashinfer_megamoe.py"
    )
    module_name = "sglang_flashinfer_megamoe_adapter_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_max_tokens_uses_runtime_context_accessor(monkeypatch):
    module = _load_megamoe_module(monkeypatch)

    assert module._resolve_max_tokens_per_rank() == 2048

    runtime_context = sys.modules["sglang.srt.runtime_context"]
    runtime_context.cutedsl_moe_max_num_tokens = lambda: 0
    assert module._resolve_max_tokens_per_rank() == 1024


def test_adapter_keeps_router_ids_int32(monkeypatch):
    module = _load_megamoe_module(monkeypatch)

    class FakeMoEEpTensors:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_moe_ep = types.ModuleType("flashinfer.moe_ep")
    fake_moe_ep.MoEEpTensors = FakeMoEEpTensors
    fake_flashinfer = types.ModuleType("flashinfer")
    fake_flashinfer.moe_ep = fake_moe_ep
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", fake_moe_ep)

    hidden_states = torch.randn((3, 4), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int32)
    topk_weights = torch.randn((3, 2), dtype=torch.float32)
    output = torch.randn_like(hidden_states)

    class Mega:
        _workspace = object()

        def forward(self, tensors):
            self.tensors = tensors
            return output

    mega = Mega()
    dispatch_output = types.SimpleNamespace(
        hidden_states=hidden_states,
        topk_output=types.SimpleNamespace(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        ),
    )
    quant_info = module.FlashInferMegaMoeQuantInfo(mega=mega)
    runner_config = types.SimpleNamespace(routed_scaling_factor=1.0)

    result = module.run_flashinfer_megamoe(
        dispatch_output,
        quant_info,
        runner_config,
    )

    assert mega.tensors.topk_ids.data_ptr() == topk_ids.data_ptr()
    assert mega.tensors.topk_ids.dtype == torch.int32
    assert result.hidden_states is output


def test_adapter_requests_workspace_output_view(monkeypatch):
    module = _load_megamoe_module(monkeypatch)

    class FakeMoEEpTensors:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_moe_ep = types.ModuleType("flashinfer.moe_ep")
    fake_moe_ep.MoEEpTensors = FakeMoEEpTensors
    fake_flashinfer = types.ModuleType("flashinfer")
    fake_flashinfer.moe_ep = fake_moe_ep
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", fake_moe_ep)

    hidden_states = torch.randn((2, 4), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    topk_weights = torch.ones((2, 2), dtype=torch.float32)
    output = torch.randn_like(hidden_states)

    class Mega:
        supports_output_view = True
        _workspace = object()

        def forward(self, tensors, *, return_workspace_view=False):
            self.tensors = tensors
            self.return_workspace_view = return_workspace_view
            return output

    mega = Mega()
    dispatch_output = types.SimpleNamespace(
        hidden_states=hidden_states,
        topk_output=types.SimpleNamespace(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        ),
    )

    result = module.run_flashinfer_megamoe(
        dispatch_output,
        module.FlashInferMegaMoeQuantInfo(mega=mega),
        types.SimpleNamespace(routed_scaling_factor=1.0),
    )

    assert result.hidden_states is output
    assert mega.tensors.topk_ids.data_ptr() == topk_ids.data_ptr()
    assert mega.tensors.topk_ids.dtype == torch.int32
    assert mega.return_workspace_view is True


def test_capture_safe_ue8m0_pack_is_scoped(monkeypatch):
    module = _load_megamoe_module(monkeypatch)

    dgm = sys.modules["deep_gemm.utils.math"]

    def original(value):
        return value

    dgm.pack_ue8m0_to_int = original
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    with module._capture_safe_ue8m0_pack():
        assert dgm.pack_ue8m0_to_int is original

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with module._capture_safe_ue8m0_pack():
        assert dgm.pack_ue8m0_to_int is not original
        packed = dgm.pack_ue8m0_to_int(torch.ones(4, dtype=torch.float32))
        assert packed.dtype == torch.int32

    assert dgm.pack_ue8m0_to_int is original


class TestMegaMoeOutputViewCapability(CustomTestCase):
    def test_unsupported_workspace_view_uses_owning_output(self):
        """A shared forward signature does not imply backend view support."""
        import pytest

        with pytest.MonkeyPatch.context() as monkeypatch:
            module = _load_megamoe_module(monkeypatch)
            output = torch.arange(4, dtype=torch.bfloat16)

            class Mega:
                supports_output_view = False

                def forward(self, tensors, *, return_workspace_view=False):
                    if return_workspace_view:
                        raise ValueError("workspace views are unsupported")
                    return output

            mega = Mega()
            result = module._select_megamoe_forward(mega)(mega, object())
            self.assertIs(result, output)


class TestMegaMoeFp4Reload(CustomTestCase):
    def test_reload_preserves_load_scales_and_cached_kernel_storage(self):
        """Reloads must accept raw scales and refresh the live kernel's tensors."""
        import pytest

        from sglang.srt.layers.utils.common import copy_or_rebind_param

        with pytest.MonkeyPatch.context() as monkeypatch:
            module = _load_megamoe_module(monkeypatch)
            fake_common = types.ModuleType("sglang.srt.layers.utils.common")
            fake_common.copy_or_rebind_param = copy_or_rebind_param
            monkeypatch.setitem(sys.modules, fake_common.__name__, fake_common)
            monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

            class Mega(torch.nn.Module):
                def __init__(self, *, bootstrap, fleet_params, weights, backend):
                    super().__init__()
                    self.transformed = backend.transformed_weights

                def forward(self, tensors):
                    return tensors

            def preprocess(weights, **kwargs):
                return (
                    (weights.w13.flip(1), weights.w13_scale[..., ::4].to(torch.int32)),
                    (weights.w2.flip(1), weights.w2_scale[..., ::4].to(torch.int32)),
                )

            fake_ep = types.ModuleType("flashinfer.moe_ep")
            for name in (
                "MoEWeightPack",
                "BootstrapConfig",
                "FleetParams",
                "MegaConfig",
                "DeepGemmMegaMoeConfig",
            ):
                setattr(fake_ep, name, types.SimpleNamespace)
            fake_ep.MoEEpMegaLayer = Mega
            fake_ep.preprocess_mega_weights = preprocess
            monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", fake_ep)

            layer = torch.nn.Module()
            layer.hidden_size = layer.intermediate_size_per_partition = 128
            layer.moe_ep_size = layer.num_experts = 8
            layer.moe_ep_rank = layer.layer_id = 0
            layer.top_k = 1
            layer.moe_runner_config = types.SimpleNamespace(swiglu_limit=10.0)
            for name, shape, dtype in (
                ("w13_weight", (1, 256, 64), torch.int8),
                ("w2_weight", (1, 128, 64), torch.int8),
                ("w13_weight_scale_inv", (1, 256, 4), torch.float32),
                ("w2_weight_scale_inv", (1, 128, 4), torch.float32),
            ):
                layer.register_parameter(
                    name,
                    torch.nn.Parameter(
                        torch.ones(shape, dtype=dtype), requires_grad=False
                    ),
                )
            raw_scales = (layer.w13_weight_scale_inv, layer.w2_weight_scale_inv)
            raw_ptrs = [scale.data_ptr() for scale in raw_scales]
            cached = None
            for value in (1, 2, 4):
                for scale in raw_scales:
                    scale.data.copy_(torch.full_like(scale, value, dtype=torch.float32))
                module.prepare_fp4_moe_weights_for_flashinfer_megamoe(layer)
                for scale, ptr in zip(raw_scales, raw_ptrs):
                    self.assertEqual(scale.dtype, torch.float32)
                    self.assertEqual(scale.shape[-1], 4)
                    self.assertEqual(scale.data_ptr(), ptr)
                mega = module.ensure_fp4_moe_layer_for_flashinfer_megamoe(layer)
                tensors = [tensor for pair in mega.transformed for tensor in pair]
                if cached is None:
                    cached = mega
                    kernel_ptrs = [tensor.data_ptr() for tensor in tensors]
                self.assertIs(mega, cached)
                self.assertEqual([tensor.data_ptr() for tensor in tensors], kernel_ptrs)
                for _, scale in cached.transformed:
                    torch.testing.assert_close(
                        scale,
                        torch.full_like(scale, value, dtype=torch.int32),
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
