import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _load_cutedsl_runner_module(monkeypatch):
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
        "sglang.srt.layers.moe.token_dispatcher.flashinfer": types.ModuleType(
            "sglang.srt.layers.moe.token_dispatcher.flashinfer"
        ),
        "sglang.srt.layers.moe.token_dispatcher.standard": types.ModuleType(
            "sglang.srt.layers.moe.token_dispatcher.standard"
        ),
        "sglang.srt.layers.moe.topk": types.ModuleType("sglang.srt.layers.moe.topk"),
        "sglang.srt.layers.quantization": types.ModuleType(
            "sglang.srt.layers.quantization"
        ),
        "sglang.srt.layers.quantization.fp4_utils": types.ModuleType(
            "sglang.srt.layers.quantization.fp4_utils"
        ),
        "sglang.srt.model_executor": types.ModuleType("sglang.srt.model_executor"),
        "sglang.srt.model_executor.cuda_graph_config": types.ModuleType(
            "sglang.srt.model_executor.cuda_graph_config"
        ),
        "sglang.srt.runtime_context": types.ModuleType("sglang.srt.runtime_context"),
        "sglang.srt.utils": types.ModuleType("sglang.srt.utils"),
        "sglang.srt.utils.log_utils": types.ModuleType("sglang.srt.utils.log_utils"),
    }
    fake_modules["sglang.srt.environ"].envs = types.SimpleNamespace(
        SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16=types.SimpleNamespace(get=lambda: True),
        SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=types.SimpleNamespace(get=lambda: False),
    )
    base = fake_modules["sglang.srt.layers.moe.moe_runner.base"]
    base.MoeQuantInfo = MoeQuantInfo
    base.MoeRunnerConfig = MoeRunnerConfig
    base.register_fused_func = register_fused_func
    fake_modules[
        "sglang.srt.layers.moe.token_dispatcher.flashinfer"
    ].FlashinferCombineInput = object
    std = fake_modules["sglang.srt.layers.moe.token_dispatcher.standard"]
    std.StandardCombineInput = object
    std.StandardDispatchOutput = object
    fake_modules["sglang.srt.layers.moe.topk"].TopKOutputChecker = object
    fake_modules["sglang.srt.layers.quantization.fp4_utils"].fp4_quantize = (
        lambda *a, **k: None
    )
    fake_modules[
        "sglang.srt.model_executor.cuda_graph_config"
    ].cuda_graph_fully_disabled = lambda: True
    rc = fake_modules["sglang.srt.runtime_context"]
    rc.cutedsl_moe_max_num_tokens = lambda: 128
    rc.get_parallel = lambda: types.SimpleNamespace(num_dp_ranks=1)
    lu = fake_modules["sglang.srt.utils.log_utils"]
    lu.log_info_on_rank0 = lambda *a, **k: None
    lu.print_warning_once = lambda *a, **k: None

    for name, mod in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, mod)

    module_path = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py"
    )
    module_name = "sglang_flashinfer_cutedsl_overheads_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, mod)
    spec.loader.exec_module(mod)
    return mod


def test_w4a16_scale_refresh_skips_scalar_reductions_and_preserves_addresses(
    monkeypatch,
):
    mod = _load_cutedsl_runner_module(monkeypatch)
    monkeypatch.setattr(
        mod,
        "resolve_cutedsl_standard_scales",
        lambda _layer: (_ for _ in ()).throw(
            AssertionError("W4A16 reload should not run full scalarized resolution")
        ),
    )

    w1_cur = torch.ones(4, dtype=torch.float32)
    fc2_cur = torch.ones(1, dtype=torch.float32)
    w2_cur = torch.ones(4, dtype=torch.float32)
    inp_cur = torch.ones(1, dtype=torch.float32)
    layer = types.SimpleNamespace(
        _cutedsl_wrapper=types.SimpleNamespace(quant_mode="w4a16"),
        _cutedsl_scales=(w1_cur, fc2_cur, w2_cur),
        _cutedsl_input_scale=inp_cur,
        g1_alphas=torch.full((4,), 2.5, dtype=torch.float32),
        g2_alphas=torch.full((4,), 3.5, dtype=torch.float32),
        quant_config=types.SimpleNamespace(use_per_token_activation=False),
    )

    w1_ptr, w2_ptr = w1_cur.data_ptr(), w2_cur.data_ptr()
    mod.refresh_cutedsl_standard_scales_for_weight_update(layer)
    assert layer._cutedsl_scales[0].data_ptr() == w1_ptr
    assert layer._cutedsl_scales[2].data_ptr() == w2_ptr
    assert torch.equal(w1_cur, layer.g1_alphas)
    assert torch.equal(w2_cur, layer.g2_alphas)


def test_w4a4_scale_refresh_preserves_addresses_and_updates_values(monkeypatch):
    mod = _load_cutedsl_runner_module(monkeypatch)
    w1_cur = torch.ones(4, dtype=torch.float32)
    fc2_cur = torch.ones(1, dtype=torch.float32)
    w2_cur = torch.ones(4, dtype=torch.float32)
    inp_cur = torch.ones(1, dtype=torch.float32)
    layer = types.SimpleNamespace(
        _cutedsl_wrapper=types.SimpleNamespace(quant_mode="w4a4"),
        _cutedsl_scales=(w1_cur, fc2_cur, w2_cur),
        _cutedsl_input_scale=inp_cur,
        w13_weight_scale_2=torch.full((4, 2), 0.5, dtype=torch.float32),
        w2_weight_scale_2=torch.full((4,), 0.75, dtype=torch.float32),
        w13_input_scale_quant=torch.tensor([2.0], dtype=torch.float32),
        w2_input_scale_quant=torch.tensor([4.0], dtype=torch.float32),
        g1_alphas=torch.full((4,), 0.25, dtype=torch.float32),
        g2_alphas=torch.full((4,), 0.1875, dtype=torch.float32),
        quant_config=types.SimpleNamespace(use_per_token_activation=False),
    )

    ptrs = (
        w1_cur.data_ptr(),
        fc2_cur.data_ptr(),
        w2_cur.data_ptr(),
        inp_cur.data_ptr(),
    )
    mod.refresh_cutedsl_standard_scales_for_weight_update(layer)
    assert (
        layer._cutedsl_scales[0].data_ptr(),
        layer._cutedsl_scales[1].data_ptr(),
        layer._cutedsl_scales[2].data_ptr(),
        layer._cutedsl_input_scale.data_ptr(),
    ) == ptrs
    assert torch.allclose(w1_cur, torch.full((4,), 0.25, dtype=torch.float32))
    assert torch.allclose(fc2_cur, torch.tensor([4.0], dtype=torch.float32))
    assert torch.allclose(w2_cur, torch.full((4,), 0.1875, dtype=torch.float32))
    assert torch.allclose(inp_cur, torch.tensor([2.0], dtype=torch.float32))


def test_ensure_cutedsl_wrapper_clears_cached_quant_info(monkeypatch):
    mod = _load_cutedsl_runner_module(monkeypatch)

    class FakeActivationType:
        Swiglu = "swiglu"
        Relu2 = "relu2"

    class FakeWrapper:
        def __init__(self, **kwargs):
            self.quant_mode = kwargs["quant_mode"]

    fake_fi = types.ModuleType("flashinfer")
    fake_fi.ActivationType = FakeActivationType
    fake_fi.CuteDslMoEWrapper = FakeWrapper
    monkeypatch.setitem(sys.modules, "flashinfer", fake_fi)

    layer = types.SimpleNamespace(
        _cutedsl_wrapper=None,
        _cutedsl_quant_info=object(),
        num_experts=4,
        num_local_experts=4,
        moe_ep_rank=0,
        top_k=2,
        hidden_size=128,
        intermediate_size_per_partition=64,
        w13_weight=torch.empty(4, 128, 64, dtype=torch.uint8),
        w13_weight_scale_2=torch.ones((4, 2), dtype=torch.float32),
        w2_weight_scale_2=torch.ones((4,), dtype=torch.float32),
        w13_input_scale_quant=torch.ones(1, dtype=torch.float32),
        w2_input_scale_quant=torch.ones(1, dtype=torch.float32),
        g1_alphas=torch.ones(4, dtype=torch.float32),
        g2_alphas=torch.ones(4, dtype=torch.float32),
        quant_config=types.SimpleNamespace(use_per_token_activation=False),
        moe_runner_config=types.SimpleNamespace(
            top_k=2,
            activation="silu",
            swiglu_limit=None,
            params_dtype=torch.bfloat16,
        ),
    )

    mod.ensure_cutedsl_wrapper(layer)
    assert layer._cutedsl_wrapper is not None
    assert layer._cutedsl_quant_info is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
