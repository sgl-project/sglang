import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def test_minimax_m3_tp4_flydsl_weight_shapes_are_complete():
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
        MXFP8_FLYDSL_WEIGHT_SHAPES,
    )

    assert MXFP8_FLYDSL_WEIGHT_SHAPES == frozenset(
        {
            (2304, 6144),
            (2560, 6144),
            (6144, 2048),
            (6144, 6144),
            (6144, 3072),
            (1536, 6144),
            (6144, 768),
        }
    )


def test_minimax_m3_flydsl_covers_all_decode_graph_buckets():
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
        MXFP8_FLYDSL_M_VALUES,
    )

    assert MXFP8_FLYDSL_M_VALUES[:12] == (
        1,
        2,
        4,
        8,
        12,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
    )


def test_dense_mxfp8_rejects_unpaired_aiter(monkeypatch):
    import builtins

    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
        get_flydsl_mxfp8_config,
    )

    real_import = builtins.__import__

    def reject_unpaired_aiter(name, *args, **kwargs):
        if name == "aiter.ops.flydsl":
            raise ImportError("missing paired API")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_unpaired_aiter)
    with pytest.raises(RuntimeError, match="paired AITER v0.1.19.post2"):
        get_flydsl_mxfp8_config(4, 2304, 6144)


def test_mxfp8_aiter_backend_dispatches_to_flydsl(monkeypatch):
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
        flydsl_mxfp8_blockscaled_linear,
    )
    from sglang.srt.layers.quantization import fp8_utils

    monkeypatch.setattr(fp8_utils, "_use_aiter", True)
    monkeypatch.setattr(fp8_utils, "_is_hip", True)
    monkeypatch.setattr(fp8_utils, "_is_gfx95_supported", True)
    monkeypatch.setattr(
        fp8_utils,
        "FP8_GEMM_RUNNER_BACKEND",
        fp8_utils.Fp8GemmRunnerBackend.AITER,
    )

    assert (
        fp8_utils.dispatch_w8a8_mxfp8_linear()
        is flydsl_mxfp8_blockscaled_linear
    )


def test_mxfp8_flydsl_unknown_topology_keeps_canonical_weights(monkeypatch):
    from sglang.kernels.ops.quantization import mxfp8_amd_gfx95
    from sglang.srt.layers.quantization import fp8 as fp8_module
    from sglang.srt.layers.quantization import fp8_utils

    fake_aiter = ModuleType("aiter")
    fake_aiter.__path__ = []
    fake_ops = ModuleType("aiter.ops")
    fake_ops.__path__ = []
    fake_shuffle = ModuleType("aiter.ops.shuffle")
    fake_shuffle.shuffle_scale_a16w4 = lambda *args: args[0]
    fake_shuffle.shuffle_weight = lambda value, **kwargs: value
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setitem(sys.modules, "aiter.ops", fake_ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.shuffle", fake_shuffle)

    monkeypatch.setattr(fp8_module, "_use_aiter", True)
    monkeypatch.setattr(fp8_module, "_is_hip", True)
    monkeypatch.setattr(fp8_module, "_is_gfx95_supported", True)
    monkeypatch.setattr(
        fp8_module,
        "get_fp8_gemm_runner_backend",
        lambda: fp8_utils.Fp8GemmRunnerBackend.AITER,
    )
    monkeypatch.setattr(
        mxfp8_amd_gfx95, "MXFP8_FLYDSL_WEIGHT_SHAPES", frozenset({(32, 256)})
    )
    monkeypatch.setattr(mxfp8_amd_gfx95, "MXFP8_FLYDSL_M_VALUES", (4,))
    monkeypatch.setattr(
        mxfp8_amd_gfx95, "get_flydsl_mxfp8_config", lambda *args: None
    )

    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.empty((32, 256), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_scale_inv",
        torch.nn.Parameter(
            torch.empty((32, 8), dtype=torch.uint8), requires_grad=False
        ),
    )
    weight_ptr = layer.weight.data_ptr()
    scale_ptr = layer.weight_scale_inv.data_ptr()

    method = fp8_module.Fp8LinearMethod.__new__(fp8_module.Fp8LinearMethod)
    method.use_mxfp8 = True
    method._process_mxfp8_linear_weight_scale(layer)

    assert layer.weight.data_ptr() == weight_ptr
    assert layer.weight_scale_inv.data_ptr() == scale_ptr
    assert not hasattr(layer, "weight_mxfp8_flydsl")
    assert not hasattr(layer, "weight_scale_inv_mxfp8_flydsl")


def test_mxfp8_flydsl_decode_requires_captured_graph_bucket(monkeypatch):
    from sglang.kernels.ops.quantization import mxfp8_amd_gfx95
    from sglang.srt import runtime_context
    from sglang.srt.layers.quantization import fp8 as fp8_module
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        CudaGraphConfig,
    )

    monkeypatch.setattr(
        fp8_module,
        "get_fp8_gemm_runner_backend",
        lambda: fp8_utils.Fp8GemmRunnerBackend.AITER,
    )
    monkeypatch.setattr(fp8_module, "_LOGGED_MXFP8_FLYDSL_SIGNATURES", set())
    config = {"kernelName": "unit_test", "splitK": 1}
    monkeypatch.setattr(
        mxfp8_amd_gfx95, "get_flydsl_mxfp8_config", lambda *args: config
    )

    fallback_calls = []
    fallback_output = torch.empty((4, 32), dtype=torch.bfloat16)

    def fallback(**kwargs):
        fallback_calls.append(kwargs)
        return fallback_output

    monkeypatch.setattr(
        mxfp8_amd_gfx95, "dot_scaled_mxfp8_blockscaled_linear", fallback
    )

    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.empty((32, 256), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_scale_inv",
        torch.nn.Parameter(
            torch.empty((32, 8), dtype=torch.uint8), requires_grad=False
        ),
    )
    layer.register_parameter(
        "weight_mxfp8_flydsl",
        torch.nn.Parameter(
            torch.empty((32, 256), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_scale_inv_mxfp8_flydsl",
        torch.nn.Parameter(
            torch.empty((32, 8), dtype=torch.uint8), requires_grad=False
        ),
    )

    method = fp8_module.Fp8LinearMethod.__new__(fp8_module.Fp8LinearMethod)
    method.use_marlin = False
    method.use_mxfp8 = True
    flydsl_calls = []
    flydsl_output = torch.empty((4, 32), dtype=torch.bfloat16)

    def flydsl(**kwargs):
        flydsl_calls.append(kwargs)
        return flydsl_output

    method.w8a8_mxfp8_linear = flydsl
    x = torch.empty((4, 256), dtype=torch.bfloat16)

    disabled = CudaGraphConfig()
    disabled.decode.backend = Backend.DISABLED
    monkeypatch.setattr(
        runtime_context,
        "get_server_args",
        lambda: SimpleNamespace(cuda_graph_config=disabled),
    )
    assert method.apply(layer, x) is fallback_output
    assert fallback_calls[-1]["weight"] is layer.weight

    enabled = CudaGraphConfig()
    enabled.decode.backend = Backend.FULL
    enabled.decode.bs = [4]
    enabled.decode.max_bs = 64
    monkeypatch.setattr(
        runtime_context,
        "get_server_args",
        lambda: SimpleNamespace(cuda_graph_config=enabled),
    )
    assert method.apply(layer, x) is flydsl_output
    assert flydsl_calls[-1]["weight"] is layer.weight_mxfp8_flydsl

    flydsl_count = len(flydsl_calls)
    fallback_count = len(fallback_calls)
    assert method.apply(layer, x.to(torch.float16)) is fallback_output
    assert len(flydsl_calls) == flydsl_count
    assert len(fallback_calls) == fallback_count + 1


def test_mxfp8_flydsl_decode_buffers_are_pooled_across_buckets(monkeypatch):
    from sglang.kernels.ops.quantization import mxfp8_amd_gfx95
    from sglang.srt.layers.quantization import fp8 as fp8_module

    monkeypatch.setattr(mxfp8_amd_gfx95, "MXFP8_FLYDSL_M_VALUES", (4, 64))
    monkeypatch.setattr(
        mxfp8_amd_gfx95,
        "get_flydsl_mxfp8_config",
        lambda m, *_: {"splitK": 2 if m == 4 else 3},
    )
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.empty((32, 256), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    method = fp8_module.Fp8LinearMethod.__new__(fp8_module.Fp8LinearMethod)

    small = method._get_mxfp8_flydsl_runtime_buffers(
        layer,
        m=4,
        n=32,
        k=256,
        output_dtype=torch.bfloat16,
        split_k=2,
    )
    large = method._get_mxfp8_flydsl_runtime_buffers(
        layer,
        m=64,
        n=32,
        k=256,
        output_dtype=torch.bfloat16,
        split_k=3,
    )
    for name in (
        "activation_scale_padded",
        "activation_scale_shuffled",
        "output_buffer",
        "splitk_workspace",
    ):
        assert small[name].data_ptr() == large[name].data_ptr()
        assert small[name].is_contiguous()
        assert large[name].is_contiguous()
    assert small["output_buffer"].shape == (4, 32)
    assert large["output_buffer"].shape == (64, 32)
    assert small["splitk_workspace"].shape == (2, 4, 32)
    assert large["splitk_workspace"].shape == (3, 64, 32)
    assert len(layer._mxfp8_flydsl_runtime_buffers) == 1
