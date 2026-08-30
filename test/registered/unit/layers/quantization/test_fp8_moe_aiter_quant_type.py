import sys
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.quantization.fp8 as fp8_quant
from sglang.srt.layers.moe import MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.aiter import AiterQuantType
from sglang.srt.layers.quantization import mxfp8_block_convert
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _method(*, use_mxfp8: bool, is_fp4_expert: bool = False):
    method = object.__new__(fp8_quant.Fp8MoEMethod)
    method.block_quant = True
    method.use_mxfp8 = use_mxfp8
    method.is_fp4_expert = is_fp4_expert
    method.moe_runner_config = SimpleNamespace(swiglu_limit=None)
    return method


def _layer():
    return SimpleNamespace(
        w13_weight=torch.empty((2, 8, 32), dtype=torch.float8_e4m3fn),
        w2_weight=torch.empty((2, 4, 32), dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.empty((2, 8, 1), dtype=torch.uint8),
        w2_weight_scale_inv=torch.empty((2, 4, 1), dtype=torch.uint8),
        dispatcher=SimpleNamespace(expert_mask_gpu=None),
    )


def test_block_fp8_keeps_aiter_128x128_quant_type(monkeypatch):
    monkeypatch.setattr(fp8_quant, "_use_aiter", True)
    monkeypatch.setattr(fp8_quant, "_use_hip_int4", False)

    quant_info = _method(use_mxfp8=False).maybe_get_hip_aiter_quant_info(_layer())

    assert quant_info is not None
    assert quant_info.quant_type == AiterQuantType.PER_128X128


def test_mxfp8_block_fallback_uses_aiter_stage2_runner(monkeypatch):
    method = _method(use_mxfp8=True)
    method.convert_mxfp8_to_block = True
    monkeypatch.setattr(fp8_quant, "_use_aiter", True)
    monkeypatch.setattr(fp8_quant, "_mxfp8_bf16_fallback_required", lambda: True)
    monkeypatch.setattr(
        fp8_quant, "get_moe_runner_backend", lambda: MoeRunnerBackend.AITER
    )
    monkeypatch.setattr(
        fp8_quant,
        "MoeRunner",
        lambda backend, config: SimpleNamespace(runner_backend=backend),
    )

    method.create_moe_runner(SimpleNamespace(), MoeRunnerConfig())

    assert method.convert_mxfp8_to_block is True
    assert method.runner.runner_backend == MoeRunnerBackend.AITER


def test_mxfp8_block_fallback_forces_triton_without_aiter(monkeypatch):
    method = _method(use_mxfp8=True)
    method.convert_mxfp8_to_block = True
    monkeypatch.setattr(fp8_quant, "_use_aiter", False)
    monkeypatch.setattr(fp8_quant, "_mxfp8_bf16_fallback_required", lambda: True)
    monkeypatch.setattr(
        fp8_quant, "get_moe_runner_backend", lambda: MoeRunnerBackend.AITER
    )
    monkeypatch.setattr(
        fp8_quant,
        "MoeRunner",
        lambda backend, config: SimpleNamespace(runner_backend=backend),
    )

    method.create_moe_runner(SimpleNamespace(), MoeRunnerConfig())

    assert method.convert_mxfp8_to_block is True
    assert method.runner.runner_backend == MoeRunnerBackend.TRITON


def test_mxfp8_block_fallback_builds_triton_quant_info(monkeypatch):
    method = _method(use_mxfp8=True)
    method.convert_mxfp8_to_block = True
    method.weight_block_size = [1, 32]
    method.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.TRITON)
    layer = _layer()
    layer.w13_input_scale = None
    layer.w2_input_scale = None

    def convert(weight, scale, block=128):
        assert scale.dtype == torch.uint8
        return weight.clone(), torch.ones(
            ((weight.shape[0] + block - 1) // block, 1), dtype=torch.float32
        )

    monkeypatch.setattr(
        mxfp8_block_convert, "convert_mxfp8_weight_to_block_fp8", convert
    )
    monkeypatch.setattr(fp8_quant, "_is_fp8_fnuz", False)

    method.process_weights_after_loading_block_quant(layer)
    quant_info = method.get_triton_quant_info(layer)

    assert method.use_mxfp8 is False
    assert method.weight_block_size == [128, 128]
    assert quant_info.use_mxfp8 is False
    assert quant_info.use_fp8_w8a8 is True
    assert quant_info.block_shape == [128, 128]
    assert quant_info.w13_scale.dtype == torch.float32
    assert quant_info.w2_scale.dtype == torch.float32


def test_mxfp8_uses_aiter_after_load_time_block_conversion(monkeypatch):
    method = _method(use_mxfp8=True)
    method.convert_mxfp8_to_block = False
    monkeypatch.setattr(fp8_quant, "_use_aiter", True)
    monkeypatch.setattr(fp8_quant, "_mxfp8_bf16_fallback_required", lambda: False)
    monkeypatch.setattr(
        fp8_quant, "get_moe_runner_backend", lambda: MoeRunnerBackend.AITER
    )
    monkeypatch.setattr(
        fp8_quant,
        "MoeRunner",
        lambda backend, config: SimpleNamespace(runner_backend=backend),
    )

    method.create_moe_runner(SimpleNamespace(), MoeRunnerConfig())

    assert method.convert_mxfp8_to_block is True
    assert method.runner.runner_backend == MoeRunnerBackend.AITER


def test_aiter_block_fp8_keeps_stage2_router_weighting(monkeypatch):
    method = _method(use_mxfp8=False)
    monkeypatch.setattr(fp8_quant, "_use_aiter", True)
    monkeypatch.setattr(fp8_quant, "_use_hip_int4", False)

    quant_info = method.maybe_get_hip_aiter_quant_info(_layer())

    assert quant_info is not None
    assert quant_info.quant_type == AiterQuantType.PER_128X128
    assert quant_info.doweight_stage1 is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
