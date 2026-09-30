from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import (
    _validate_gluon_quant_method,
)
from sglang.srt.layers.moe.gluon_backend import (
    should_use_gluon_moe,
    use_native_moe_with_gluon,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Gluon:
    @staticmethod
    def is_gluon():
        return True


class _NoBackend:
    def __getattr__(self, _name):
        return lambda: False


class _PrepareRecorder:
    def __init__(self):
        self.prepared = False

    def prepare_weights(self):
        self.prepared = True


class _TensorMeta:
    def __init__(self, shape, *, shuffled=False):
        self.shape = shape
        self.is_shuffled = shuffled

    def numel(self):
        result = 1
        for value in self.shape:
            result *= value
        return result


def _backend_shell(monkeypatch):
    from sglang.srt import utils
    from sglang.srt.layers.moe.deepseek_v4_pro_gluon import (
        DeepseekV4ProGluonMoeBackend,
    )
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

    monkeypatch.setattr(utils, "is_gfx95_supported", lambda: True)
    quant_method = object.__new__(Fp8MoEMethod)
    quant_method.is_fp4_expert = True
    quant_method.quant_config = SimpleNamespace(
        is_dsv4_fp4_experts=True,
        is_checkpoint_fp8_serialized=True,
        scale_fmt="ue8m0",
        weight_block_size=[128, 128],
    )
    experts = SimpleNamespace(
        quant_method=quant_method,
        w13_weight=_TensorMeta((384, 768, 3584), shuffled=True),
        w13_weight_scale_inv=_TensorMeta((384, 768, 224)),
        w2_weight=_TensorMeta((384, 7168, 192), shuffled=True),
        w2_weight_scale_inv=_TensorMeta((384, 7168, 16)),
    )
    config = SimpleNamespace(
        model_type="deepseek_v4",
        hidden_size=7168,
        n_routed_experts=384,
        num_experts_per_tok=6,
        moe_intermediate_size=3072,
        num_hidden_layers=61,
        num_hash_layers=3,
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        swiglu_limit=10.0,
    )
    layer = SimpleNamespace(
        config=config,
        layer_id=3,
        tp_size=8,
        moe_ep_size=1,
        is_hash=False,
        gate=SimpleNamespace(weight=object(), e_score_correction_bias=object()),
        _enable_a2a_moe=False,
        alt_stream=None,
        num_fused_shared_experts=0,
        _fuse_shared_experts_inside_sbo=False,
        _shared_expert_tp1=False,
        shared_experts=object(),
        routed_scaling_factor=2.5,
    )
    return DeepseekV4ProGluonMoeBackend(), layer, experts


def test_gluon_quant_validation_accepts_dsv4_native_packed_fp4(monkeypatch):
    from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

    monkeypatch.setattr(fused_moe_layer, "get_moe_runner_backend", lambda: _Gluon())
    quant_method = object.__new__(Fp8MoEMethod)
    quant_method.is_fp4_expert = True
    quant_method.quant_config = SimpleNamespace(
        is_dsv4_fp4_experts=True,
        is_checkpoint_fp8_serialized=True,
    )

    _validate_gluon_quant_method(object(), quant_method)


def test_backend_binds_and_prepares_exact_contract(monkeypatch):
    backend, layer, experts = _backend_shell(monkeypatch)

    backend.bind(layer, experts)
    backend.prepare_weights()

    assert backend.layer is layer
    assert backend.experts is experts
    assert backend.parameters[2:] == (
        experts.w13_weight,
        experts.w13_weight_scale_inv,
        experts.w2_weight,
        experts.w2_weight_scale_inv,
    )


def test_backend_rejects_other_dsv4_variants(monkeypatch):
    backend, layer, experts = _backend_shell(monkeypatch)
    layer.config.hidden_size = 5120

    with pytest.raises(RuntimeError, match="hidden size 7168"):
        backend.bind(layer, experts)


def test_backend_rejects_uncovered_decode_shape(monkeypatch):
    backend, layer, experts = _backend_shell(monkeypatch)
    backend.bind(layer, experts)

    with pytest.raises(RuntimeError, match="supports only c=1 decode shapes"):
        backend.forward(torch.empty((2, 7168), dtype=torch.bfloat16))


def test_fp8_post_load_prepares_attached_gluon_backend(monkeypatch):
    from sglang.srt.layers.quantization import fp8
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

    method = object.__new__(Fp8MoEMethod)
    method.block_quant = True
    method.use_mxfp8 = False
    method.process_weights_after_loading_block_quant = lambda _layer: None
    recorder = _PrepareRecorder()
    layer = SimpleNamespace(_gluon_moe_backend=recorder)
    monkeypatch.setattr(fp8, "get_moe_runner_backend", lambda: _NoBackend())

    method.process_weights_after_loading(layer)

    assert recorder.prepared


def test_hash_prefix_uses_explicit_native_subpath(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.layers.moe.utils.get_moe_runner_backend", lambda: _Gluon()
    )
    layer = SimpleNamespace(layer_id=0)

    use_native_moe_with_gluon(layer)

    assert not should_use_gluon_moe(layer)
