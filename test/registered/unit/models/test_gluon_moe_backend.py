from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import (
    _validate_gluon_quant_method,
)
from sglang.srt.layers.moe.gluon_backend import GluonMoeBackend
from sglang.srt.models import deepseek_v2
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Backend(GluonMoeBackend):
    def __init__(self, result=None):
        self.result = result
        self.bound = False

    def bind(self, layer, experts):
        self.bound = True

    def forward(self, hidden_states, **kwargs):
        if self.result is None:
            return hidden_states + 2
        return self.result(hidden_states)


def _moe_shell() -> DeepseekV2MoE:
    moe = DeepseekV2MoE.__new__(DeepseekV2MoE)
    torch.nn.Module.__init__(moe)
    moe._gluon_moe_backend = None
    moe.is_deepseek_v4 = False
    moe.is_hash = False
    moe.tp_size = 8
    moe.layer_id = 7
    moe.experts = torch.nn.Identity()
    moe._shared_expert_tp1 = False
    return moe


def test_gluon_backend_is_strict_and_uses_native_finalize(monkeypatch):
    moe = _moe_shell()
    monkeypatch.setattr(deepseek_v2, "get_moe_runner_backend", lambda: _Gluon())
    monkeypatch.setattr(deepseek_v2, "post_experts_all_reduce", lambda value: value + 3)
    backend = _Backend()

    moe.bind_gluon_moe_backend(backend)
    output = moe.forward_normal(torch.zeros((2, 4), dtype=torch.bfloat16))

    assert backend.bound
    torch.testing.assert_close(output, torch.full_like(output, 5))


def test_gluon_backend_missing_implementation_raises(monkeypatch):
    moe = _moe_shell()
    monkeypatch.setattr(deepseek_v2, "get_moe_runner_backend", lambda: _Gluon())

    with pytest.raises(RuntimeError, match="no Gluon implementation was bound"):
        moe.forward_normal(torch.zeros((2, 4), dtype=torch.bfloat16))


def test_gluon_backend_is_selected_before_other_forward_paths(monkeypatch):
    moe = _moe_shell()
    monkeypatch.setattr(deepseek_v2, "get_moe_runner_backend", lambda: _Gluon())

    def reject_mega_moe(*_args, **_kwargs):
        raise AssertionError("MegaMoE selection must not run for Gluon")

    monkeypatch.setattr(
        "sglang.srt.layers.moe.mega_moe.should_use_mega_moe",
        reject_mega_moe,
    )

    with pytest.raises(RuntimeError, match="no Gluon implementation was bound"):
        moe.forward(torch.zeros((2, 4), dtype=torch.bfloat16))


def test_gluon_backend_rejects_invalid_output(monkeypatch):
    moe = _moe_shell()
    monkeypatch.setattr(deepseek_v2, "get_moe_runner_backend", lambda: _Gluon())
    moe.bind_gluon_moe_backend(_Backend(lambda value: value[:, :2]))

    with pytest.raises(RuntimeError, match="must match the input tensor contract"):
        moe.forward_normal(torch.zeros((2, 4), dtype=torch.bfloat16))


class _Gluon:
    @staticmethod
    def is_gluon():
        return True

    @staticmethod
    def is_auto():
        return False


def test_gluon_keeps_shared_experts_out_of_generic_fusion(monkeypatch):
    model = deepseek_v2.DeepseekV2ForCausalLM.__new__(deepseek_v2.DeepseekV2ForCausalLM)
    model.config = SimpleNamespace(n_shared_experts=1)
    monkeypatch.setattr(deepseek_v2, "is_shared_experts_fusion_disabled", lambda: False)
    monkeypatch.setattr(deepseek_v2, "get_moe_runner_backend", lambda: _Gluon())

    model.determine_num_fused_shared_experts()

    assert model.num_fused_shared_experts == 0


def test_gluon_backend_rejects_fp8_quant_method(monkeypatch):
    from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "get_moe_runner_backend", lambda: _Gluon())

    with pytest.raises(ValueError, match="only serialized Quark W4A4"):
        _validate_gluon_quant_method(object(), object())


def test_gluon_backend_accepts_serialized_quark_mxfp4(monkeypatch):
    from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer
    from sglang.srt.layers.quantization.quark.quark import QuarkFusedMoEMethod
    from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
        QuarkW4A4MXFp4MoE,
    )

    monkeypatch.setattr(fused_moe_layer, "get_moe_runner_backend", lambda: _Gluon())
    scheme = object.__new__(QuarkW4A4MXFp4MoE)
    scheme.is_checkpoint_mxfp4_serialized = True
    layer = type("Layer", (), {"scheme": scheme})()
    quant_method = object.__new__(QuarkFusedMoEMethod)

    _validate_gluon_quant_method(layer, quant_method)


@pytest.mark.parametrize(
    ("ep_size", "tp_size", "local_experts", "intermediate"),
    (
        (1, 4, 256, 512),
        (1, 8, 256, 256),
        (2, 4, 128, 512),
        (4, 2, 64, 1024),
        (8, 1, 32, 2048),
    ),
)
def test_gluon_backend_accepts_audited_glm_nextn_bf16_experts(
    monkeypatch, ep_size, tp_size, local_experts, intermediate
):
    from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    monkeypatch.setattr(fused_moe_layer, "get_moe_runner_backend", lambda: _Gluon())
    w13 = type(
        "Weight",
        (),
        {"dtype": torch.bfloat16, "shape": (local_experts, 2 * intermediate, 6144)},
    )()
    w2 = type(
        "Weight",
        (),
        {"dtype": torch.bfloat16, "shape": (local_experts, 6144, intermediate)},
    )()
    layer = type(
        "Layer",
        (),
        {
            "layer_name": "model.decoder.mlp.experts",
            "num_experts": 256,
            "hidden_size": 6144,
            "top_k": 8,
            "moe_ep_size": ep_size,
            "moe_tp_size": tp_size,
            "_num_local_routed": local_experts,
            "intermediate_size_per_partition": intermediate,
            "w13_weight": w13,
            "w2_weight": w2,
        },
    )()

    _validate_gluon_quant_method(layer, object.__new__(UnquantizedFusedMoEMethod))


def test_gluon_backend_owns_quark_mxfp4_weight_layout(monkeypatch):
    from sglang.srt.layers.moe import utils as moe_utils
    from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
        QuarkW4A4MXFp4MoE,
    )

    monkeypatch.setattr(moe_utils, "get_moe_runner_backend", lambda: _Gluon())
    scheme = object.__new__(QuarkW4A4MXFp4MoE)

    scheme.create_moe_runner(layer=object(), moe_runner_config=object())

    assert scheme.runner is None
    assert not scheme._owns_moe_runner
    assert scheme._owns_moe_weight_layout


def _glm_backend_shell(monkeypatch, total_tp=8, ep_size=1, ep_rank=0, is_nextn=False):
    from sglang.srt import utils
    from sglang.srt.layers.moe.glm_mxfp4_gluon import (
        GlmMxfp4GluonMoeBackend,
    )
    from sglang.srt.layers.quantization.quark.quark import QuarkFusedMoEMethod
    from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
        QuarkW4A4MXFp4MoE,
    )

    monkeypatch.setattr(utils, "is_gfx95_supported", lambda: True)
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(enable_eplb=False)),
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(moe_dp_size=1),
    )
    quant_method = object.__new__(QuarkFusedMoEMethod)
    scheme = object.__new__(QuarkW4A4MXFp4MoE)
    scheme.is_checkpoint_mxfp4_serialized = True
    local_experts = 256 // ep_size
    local_intermediate = 2048 // (total_tp // ep_size)
    experts = SimpleNamespace(
        quant_method=quant_method,
        scheme=scheme,
        _num_local_routed=local_experts,
        intermediate_size_per_partition=local_intermediate,
        moe_ep_rank=ep_rank,
    )
    config = SimpleNamespace(
        model_type="glm_moe_dsa",
        hidden_size=6144,
        n_routed_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=2048,
        n_shared_experts=1,
        scoring_func="sigmoid",
        norm_topk_prob=True,
    )
    layer = SimpleNamespace(
        config=config,
        layer_id=7,
        tp_size=total_tp,
        moe_ep_size=ep_size,
        is_nextn=is_nextn,
        _enable_a2a_moe=False,
        alt_stream=None,
        num_fused_shared_experts=0,
        _fuse_shared_experts_inside_sbo=False,
        _shared_expert_tp1=False,
        shared_experts=object(),
        routed_scaling_factor=2.5,
    )
    backend = GlmMxfp4GluonMoeBackend()
    return backend, layer, experts


@pytest.mark.parametrize(
    ("total_tp", "ep_size", "ep_rank", "local_experts", "local_intermediate"),
    (
        (4, 1, 0, 256, 512),
        (8, 1, 0, 256, 256),
        (8, 2, 1, 128, 512),
        (8, 4, 3, 64, 1024),
        (8, 8, 7, 32, 2048),
    ),
)
def test_glm_backend_binds_supported_tp_ep_contracts(
    monkeypatch, total_tp, ep_size, ep_rank, local_experts, local_intermediate
):
    backend, layer, experts = _glm_backend_shell(
        monkeypatch, total_tp, ep_size, ep_rank
    )

    backend.bind(layer, experts)

    assert backend.layer is layer
    assert backend.experts is experts
    assert backend.local_experts == local_experts
    assert backend.local_intermediate == local_intermediate
    assert backend.expert_start == ep_rank * local_experts


def test_glm_backend_rejects_unsupported_topology(monkeypatch):
    backend, layer, experts = _glm_backend_shell(monkeypatch, total_tp=4)
    layer.moe_ep_size = 2

    with pytest.raises(RuntimeError, match="TP4/EP1 or TP8/EP1/2/4/8 topology"):
        backend.bind(layer, experts)


def test_glm_backend_rejects_fp8_checkpoint(monkeypatch):
    backend, layer, experts = _glm_backend_shell(monkeypatch)
    experts.quant_method = object()

    with pytest.raises(RuntimeError, match="serialized MXFP4 target"):
        backend.bind(layer, experts)


def test_glm_dispatch_covers_tp4_and_tp8_target_and_nextn_ranges():
    from sglang.srt.layers.moe.glm_mxfp4_gluon import _kernel_name

    for m in range(1, 32769):
        assert _kernel_name(8, 1, False, m).startswith("fused_moe_tp8_")
        assert _kernel_name(8, 1, True, m).startswith("fused_moe_tp8_")
    for total_tp, ep_size in ((4, 1), (8, 2), (8, 4), (8, 8)):
        for m in range(1, 16769):
            assert _kernel_name(total_tp, ep_size, False, m).startswith(
                "fused_moe_tp4_"
            )
            assert _kernel_name(total_tp, ep_size, True, m).startswith("fused_moe_tp4_")


@pytest.mark.parametrize(
    ("total_tp", "ep_size", "is_nextn", "m"),
    ((8, 1, False, 32769), (8, 1, True, 32769), (4, 1, False, 16769)),
)
def test_glm_dispatch_rejects_uncovered_shapes(total_tp, ep_size, is_nextn, m):
    from sglang.srt.layers.moe.glm_mxfp4_gluon import _kernel_name

    with pytest.raises(RuntimeError, match="no specialization"):
        _kernel_name(total_tp, ep_size, is_nextn, m)
