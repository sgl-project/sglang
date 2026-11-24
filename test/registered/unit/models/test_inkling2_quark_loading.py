"""The Quark experts of Inkling-2: loading the packed tensors of an export, the MoE
layer around them, and the real model built from the config.json of an export (CPU only).
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load, save
from torch import nn

from sglang.srt.layers.quantization.inkling2 import Inkling2QuantConfig
from sglang.srt.layers.quantization.inkling2_quark import Inkling2QuarkLayout
from sglang.srt.models.inkling2 import Inkling2ForConditionalGeneration
from sglang.srt.models.inkling_common import inkling2_moe as moe
from sglang.srt.models.inkling_common import inkling2_moe_quark as quark_moe
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.inkling2_quark_test_utils import (
    EXPERT_FORMAT,
    Materialization,
    expert_checkpoint_tensors,
    native_quant_dict,
    quantization_config,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

EXPERTS, HIDDEN, INTERMEDIATE = 4, 256, 128
EXPERT_PREFIX = "model.llm.layers.0.mlp.experts."
BF16, F32, U8, I32 = torch.bfloat16, torch.float32, torch.uint8, torch.int32
E4M3 = torch.float8_e4m3fn


def _config():
    return SimpleNamespace(
        intermediate_size=INTERMEDIATE,
        hidden_size=HIDDEN,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        num_experts_per_tok=2,
        route_scale=8.0,
        n_routed_experts=EXPERTS,
        use_gate_bias=True,
    )


def _parallel(tp_size=1, ep_size=1, ep_rank=0, group=None):
    return lambda: SimpleNamespace(
        moe_tp_rank=0,
        moe_tp_size=tp_size,
        moe_ep_rank=ep_rank,
        moe_ep_size=ep_size,
        moe_dp_size=1,
        moe_ep_group=group
        or SimpleNamespace(world_size=ep_size, rank_in_group=ep_rank),
    )


def _layout(variant):
    return Inkling2QuarkLayout.from_quantization_dict(quantization_config(variant), 1)


def _experts(monkeypatch, variant="MXFP4-RTN", *, tp_size=1, ep_size=1, ep_rank=0):
    monkeypatch.setattr(quark_moe, "get_parallel", _parallel(tp_size, ep_size, ep_rank))
    return quark_moe.Inkling2QuarkExperts(
        _config(), EXPERTS, EXPERT_PREFIX, _layout(variant)
    )


def _target(monkeypatch, variant="MXFP4-RTN", *, ep_size=1, ep_rank=0):
    """The model, assembled with __new__ around one layer of experts (as test_inkling2.py)."""
    monkeypatch.setattr(quark_moe, "get_parallel", _parallel(1, ep_size, ep_rank))
    layout = _layout(variant)
    target = Inkling2ForConditionalGeneration.__new__(Inkling2ForConditionalGeneration)
    nn.Module.__init__(target)
    target.visual, target.quant_config = None, layout
    target.llm = nn.Module()
    layer = nn.Module()
    layer.mlp = nn.Module()
    target.llm.layers = nn.ModuleList([layer])
    layer.mlp.experts = quark_moe.Inkling2QuarkExperts(
        _config(), EXPERTS, EXPERT_PREFIX, layout
    )
    return target


def _export(variant, scale_dtype=BF16):
    """The expert tensors of one layer of a tiny export."""
    return expert_checkpoint_tensors(
        EXPERT_FORMAT[variant],
        experts=EXPERTS,
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        scale_dtype=scale_dtype,
    )


def _leaf(checkpoint_name):
    return checkpoint_name.rsplit(".", 1)[1].replace("_weight_scale", "_scale")


@pytest.mark.parametrize("ep_size", [1, 2])
@pytest.mark.parametrize(
    "variant,scale_dtype",
    [("MXFP4-RTN", BF16), ("UINT4-AWQ", BF16), ("UINT4-AWQ", F32)],
    ids=["mxfp4", "uint4-bf16-scales", "uint4-f32-scales"],
)
def test_packed_experts_load_bitwise_without_dequantizing(
    monkeypatch, variant, scale_dtype, ep_size
):
    # through safetensors, as the checkpoint reader hands them over
    weights = list(load(save(_export(variant, scale_dtype))).items())
    for rank in range(ep_size):
        target = _target(monkeypatch, variant, ep_size=ep_size, ep_rank=rank)
        assert target.load_weights(weights) == set(dict(target.named_parameters()))
        experts = target.llm.layers[0].mlp.experts
        for name, value in weights:
            actual = getattr(experts, _leaf(name))
            expected = value.chunk(ep_size, dim=0)[rank]
            if name.endswith("_weight_scale") and variant == "UINT4-AWQ":
                expected = expected.to(F32)  # the one lossless upcast
            assert actual.dtype == expected.dtype
            assert torch.equal(actual.view(U8), expected.contiguous().view(U8)), name
        assert experts._identity is not None  # finalize_load ran


@pytest.mark.parametrize("variant", ["MXFP4-RTN", "UINT4-AWQ"])
def test_finalize_load_makes_no_copy_of_the_packed_weights(monkeypatch, variant):
    experts = _experts(monkeypatch, variant)
    with Materialization(*experts.parameters()) as watch:
        experts.finalize_load()
        experts.finalize_load()
    assert not watch.copies, watch.copies
    experts.w13_weight.data = experts.w13_weight.data.to(BF16)
    with pytest.raises(ValueError, match="must stay packed: w13_weight"):
        experts.finalize_load()


def test_unexpected_tensors_are_refused(monkeypatch):
    for extra in (
        EXPERT_PREFIX + "w13_weight_amax",  # an NVFP4 tensor
        "model.llm.layers.1.mlp.experts.w13_weight",  # a layer the model does not have
    ):
        tensors = _export("MXFP4-RTN")
        tensors[extra] = torch.zeros(EXPERTS, 8, dtype=U8)
        with pytest.raises(ValueError, match="Unrecognized checkpoint tensor"):
            _target(monkeypatch, "MXFP4-RTN").load_weights(list(tensors.items()))


@pytest.mark.parametrize("variant", ["MXFP4-RTN", "UINT4-AWQ"])
def test_missing_tensors_are_refused(monkeypatch, variant):
    tensors = _export(variant)
    for name in tensors:
        rest = [(key, value) for key, value in tensors.items() if key != name]
        with pytest.raises(ValueError, match="Missing checkpoint tensors") as error:
            _target(monkeypatch, variant).load_weights(rest)
        assert f"mlp.experts.{_leaf(name)}" in str(error.value)


def test_moe_tensor_parallelism_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="need expert parallelism"):
        _experts(monkeypatch, tp_size=2)
    with pytest.raises(ValueError, match="divisible by expert parallelism"):
        _experts(monkeypatch, ep_size=3)


def test_experts_run_on_gfx950_only(monkeypatch):
    experts = _experts(monkeypatch)
    experts._check_platform()  # CPU weights (tests, meta dry runs) are allowed
    experts._parameters["w13_weight"] = SimpleNamespace(is_cuda=True)
    for facts in (
        dict(is_hip=False, is_gfx95=False),
        dict(is_hip=True, is_gfx95=False),
    ):
        with override_platform(**facts):
            with pytest.raises(ValueError, match="ROCm gfx950 only"):
                experts.finalize_load()
    with override_platform(is_hip=True, is_gfx95=True):
        experts._check_platform()


def test_expert_parallel_sums_are_reduced_over_the_expert_group(monkeypatch):
    reduced = []
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=1,
        all_reduce=lambda x: reduced.append(x) or x * 4,
    )
    monkeypatch.setattr(
        quark_moe, "get_parallel", _parallel(ep_size=2, ep_rank=1, group=group)
    )
    experts = quark_moe.Inkling2QuarkExperts(
        _config(), EXPERTS, EXPERT_PREFIX, _layout("MXFP4-RTN")
    )
    experts.finalize_load()
    sums = torch.randn(5, HIDDEN, dtype=F32)
    assert torch.equal(experts.finish_forward(sums, True, BF16), (sums * 4).to(BF16))
    routed = experts.finish_forward(sums, False, BF16)
    # one all-reduce of the FP32 sums per call
    assert len(reduced) == 2 and all(t is sums for t in reduced)
    assert isinstance(routed, moe.RoutedOutput) and routed.top_k == 1
    assert routed.rows.dtype == F32 and torch.equal(routed.rows, sums * 4)
    assert routed.src2dst.dtype == I32
    assert torch.equal(routed.src2dst, torch.arange(5, dtype=I32))


def _native_config():
    return Inkling2QuantConfig(
        native_quant_dict(
            "mxfp8", [f"{EXPERT_PREFIX}{leaf}_weight" for leaf in ("w13", "w2")]
        )
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
