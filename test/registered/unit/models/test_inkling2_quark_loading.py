"""The Quark experts of Inkling-2: loading the packed tensors of an export, the MoE
layer around them, and the real model built from the config.json of an export (CPU only).
"""

import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load, save
from torch import nn

from sglang.srt.layers.quantization.inkling2 import Inkling2QuantConfig
from sglang.srt.layers.quantization.inkling2_quark import Inkling2QuarkLayout
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.inkling2 import Inkling2ForConditionalGeneration
from sglang.srt.models.inkling_common import inkling2_moe as moe
from sglang.srt.models.inkling_common import inkling2_moe_quark as quark_moe
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.inkling2_quark_test_utils import (
    EXPERT_FORMAT,
    MINI_CONFIG,
    VARIANTS,
    Materialization,
    expert_checkpoint_tensors,
    has_fp8_attention,
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


# the modules that provide forward_experts for each expert format
FORWARD_MODULES = {
    "mxfp4": "sglang.srt.models.inkling_common.inkling2_moe_quark_mxfp4",
    "uint4_awq": "sglang.srt.models.inkling_common.inkling2_moe_quark_uint4",
}


def _kernels(monkeypatch, expert_format, forward_experts):
    name = FORWARD_MODULES[expert_format]
    module = types.ModuleType(name)
    module.forward_experts = forward_experts
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("variant", ["MXFP4-RTN", "UINT4-AWQ"])
def test_moe_forward_routes_unquantized_input(monkeypatch, variant):
    monkeypatch.setattr(moe, "get_parallel", _parallel())
    monkeypatch.setattr(quark_moe, "get_parallel", _parallel())
    layer = moe.Inkling2MoE(_config(), 0, "llm.layers.0.mlp", _layout(variant))
    experts = layer.experts
    other = "uint4_awq" if experts.expert_format == "mxfp4" else "mxfp4"
    calls, routed = [], []

    def forward_experts(module, x, weights, ids, combine):
        calls.append((module, x, weights, ids, combine))
        return f"kernel output, combine={combine}"

    def route(x):
        routed.append(x)
        return "weights", "ids"

    def wrong_format(*args):
        raise AssertionError("dispatched to the kernels of the other format")

    _kernels(monkeypatch, experts.expert_format, forward_experts)
    _kernels(monkeypatch, other, wrong_format)
    monkeypatch.setattr(layer.gate, "forward", route)
    x = torch.randn(3, HIDDEN)
    for combine in (True, False):
        assert layer(x, combine=combine) == f"kernel output, combine={combine}"
        assert routed[-1] is x
        module, seen_x, *rest = calls[-1]
        assert module is experts and seen_x is x and rest == ["weights", "ids", combine]
    # a prefill step on this rank reads every input row
    assert layer.local_input_rows(4096, combine=True) is None


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


def _poison_native_experts(monkeypatch):
    """Make every method of Inkling2Experts and of the HIP MoE raise: the native experts
    read uint8 weights as NVFP4, and MXFP4 weights are uint8 too."""

    def boom_for(name):
        def boom(*args, **kwargs):
            raise AssertionError(f"the Quark path reached native expert code: {name}")

        return boom

    for method, value in list(vars(moe.Inkling2Experts).items()):
        if callable(value):
            monkeypatch.setattr(moe.Inkling2Experts, method, boom_for(method))
    name = "sglang.srt.models.inkling_common.inkling2_moe_hip"
    hip = types.ModuleType(name)
    for function in ("moe_forward", "forward_experts", "prepare_experts"):
        setattr(hip, function, boom_for(f"inkling2_moe_hip.{function}"))
    monkeypatch.setitem(sys.modules, name, hip)


@pytest.mark.parametrize("variant", ["MXFP4-RTN", "UINT4-AWQ"])
def test_the_quark_path_never_reaches_native_expert_code(monkeypatch, variant):
    _poison_native_experts(monkeypatch)
    monkeypatch.setattr(moe, "get_parallel", _parallel())
    monkeypatch.setattr(quark_moe, "get_parallel", _parallel())
    with pytest.raises(AssertionError, match="native expert code"):
        moe.Inkling2MoE(_config(), 0, "llm.layers.0.mlp", _native_config())
    layer = moe.Inkling2MoE(_config(), 0, "llm.layers.0.mlp", _layout(variant))
    _kernels(monkeypatch, layer.experts.expert_format, lambda *args: "kernel output")
    monkeypatch.setattr(layer.gate, "forward", lambda x: ("weights", "ids"))
    for combine in (True, False):
        assert layer(torch.randn(2, HIDDEN), combine=combine) == "kernel output"
    _target(monkeypatch, variant).load_weights(list(_export(variant).items()))


# the experts of one layer of Inkling-2-Mini: 128 experts, hidden 2304, intermediate 1152
REAL_EXPERT_PARAMETERS = {
    "mxfp4": {
        "w13_weight": ((128, 2304, 1152), U8),
        "w13_scale": ((128, 2304, 72), U8),
        "w2_weight": ((128, 2304, 576), U8),
        "w2_scale": ((128, 2304, 36), U8),
    },
    "uint4_awq": {
        "w13_weight": ((128, 2304, 288), I32),
        "w13_scale": ((128, 18, 2304), F32),
        "w13_weight_zero_point": ((128, 18, 288), I32),
        "w2_weight": ((128, 1152, 288), I32),
        "w2_scale": ((128, 9, 2304), F32),
        "w2_weight_zero_point": ((128, 9, 288), I32),
    },
}


@pytest.fixture
def published_context():
    from sglang.srt.runtime_context import reset_context
    from sglang.srt.server_args import ServerArgs
    from sglang.test.parallel_groups import publish

    publish(ServerArgs(model_path="dummy"), role="test")
    yield
    reset_context()


@pytest.mark.usefixtures("published_context")
@pytest.mark.parametrize("variant", VARIANTS)
def test_config_json_to_layout_to_the_real_model(tmp_path, variant):
    """config.json of an export -> generic Quark config -> layout -> the model on the
    meta device, with the packed experts in every layer."""
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_loader.loader import _get_quantization_config
    from sglang.srt.model_loader.weight_utils import get_quant_config
    from sglang.srt.models.inkling2 import Inkling2LoopForConditionalGeneration
    from sglang.test.parallel_groups import parallel_scope

    config = {**MINI_CONFIG, "quantization_config": quantization_config(variant)}
    (tmp_path / "config.json").write_text(json.dumps(config))
    model_config = ModelConfig(str(tmp_path), trust_remote_code=False)
    assert model_config.quantization == "quark"
    generic = get_quant_config(model_config, LoadConfig(), packed_modules_mapping={})
    pipeline = SimpleNamespace(world_size=1, rank_in_group=0)
    with parallel_scope(pp_group=pipeline), torch.device("meta"):
        with set_default_torch_dtype(BF16):
            model = Inkling2LoopForConditionalGeneration(
                model_config.hf_config, generic
            )
    layout = model.quant_config
    assert isinstance(layout, Inkling2QuarkLayout)
    assert layout.expert_format == EXPERT_FORMAT[variant]
    assert model.llm.logits_processor.use_fp32_lm_head
    expected = REAL_EXPERT_PARAMETERS[layout.expert_format]
    assert len(model.llm.layers) == 24
    for layer in model.llm.layers:
        assert isinstance(layer.mlp.experts, quark_moe.Inkling2QuarkExperts)
        assert {
            name: (tuple(p.shape), p.dtype)
            for name, p in layer.mlp.experts.named_parameters()
        } == expected
        if has_fp8_attention(variant):
            assert layer.attn.qkvr["fp8"].weight.dtype == E4M3
            assert layer.attn.qkvr["bf16"].weight.dtype == BF16
            assert layer.attn.wo_ud.weight.dtype == E4M3
        else:
            assert layer.attn.qkvr.weight.dtype == BF16
            assert layer.attn.wo_ud.weight.dtype == BF16
    # the bundled MTP head is BF16: the ModelConfig of the draft has no quantization
    draft = ModelConfig(
        str(tmp_path),
        trust_remote_code=False,
        is_draft_model=True,
        speculative_algorithm="DSPARK",
    )
    assert draft.quantization is None
    assert _get_quantization_config(draft, LoadConfig()) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
