# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native MoVA FP8 admission and loading with synthetic CPU parameters."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.k2_horizon import K2HorizonConfig
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.xllm import (
    XllmForCausalLM,
    XllmGatedAttention,
    XllmMLP,
    XllmMoVAAttention,
    _normalize_k2_horizon_config,
    _validate_mova_config,
)
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _public_metadata():
    # IFM/K2-Horizon-MoVA-36B-A4B-FP8 at feffd71999eb06bfa2fbb8ad220059b694077457.
    path = Path(__file__).parent / "fixtures/k2_horizon_mova_fp8_config.json"
    return json.loads(path.read_text())


def _small_config():
    metadata = _public_metadata()
    metadata.update(
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        rope_head_dim=64,
        vocab_size=32,
        max_position_embeddings=32,
        num_experts=4,
        num_experts_per_tok=2,
        mova_num_experts=4,
        mova_num_experts_per_tok=2,
        mlp_only_layers=[0],
        # This value is synthetic. It is not public checkpoint provenance.
        xllm_source_router_gemm_partitions=2,
    )
    ignored = []
    for name in metadata["quantization_config"]["ignored_layers"]:
        if ".layers." not in name or name.startswith("model.layers.0."):
            ignored.append(name)
        elif name.startswith("model.layers.3."):
            if ".v_experts." not in name or int(name.rsplit(".", 1)[1]) < 4:
                ignored.append(name.replace("model.layers.3.", "model.layers.1."))
    metadata["quantization_config"]["ignored_layers"] = ignored
    config = K2HorizonConfig.from_dict(metadata)
    _normalize_k2_horizon_config(config)
    return config


def _quantization(config):
    metadata = copy.deepcopy(config.quantization_config)
    metadata["packed_modules_mapping"] = XllmForCausalLM.packed_modules_mapping
    return Fp8Config.from_config(metadata)


@pytest.fixture(autouse=True)
def native_cpu_context(monkeypatch):
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with (
            get_context().override_server_args(
                enable_eplb=False,
                init_expert_location="trivial",
                ep_num_redundant_experts=0,
                enable_two_batch_overlap=False,
                moe_runner_backend="triton",
                moe_a2a_backend="none",
            ),
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                moe_ep_size=1,
                moe_ep_rank=0,
            ),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.TRITON,
                a2a_backend=MoeA2ABackend.NONE,
            ),
        ):
            yield
    finally:
        torch.set_default_dtype(previous)


def test_public_metadata_requires_source_router_provenance():
    config = K2HorizonConfig.from_dict(_public_metadata())
    with pytest.raises(ValueError, match="source router GEMM provenance"):
        _normalize_k2_horizon_config(config)


def test_public_precision_partition_with_explicitly_synthetic_provenance():
    metadata = _public_metadata()
    metadata["xllm_source_router_gemm_partitions"] = 2
    config = K2HorizonConfig.from_dict(metadata)
    _normalize_k2_horizon_config(config)
    _validate_mova_config(config=config, quant_config=_quantization(config))


@pytest.mark.parametrize("prefix", ["model.", ""])
def test_native_mova_accepts_serialized_expert_only_fp8(prefix):
    config = _small_config()
    config.quantization_config["ignored_layers"] = [
        prefix + name.removeprefix("model.")
        for name in config.quantization_config["ignored_layers"]
    ]
    _validate_mova_config(config=config, quant_config=_quantization(config))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("is_checkpoint_fp8_serialized", False),
        ("activation_scheme", "static"),
        ("weight_block_size", None),
        ("weight_block_size", [64, 128]),
        ("weight_block_size", [128, 64]),
        ("use_mxfp8", True),
        ("is_fp4_experts", True),
        ("dequant_fp4_to_fp8", True),
    ],
)
def test_native_mova_rejects_incompatible_fp8_method(field, value):
    config = _small_config()
    quantization = _quantization(config)
    setattr(quantization, field, value)
    with pytest.raises(ValueError):
        _validate_mova_config(config=config, quant_config=quantization)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("activation_scheme", "static"),
        ("activation_scheme", None),
        ("activation_scheme", ["dynamic"]),
        ("weight_block_size", [64, 128]),
        ("weight_block_size", None),
        ("weight_block_size", "128,128"),
        ("weight_block_size", (128, 128)),
        ("weight_block_size", [128.0, 128]),
    ],
)
def test_native_mova_declared_precision_regression(field, value):
    config = _small_config()
    quantization = _quantization(config)
    config.quantization_config[field] = value
    with pytest.raises(ValueError):
        _validate_mova_config(config=config, quant_config=quantization)


@pytest.mark.parametrize(
    ("join_mode", "storage_size"),
    [("scale", 3), ("scale", 2), ("scale", None), ("recover", 2)],
)
def test_native_mova_elastic_join_regression(join_mode, storage_size):
    config = _small_config()
    quantization = _quantization(config)
    with (
        get_context().override_server_args(
            ep_join_mode=join_mode, elastic_ep_initial_size=storage_size
        ),
        get_parallel().override(moe_ep_size=2),
    ):
        assert get_parallel().elastic_ep_initial_size == storage_size
        with pytest.raises(ValueError):
            _validate_mova_config(config=config, quant_config=quantization)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_native_mova_rejects_non_bf16_runtime(dtype):
    config = _small_config()
    torch.set_default_dtype(dtype)
    with pytest.raises(ValueError):
        _validate_mova_config(config=config, quant_config=_quantization(config))


@pytest.mark.parametrize(
    "missing",
    [
        "lm_head",
        "model.embed_tokens",
        "model.norm",
        "model.layers.0.input_layernorm",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.self_attn.gate_proj",
        "model.layers.1.self_attn.v_router",
        "model.layers.1.self_attn.v_experts.3",
        "model.layers.1.mlp.gate",
        "model.layers.1.mlp.shared_experts.up_proj",
        "model.layers.1.mlp.shared_experts.down_proj",
    ],
)
def test_native_mova_rejects_missing_source_precision_exclusion(missing):
    config = _small_config()
    config.quantization_config["ignored_layers"].remove(missing)
    with pytest.raises(ValueError):
        _validate_mova_config(config=config, quant_config=_quantization(config))


@pytest.mark.parametrize(
    "extra",
    [
        "model.layers.1.mlp.experts",
        "model.layers.1.mlp.experts.0.gate_proj",
        "model.layers.1.mlp.experts.0.up_proj",
        "model.layers.1.mlp.experts.3.down_proj",
    ],
)
@pytest.mark.parametrize("source", ["metadata", "environment"])
def test_native_mova_rejects_excluded_routed_experts(monkeypatch, extra, source):
    config = _small_config()
    if source == "metadata":
        config.quantization_config["ignored_layers"].append(extra)
    else:
        monkeypatch.setenv("SGLANG_FP8_IGNORED_LAYERS", extra)
    with pytest.raises(ValueError):
        _validate_mova_config(config=config, quant_config=_quantization(config))


@pytest.mark.parametrize(
    ("moe_tp_size", "moe_ep_size"), [(1, 1), (2, 1), (1, 2), (1, 4)]
)
def test_native_mova_accepts_complete_block_and_expert_partitions(
    moe_tp_size, moe_ep_size
):
    config = _small_config()
    with get_parallel().override(moe_tp_size=moe_tp_size, moe_ep_size=moe_ep_size):
        _validate_mova_config(config=config, quant_config=_quantization(config))


@pytest.mark.parametrize(
    ("intermediate_size", "moe_tp_size", "moe_ep_size"),
    [(256, 4, 1), (768, 8, 1), (256, 1, 3)],
)
def test_native_mova_rejects_partial_block_or_expert_partitions(
    intermediate_size, moe_tp_size, moe_ep_size
):
    config = _small_config()
    config.moe_intermediate_size = intermediate_size
    with (
        get_parallel().override(moe_tp_size=moe_tp_size, moe_ep_size=moe_ep_size),
        pytest.raises(ValueError),
    ):
        _validate_mova_config(config=config, quant_config=_quantization(config))


@pytest.mark.parametrize("layer_id", [0, 1])
@pytest.mark.parametrize("fp8", [False, True])
def test_attention_constructors_keep_source_precision(monkeypatch, layer_id, fp8):
    from sglang.srt.layers.rotary_embedding import base, factory

    config = _small_config()
    attention_class = XllmGatedAttention if layer_id == 0 else XllmMoVAAttention
    # Select native RoPE construction without the optional vLLM kernel import.
    # No kernel runs here. Keep the generic parameter-loader flags unchanged.
    with monkeypatch.context() as rope_context:
        rope_context.setattr(base, "_is_cpu", True)
        rope_context.setattr(factory, "_ROPE_DICT", {})
        attention = attention_class(
            config=config,
            layer_id=layer_id,
            quant_config=_quantization(config) if fp8 else None,
            prefix=f"model.layers.{layer_id}.self_attn",
        )
    for name in ("q_proj", "k_proj", "gate_proj", "o_proj"):
        projection = getattr(attention, name)
        assert isinstance(projection.quant_method, UnquantizedLinearMethod)
        assert projection.weight.dtype == torch.bfloat16
    if layer_id == 0:
        assert attention.v_proj.weight.shape == (128, 256)
    else:
        assert attention.v_experts.weight.shape == (4, 128, 256)
        assert attention.v_experts.weight.dtype == torch.bfloat16
        assert attention.v_router.bias.dtype == (
            torch.bfloat16 if fp8 else torch.float32
        )


@pytest.mark.parametrize("attention_class", [XllmGatedAttention, XllmMoVAAttention])
def test_direct_attention_rejects_incompatible_precision_partition(attention_class):
    config = _small_config()
    config.quantization_config["ignored_layers"].remove(
        "model.layers.1.self_attn.q_proj"
    )
    with pytest.raises(ValueError):
        attention_class(
            config=config,
            layer_id=0 if attention_class is XllmGatedAttention else 1,
            quant_config=_quantization(config),
            prefix="model.layers.1.self_attn",
        )


@pytest.mark.parametrize("attention_class", [XllmGatedAttention, XllmMoVAAttention])
def test_native_mova_direct_provenance_regression(monkeypatch, attention_class):
    config = _small_config()
    quantization = _quantization(config)
    del config.xllm_source_router_gemm_partitions
    assert config._sglang_xllm_checkpoint_format == "k2_horizon_hf"
    assert not hasattr(config, "xllm_source_router_gemm_partitions")

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("Missing router provenance reached parameter allocation")

    monkeypatch.setattr(torch, "empty", unexpected_allocation)
    with pytest.raises(ValueError):
        attention_class(
            config=config,
            layer_id=0 if attention_class is XllmGatedAttention else 1,
            quant_config=quantization,
            prefix="model.layers.1.self_attn",
        )


def test_fixture_selects_real_unquantized_linears_and_fp8_experts():
    config = _small_config()
    quantization = _quantization(config)
    for prefix in ("model.layers.0.mlp", "model.layers.1.mlp.shared_experts"):
        mlp = XllmMLP(
            hidden_size=256,
            intermediate_size=256,
            hidden_act="silu",
            quant_config=quantization,
            prefix=prefix,
            tp_rank=0,
            tp_size=1,
        )
        for projection in (mlp.gate_up_proj, mlp.down_proj):
            assert isinstance(projection.quant_method, UnquantizedLinearMethod)
            assert projection.weight.dtype == torch.bfloat16
    experts = object.__new__(FusedMoE)
    torch.nn.Module.__init__(experts)
    method = quantization.get_quant_method(
        layer=experts, prefix="model.layers.1.mlp.experts"
    )
    assert isinstance(method, Fp8MoEMethod)
    assert method.block_quant


def _source_tensor_inventory(config):
    shapes = {
        "model.embed_tokens.weight": (config.vocab_size, config.hidden_size),
        "lm_head.weight": (config.vocab_size, config.hidden_size),
        "model.norm.weight": (config.hidden_size,),
    }
    hidden = config.hidden_size
    query = config.num_attention_heads * config.head_dim
    value = config.num_key_value_heads * config.head_dim
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        for name in ("input_layernorm", "post_attention_layernorm"):
            shapes[f"{prefix}.{name}.weight"] = (hidden,)
        for name, shape in (
            ("q_proj", (query, hidden)),
            ("k_proj", (value, hidden)),
            ("o_proj", (hidden, query)),
            ("gate_proj", (query, hidden)),
        ):
            shapes[f"{prefix}.self_attn.{name}.weight"] = shape
        dense = layer < config.num_dense_layers
        if dense:
            shapes[f"{prefix}.self_attn.v_proj.weight"] = (value, hidden)
        else:
            for expert in range(config.num_values):
                shapes[f"{prefix}.self_attn.v_experts.{expert}.weight"] = (
                    value,
                    hidden,
                )
            for router, count in (
                ("self_attn.v_router", config.num_values),
                ("mlp.gate", config.num_experts),
            ):
                shapes[f"{prefix}.{router}.weight"] = (count, hidden)
                shapes[f"{prefix}.{router}.bias"] = (count,)
        mlp = f"{prefix}.mlp" + ("" if dense else ".shared_experts")
        width = (
            config.intermediate_size
            if dense
            else config.moe_intermediate_size * config.num_shared_experts
        )
        for projection in ("gate_proj", "up_proj", "down_proj"):
            shapes[f"{mlp}.{projection}.weight"] = (
                (hidden, width) if projection == "down_proj" else (width, hidden)
            )
        if not dense:
            for expert in range(config.num_experts):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    name = f"{prefix}.mlp.experts.{expert}.{projection}"
                    shape = (
                        (hidden, config.moe_intermediate_size)
                        if projection == "down_proj"
                        else (config.moe_intermediate_size, hidden)
                    )
                    shapes[f"{name}.weight"] = shape
                    shapes[f"{name}.weight_scale_inv"] = tuple(
                        size // 128 for size in shape
                    )
    tensors = {}
    for index, (name, shape) in enumerate(shapes.items()):
        dtype = torch.float32 if name.endswith("weight_scale_inv") else torch.bfloat16
        if ".mlp.experts." in name and name.endswith(".weight"):
            dtype = torch.float8_e4m3fn
        values = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(
            shape
        )
        tensors[name] = ((values % 7) + (index % 5) + 1).to(dtype)
    return tensors


@pytest.fixture
def native_loader_model(monkeypatch):
    from sglang.srt.layers import dp_attention
    from sglang.srt.layers.rotary_embedding import base, factory
    from sglang.srt.models import xllm

    def build(tp=1, tp_rank=0, ep=1, ep_rank=0, pp=1, pp_rank=0, fp8=True):
        config = _small_config()
        quantization = _quantization(config) if fp8 else None
        group = SimpleNamespace(
            world_size=pp,
            rank_in_group=pp_rank,
            is_first_rank=pp_rank == 0,
            is_last_rank=pp_rank == pp - 1,
        )
        # Group getters supply metadata only. No collective or model forward runs.
        with (
            monkeypatch.context() as context,
            get_context().override_server_args(enable_dp_lm_head=True),
            get_parallel().override(
                tp_size=tp,
                tp_rank=tp_rank,
                attn_tp_size=tp,
                attn_tp_rank=tp_rank,
                moe_tp_size=tp,
                moe_tp_rank=tp_rank,
                moe_ep_size=ep,
                moe_ep_rank=ep_rank,
                attn_dp_size=1,
                attn_dp_rank=0,
                attn_cp_size=1,
                attn_cp_rank=0,
                moe_dp_size=1,
            ),
        ):
            context.setattr(xllm, "get_pp_group", lambda: group)
            context.setattr(
                dp_attention, "_get_moe_dp_group", lambda: SimpleNamespace(world_size=1)
            )
            context.setattr(base, "_is_cpu", True)
            context.setattr(factory, "_ROPE_DICT", {})
            model = XllmForCausalLM(config, quant_config=quantization)
        tensors = _source_tensor_inventory(config)
        if not fp8:
            tensors = {
                name: tensor.to(torch.bfloat16)
                for name, tensor in tensors.items()
                if not name.endswith("weight_scale_inv")
            }
        assert sum(t.numel() * t.element_size() for t in tensors.values()) < 16 << 20
        return model, tensors

    return build


def _dequantize_blocks(weight, scales):
    return weight.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)


@pytest.mark.parametrize(
    ("tp", "rank", "ep", "ep_rank", "reverse"),
    [(1, 0, 1, 0, False), (1, 0, 1, 0, True), (2, 0, 2, 0, False), (2, 1, 2, 1, True)],
)
def test_native_loader_complete_storage_slices(
    native_loader_model, tp, rank, ep, ep_rank, reverse
):
    model, tensors = native_loader_model(tp=tp, tp_rank=rank, ep=ep, ep_rank=ep_rank)
    model.load_weights(reversed(tuple(tensors.items())) if reverse else tensors.items())
    block = model.model.layers[1].mlp
    experts = block.experts
    assert isinstance(experts, FusedMoE)
    assert isinstance(experts.quant_method, Fp8MoEMethod)
    width = model.config.moe_intermediate_size // tp
    for local in range(model.config.num_experts // ep):
        global_id = ep_rank * (model.config.num_experts // ep) + local
        for projection, offset in (
            ("gate_proj", 0),
            ("up_proj", width),
            ("down_proj", 0),
        ):
            name = f"model.layers.1.mlp.experts.{global_id}.{projection}"
            expected = _dequantize_blocks(
                tensors[f"{name}.weight"], tensors[f"{name}.weight_scale_inv"]
            )
            if projection == "down_proj":
                actual = _dequantize_blocks(
                    experts.w2_weight[local], experts.w2_weight_scale_inv[local]
                )
                expected = expected[:, rank * width : (rank + 1) * width]
            else:
                actual = _dequantize_blocks(
                    experts.w13_weight[local, offset : offset + width],
                    experts.w13_weight_scale_inv[
                        local, offset // 128 : (offset + width) // 128
                    ],
                )
                expected = expected[rank * width : (rank + 1) * width]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    values = model.model.layers[1].self_attn.v_experts.weight
    local_width = 128 // tp
    for expert in range(model.config.num_values):
        expected = tensors[f"model.layers.1.self_attn.v_experts.{expert}.weight"]
        torch.testing.assert_close(
            values[expert], expected[rank * local_width : (rank + 1) * local_width]
        )
    assert not any(
        "scale" in name
        for name, _ in model.model.layers[1].mlp.shared_experts.named_parameters()
    )


@pytest.mark.parametrize("operation", ["missing", "duplicate"])
@pytest.mark.parametrize(
    "name",
    [
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.experts.0.up_proj.weight",
        "model.layers.1.mlp.experts.0.down_proj.weight",
        "model.layers.1.mlp.experts.0.gate_proj.weight_scale_inv",
        "model.layers.1.mlp.experts.0.up_proj.weight_scale_inv",
        "model.layers.1.mlp.experts.0.down_proj.weight_scale_inv",
        "model.layers.1.self_attn.v_experts.3.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.gate.bias",
        "model.layers.1.self_attn.v_router.bias",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ],
)
def test_native_loader_requires_complete_unique_inventory(
    native_loader_model, operation, name
):
    model, tensors = native_loader_model()
    items = [
        (key, value)
        for key, value in tensors.items()
        if operation != "missing" or key != name
    ]
    if operation == "duplicate":
        items.append((name, tensors[name]))
    with pytest.raises((ValueError, RuntimeError)):
        model.load_weights(items)


@pytest.mark.parametrize(
    "mutation",
    [
        "weight-dtype",
        "weight-shape",
        "scale-dtype",
        "scale-shape",
        "negative",
        "zero",
        "nan",
        "inf",
        "attention-dtype",
        "bias-dtype",
        "excluded-scale",
        "unknown",
    ],
)
def test_native_loader_rejects_invalid_source_tensor(native_loader_model, mutation):
    model, tensors = native_loader_model()
    name = "model.layers.1.mlp.experts.0.gate_proj."
    name += "weight" if mutation.startswith("weight") else "weight_scale_inv"
    if mutation == "attention-dtype":
        name = "model.layers.1.self_attn.q_proj.weight"
    elif mutation == "bias-dtype":
        name = "model.layers.1.mlp.gate.bias"
    elif mutation == "excluded-scale":
        name = "model.layers.1.mlp.shared_experts.gate_proj.weight_scale_inv"
    elif mutation == "unknown":
        name = "model.layers.1.unknown.weight"
    candidate = tensors.get(name, torch.ones((2, 2))).clone()
    if mutation.endswith("dtype"):
        dtype = {
            "weight-dtype": torch.bfloat16,
            "scale-dtype": torch.float16,
            "attention-dtype": torch.float8_e4m3fn,
            "bias-dtype": torch.float32,
        }[mutation]
        candidate = candidate.to(dtype)
    elif mutation.endswith("shape"):
        candidate = candidate[:1]
    elif mutation in {"negative", "zero", "nan", "inf"}:
        candidate.fill_(
            {"negative": -1, "zero": 0, "nan": float("nan"), "inf": float("inf")}[
                mutation
            ]
        )
    before = {
        key: parameter.detach().clone() for key, parameter in model.named_parameters()
    }
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        model.load_weights(
            [
                (name, candidate),
                *((key, value) for key, value in tensors.items() if key != name),
            ]
        )
    for key, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.float(), before[key].float(), rtol=0, atol=0, equal_nan=True
        )


@pytest.mark.parametrize("first_load", ["complete", "incomplete", "iterator-error"])
def test_native_loader_rejects_second_attempt_before_iterator(
    native_loader_model, first_load
):
    model, tensors = native_loader_model()

    def interrupted():
        yield next(iter(tensors.items()))
        raise RuntimeError("synthetic source iterator failure")

    first = tensors.items() if first_load == "complete" else []
    if first_load == "iterator-error":
        first = interrupted()
    try:
        model.load_weights(first)
    except (ValueError, RuntimeError):
        if first_load == "complete":
            raise

    def forbidden():
        pytest.fail("A second load consumed its source iterator")
        yield

    with pytest.raises((ValueError, RuntimeError)):
        model.load_weights(forbidden())


@pytest.mark.parametrize("pp_rank", [0, 1])
def test_native_loader_uses_owned_pipeline_and_expert_inventory(
    native_loader_model, pp_rank
):
    model, tensors = native_loader_model(ep=2, ep_rank=1, pp=2, pp_rank=pp_rank)
    selected = {
        name: tensor
        for name, tensor in tensors.items()
        if not any(f".mlp.experts.{expert}." in name for expert in (0, 1))
        and (
            name.startswith(f"model.layers.{pp_rank}.")
            or name
            in (
                ("model.embed_tokens.weight",)
                if pp_rank == 0
                else ("model.norm.weight", "lm_head.weight")
            )
        )
    }
    model.load_weights(selected.items())
    if pp_rank == 1:
        values = model.model.layers[1].self_attn.v_experts.weight
        for expert in range(4):
            torch.testing.assert_close(
                values[expert],
                tensors[f"model.layers.1.self_attn.v_experts.{expert}.weight"],
            )
    else:
        torch.testing.assert_close(
            model.model.embed_tokens.weight[:32], tensors["model.embed_tokens.weight"]
        )


def test_native_loader_preserves_legacy_repeated_loads(native_loader_model):
    model, tensors = native_loader_model(fp8=False)
    model.load_weights(tensors.items())
    tensors["model.norm.weight"] = torch.zeros(256, dtype=torch.bfloat16)
    model.load_weights(tensors.items())
    torch.testing.assert_close(model.model.norm.weight, tensors["model.norm.weight"])


def test_native_loader_router_bias_preserves_source_and_derived_values(
    native_loader_model,
):
    from sglang.srt.layers.mova import mova_router_topk

    derived = []
    for offset in (0, 1):
        model, tensors = native_loader_model()
        bias = torch.tensor([0, 1, -1, 2], dtype=torch.bfloat16).roll(offset)
        tensors["model.layers.1.mlp.gate.bias"] = bias
        tensors["model.layers.1.self_attn.v_router.bias"] = bias.clone()
        model.load_weights(tensors.items())
        mlp, attention = model.model.layers[1].mlp, model.model.layers[1].self_attn
        assert mlp.gate.bias.dtype == attention.v_router.bias.dtype == torch.bfloat16
        source = mlp.gate.bias.detach().clone()
        correction = mlp.topk.topk_config.correction_bias_for_dtype(torch.float32)
        derived.append(correction)
        torch.testing.assert_close(correction, bias.float())
        logits = torch.tensor([[0.1, 0.3, -0.2, 0.4]], dtype=torch.float32)
        weights, ids = mova_router_topk(
            logits,
            attention.v_router.bias,
            score_func="sigmoid",
            top_k=2,
            scaling_factor=1.5,
            renormalize=True,
        )
        scores = logits.sigmoid()
        expected_ids = (scores + bias.float()).topk(2).indices
        expected_weights = scores.gather(1, expected_ids)
        expected_weights = (
            expected_weights / expected_weights.sum(-1, keepdim=True) * 1.5
        )
        torch.testing.assert_close(ids, expected_ids.int())
        torch.testing.assert_close(weights, expected_weights)
        output = mlp.topk.forward_native(torch.zeros((1, 256)), logits)
        torch.testing.assert_close(output.topk_ids, expected_ids)
        torch.testing.assert_close(output.topk_weights, expected_weights / 1.5)
        torch.testing.assert_close(mlp.gate.bias, source)
        assert mlp.gate.bias.dtype == torch.bfloat16
    assert derived[0].data_ptr() != derived[1].data_ptr()
    assert not torch.equal(derived[0], derived[1])


def test_native_mova_selects_existing_deepep_implementation():
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, get_moe_impl_class

    quantization = _quantization(_small_config())
    with get_flags().moe.override(a2a_backend=MoeA2ABackend.DEEPEP):
        assert get_moe_impl_class(quantization) is DeepEPMoE


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
