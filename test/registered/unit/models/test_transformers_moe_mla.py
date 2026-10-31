# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import gc
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from transformers import DeepseekV3Config, Qwen2MoeConfig, Qwen3MoeConfig
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.models.transformers.execution_context import (
    get_execution_layer,
    get_transformers_execution_context,
    register_execution_layer,
    transformers_execution_context,
)
from sglang.srt.models.transformers.mla import (
    absorb_mla_query,
    expand_mla_output,
    install_mla_adapters,
    refresh_mla_weights,
    split_mla_projection,
)
from sglang.srt.models.transformers.moe import (
    MoEMixin,
    TransformersFusedMoE,
    TransformersNativeMoE,
    native_router_contract,
    native_routing_method_type,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def test_execution_registry_uses_unique_weak_handles():
    first, second = nn.Identity(), nn.Identity()
    first_handle = register_execution_layer(first)
    second_handle = register_execution_layer(second)
    assert first_handle != second_handle
    assert get_execution_layer(first_handle) is first
    del first
    gc.collect()
    with pytest.raises(RuntimeError, match="no longer alive"):
        get_execution_layer(first_handle)
    assert get_execution_layer(second_handle) is second


def test_execution_context_resets_after_exception():
    outer, inner = object(), object()
    with transformers_execution_context(outer):
        with pytest.raises(ValueError):
            with transformers_execution_context(inner):
                assert get_transformers_execution_context().forward_batch is inner
                raise ValueError("inner")
        assert get_transformers_execution_context().forward_batch is outer
    with pytest.raises(RuntimeError, match="requires an execution context"):
        get_transformers_execution_context()


def test_execution_context_is_task_local():
    async def run(value):
        with transformers_execution_context(value):
            await asyncio.sleep(0)
            return get_transformers_execution_context().forward_batch

    async def concurrent():
        return await asyncio.gather(run("first"), run("second"))

    assert asyncio.run(concurrent()) == ["first", "second"]


@pytest.mark.parametrize("causal", [False, True])
def test_latent_attention_equals_expanded_attention(causal):
    torch.manual_seed(37)
    tokens, heads, nope, rope, latent, values = 7, 3, 4, 2, 5, 6
    projection = torch.randn(heads * (nope + values), latent, dtype=torch.float64)
    compressed = torch.randn(tokens, latent, dtype=torch.float64)
    q_nope = torch.randn(tokens, heads, nope, dtype=torch.float64)
    q_rope = torch.randn(tokens, heads, rope, dtype=torch.float64)
    k_rope = torch.randn(tokens, 1, rope, dtype=torch.float64)
    w_kc, w_vc = split_mla_projection(projection, heads, nope, values)
    q_latent = absorb_mla_query(q_nope, w_kc)
    expanded = torch.nn.functional.linear(compressed, projection).reshape(
        tokens, heads, nope + values
    )
    k_nope, v = expanded.split((nope, values), dim=-1)
    q = torch.cat((q_nope, q_rope), dim=-1).transpose(0, 1)
    k = torch.cat((k_nope, k_rope.expand(-1, heads, -1)), dim=-1).transpose(0, 1)
    compressed_q = torch.cat((q_latent, q_rope), dim=-1).transpose(0, 1)
    compressed_k = torch.cat((compressed[:, None, :], k_rope), dim=-1).transpose(0, 1)
    scale = (nope + rope) ** -0.5
    full_output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v.transpose(0, 1), is_causal=causal, scale=scale
    )
    latent_output = torch.nn.functional.scaled_dot_product_attention(
        compressed_q,
        compressed_k,
        compressed[None, :, :],
        is_causal=causal,
        scale=scale,
    )
    actual = expand_mla_output(latent_output.transpose(0, 1), w_vc)
    torch.testing.assert_close(
        actual, full_output.transpose(0, 1), rtol=1e-12, atol=1e-12
    )


def test_mla_installation_and_in_place_weight_refresh():
    config = DeepseekV3Config(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        kv_lora_rank=5,
        v_head_dim=3,
        q_lora_rank=None,
    )
    source = DeepseekV3Attention(config, 0)
    model = nn.ModuleDict({"attention": source})
    instances = nn.ModuleDict({"0": RadixAttention(2, 6, 6**-0.5, 2, 0)})
    assert install_mla_adapters(model, instances) == 1
    compressed = torch.randn(1, 1, 3, 5)
    rope = torch.randn(1, 1, 3, 2)
    assert source.expand_kv(compressed, rope) == (compressed, rope)
    refresh_mla_weights(instances)
    adapter = instances["0"]
    pointer = adapter.w_kc.data_ptr()
    before = adapter.w_kc.clone()
    with torch.no_grad():
        source.kv_b_proj.weight.add_(1)
    refresh_mla_weights(instances)
    assert adapter.w_kc.data_ptr() == pointer
    torch.testing.assert_close(adapter.w_kc, before + 1)
    assert not any("w_kc" in name or "w_vc" in name for name in instances.state_dict())


def test_mla_rejects_shape_and_quantization_before_mutation():
    config = DeepseekV3Config(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        kv_lora_rank=5,
        v_head_dim=3,
        q_lora_rank=None,
    )
    source = DeepseekV3Attention(config, 0)
    model = nn.ModuleDict({"attention": source})
    instances = nn.ModuleDict({"0": RadixAttention(1, 6, 6**-0.5, 1, 0)})
    original_method = source.expand_kv.__func__
    with pytest.raises(ValueError, match="TP layout"):
        install_mla_adapters(model, instances)
    assert source.expand_kv.__func__ is original_method
    with pytest.raises(ValueError, match="unquantized"):
        install_mla_adapters(model, instances, quant_config=object())


class ReferenceExpertDispatch(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = experts

    def configure_router(self, gate, contract):
        self.gate = gate

    def forward_router(self, hidden_states, logits):
        expected_logits, weights, indices = self.gate(hidden_states)
        torch.testing.assert_close(logits, expected_logits)
        return self.experts(hidden_states, indices, weights)


@pytest.mark.parametrize("shared", [False, True])
def test_native_moe_preserves_hf_block_and_shared_expert_outputs(shared):
    torch.manual_seed(19)
    config_type = Qwen2MoeConfig if shared else Qwen3MoeConfig
    block_type = Qwen2MoeSparseMoeBlock if shared else Qwen3MoeSparseMoeBlock
    config = config_type(
        hidden_size=8,
        moe_intermediate_size=12,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    )
    original = block_type(config).eval()
    for parameter in original.parameters():
        nn.init.normal_(parameter, std=0.1)
    contract = native_router_contract(original)
    assert contract == ("qwen2" if shared else "qwen3")
    replacement = TransformersNativeMoE(
        original, ReferenceExpertDispatch(copy.deepcopy(original.experts)), contract
    )
    hidden_states = torch.randn(1, 7, 8)
    torch.testing.assert_close(replacement(hidden_states), original(hidden_states))
    original.extra = nn.Identity()
    assert native_router_contract(original) is None


def test_native_router_contract_maps_to_runner_routing_method():
    from sglang.srt.layers.moe.utils import RoutingMethodType

    def block(norm_topk_prob):
        config = Qwen3MoeConfig(
            hidden_size=8,
            moe_intermediate_size=12,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=norm_topk_prob,
        )
        return Qwen3MoeSparseMoeBlock(config)

    renormalized, plain = block(True), block(False)
    assert (
        native_routing_method_type(renormalized, "qwen3")
        == RoutingMethodType.Renormalize
    )
    assert native_routing_method_type(plain, "qwen3") == RoutingMethodType.Default
    assert (
        native_routing_method_type(renormalized, "deepseek_v3")
        == RoutingMethodType.DeepSeekV3
    )
    assert native_routing_method_type(renormalized, None) is None


def test_moe_custom_ops_are_non_mutating_and_have_fake_outputs():
    with FakeTensorMode():
        hidden = torch.empty(3, 8)
        selected = torch.empty(3, 2, dtype=torch.int32)
        scores = torch.empty(3, 2)
        output = torch.ops.sglang.transformers_moe_forward(
            hidden, selected, scores, "no-eager-lookup"
        )
        native = torch.ops.sglang.transformers_native_moe_forward(
            hidden, torch.empty(3, 4), "no-eager-lookup"
        )
        assert output.shape == native.shape == hidden.shape
    schema = str(torch.ops.sglang.transformers_moe_forward.default._schema)
    assert "!" not in schema


def test_packed_expert_weights_use_logical_expert_ids_and_shards():
    wrapper = TransformersFusedMoE.__new__(TransformersFusedMoE)
    nn.Module.__init__(wrapper)
    wrapper.num_experts = 3
    wrapper.layer_name = "model.layers.0.mlp.experts"
    wrapper._expert_mapping = []
    wrapper.experts = nn.Module()
    wrapper.experts.w13_weight = nn.Parameter(torch.empty(3, 8, 5))
    wrapper.experts.w2_weight = nn.Parameter(torch.empty(3, 5, 4))
    seen = []

    def load(parameter, weight, name, *, shard_id, expert_id):
        seen.append((name, shard_id, expert_id, weight.clone()))

    wrapper.experts.w13_weight.weight_loader = load
    wrapper.experts.w2_weight.weight_loader = load
    gate_up = torch.randn(3, 8, 5)
    down = torch.randn(3, 5, 4)
    assert wrapper.load_weights((("gate_up_proj", gate_up), ("down_proj", down))) == {
        "gate_up_proj",
        "down_proj",
        "experts.w13_weight",
        "experts.w2_weight",
    }
    wrapper.validate_loaded_weights()
    wrapper._loaded_expert_shards.remove((1, "w3"))
    with pytest.raises(ValueError, match="Incomplete expert weights"):
        wrapper.validate_loaded_weights()
    initial_model = SimpleNamespace(moe_layers=[wrapper], _weights_loaded=False)
    with pytest.raises(ValueError, match="Incomplete expert weights"):
        MoEMixin._validate_expert_weights(initial_model)
    initial_model._weights_loaded = True
    MoEMixin._validate_expert_weights(initial_model)
    assert [(shard, expert) for _, shard, expert, _ in seen] == [
        (shard, expert) for expert in range(3) for shard in ("w1", "w3")
    ] + [("w2", expert) for expert in range(3)]
    torch.testing.assert_close(seen[0][3], gate_up[0, :4])
    torch.testing.assert_close(seen[1][3], gate_up[0, 4:])
    with pytest.raises(ValueError, match="Unrecognized MoE weight"):
        wrapper.load_weights((("unknown", torch.empty(1)),))

    from sglang.srt.models.utils import AutoWeightsLoader

    parent = nn.Module()
    parent.mlp = wrapper
    actual = AutoWeightsLoader(parent).load_weights(
        (("mlp.gate_up_proj", gate_up), ("mlp.down_proj", down))
    )
    assert {"mlp.experts.w13_weight", "mlp.experts.w2_weight"} <= actual
    wrapper.validate_loaded_weights()


def test_hf_mla_forward_uses_compressed_native_attention_boundary():
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )
    from sglang.srt.models.transformers import attention as attention_interface

    class CPUReferenceAttention:
        def forward(self, q, k, v, layer, batch, save_kv_cache, *, q_rope, k_rope):
            assert k.shape[-1] == v.shape[-1] == 5
            query = torch.cat((q, q_rope), dim=-1).transpose(0, 1)
            key = torch.cat((k, k_rope), dim=-1).transpose(0, 1)
            value = v.transpose(0, 1)
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=True, scale=layer.scaling
            ).transpose(0, 1)

    torch.manual_seed(83)
    config = DeepseekV3Config(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        kv_lora_rank=5,
        v_head_dim=3,
        q_lora_rank=None,
    )
    config._attn_implementation = "eager"
    original = DeepseekV3Attention(config, 0).eval()
    source = copy.deepcopy(original)
    source.config._attn_implementation = "sglang"
    instances = nn.ModuleDict({"0": RadixAttention(2, 6, source.scaling, 2, 0)})
    install_mla_adapters(nn.ModuleDict({"attention": source}), instances)
    refresh_mla_weights(instances)
    hidden = torch.randn(1, 4, 16)
    positions = torch.arange(4).float().reshape(1, 4, 1).expand(-1, -1, 2)
    embeddings = positions.cos(), positions.sin()
    mask = torch.full((4, 4), float("-inf")).triu(1)[None, None]
    expected, _ = original(hidden, embeddings, attention_mask=mask)
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: False))
    with forward_context(ForwardContext(CPUReferenceAttention())):
        actual, _ = source(
            hidden,
            embeddings,
            attention_mask=None,
            forward_batch=batch,
            attention_instances=instances,
        )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert attention_interface.sglang_flash_attention_forward is not None


def test_fullgraph_forward_context_is_bound_outside_compiled_model():
    from sglang.srt.models.transformers.execution_context import (
        wrap_forward_with_context,
    )

    @torch.library.custom_op("sglang_test::transformers_context_probe", mutates_args=())
    def context_probe(value: torch.Tensor) -> torch.Tensor:
        return value + get_transformers_execution_context().forward_batch.offset

    @context_probe.register_fake
    def fake_context_probe(value):
        return torch.empty_like(value)

    def forward(input_ids, positions, forward_batch):
        with transformers_execution_context(forward_batch):
            return context_probe(input_ids)

    compiled = wrap_forward_with_context(
        torch.compile(forward, fullgraph=True, backend="eager")
    )
    value = torch.arange(3)
    torch.testing.assert_close(
        compiled(value, value, SimpleNamespace(offset=2)), value + 2
    )
    torch.testing.assert_close(
        compiled(
            input_ids=value, positions=value, forward_batch=SimpleNamespace(offset=5)
        ),
        value + 5,
    )
    with pytest.raises(RuntimeError, match="requires an execution context"):
        get_transformers_execution_context()


def test_native_deepseek_router_keeps_fp32_logits_and_shared_experts():
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

    torch.manual_seed(29)
    config = DeepseekV3Config(
        hidden_size=8,
        moe_intermediate_size=12,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=2,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
    )
    original = DeepseekV3MoE(config).eval()
    for parameter in original.parameters():
        nn.init.normal_(parameter, std=0.1)
    original.gate.e_score_correction_bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.1]))
    contract = native_router_contract(original)
    assert contract == "deepseek_v3"
    replacement = TransformersNativeMoE(
        original, ReferenceExpertDispatch(copy.deepcopy(original.experts)), contract
    )
    hidden = torch.randn(1, 7, 8)
    torch.testing.assert_close(replacement(hidden), original(hidden))


@pytest.mark.parametrize(
    "model_impl, is_mla, already_set, expected",
    [
        ("transformers", True, False, {"flashinfer_mla_disable_ragged": True}),
        ("transformers", True, True, {}),
        ("transformers", False, False, {}),
        ("sglang", True, False, {}),
    ],
)
def test_transformers_mla_forces_paged_prefill(
    monkeypatch, model_impl, is_mla, already_set, expected
):
    from sglang.srt.arg_groups import overrides

    monkeypatch.setattr(overrides, "use_mla_backend", lambda view: is_mla)
    view = SimpleNamespace(
        model_impl=model_impl, flashinfer_mla_disable_ragged=already_set
    )
    assert overrides._transformers_mla_paged_prefill(view) == expected


@pytest.mark.parametrize(
    "options",
    [
        dict(prefill_backend="flashmla", decode_backend="flashmla"),
        dict(prefill_backend="flashinfer", decode_backend="tokenspeed_mla"),
        dict(
            prefill_backend="flashinfer",
            decode_backend="flashinfer",
            kv_cache_dtype="fp8_e4m3",
        ),
        dict(
            prefill_backend="flashinfer", decode_backend="flashinfer", dcp_enabled=True
        ),
        dict(
            prefill_backend="flashinfer",
            decode_backend="flashinfer",
            enable_dp_attention=True,
        ),
        dict(
            prefill_backend="flashinfer", decode_backend="flashinfer", enable_lora=True
        ),
    ],
)
def test_mla_backend_options_reject_unsupported_contracts(options):
    from sglang.srt.models.transformers.mla import validate_mla_backend_options

    with pytest.raises(ValueError, match="Transformers native MLA"):
        validate_mla_backend_options(**options)
    validate_mla_backend_options("flashinfer", "flashinfer")


@pytest.mark.parametrize("combine_complete", [False, True])
def test_routed_output_has_exactly_one_reduction_owner(monkeypatch, combine_complete):
    from sglang.srt.models.transformers import moe

    wrapper = TransformersFusedMoE.__new__(TransformersFusedMoE)
    nn.Module.__init__(wrapper)
    wrapper._combine_completes_output = combine_complete
    calls = []

    def reduce(output):
        calls.append(output)
        return output + 1

    monkeypatch.setattr(moe, "post_experts_all_reduce", reduce)
    output = torch.zeros(2, 4)
    actual = wrapper.complete_output(output)
    assert len(calls) == (0 if combine_complete else 1)
    torch.testing.assert_close(actual, output if combine_complete else output + 1)


@pytest.mark.parametrize("tp_size,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("q_lora_rank", [None, 8])
def test_deepseek_full_wrapper_checkpoint_and_mla_shards(
    monkeypatch, tmp_path, tp_size, rank, q_lora_rank
):
    from transformers import AutoModelForCausalLM

    from sglang.srt.configs.model_config import AttentionArch, ModelConfig
    from sglang.srt.models.transformers import TransformersMoEForCausalLM
    from sglang.srt.runtime_context import get_context, get_parallel

    config = DeepseekV3Config(
        vocab_size=48,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        kv_lora_rank=5,
        v_head_dim=4,
        q_lora_rank=q_lora_rank,
        first_k_dense_replace=1,
        max_position_embeddings=128,
        architectures=["DeepseekV3ForCausalLM"],
    )
    config.save_pretrained(tmp_path)
    reference = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    pp_group = SimpleNamespace(
        world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True
    )
    tp_group = SimpleNamespace(world_size=tp_size, rank_in_group=rank, cpu_group=None)
    monkeypatch.setattr("sglang.srt.models.transformers.base.get_device", lambda: "cpu")
    with (
        get_context().override_server_args(device="cpu"),
        get_parallel().override(
            tp_size=tp_size,
            tp_rank=rank,
            attn_tp_size=tp_size,
            attn_tp_rank=rank,
            moe_tp_size=tp_size,
            moe_tp_rank=rank,
            moe_ep_size=1,
            moe_ep_rank=0,
            pp_group=pp_group,
            tp_group=tp_group,
        ),
    ):
        model_config = ModelConfig(
            str(tmp_path),
            trust_remote_code=False,
            model_impl="transformers",
            dtype="float32",
        )
        assert model_config.attention_arch == AttentionArch.MLA
        wrapper = TransformersMoEForCausalLM(
            config=model_config.hf_config, model_config=model_config
        )
        assert wrapper.uses_native_mla
        assert not any(parameter.is_meta for parameter in wrapper.parameters())
        loaded = wrapper.load_weights(reference.state_dict().items())
        assert set(dict(wrapper.named_parameters())) <= loaded
        assert wrapper._weights_loaded
        assert len(wrapper.moe_layers) == 1
        for key, adapter in wrapper.attention_instances.items():
            original = reference.model.layers[int(key)].self_attn
            actual = wrapper.model.layers[int(key)].self_attn
            kv_weight = original.kv_b_proj.weight.chunk(tp_size, dim=0)[rank]
            torch.testing.assert_close(actual.kv_b_proj.weight, kv_weight)
            expected_k, expected_v = split_mla_projection(kv_weight, 2 // tp_size, 4, 4)
            torch.testing.assert_close(adapter.w_kc, expected_k)
            torch.testing.assert_close(adapter.w_vc, expected_v)
            torch.testing.assert_close(
                actual.kv_a_proj_with_mqa.weight, original.kv_a_proj_with_mqa.weight
            )
            query_name = "q_proj" if q_lora_rank is None else "q_b_proj"
            torch.testing.assert_close(
                getattr(actual, query_name).weight,
                getattr(original, query_name).weight.chunk(tp_size, dim=0)[rank],
            )
            torch.testing.assert_close(
                actual.o_proj.weight, original.o_proj.weight.chunk(tp_size, dim=1)[rank]
            )
            if q_lora_rank is not None:
                torch.testing.assert_close(
                    actual.q_a_proj.weight, original.q_a_proj.weight
                )

        adapter = wrapper.attention_instances["0"]
        pointer = adapter.w_kc.data_ptr()
        updated = reference.model.layers[0].self_attn.kv_b_proj.weight.detach() + 1
        wrapper.load_weights((("model.layers.0.self_attn.kv_b_proj.weight", updated),))
        assert adapter.w_kc.data_ptr() == pointer
        torch.testing.assert_close(
            adapter.w_kc,
            split_mla_projection(updated.chunk(tp_size, 0)[rank], 2 // tp_size, 4, 4)[
                0
            ],
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
