"""Inference-only Intern-S2-Mobius model."""

import logging
from collections.abc import Iterable

import torch
from torch import nn

from sglang.srt.configs.interns2_mobius import (
    InternS2MobiusConfig,
    InternS2MobiusTextConfig,
)
from sglang.srt.distributed import get_pp_group, tensor_model_parallel_all_reduce
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe import (
    should_skip_post_experts_all_reduce,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    filter_moe_weight_param_global_expert,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.srt.models.qwen3_5 import (
    QWEN3_5_KV_SCALE_MAPPER,
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    _enable_qwen35_fused_ar_quant,
    _linear_accepts_fp8_tuple,
)
from sglang.srt.runtime_context import get_forward, get_parallel, get_stream
from sglang.srt.utils import add_prefix, is_cuda, make_layers
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()

_MOBIUS_PACKED_WEIGHT_MAPPING = (
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
    ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
    ("in_proj_qkvz.", "in_proj_z.", 3),
    ("in_proj_ba.", "in_proj_b.", 0),
    ("in_proj_ba.", "in_proj_a.", 1),
)


def _is_intentional_mobius_skip(name: str, tie_word_embeddings: bool) -> bool:
    return (
        name.startswith("mtp.")
        or name.endswith(
            (
                ".rotary_emb.inv_freq",
                ".rotary_emb.cos_cached",
                ".rotary_emb.sin_cached",
            )
        )
        or (tie_word_embeddings and name == "lm_head.weight")
    )


def _normalize_mobius_weight_name(name: str) -> str:
    if name.startswith("model.language_model."):
        name = "model." + name.removeprefix("model.language_model.")
    elif name.startswith("model.visual."):
        name = "visual." + name.removeprefix("model.visual.")

    if ".self_attn." in name:
        name = name.replace(".self_attn.", ".")
    if name.startswith("visual."):
        name = name.replace(".attn.qkv.", ".attn.qkv_proj.")
    return name


def _is_optional_mobius_parameter(name: str) -> bool:
    return name.endswith((".attn.k_scale", ".attn.v_scale"))


def _load_fused_mobius_expert_weight(
    *,
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict[str, nn.Parameter],
    num_experts: int,
    record_slot,
) -> None:
    gate_up_suffixes = {
        "experts.gate_up_proj": "experts.w13_weight",
        "experts.gate_up_proj_scale_inv": "experts.w13_weight_scale_inv",
    }
    gate_up_suffix = next(
        (suffix for suffix in gate_up_suffixes if name.endswith(suffix)), None
    )
    if gate_up_suffix is not None:
        parameter_name = (
            name.removesuffix(gate_up_suffix) + gate_up_suffixes[gate_up_suffix]
        )
        if parameter_name not in params_dict:
            raise KeyError(
                f"Mobius fused gate/up destination is missing: {parameter_name}"
            )
        if loaded_weight.shape[0] != num_experts:
            raise ValueError(
                f"Expected {num_experts} experts in {name}, got {loaded_weight.shape[0]}"
            )
        gate_weights, up_weights = loaded_weight.chunk(2, dim=-2)
        parameter = params_dict[parameter_name]
        loader = parameter.weight_loader
        for expert_id in range(num_experts):
            for shard_id, expert_weight in (
                ("w1", gate_weights[expert_id]),
                ("w3", up_weights[expert_id]),
            ):
                record_slot(parameter_name, shard_id, expert_id)
                loader(
                    parameter,
                    expert_weight,
                    parameter_name,
                    shard_id,
                    expert_id,
                )
        return

    down_suffixes = {
        "experts.down_proj": "experts.w2_weight",
        "experts.down_proj_scale_inv": "experts.w2_weight_scale_inv",
    }
    down_suffix = next(
        (suffix for suffix in down_suffixes if name.endswith(suffix)), None
    )
    if down_suffix is not None:
        parameter_name = name.removesuffix(down_suffix) + down_suffixes[down_suffix]
        if parameter_name not in params_dict:
            raise KeyError(
                f"Mobius fused down destination is missing: {parameter_name}"
            )
        if loaded_weight.shape[0] != num_experts:
            raise ValueError(
                f"Expected {num_experts} experts in {name}, got {loaded_weight.shape[0]}"
            )
        parameter = params_dict[parameter_name]
        loader = parameter.weight_loader
        for expert_id in range(num_experts):
            record_slot(parameter_name, "w2", expert_id)
            loader(
                parameter,
                loaded_weight[expert_id],
                parameter_name,
                "w2",
                expert_id,
            )
        return

    raise KeyError(f"Unexpected Mobius fused expert tensor: {name}")


def _expected_mobius_load_slots(
    params_dict: dict[str, nn.Parameter], num_experts: int
) -> set[tuple[str, object, int | None]]:
    expected = set()
    seen_parameters = set()
    for name, parameter in params_dict.items():
        # RadixLinearAttention exposes A_log/dt_bias aliases that refer to the
        # same Parameter already owned by Qwen3_5GatedDeltaNet. Require one
        # canonical load, not one load per module alias.
        parameter_id = id(parameter)
        if parameter_id in seen_parameters:
            continue
        seen_parameters.add(parameter_id)
        if ".meta_mlp." in name and name.endswith(
            ("experts.w13_weight", "experts.w13_weight_scale_inv")
        ):
            for expert_id in range(num_experts):
                expected.add((name, "w1", expert_id))
                expected.add((name, "w3", expert_id))
        elif ".meta_mlp." in name and name.endswith(
            ("experts.w2_weight", "experts.w2_weight_scale_inv")
        ):
            for expert_id in range(num_experts):
                expected.add((name, "w2", expert_id))
        elif ".qkv_proj." in name and name.startswith("model.layers."):
            for shard_id in ("q", "k", "v"):
                expected.add((name, shard_id, None))
        elif ".mlp.shared_expert.gate_up_proj." in name:
            for shard_id in (0, 1):
                expected.add((name, shard_id, None))
        elif ".in_proj_qkvz." in name:
            expected.add((name, (0, 1, 2), None))
            expected.add((name, 3, None))
        elif ".in_proj_ba." in name:
            expected.add((name, 0, None))
            expected.add((name, 1, None))
        elif _is_optional_mobius_parameter(name):
            continue
        else:
            expected.add((name, None, None))
    return expected


def _load_mobius_weights_strict(
    owner: nn.Module,
    config: InternS2MobiusTextConfig,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    params_dict = dict(owner.named_parameters(remove_duplicate=False))
    expected_slots = _expected_mobius_load_slots(params_dict, config.num_experts)
    loaded_slots: set[tuple[str, object, int | None]] = set()
    loaded_sources: set[str] = set()

    def record_slot(name, shard_id=None, expert_id=None):
        slot = (name, shard_id, expert_id)
        if slot in loaded_slots:
            raise ValueError(f"Mobius destination load is duplicated: {slot}")
        loaded_slots.add(slot)

    for source_name, loaded_weight in weights:
        if _is_intentional_mobius_skip(source_name, config.tie_word_embeddings):
            continue
        if source_name in loaded_sources:
            raise ValueError(f"Mobius checkpoint key is duplicated: {source_name}")
        loaded_sources.add(source_name)

        name = _normalize_mobius_weight_name(source_name)
        if ".meta_mlp." in name and name.endswith(
            (
                "experts.gate_up_proj",
                "experts.down_proj",
                "experts.gate_up_proj_scale_inv",
                "experts.down_proj_scale_inv",
            )
        ):
            _load_fused_mobius_expert_weight(
                name=name,
                loaded_weight=loaded_weight,
                params_dict=params_dict,
                num_experts=config.num_experts,
                record_slot=record_slot,
            )
            continue

        for parameter_name, weight_name, shard_id in _MOBIUS_PACKED_WEIGHT_MAPPING:
            if weight_name not in name:
                continue
            # Vision qkv is already fused in the checkpoint.
            if name.startswith("visual."):
                continue
            destination = name.replace(weight_name, parameter_name)
            if destination not in params_dict:
                raise KeyError(
                    f"Mobius packed destination is missing: {destination} "
                    f"(from {source_name})"
                )
            parameter = params_dict[destination]
            loader = parameter.weight_loader
            record_slot(destination, shard_id)
            loader(parameter, loaded_weight, shard_id)
            break
        else:
            if name not in params_dict:
                raise KeyError(
                    f"Mobius destination is missing: {name} (from {source_name})"
                )
            parameter = params_dict[name]
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            if _is_optional_mobius_parameter(name):
                expected_slots.add((name, None, None))
            record_slot(name)
            loader(parameter, loaded_weight)

    missing = expected_slots - loaded_slots
    unexpected = loaded_slots - expected_slots
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing destinations: {sorted(map(str, missing))[:20]}")
        if unexpected:
            details.append(
                f"unexpected destinations: {sorted(map(str, unexpected))[:20]}"
            )
        raise ValueError("Mobius weight coverage failure; " + "; ".join(details))
    return loaded_sources


class InternS2MobiusRoutedExpertBank(nn.Module):
    """One physical routed-expert bank, without a shared expert or reduction."""

    def __init__(
        self,
        bank_id: int,
        config: InternS2MobiusTextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.bank_id = bank_id
        self.tp_size = get_parallel().tp_size
        self.num_experts = config.num_experts
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            layer_id=bank_id,
        )
        self.experts = get_moe_impl_class(quant_config)(
            layer_id=bank_id,
            top_k=config.num_experts_per_tok,
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            routing_method_type=RoutingMethodType.RenormalizeNaive,
            num_fused_shared_experts=0,
            # The layer-local shared path consumes the same post-attention input.
            inplace=False,
        )
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

    def get_moe_weights(self):
        return [
            parameter.data
            for name, parameter in self.experts.named_parameters()
            if name != "correction_bias"
            and filter_moe_weight_param_global_expert(
                name, parameter, self.experts.num_local_experts
            )
        ]

    def forward_routed(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch | None = None,
    ) -> torch.Tensor:
        """Return an unreduced TP-partial routed result without mutating input."""
        del forward_batch
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, original_shape[-1])

        if hidden_states.shape[0] == 0:
            # Always enter the expert implementation so collective-capable
            # implementations keep identical participation on idle ranks.
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        else:
            router_logits, _ = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)

        output = self.experts(hidden_states, topk_output)
        return output.reshape(original_shape)


def _mobius_reduce_combined_output(combined: torch.Tensor) -> torch.Tensor:
    """Apply the one ordinary TP reduction unless the scoped runtime owns it."""
    if get_parallel().tp_size > 1 and not should_skip_post_experts_all_reduce(
        is_tp_path=True
    ):
        return tensor_model_parallel_all_reduce(combined)
    return combined


def _get_mobius_routed_bank(meta_mlp: nn.ModuleList, layer_id: int) -> nn.Module:
    if not meta_mlp:
        raise ValueError(
            "Intern-S2-Mobius requires at least one physical routed-expert bank"
        )
    return meta_mlp[layer_id % len(meta_mlp)]


class _InternS2MobiusDecoderMixin:
    def _init_mobius_mlp(
        self,
        config: InternS2MobiusTextConfig,
        quant_config: QuantizationConfig | None,
        layer_prefix: str,
    ) -> None:
        self.mlp = InternS2MobiusLayerMlp(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", layer_prefix),
        )

    def _forward_mobius_mlp(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        meta_mlp: nn.ModuleList,
    ) -> torch.Tensor:
        # Evaluate the layer-local branch first. Besides matching the intended
        # equation, this prevents a broken/in-place routed backend from
        # corrupting the input consumed by the shared expert or its gate.
        if hidden_states.shape[0] == 0:
            shared = torch.zeros_like(hidden_states)
        else:
            shared = self.mlp.shared_expert(hidden_states)
            gate, _ = self.mlp.shared_expert_gate(hidden_states)
            shared = torch.sigmoid(gate) * shared

        routed = _get_mobius_routed_bank(meta_mlp, self.layer_id).forward_routed(
            hidden_states, forward_batch
        )
        return _mobius_reduce_combined_output(routed + shared)

    def _forward_after_attention(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        forward_batch: ForwardBatch,
        meta_mlp: nn.ModuleList,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        mlp_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        # Model-side all-reduce fusion is intentionally disabled for baseline.
        with get_forward().scoped(
            fuse_mlp_allreduce=False,
            mlp_reduce_scatter=mlp_reduce_scatter,
        ):
            hidden_states = self._forward_mobius_mlp(
                hidden_states, forward_batch, meta_mlp
            )
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual


class InternS2MobiusLayerMlp(nn.Module):
    """Layer-local shared branch; routed banks live on the causal model."""

    def __init__(
        self,
        config: InternS2MobiusTextConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.shared_expert = Qwen2MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("shared_expert", prefix),
        )
        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=add_prefix("shared_expert_gate", prefix),
        )


class InternS2MobiusLinearDecoderLayer(_InternS2MobiusDecoderMixin, nn.Module):
    def __init__(
        self,
        config: InternS2MobiusTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.linear_attn = Qwen3_5GatedDeltaNet(
            config, layer_id, quant_config, alt_stream, prefix
        )

        layer_prefix = prefix.removesuffix(".linear_attn")
        self._init_mobius_mlp(config, quant_config, layer_prefix)
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=True,
            is_previous_layer_sparse=True,
            is_next_layer_sparse=True,
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        enable_fused_ar_quant = (
            _enable_qwen35_fused_ar_quant()
            and _linear_accepts_fp8_tuple(self.linear_attn.in_proj_qkvz)
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
            enable_fused_ar_quant=enable_fused_ar_quant,
            fused_ar_quant_keep_bf16=enable_fused_ar_quant,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        meta_mlp: nn.ModuleList,
        **kwargs,
    ):
        forward_batch = kwargs["forward_batch"]
        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=kwargs.get("captured_last_layer_outputs"),
            )
        )
        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.linear_attn(hidden_states, forward_batch)
        return self._forward_after_attention(
            hidden_states, residual, forward_batch, meta_mlp
        )


class InternS2MobiusAttentionDecoderLayer(
    _InternS2MobiusDecoderMixin, Qwen3_5AttentionDecoderLayer
):
    """Mobius-owned full-attention constructor reusing Qwen3.5 methods."""

    def __init__(
        self,
        config: InternS2MobiusTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.attn_tp_rank = get_parallel().attn_tp_rank
        self.attn_tp_size = get_parallel().attn_tp_size
        self.total_num_heads = config.num_attention_heads
        if self.total_num_heads % self.attn_tp_size != 0:
            raise ValueError("num_attention_heads must be divisible by attention TP")
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.attn_tp_size:
            if self.total_num_kv_heads % self.attn_tp_size != 0:
                raise ValueError(
                    "num_key_value_heads must be divisible by attention TP"
                )
        elif self.attn_tp_size % self.total_num_kv_heads != 0:
            raise ValueError("attention TP must be divisible by num_key_value_heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rope_theta, rope_scaling = get_rope_config(config)
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.layer_id = layer_id
        if rope_scaling and not ("rope_type" in rope_scaling or "type" in rope_scaling):
            rope_scaling = None
        self.attn_output_gate = getattr(config, "attn_output_gate", True)
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            rope_scaling=rope_scaling,
            base=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
            quant_config=quant_config,
        )

        layer_prefix = prefix.removesuffix(".self_attn")
        self._init_mobius_mlp(config, quant_config, layer_prefix)
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=True,
            is_previous_layer_sparse=True,
            is_next_layer_sparse=True,
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        enable_fused_ar_quant = (
            _enable_qwen35_fused_ar_quant() and _linear_accepts_fp8_tuple(self.qkv_proj)
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
            enable_fused_ar_quant=enable_fused_ar_quant,
            fused_ar_quant_keep_bf16=False,
        )
        self.alt_stream = alt_stream

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        forward_batch: ForwardBatch,
        meta_mlp: nn.ModuleList,
        captured_last_layer_outputs: list[torch.Tensor] | None = None,
        **kwargs,
    ):
        del kwargs
        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
            )
        )
        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.self_attention(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        return self._forward_after_attention(
            hidden_states, residual, forward_batch, meta_mlp
        )


class InternS2MobiusForCausalLM(Qwen3_5ForCausalLM):
    def __init__(
        self,
        config: InternS2MobiusTextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.pp_group = get_pp_group()
        if self.pp_group.world_size != 1:
            raise ValueError(
                "Intern-S2-Mobius baseline does not support pipeline parallelism"
            )

        alt_stream = get_stream("alt") if _is_cuda else None
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            enable_tp=not is_dp_attention_enabled(),
        )

        bank_prefix = prefix.replace("model.language_model", "model")
        self.meta_mlp = nn.ModuleList(
            [
                InternS2MobiusRoutedExpertBank(
                    bank_id=bank_id,
                    config=config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"meta_mlp.{bank_id}", bank_prefix),
                )
                for bank_id in range(config.num_blocks)
            ]
        )
        if len(self.meta_mlp) != config.num_blocks:
            raise AssertionError(
                "physical routed-expert bank count does not match num_blocks"
            )

        def get_layer(idx: int, prefix: str):
            checkpoint_type = config.layer_types[idx]
            if checkpoint_type == "full_attention":
                return InternS2MobiusAttentionDecoderLayer(
                    config=config,
                    layer_id=idx,
                    quant_config=quant_config,
                    prefix=add_prefix("self_attn", prefix),
                    alt_stream=alt_stream,
                )
            if checkpoint_type == "linear_attention":
                return InternS2MobiusLinearDecoderLayer(
                    config=config,
                    layer_id=idx,
                    quant_config=quant_config,
                    prefix=add_prefix("linear_attn", prefix),
                    alt_stream=alt_stream,
                )
            raise ValueError(f"Unsupported Mobius layer type: {checkpoint_type}")

        self.layers, self._start_layer, self._end_layer = make_layers(
            config.num_hidden_layers,
            get_layer,
            pp_rank=0,
            pp_size=1,
            prefix=f"{prefix}.layers",
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture = []

    def get_hidden_dim(self, module_name: str, layer_idx: int):
        if module_name == "gate_up_proj":
            return (
                self.config.hidden_size,
                self.config.shared_expert_intermediate_size * 2,
            )
        if module_name == "down_proj":
            return self.config.shared_expert_intermediate_size, self.config.hidden_size
        return super().get_hidden_dim(module_name, layer_idx)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: PPProxyTensors | None = None,
        input_deepstack_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | PPProxyTensors:
        if pp_proxy_tensors is not None:
            raise ValueError("Intern-S2-Mobius baseline does not support PP tensors")
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        residual = None
        aux_hidden_states = []
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
                meta_mlp=self.meta_mlp,
                captured_last_layer_outputs=(
                    aux_hidden_states
                    if getattr(layer, "_is_layer_to_capture", False)
                    else None
                ),
            )
            if (
                input_deepstack_embeds is not None
                and input_deepstack_embeds.numel() > 0
                and layer_idx < 3
            ):
                start = self.hidden_size * layer_idx
                hidden_states.add_(
                    input_deepstack_embeds[:, start : start + self.hidden_size]
                )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return (
            hidden_states
            if not aux_hidden_states
            else (hidden_states, aux_hidden_states)
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        raise ValueError(
            "Load Intern-S2-Mobius through its conditional-generation wrapper "
            "so vision, language, lm_head, and strict coverage are handled together"
        )


class InternS2MobiusForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    packed_modules_mapping = InternS2MobiusForCausalLM.packed_modules_mapping
    supported_lora_modules = InternS2MobiusForCausalLM.supported_lora_modules

    def __init__(
        self,
        config: InternS2MobiusConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        language_model_cls=InternS2MobiusForCausalLM,
    ) -> None:
        ignored_layers = getattr(quant_config, "ignored_layers", None)
        if (
            getattr(quant_config, "is_checkpoint_fp8_serialized", False)
            and ignored_layers
        ):
            # HF treats these parent entries as exact names; SGLang prefix matching
            # would also skip their quantized qkv/z and output projections.
            quant_config.ignored_layers = [
                name for name in ignored_layers if not name.endswith(".linear_attn")
            ]
        super().__init__(config, quant_config, prefix, language_model_cls)

    def should_apply_lora(self, module_name: str) -> bool:
        # Meta banks require bank-aware adapter ownership and are excluded.
        return module_name.startswith("model.layers.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return _load_mobius_weights_strict(
            self,
            self.config,
            QWEN3_5_KV_SCALE_MAPPER.apply(weights),
        )


EntryClass = [InternS2MobiusForConditionalGeneration]
