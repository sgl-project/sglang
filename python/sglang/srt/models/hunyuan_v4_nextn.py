import copy
import functools
from typing import Iterable, Set, Tuple

import torch
from torch import nn

from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.layer_boundary import (
    append_stages,
    declare_attn,
    declare_ffn,
    get_attn_tp_context,
    layer_stack,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.models.hunyuan_v4 import (
    HYV4_PACKED_MODULES_MAPPING,
    HYV4Attention,
    HYV4ForCausalLM,
    check_hyv4_indexer_pairs,
    hyv4_mrope_rows_are_equal,
    normalize_hyv4_weight_scale,
    permute_hyv4_indexer_weight,
)
from sglang.srt.runtime_context import get_parallel, get_stream
from sglang.srt.utils import BumpAllocator, is_cuda


def _check_mtp_quant_algo(mtp_quant_algo, quant_config):
    # Reference semantics: NONE inherits the target quantization, BF16/FP16
    # leave the MTP layer unquantized; only the inherited layout is supported.
    if mtp_quant_algo is None:
        return
    algo = str(mtp_quant_algo).upper()
    if algo == "NONE" or (algo in ("BF16", "FP16") and quant_config is None):
        return
    target = "unquantized" if quant_config is None else quant_config.get_name()
    raise ValueError(
        f"HYV4 MTP inherits the target quantization ({target}); "
        f"mtp_quant_algo={mtp_quant_algo!r} is not supported"
    )


def _check_single_mtp_head_weight(name, num_hidden_layers):
    # The draft loads only model.mtp_layers.0; MTP weights under another head or
    # appended as model.layers.<num_hidden_layers + i> would be skipped unloaded.
    parts = name.split(".", 3)
    if len(parts) < 4 or parts[0] != "model" or not parts[2].isdigit():
        return
    if (parts[1] == "mtp_layers" and parts[2] != "0") or (
        parts[1] == "layers" and int(parts[2]) >= num_hidden_layers
    ):
        raise ValueError(
            "HYV4 MTP loads a single draft head from model.mtp_layers.0; "
            f"got checkpoint weight {name}"
        )


def _mtp_checkpoint_layer_name(name, layer_id):
    # compressed-tensors targets name the MTP block model.layers.<num_hidden_layers>.
    _, found, rest = name.partition("model.decoder.")
    return f"model.layers.{layer_id}.{rest}" if found else name


def _mtp_quant_config(quant_config, num_hidden_layers):
    if quant_config is None:
        return None
    quant_config = copy.deepcopy(quant_config)
    if isinstance(quant_config, CompressedTensorsConfig):
        quant_config.checkpoint_layer_name = functools.partial(
            _mtp_checkpoint_layer_name, layer_id=num_hidden_layers
        )
        # Lookup stays strict, which rejects an unmatched expert 0 but never
        # queries the later experts of the bank.
        quant_config.check_all_experts = True
    ignored_layers = getattr(quant_config, "ignored_layers", None)
    if ignored_layers is not None:
        quant_config.ignored_layers = [
            name.replace("model.mtp_layers.0", "model.decoder").replace(
                "mtp_layers.0", "model.decoder"
            )
            for name in ignored_layers
        ]
    return quant_config


class HYV4MTPDecoderLayer(nn.Module):
    def __init__(
        self, config, quant_config=None, prefix="", alt_stream=None, *, use_mrope=False
    ):
        super().__init__()
        self.self_attn = HYV4Attention(
            config,
            0,
            quant_config,
            f"{prefix}.self_attn",
            alt_stream,
            is_nextn=True,
            use_mrope=use_mrope,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = DeepseekV2MoE(
            config,
            0,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            alt_stream=alt_stream,
            is_nextn=True,
            reduce_results=False,
        )
        if self.mlp.shared_experts is not None:
            self.mlp.shared_experts.swiglu_limit = None
        self.attn_boundary, self.ffn_boundary = append_stages(
            (
                declare_attn(),
                self.input_layernorm,
                {"qkv_latent_func": self.self_attn.prepare_qkv_latent},
            ),
            (declare_ffn(sparse=True), self.post_attention_layernorm),
        )

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        prev_topk_indices=None,
    ):
        hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch)
        try:
            hidden_states = self.self_attn(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                prev_topk_indices=prev_topk_indices,
            )
        finally:
            get_attn_tp_context().clear_attn_inputs()
        if isinstance(hidden_states, tuple):
            hidden_states, topk_indices = hidden_states
        else:
            topk_indices = None
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.mlp(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.finish(hidden_states, forward_batch)
        return hidden_states, topk_indices


class HYV4ModelNextN(nn.Module):
    def __init__(self, config, quant_config=None, prefix="", *, use_mrope=False):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
            **get_embedding_tp_kwargs(),
        )
        self.enorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.alt_stream = get_stream("alt") if is_cuda() else None
        with layer_stack():
            self.decoder = HYV4MTPDecoderLayer(
                config,
                quant_config,
                f"{prefix}.decoder",
                self.alt_stream,
                use_mrope=use_mrope,
            )
        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def embed_input_ids(self, input_ids, forward_batch):
        # Media spans hold hashed pad ids outside the vocab; the worker shifts the
        # target's merged embeddings with input_ids, leaving each last slot to fill.
        if (
            forward_batch.forward_mode.is_extend()
            and forward_batch.contains_mm_inputs()
        ):
            input_embeds = forward_batch.mm_input_embeds
            assert input_embeds is not None
            last_indices = (
                forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
            ).long()
            input_embeds[last_indices] = self.embed_tokens(input_ids[last_indices])
            return input_embeds
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden_states = (
            self.embed_input_ids(input_ids, forward_batch)
            if input_embeds is None
            else input_embeds
        )
        if hidden_states.shape[0] > 0:
            hidden_states = self.eh_proj(
                torch.cat(
                    (
                        self.enorm(hidden_states),
                        self.hnorm(forward_batch.spec_info.hidden_states),
                    ),
                    dim=-1,
                )
            )
        zero_allocator = BumpAllocator(
            buffer_size=2,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        residual_batch.start(forward_batch)
        topk_share = IndexTopKShareState.from_mtp_carry(forward_batch)
        hidden_states, topk_indices = self.decoder(
            positions,
            hidden_states,
            forward_batch,
            zero_allocator,
            topk_share.topk_indices,
        )
        topk_share.update(topk_indices)
        topk_share.publish()
        hidden_states = residual_batch.complete_output(hidden_states, forward_batch)
        if not forward_batch.forward_mode.is_idle():
            hidden_states = residual_batch.final_norm(
                hidden_states, forward_batch, self.shared_head.norm
            )
        return hidden_states


class HYV4ForCausalLMNextN(nn.Module, DeepseekV2WeightLoaderMixin):
    packed_modules_mapping = HYV4_PACKED_MODULES_MAPPING

    @staticmethod
    def shared_experts_fusion_disable_reason(hf_config, quant_config):
        return HYV4ForCausalLM.shared_experts_fusion_disable_reason(
            hf_config, quant_config
        )

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        _check_mtp_quant_algo(config.mtp_quant_algo, quant_config)
        self.is_mrope_enabled = "mrope_section" in config.rope_parameters and not (
            getattr(config, "language_model_only", False)
        )
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        nextn_quant_config = _mtp_quant_config(quant_config, config.num_hidden_layers)
        self.model = HYV4ModelNextN(
            config,
            nextn_quant_config,
            prefix=f"{prefix}.model",
            use_mrope=self.is_mrope_enabled,
        )
        self.num_fused_shared_experts = self.model.decoder.mlp.num_fused_shared_experts
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head",
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch):
        # Attention DP pads an idle rank to EXTEND with 1-D positions only.
        if (
            self.is_mrope_enabled
            and not forward_batch.forward_mode.is_idle()
            and forward_batch.mrope_positions is not None
        ):
            positions = forward_batch.mrope_positions
            if hyv4_mrope_rows_are_equal(forward_batch.forward_mode):
                positions = positions[0]
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        layer_prefix = f"model.layers.{self.config.num_hidden_layers}"

        def mapped_weights():
            for name, loaded_weight in weights:
                _check_single_mtp_head_weight(name, self.config.num_hidden_layers)
                if not name.startswith("model.mtp_layers.0."):
                    continue
                name = name.replace("model.mtp_layers.0", layer_prefix)
                if name.endswith(".final_layernorm.weight"):
                    name = name.replace(
                        ".final_layernorm.weight", ".shared_head.norm.weight"
                    )
                name, loaded_weight = normalize_hyv4_weight_scale(
                    name, loaded_weight, self.quant_config
                )
                loaded_weight = permute_hyv4_indexer_weight(
                    name, loaded_weight, self.config
                )
                yield name, loaded_weight

        self.do_load_weights(mapped_weights(), is_nextn=True)

    def _check_loaded_weights(
        self, unloaded_params: Set[str], unresolved_indexer_pairs: Set[str]
    ) -> None:
        check_hyv4_indexer_pairs(unresolved_indexer_pairs)
        # Weight scales only, per whole parameter: one loaded shard marks a fused
        # scale present. KV-cache and input scales are not checked here.
        missing_scales = sorted(
            name for name in unloaded_params if "weight_scale" in name
        )
        if missing_scales:
            raise ValueError(
                "HYV4 MTP checkpoint is missing weight scales: "
                + ", ".join(missing_scales)
            )

    def post_load_weights(self, is_nextn=True, weight_names=None):
        super().post_load_weights(is_nextn=True, weight_names=weight_names)


EntryClass = [HYV4ForCausalLMNextN]
