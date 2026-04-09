import logging
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.step3p5 import Step3p5DecoderLayer, Step3p5ForCausalLM
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


def get_spec_layer_idx_from_weight_name(
    config: PretrainedConfig, weight_name: str
) -> Optional[int]:
    """Return MTP/nextn layer index if this weight belongs to spec layers.

    Step3p5 MTP/nextn checkpoints append extra layers after the main decoder:
      model.layers.[num_hidden_layers ... num_hidden_layers + num_nextn_predict_layers)
    """
    if hasattr(config, "num_nextn_predict_layers") and (
        getattr(config, "num_nextn_predict_layers", 0) > 0
    ):
        base = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{base + i}."):
                return base + i
    return None


class SharedHead(nn.Module):

    def __init__(
        self,
        config,
        quant_config=None,
    ) -> None:
        super().__init__()
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=quant_config
        )
        self.lm_head = self.head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class Step3p5AMultiTokenPredictor(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers

        layer_id = 45  # FIXME

        self.enorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = SharedHead(config=config, quant_config=quant_config)
        self.mtp_block = Step3p5DecoderLayer(
            config=config, layer_id=layer_id, prefix=f"{prefix}.mtp_block"
        )
        self.lm_head = self.shared_head.head

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

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
        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=None,
        )
        hidden_states_before_norm = None
        if not forward_batch.forward_mode.is_idle():
            # if forward_batch.return_hidden_states_before_norm:
            hidden_states_before_norm = (
                hidden_states if residual is None else hidden_states + residual
            )
            if residual is not None:
                hidden_states, _ = self.shared_head.norm(hidden_states, residual)
            else:
                hidden_states = self.shared_head.norm(hidden_states)

        return hidden_states, hidden_states_before_norm

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


# The current implementation differs slightly from the standard MTP implementation in Step3.5 Flash.
# In the standard multi-layer MTP design of Step3.5 Flash,
# the hidden states of each MTP layer are passed from the preceding MTP layer
# (the hidden states of the initial (layer-0) MTP still being provided by the target model).
# In contrast, the current SGL implementation obtains hidden states directly from the target model for all MTP layers.
# Empirical evaluations indicate that the overall performance remains strong;
# however, this design choice may lead to a slight reduction in acceptance rate in certain scenarios.
# This behavior will be corrected shortly, and we expect to implement the standard multi-layer MTP design of Step3.5 Flash in the near future.
# FIXME(yhyang201)
class Step3p5MTP(Step3p5ForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        draft_model_idx: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.draft_model_idx = draft_model_idx

        self.model = Step3p5AMultiTokenPredictor(
            config=config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.logits_processor = LogitsProcessor(config)
        self.lm_head = self.model.lm_head

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states, hidden_states_before_norm = self.model(
            input_ids, positions, forward_batch
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.model.shared_head.head,
            forward_batch,
            hidden_states_before_norm=hidden_states_before_norm,
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.model.shared_head.head.weight

    def set_embed_and_head(self, embed, head):
        return

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = [
            (".moe.experts.w13_weight", ".moe.gate_proj.weight", "w1"),
            (".moe.experts.w13_weight", ".moe.up_proj.weight", "w3"),
            (".moe.experts.w2_weight", ".moe.down_proj.weight", "w2"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None and spec_layer != (
                self.config.num_hidden_layers + self.draft_model_idx
            ):
                continue
            if "embed_tokens" not in name and spec_layer is None:
                continue
            name = self._rewrite_spec_layer_name(spec_layer, name)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if "experts" in name or "moe" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    for expert_id in range(loaded_weight.shape[0]):
                        loaded_weight_expert = loaded_weight[expert_id]
                        weight_loader(
                            param,
                            loaded_weight_expert,
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    loaded_params.add(name)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        or "tok_embeddings" in name
                    ):
                        continue

                    if "shared_head" in name:
                        name = name.replace("shared_head.output", "shared_head.head")
                    if "embed_tokens" in name:
                        assert (
                            hasattr(self.config, "num_nextn_predict_layers")
                            and self.config.num_nextn_predict_layers > 0
                        )
                        name = "model.embed_tokens.weight"
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        params_need_to_load = set(params_dict.keys())
        if params_need_to_load != loaded_params:
            missing_params = list(params_need_to_load - loaded_params)
            param_name_example = missing_params[0]
            raise RuntimeError(
                f"Some parameters like {param_name_example} are not in the checkpoint and will falsely use random initialization"
            )
        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: Optional[int], name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        """
        if spec_layer is None:
            return name

        # Some checkpoints place MTP weights under "model.layers.<id>.transformer.*".
        # Our modules use "model.layers.<id>.*", so drop the ".transformer." segment.
        transformer_prefix = f"model.layers.{spec_layer}.transformer."
        if name.startswith(transformer_prefix):
            name = name.replace(".transformer.", ".", 1)

        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        spec_layer_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                break
        if not spec_layer_weight:
            # treat rest weights as weights for transformer layer block
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )

        # NEW: drop "layers.<idx>." from the rewritten name (minimal change).
        layers_prefix = f"model.layers.{spec_layer}."
        if name.startswith(layers_prefix):
            name = name.replace(layers_prefix, "model.", 1)

        return name


EntryClass = [Step3p5MTP]
