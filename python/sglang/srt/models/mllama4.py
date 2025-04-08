# TODO: add Aapted from vllm/mllama4.py
from collections.abc import Iterable
from typing import Optional, Set, Tuple

import torch
from torch import nn
from transformers import Llama4Config

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix


class Llama4ForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(
        self,
        config: Llama4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        # Initialize the language model
        from sglang.srt.models.llama4 import Llama4ForCausalLM

        self.language_model = Llama4ForCausalLM(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        self.logits_processor = LogitsProcessor(config.text_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ) -> torch.Tensor:

        return self.language_model(input_ids, positions, forward_batch)

    def permute_qk_weight_for_rotary(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:

        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.language_model.config.head_dim * n_heads
            attn_out = self.language_model.config.hidden_size

            return (
                w.view(n_heads, attn_in // n_heads // 2, 2, attn_out)
                .transpose(1, 2)
                .reshape(attn_in, attn_out)
            )

        modules = name.split(".")

        # rotary embeds should be sliced
        if ("wk" in modules or "k_proj" in modules) and modules[-1] == "weight":
            loaded_weight = permute(
                loaded_weight, self.language_model.config.num_key_value_heads
            )
        elif ("wq" in modules or "q_proj" in modules) and modules[-1] == "weight":
            loaded_weight = permute(
                loaded_weight, self.language_model.config.num_attention_heads
            )

        return name, loaded_weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".shared_expert.gate_up_proj", ".shared_expert.gate_proj", 0),
            (".shared_expert.gate_up_proj", ".shared_expert.up_proj", 1),
            (".feed_forward.gate_up_proj", ".feed_forward.gate_proj", 0),
            (".feed_forward.gate_up_proj", ".feed_forward.up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        num_experts = self.config.text_config.num_local_experts

        for name, loaded_weight in weights:

            if name.startswith("vision_model") or name.startswith(
                "multi_modal_projector"
            ):
                continue

            name, loaded_weight = self.permute_qk_weight_for_rotary(name, loaded_weight)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if ".experts" in name:
                    if ".gate_up_proj" in name:
                        name_list = [
                            name.replace(".experts.gate_up_proj", ".experts.w13_weight")
                        ] * 2
                        loaded_weight_list = loaded_weight.chunk(2, dim=-1)
                        shard_id_list = ["w1", "w3"]
                    else:
                        name_list = [
                            name.replace(".experts.down_proj", ".experts.w2_weight")
                        ]
                        shard_id_list = ["w2"]
                        loaded_weight_list = [loaded_weight]
                    for name, loaded_weight, shard_id in zip(
                        name_list, loaded_weight_list, shard_id_list
                    ):
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        for expert_id in range(num_experts):
                            weight_loader(
                                param,
                                loaded_weight[expert_id].T,
                                name,
                                shard_id=shard_id,
                                expert_id=expert_id,
                            )
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


EntryClass = Llama4ForConditionalGeneration
