from typing import Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer, DeepseekV2Model
from sglang.srt.models.mistral_large_3 import MistralLarge3ForCausalLM
from sglang.srt.utils import add_prefix


class MistralLarge3Model(DeepseekV2Model):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        nn.Module.__init__(self)

        self.config = config
        self.vocab_size = config.vocab_size
        assert get_pp_group().world_size == 1
        self.pp_group = get_pp_group()
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(
                    config=config,
                    prefix=add_prefix(prefix, f"layers.{i}"),
                    quant_config=quant_config,
                    layer_id=i,
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.start_layer = 0
        self.end_layer = self.config.num_hidden_layers

        self.fc = RowParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix(prefix, "fc"),
            input_is_parallel=False,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture = []
        self.llama_4_scaling_config = getattr(config, "llama_4_scaling", None)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        input_embeds, _ = self.fc(
            torch.cat((input_embeds, forward_batch.spec_info.hidden_states), dim=-1)
        )
        output = super().forward(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
        assert isinstance(output, torch.Tensor)
        return output


class MistralLarge3ForCausalLMEagle(MistralLarge3ForCausalLM):
    remapping = MistralLarge3ForCausalLM.remapping | {
        r"eagle_linear\.weight": r"model.fc.weight",
        r"eagle_linear\.qscale_act": r"model.fc.input_scale",
        r"eagle_linear\.qscale_weight": r"model.fc.weight_scale",
    }

    def __init__(
        self,
        *,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        config.quant_config = quant_config
        self.model_cls = MistralLarge3Model
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)


EntryClass = [MistralLarge3ForCausalLMEagle]
