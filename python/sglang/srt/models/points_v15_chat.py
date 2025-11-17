import copy
from typing import Iterable, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.points_v15_chat import POINTSV15ChatConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.models.qwen2_vl import Qwen2VisionPatchMerger, Qwen2VisionTransformer
from sglang.srt.utils import add_prefix


class Qwen2VisionTransformerForNavitPOINTS(Qwen2VisionTransformer):
    def __init__(
        self,
        vision_config: POINTSV15ChatConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vision_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        # transformers
        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

        return x


class POINTSV15ChatModel(nn.Module):
    def __init__(
        self,
        config: POINTSV15ChatConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        config.llm_config._attn_implementation = "flash_attention_2"
        config._attn_implementation_autoset = False
        self.config = config
        self.quant_config = quant_config

        llm_config = copy.deepcopy(config.llm_config)
        llm_config.architectures = ["Qwen2ForCausalLM"]
        self.llm = Qwen2ForCausalLM(
            config=llm_config,
            quant_config=quant_config,
            prefix=add_prefix("llm", prefix),
        )

        self.vision_encoder = Qwen2VisionTransformerForNavitPOINTS(
            config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_encoder", prefix),
        )

        self.vision_projector = Qwen2VisionPatchMerger(
            d_model=config.llm_config.hidden_size,
            context_dim=1280,
            quant_config=quant_config,
            prefix=add_prefix("vision_projector", prefix),
        )

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.vision_encoder.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)

        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()

        image_features = self.vision_encoder(pixel_values, grid_thw=image_grid_thw)
        image_features = self.vision_projector(image_features)
        return image_features

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.llm,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            positions=positions,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "vision_encoder" in name:
                    # adapt to VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")

                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [POINTSV15ChatModel]
