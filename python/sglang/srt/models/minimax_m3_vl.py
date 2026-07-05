# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.utils.common import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.minimax_m3 import (
    MiniMaxM3Model,
    MiniMaxM3SparseForCausalLM,
    build_minimax_fused_qkv_index,
    get_spec_layer_idx_from_weight_name,
)
from sglang.srt.models.minimax_vl_common import (
    CLIPVisionConfig,
    MiniMaxVLVisionModel,
    get_image_feature,
    get_video_feature,
    load_vision_weight,
    merge_vit_qkv_weights,
)
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils import add_prefix, get_device_sm, is_cuda, log_info_on_rank0
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)


_is_cuda = is_cuda()
_device_sm = get_device_sm()


class MiniMaxM3SparseForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.use_data_parallel = get_server_args().mm_enable_dp_encoder

        self.num_fused_shared_experts = 0
        self._determine_num_fused_shared_experts()

        vision_config_raw = config.vision_config
        assert vision_config_raw is not None, "vision_config is required"
        if hasattr(vision_config_raw, "to_dict"):
            vision_config_dict = vision_config_raw.to_dict()
        else:
            vision_config_dict = vision_config_raw
        vision_config = CLIPVisionConfig.from_dict(vision_config_dict)
        self.vision_config = vision_config

        text_hidden_size = getattr(config.text_config, "hidden_size", None)
        assert text_hidden_size is not None, "text_hidden_size is required"
        projector_hidden_size = getattr(config, "projector_hidden_size", None)

        # Vision model skips quantization: CLIP dimensions (head_dim=80) are not
        # compatible with MXFP8 kernel alignment requirements (128).
        self.vision_tower = MiniMaxVLVisionModel(
            config=vision_config,
            text_hidden_size=text_hidden_size,
            projector_hidden_size=projector_hidden_size,
            quant_config=None,
            prefix=add_prefix("vision_tower", prefix),
            multimodal_projector_bias=getattr(
                config, "multimodal_projector_bias", True
            ),
            patch_merge_bias=getattr(config, "patch_merge_bias", True),
        )

        text_config = config.text_config
        self.model = MiniMaxM3Model(
            config=text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model.model", prefix),
        )

        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                text_config.vocab_size,
                text_config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("language_model.lm_head", prefix),
                use_attn_tp_group=get_server_args().enable_dp_lm_head,
            )
        else:
            self.lm_head = PPMissingLayer()

        _, text_rope_scaling = get_rope_config(text_config)
        self.is_mrope_enabled = (
            text_rope_scaling is not None and "mrope_section" in text_rope_scaling
        )

        self.logits_processor = LogitsProcessor(text_config)

    def _determine_num_fused_shared_experts(self) -> None:
        text_config = self.config.text_config
        server_args = get_server_args()
        if server_args.disable_shared_experts_fusion:
            return

        disable_reason = None
        if not getattr(text_config, "n_shared_experts", None):
            disable_reason = "No shared experts are defined in the config."
        elif not _is_cuda:
            disable_reason = "Shared experts fusion currently requires CUDA devices."
        elif (_device_sm is not None) and (_device_sm < 80):
            disable_reason = "Shared experts fusion requires SM80 or newer GPUs."
        elif get_moe_expert_parallel_world_size() > 1:
            disable_reason = (
                "Shared experts fusion is not supported together with expert "
                "parallelism yet."
            )
        elif get_moe_a2a_backend().is_deepep():
            disable_reason = (
                "Shared experts fusion is not supported when Deepep MoE backend "
                "is enabled."
            )

        if disable_reason is not None:
            server_args.disable_shared_experts_fusion = True
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = text_config.n_shared_experts
        assert (
            self.num_fused_shared_experts == 1
        ), "Only 1 fused shared expert is supported"
        log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        # EP asserts if this hook is absent on the top-level arch; VL nests the
        # LM config under text_config, so delegate there (fall back to config).
        text_config = getattr(config, "text_config", None) or config
        return MiniMaxM3SparseForCausalLM.get_model_config_for_expert_location(
            text_config
        )

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
            input_ids, mm_inputs
        )

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return get_image_feature(self.vision_tower, items, self.use_data_parallel)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return get_video_feature(self.vision_tower, items, self.use_data_parallel)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank and not get_embedding:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
            )
        return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        # ``.qkv_proj`` (with the leading dot) prevents matching e.g.
        # ``index_q_proj`` in the sparse-attention branch.
        llm_stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        if (
            getattr(self.config.text_config, "sparse_attention_config", None)
            is not None
        ):
            llm_stacked_params_mapping += [
                (".index_qkv_proj", ".index_q_proj", "q"),
                (".index_qkv_proj", ".index_k_proj", "k"),
                (".index_qkv_proj", ".index_v_proj", "v"),
            ]

        num_experts = getattr(self.config.text_config, "num_local_experts", 0)
        expert_params_mapping = (
            FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=num_experts + self.num_fused_shared_experts,
            )
            if num_experts > 0
            else []
        )

        params_dict = dict(self.named_parameters())
        vit_qkv_weights: dict = {}
        vit_qkv_biases: dict = {}

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("language_model."):
                self._load_llm_weight(
                    name[len("language_model.") :],
                    loaded_weight,
                    params_dict,
                    llm_stacked_params_mapping,
                    expert_params_mapping,
                )
                continue

            load_vision_weight(
                name, loaded_weight, params_dict, vit_qkv_weights, vit_qkv_biases
            )

        merge_vit_qkv_weights(vit_qkv_weights, vit_qkv_biases, params_dict)

        build_minimax_fused_qkv_index(self)

    def _load_llm_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict,
        llm_stacked_params_mapping: list,
        expert_params_mapping: list,
    ) -> None:
        if "block_sparse_moe" in name:
            name = name.replace("block_sparse_moe", "mlp")

        layer_id = get_layer_id(name)
        if layer_id is not None and (
            layer_id < self.model.start_layer or layer_id >= self.model.end_layer
        ):
            return

        if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
            name = name.replace(
                "mlp.shared_experts",
                f"mlp.experts.{self.config.text_config.num_local_experts}",
            )
            name = name.replace("gate_proj", "w1")
            name = name.replace("down_proj", "w2")
            name = name.replace("up_proj", "w3")

        if (
            get_spec_layer_idx_from_weight_name(self.config.text_config, name)
            is not None
        ):
            return

        for param_name, weight_name, shard_id in llm_stacked_params_mapping:
            if weight_name not in name:
                continue
            if "mlp.experts." in name:
                continue
            new_name = name.replace(weight_name, param_name)
            if new_name.endswith(".bias") and new_name not in params_dict:
                continue
            if new_name not in params_dict:
                continue
            param = params_dict[new_name]
            param.weight_loader(param, loaded_weight, shard_id)
            return

        is_expert_weight = False
        for mapping in expert_params_mapping:
            param_name, weight_name, expert_id, shard_id = mapping
            if weight_name not in name:
                continue
            is_expert_weight = True
            new_name = name.replace(weight_name, param_name)
            if new_name not in params_dict:
                continue
            param = params_dict[new_name]
            param.weight_loader(
                param,
                loaded_weight,
                new_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            return
        if is_expert_weight:
            return

        if name.endswith(".bias") and name not in params_dict:
            return
        remapped = maybe_remap_kv_scale_name(name, params_dict)
        if remapped is None:
            return
        if remapped not in params_dict:
            logger.warning(f"Parameter {remapped} not found in params_dict")
            return
        param = params_dict[remapped]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        try:
            weight_loader(param, loaded_weight)
        except Exception as e:
            logger.warning(f"Error loading weight {remapped}: {e}")


EntryClass = [MiniMaxM3SparseForConditionalGeneration]
