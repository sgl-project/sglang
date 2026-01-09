import logging
from functools import lru_cache
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.glm4v_moe.configuration_glm4v_moe import Glm4vMoeConfig

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers.attention import vision_utils
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.glm4_moe import Glm4MoeModel
from sglang.srt.models.glm4v import Glm4vForConditionalGeneration, Glm4vVisionModel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, get_device_sm, is_cuda, log_info_on_rank0
from sglang.srt.utils.hf_transformers_utils import get_processor

_is_cuda = is_cuda()
_device_sm = get_device_sm()

logger = logging.getLogger(__name__)

cached_get_processor = lru_cache(get_processor)


class Glm4vMoeForConditionalGeneration(Glm4vForConditionalGeneration):
    def __init__(
        self,
        config: Glm4vMoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.pp_group = get_pp_group()
        self.config = config
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
        vision_utils.update_vit_attn_dummy_heads_config(self.config)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.num_fused_shared_experts = 0
        self.determine_num_fused_shared_experts()

        self.model = Glm4MoeModel(
            config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.visual = Glm4vVisionModel(
            config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
            use_data_parallel=self.use_data_parallel,
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and self.config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def determine_num_fused_shared_experts(self):
        if get_global_server_args().disable_shared_experts_fusion:
            return

        disable_reason = None
        if not getattr(self.config, "n_shared_experts", None):
            disable_reason = "No shared experts are defined in the config."
        elif not _is_cuda:
            disable_reason = "Shared experts fusion currently requires CUDA devices."
        elif _is_cuda and (_device_sm is not None) and (_device_sm < 80):
            disable_reason = "Shared experts fusion requires SM80 or newer GPUs."
        elif get_moe_expert_parallel_world_size() > 1:
            disable_reason = "Shared experts fusion is not supported together with expert parallelism yet."
        elif get_moe_a2a_backend().is_deepep():
            disable_reason = "Shared experts fusion is not supported when Deepep MoE backend is enabled."

        if disable_reason is not None:
            get_global_server_args().disable_shared_experts_fusion = True
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = self.config.n_shared_experts
        assert (
            self.num_fused_shared_experts == 1
        ), "Only 1 fused shared expert is supported for Glm4vMoeForConditionalGeneration"
        log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                # compatible with old design
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]

        params_dict = dict(self.named_parameters())
        weight_names = []
        for name, loaded_weight in weights:
            weight_names.append(name)

            if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
                # Shared expert becomes expert ID = n_routed_experts
                name = name.replace(
                    "mlp.shared_experts",
                    f"mlp.experts.{self.config.n_routed_experts}",
                )

            if not is_nextn:
                if hasattr(self.config, "num_nextn_predict_layers"):
                    num_nextn_layers = self.config.num_nextn_predict_layers
                    if num_nextn_layers > 0 and name.startswith("model.layers"):
                        name_list = name.split(".")
                        if (
                            len(name_list) >= 3
                            and int(name_list[2]) >= self.config.num_hidden_layers
                        ):
                            continue
            else:
                if not name.startswith(nextn_layer_prefix):
                    continue

                # Use shared head and embed weights from target model
                if "shared_head.head" in name or "embed_tokens" in name:
                    continue

                is_decoder = True
                # For nextn specific weights
                for weight_name in nextn_spec_weight_names:
                    if weight_name in name:
                        name = name.replace(nextn_layer_prefix, "model")
                        is_decoder = False
                        break
                # For decoder layer weights
                if is_decoder:
                    name = name.replace(nextn_layer_prefix, "model.decoder")

            if "language_model." in name:
                name = name.replace("language_model.", "")
            if "model.visual." in name:
                name = name.replace("model.visual.", "visual.")
            if "rotary_emb.inv_freq" in name:
                continue

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
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Mark as expert weight regardless of whether we can process it
                    is_expert_weight = True

                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        # Expert weight not on this rank, will be skipped below
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue

                    if "visual" in name:
                        # adapt to VisionAttention for GLM-V
                        name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        if "visual" in name:
                            loaded_weight = vision_utils.pad_vit_attn_dummy_heads(
                                self.config, name, loaded_weight
                            )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")


EntryClass = [Glm4vMoeForConditionalGeneration]
