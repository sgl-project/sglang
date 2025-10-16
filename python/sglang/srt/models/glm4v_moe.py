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
from sglang.srt.layers.attention import vision_utils
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.glm4_moe import Glm4MoeModel
from sglang.srt.models.glm4v import Glm4vForConditionalGeneration, Glm4vVisionModel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, log_info_on_rank0
from sglang.srt.utils.hf_transformers_utils import get_processor

_is_cuda = is_cuda()

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

        config.moe_layer_freq = 1
        self.config = config
        vision_utils.update_vit_attn_dummy_heads_config(self.config)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts("Glm4MoeForCausalLM")
        self.num_fused_shared_experts = (
            0
            if get_global_server_args().disable_shared_experts_fusion
            else config.n_shared_experts
        )

        self.model = Glm4MoeModel(
            config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.visual = Glm4vVisionModel(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-5),
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def determine_num_fused_shared_experts(
        self, architecture: str = "Glm4MoeForCausalLM"
    ):
        self.num_fused_shared_experts = 0
        if get_global_server_args().disable_shared_experts_fusion:
            return

        # Only Deepseek V3/R1 can use shared experts fusion optimization now.
        disable_reason = None
        if (
            not _is_cuda
            or torch.cuda.get_device_capability("cuda") < (8, 0)
            or self.config.architectures[0] != architecture
            or self.config.n_shared_experts != 1
        ):
            disable_reason = "Only GLM-4.5 on NV-platform with capability >= 80 can use shared experts fusion optimization."
        elif get_moe_expert_parallel_world_size() > 1:
            disable_reason = "Deepseek and GLM-4.5 can not use shared experts fusion optimization under expert parallelism."

        if disable_reason is not None:
            get_global_server_args().disable_shared_experts_fusion = True
            self.num_fused_shared_experts = 0
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = self.config.n_shared_experts

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
        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            weights_list = list(weights)
            weights_dict = dict(weights_list)
            if self.quant_config is not None:
                if self.quant_config.get_name() == "w8a8_int8":
                    suffix_list = [
                        "down_proj.weight",
                        "down_proj.weight_scale",
                        "gate_proj.weight",
                        "gate_proj.weight_scale",
                        "up_proj.weight",
                        "up_proj.weight_scale",
                    ]
                elif (
                    self.quant_config.get_name() == "fp8"
                    or self.quant_config.get_name() == "blockwise_int8"
                    or self.quant_config.get_name() == "compressed_tensors"
                ):
                    suffix_list = [
                        "down_proj.weight",
                        "down_proj.weight_scale",
                        "gate_proj.weight",
                        "gate_proj.weight_scale",
                        "up_proj.weight",
                        "up_proj.weight_scale",
                    ]
                elif self.quant_config.get_name() == "awq":
                    suffix_list = [
                        "down_proj.qweight",
                        "down_proj.qzeros",
                        "down_proj.scales",
                        "gate_proj.qweight",
                        "gate_proj.qzeros",
                        "gate_proj.scales",
                        "up_proj.qweight",
                        "up_proj.qzeros",
                        "up_proj.scales",
                    ]
                elif self.quant_config.get_name() == "modelopt_fp4":
                    suffix_list = [
                        "down_proj.weight",
                        "down_proj.weight_scale",
                        "down_proj.weight_scale_2",
                        "down_proj.input_scale",
                        "gate_proj.weight",
                        "gate_proj.weight_scale",
                        "gate_proj.weight_scale_2",
                        "gate_proj.input_scale",
                        "up_proj.weight",
                        "up_proj.weight_scale",
                        "up_proj.weight_scale_2",
                        "up_proj.input_scale",
                    ]
                else:
                    raise ValueError(
                        f"Unsupported shared expert fusion for quantization: {self.quant_config.get_name()}."
                    )
            else:
                suffix_list = [
                    "down_proj.weight",
                    "gate_proj.weight",
                    "up_proj.weight",
                ]
            names_to_remove = []

            moe_layers = (
                range(
                    self.config.first_k_dense_replace,
                    self.config.num_hidden_layers,
                    self.config.moe_layer_freq,
                )
                if not is_nextn
                else [nextn_layer_id]
            )

            for moe_layer in moe_layers:
                for suffix in suffix_list:
                    shared_expert_weight_name = (
                        f"model.layers.{moe_layer}.mlp.shared_experts.{suffix}"
                    )
                    # online fp8 quantization does not load weight_scale
                    if shared_expert_weight_name not in weights_dict:
                        continue
                    weights_list.append(
                        (
                            f"model.layers.{moe_layer}."
                            f"mlp.experts."
                            f"{self.config.n_routed_experts + 0}"
                            f".{suffix}",
                            weights_dict[shared_expert_weight_name],
                        )
                    )
                    names_to_remove += [shared_expert_weight_name]
            weights = [w for w in weights_list if w[0] not in names_to_remove]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None

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
                if ("mlp.experts." in name) and name not in params_dict:
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
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
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
                    if "visual" in name:
                        # adapt to VisionAttention
                        name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if fuse_qkv_a_proj and (
                        "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                    ):
                        cached_a_proj[name] = loaded_weight
                        q_a_proj_name = (
                            name
                            if "q_a_proj" in name
                            else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                        )
                        kv_a_proj_name = (
                            name
                            if "kv_a_proj_with_mqa" in name
                            else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                        )

                        # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter
                        if (
                            q_a_proj_name in cached_a_proj
                            and kv_a_proj_name in cached_a_proj
                        ):
                            q_a_proj_weight = cached_a_proj[q_a_proj_name]
                            kv_a_proj_weight = cached_a_proj[kv_a_proj_name]
                            fused_weight = torch.cat(
                                [q_a_proj_weight, kv_a_proj_weight], dim=0
                            )
                            param_name = (
                                name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                                if "q_a_proj" in name
                                else name.replace(
                                    "kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa"
                                )
                            )
                            param = params_dict[param_name]

                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, fused_weight)
                            cached_a_proj.pop(q_a_proj_name)
                            cached_a_proj.pop(kv_a_proj_name)
                    else:
                        if (
                            "k_scale" in name or "v_scale" in name
                        ) and name not in params_dict:
                            # modelopt attn kv scale is named differently
                            if any(scale in name for scale in ["k_scale", "v_scale"]):
                                name = name.replace("_proj", "attn_mqa")
                            else:
                                logger.warning(
                                    f"Unknown scale found in checkpoint: {name}"
                                )
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        if "visual" in name:
                            loaded_weight = vision_utils.pad_vit_attn_dummy_heads(
                                self.config, name, loaded_weight
                            )
                        weight_loader(param, loaded_weight)


EntryClass = [Glm4vMoeForConditionalGeneration]
