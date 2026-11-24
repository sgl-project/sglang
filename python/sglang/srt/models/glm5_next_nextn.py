# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging

import torch

from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.utils import is_gfx95_supported, is_hip

logger = logging.getLogger(__name__)

_is_gfx95 = is_hip() and is_gfx95_supported()


class Glm5NextForConditionalGenerationNextN(DeepseekV3ForCausalLMNextN):
    def prepare_cp_inputs(self, forward_batch, **kwargs):
        # The worker has rotated the target's multimodal embeddings per request.
        # Fill each appended token while indices still address full sequences,
        # then let the runner shard embeddings and target states together.
        input_embeds = self.model.embed_input_ids(
            forward_batch.input_ids, forward_batch
        )
        return input_embeds, forward_batch.positions, {}

    @classmethod
    def get_hf_to_sglang_mapper(cls, config) -> WeightsMapper:
        text_config = getattr(config, "text_config", config)
        n = text_config.num_hidden_layers
        # lookups arrive as checkpoint and normalized names, so every rule has both forms
        draft_rules: dict[str, str] = {}
        for ckpt_prefix in (f"model.language_model.layers.{n}", f"model.layers.{n}"):
            # eh_proj/enorm/hnorm sit beside the block under `model`, not in `decoder`
            draft_rules[f"{ckpt_prefix}.eh_proj"] = "model.eh_proj"
            draft_rules[f"{ckpt_prefix}.enorm"] = "model.enorm"
            draft_rules[f"{ckpt_prefix}.hnorm"] = "model.hnorm"
            draft_rules[ckpt_prefix] = "model.decoder"
        # target rules still normalize the non-draft layers and vision tower in `exclude`
        return Glm5NextForConditionalGeneration.hf_to_sglang_mapper | WeightsMapper(
            orig_to_new_substr=draft_rules,
        )

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        if not _is_gfx95:
            return super().shared_experts_fusion_disable_reason(hf_config, quant_config)
        return Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason(
            hf_config, quant_config
        )

    @classmethod
    def num_fusable_shared_experts(cls, hf_config) -> int:
        if not _is_gfx95:
            return super().num_fusable_shared_experts(hf_config)
        text_config = getattr(hf_config, "text_config", hf_config)
        assert text_config.n_shared_experts == 1, (
            f"Only 1 fused shared expert is supported for {cls.__name__}"
        )
        return text_config.n_shared_experts

    def _resolve_nextn_quant_config(self, config, quant_config):
        """Mixed checkpoints list the BF16 NextN block in ``quantization_config.ignore``;
        inheriting global FP8 quantization would corrupt its QKV weights."""
        raw_quant_config = getattr(config, "quantization_config", None) or {}
        if hasattr(raw_quant_config, "to_dict"):
            raw_quant_config = raw_quant_config.to_dict()
        ignored = (
            raw_quant_config.get("ignore", [])
            if isinstance(raw_quant_config, dict)
            else []
        )
        nextn_layer_pattern = f"model.layers.{config.num_hidden_layers}.*"
        if nextn_layer_pattern in ignored:
            logger.warning(
                "GLM5 NextN layer %s is checkpoint-declared unquantized; "
                "using BF16 draft modules",
                nextn_layer_pattern,
            )
            return None
        return super()._resolve_nextn_quant_config(config, quant_config)

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        Glm5NextForConditionalGeneration.register_modelslim_projection_aliases(
            quant_config
        )
        super().__init__(
            getattr(config, "text_config", config),
            quant_config=quant_config,
            prefix=prefix,
        )
        self.logits_processor.draft_fp8_lm_head = True

    def load_weights(self, weights):
        if not hasattr(self, "fuse_qkv_a_proj"):
            self.fuse_qkv_a_proj = getattr(self.config, "q_lora_rank", None) is not None
        layer_id = self.config.num_hidden_layers
        layer_prefixes = (
            f"model.layers.{layer_id}.",
            f"model.language_model.layers.{layer_id}.",
        )
        nextn_weights = (
            (name, weight)
            for name, weight in weights
            if name.startswith(layer_prefixes)
        )
        result = Glm5NextForConditionalGeneration.load_weights(
            self, nextn_weights, is_nextn=True
        )
        if _is_gfx95:
            self._experts_to_mxfp4()
        return result

    def _experts_to_mxfp4(self):
        """Swap FP8 block-128 experts for MXFP4 before the loader's post-load pass.

        The weights are dequantized to bf16 here; Mxfp4DynamicQuantMoEMethod's
        process_weights_after_loading then quantizes and shuffles them.
        """
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            FusedMoE,
            _fuses_routed_scaling_factor_in_topk,
        )
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4DynamicQuantMoEMethod

        for module in self.modules():
            method = getattr(module, "quant_method", None)
            if not (
                isinstance(module, FusedMoE)
                and type(method) is Fp8MoEMethod
                and list(method.weight_block_size or []) == [128, 128]
                and method.w2_weight_block_size in (None, [128, 128])
            ):
                continue
            for name in ("w13", "w2"):
                weight = getattr(module, f"{name}_weight")
                scale = getattr(module, f"{name}_weight_scale_inv")
                rows, cols = weight.shape[1:]
                out = torch.empty(
                    weight.shape, dtype=torch.bfloat16, device=weight.device
                )
                for e in range(weight.shape[0]):
                    s = scale[e].repeat_interleave(128, 0)[:rows]
                    s = s.repeat_interleave(128, 1)[:, :cols]
                    out[e] = (weight[e].float() * s).to(torch.bfloat16)
                setattr(
                    module,
                    f"{name}_weight",
                    torch.nn.Parameter(out, requires_grad=False),
                )
                delattr(module, f"{name}_weight_scale_inv")
            mxfp4 = Mxfp4DynamicQuantMoEMethod()
            module.quant_method = mxfp4
            mxfp4.create_moe_runner(module, module.moe_runner_config)
            module.runner_backend = mxfp4.get_moe_runner_backend(module)
            module.moe_runner_config.runner_backend = module.runner_backend
            module._set_runner_flags()
            module.should_fuse_routed_scaling_factor_in_topk = (
                _fuses_routed_scaling_factor_in_topk(mxfp4, module.runner_backend)
            )
            module.runner = mxfp4.runner
            logger.info(
                "NextN experts %s: FP8 block-128 -> MXFP4", type(module).__name__
            )
            torch.cuda.empty_cache()


EntryClass = [Glm5NextForConditionalGenerationNextN]
