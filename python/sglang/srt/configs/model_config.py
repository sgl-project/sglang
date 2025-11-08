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

import json
import logging
import math
import os
from enum import Enum, IntEnum, auto
from typing import Any, List, Optional, Set, Union

import torch
from transformers import PretrainedConfig

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import QUANTIZATION_METHODS
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_hip, retry
from sglang.srt.utils.hf_transformers_utils import (
    get_config,
    get_context_length,
    get_generation_config,
    get_hf_text_config,
    get_sparse_attention_config,
)
from sglang.utils import is_in_ci

logger = logging.getLogger(__name__)


class AttentionArch(IntEnum):
    MLA = auto()
    MHA = auto()


class ModelImpl(str, Enum):
    AUTO = "auto"
    SGLANG = "sglang"
    TRANSFORMERS = "transformers"


def is_deepseek_nsa(config: PretrainedConfig) -> bool:
    return (
        config.architectures is not None
        and config.architectures[0]
        in [
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
            "DeepseekV3ForCausalLMNextN",
        ]
        and getattr(config, "index_topk", None) is not None
    )


def get_nsa_index_head_dim(config: PretrainedConfig) -> int:
    assert is_deepseek_nsa(config)
    return config.index_head_dim


def get_nsa_index_topk(config: PretrainedConfig) -> int:
    assert is_deepseek_nsa(config)
    return config.index_topk


def get_nsa_index_n_heads(config: PretrainedConfig) -> int:
    assert is_deepseek_nsa(config)
    return config.index_n_heads


class ModelConfig:
    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = True,
        revision: Optional[str] = None,
        context_length: Optional[int] = None,
        model_override_args: str = "{}",
        is_embedding: Optional[bool] = None,
        enable_multimodal: Optional[bool] = None,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        override_config_file: Optional[str] = None,
        is_draft_model: bool = False,
        hybrid_kvcache_ratio: Optional[
            float
        ] = None,  # TODO: remove this, it is not a model config
        model_impl: Union[str, ModelImpl] = ModelImpl.AUTO,
        sampling_defaults: str = "openai",
        quantize_and_serve: bool = False,
    ) -> None:
        # Parse args
        self.model_path = model_path
        self.revision = revision
        self.quantization = quantization
        self.is_draft_model = is_draft_model
        self.model_impl = model_impl
        self.sampling_defaults = sampling_defaults
        self.quantize_and_serve = quantize_and_serve

        # Validate quantize_and_serve configuration
        self._validate_quantize_and_serve_config()

        # Get hf config
        self._maybe_pull_model_tokenizer_from_remote()
        self.model_override_args = json.loads(model_override_args)
        kwargs = {}
        if override_config_file and override_config_file.strip():
            kwargs["_configuration_file"] = override_config_file.strip()
        self.hf_config = get_config(
            self.model_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            model_override_args=self.model_override_args,
            **kwargs,
        )
        self.hf_text_config = get_hf_text_config(self.hf_config)
        self.hf_generation_config = get_generation_config(
            self.model_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )

        # Set enable_multimodal
        if enable_multimodal is None:
            mm_disabled_models = [
                "Gemma3ForConditionalGeneration",
                "Llama4ForConditionalGeneration",
                "Step3VLForConditionalGeneration",
            ]
            if self.hf_config.architectures[0] in mm_disabled_models:
                enable_multimodal = False
                logger.info(
                    f"Multimodal is disabled for {self.hf_config.model_type}. To enable it, set --enable-multimodal."
                )
            else:
                enable_multimodal = True

        # Config draft model
        self._config_draft_model()

        # Check model type
        self.attention_chunk_size = getattr(
            self.hf_text_config, "attention_chunk_size", None
        )
        self.is_hybrid = is_hybrid_model(
            self.hf_config.architectures,
            hybrid_kvcache_ratio=hybrid_kvcache_ratio,
            context_length=context_length,
            attention_chunk_size=self.attention_chunk_size,
        )
        if self.is_hybrid is not None:
            self.swa_attention_layer_ids, self.full_attention_layer_ids = (
                get_hybrid_layer_ids(
                    self.hf_config.architectures, self.hf_text_config.num_hidden_layers
                )
            )
        self.is_generation = is_generation_model(
            self.hf_config.architectures, is_embedding
        )
        self.is_multimodal = enable_multimodal and is_multimodal_model(
            self.hf_config.architectures
        )
        self.is_multimodal_gen = enable_multimodal and is_multimodal_gen_model(
            self.hf_config.architectures
        )
        self.is_image_gen = enable_multimodal and is_image_gen_model(
            self.hf_config.architectures
        )
        self.is_audio_model = enable_multimodal and is_audio_model(
            self.hf_config.architectures
        )
        self.is_multimodal_chunked_prefill_supported = (
            enable_multimodal
            and is_multimodal_chunked_prefill_supported(self.hf_config.architectures)
        )
        self.is_encoder_decoder = is_encoder_decoder_model(self.hf_config.architectures)
        self.dtype = _get_and_verify_dtype(self.hf_text_config, dtype)

        # Derive context length and model shapes
        self._derive_context_length(context_length)
        self._derive_model_shapes()

        # Verify quantization
        self._verify_quantization()

        # Verify dual-chunk attention config
        self._verify_dual_chunk_attention_config()

        # Cache attributes
        self.hf_eos_token_id = self._get_hf_eos_token_id()

        # multimodal
        self.image_token_id = getattr(
            self.hf_config, "image_token_id", None
        ) or getattr(self.hf_config, "image_token_index", None)

        # matryoshka embeddings
        self.matryoshka_dimensions = getattr(
            self.hf_config, "matryoshka_dimensions", None
        )
        self.is_matryoshka = self.matryoshka_dimensions or getattr(
            self.hf_config, "is_matryoshka", False
        )

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
        model_path: str = None,
        model_revision: str = None,
        **kwargs,
    ):
        return ModelConfig(
            model_path=model_path or server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=model_revision or server_args.revision,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
            is_embedding=server_args.is_embedding,
            enable_multimodal=server_args.enable_multimodal,
            dtype=server_args.dtype,
            quantization=server_args.quantization,
            hybrid_kvcache_ratio=server_args.hybrid_kvcache_ratio,
            model_impl=server_args.model_impl,
            sampling_defaults=server_args.sampling_defaults,
            quantize_and_serve=server_args.quantize_and_serve,
            override_config_file=server_args.decrypted_config_file,
            **kwargs,
        )

    def _config_draft_model(self):
        is_draft_model = self.is_draft_model

        if (
            is_draft_model
            and self.hf_config.architectures[0] == "DeepseekV3ForCausalLM"
        ):
            self.hf_config.architectures[0] = "DeepseekV3ForCausalLMNextN"

        if is_draft_model and self.hf_config.architectures[0] == "Glm4MoeForCausalLM":
            self.hf_config.architectures[0] = "Glm4MoeForCausalLMNextN"

        if (
            is_draft_model
            and self.hf_config.architectures[0] == "LongcatFlashForCausalLM"
        ):
            self.hf_config.architectures[0] = "LongcatFlashForCausalLMNextN"
            self.hf_config.num_hidden_layers = self.hf_config.num_nextn_predict_layers

        if is_draft_model and self.hf_config.architectures[0] == "MiMoForCausalLM":
            self.hf_config.architectures[0] = "MiMoMTP"
        if is_draft_model and self.hf_config.architectures[0] in [
            "BailingMoeV2ForCausalLM",
            "BailingMoeForCausalLM",
        ]:
            self.hf_config.architectures[0] = "BailingMoeForCausalLMNextN"
        if (
            is_draft_model
            and self.hf_config.architectures[0] == "Ernie4_5_MoeForCausalLM"
        ):
            self.hf_config.architectures[0] = "Ernie4_5_MoeForCausalLMMTP"

        if is_draft_model and self.hf_config.architectures[0] == "Qwen3NextForCausalLM":
            self.hf_config.architectures[0] = "Qwen3NextForCausalLMMTP"
            self.hf_config.num_nextn_predict_layers = 1

    def _derive_context_length(self, context_length: int):
        is_draft_model = self.is_draft_model
        derived_context_len = get_context_length(self.hf_text_config)

        if context_length is not None:
            if context_length > derived_context_len:
                reason = "Target model's" if is_draft_model else "User-specified"
                msg = (
                    f"Warning: {reason} context_length ({context_length}) is greater than the derived context_length ({derived_context_len}). "
                    f"This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config."
                )
                if (
                    envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.get()
                    or is_in_ci()  # FIXME: fix this special case
                ):
                    logger.warning(msg)
                    self.context_len = context_length
                    if is_draft_model:
                        self.hf_text_config.max_position_embeddings = context_length
                        logger.warning(
                            f"Overriding the draft model's max_position_embeddings to {context_length}."
                        )
                else:
                    raise ValueError(
                        f"{msg} To allow overriding this maximum, set the env var SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"
                    )
            else:
                self.context_len = context_length
        else:
            self.context_len = derived_context_len

        # Transfer context_len to HuggingFace config so models can access it
        self.hf_config.context_len = self.context_len

    def _derive_model_shapes(self):
        # Unify the config keys for hf_text_config
        self.head_dim = getattr(
            self.hf_text_config,
            "head_dim",
            self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads,
        )

        # FIXME: temporary special judge for MLA architecture
        if (
            "DeepseekV2ForCausalLM" in self.hf_config.architectures
            or "DeepseekV32ForCausalLM" in self.hf_config.architectures
            or "DeepseekV3ForCausalLM" in self.hf_config.architectures
            or "DeepseekV3ForCausalLMNextN" in self.hf_config.architectures
            or "LongcatFlashForCausalLM" in self.hf_config.architectures
            or "LongcatFlashForCausalLMNextN" in self.hf_config.architectures
            or "DotsVLMForCausalLM" in self.hf_config.architectures
        ):
            self.head_dim = 256
            self.attention_arch = AttentionArch.MLA
            self.kv_lora_rank = self.hf_config.kv_lora_rank
            self.qk_nope_head_dim = self.hf_config.qk_nope_head_dim
            self.qk_rope_head_dim = self.hf_config.qk_rope_head_dim
            self.v_head_dim = self.hf_config.v_head_dim
            self.index_head_dim = (
                get_nsa_index_head_dim(self.hf_config)
                if is_deepseek_nsa(self.hf_config)
                else None
            )

            # Handle rope scaling with yarn
            self.scaling = 1 / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)
            if self.hf_config.rope_scaling:
                mscale_all_dim = self.hf_config.rope_scaling.get(
                    "mscale_all_dim", False
                )
                scaling_factor = self.hf_config.rope_scaling["factor"]
                mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                self.scaling = self.scaling * mscale * mscale

        elif "MiniCPM3ForCausalLM" in self.hf_config.architectures:
            self.head_dim = 128
            self.attention_arch = AttentionArch.MLA
            self.kv_lora_rank = self.hf_config.kv_lora_rank
            self.qk_rope_head_dim = self.hf_config.qk_rope_head_dim
        elif "DeepseekVL2ForCausalLM" in self.hf_config.architectures and getattr(
            self.hf_text_config, "use_mla", True
        ):
            self.head_dim = 256
            self.attention_arch = AttentionArch.MLA
            self.kv_lora_rank = self.hf_text_config.kv_lora_rank
            self.qk_rope_head_dim = self.hf_text_config.qk_rope_head_dim
        elif "KimiVLForConditionalGeneration" in self.hf_config.architectures:
            self.head_dim = 256
            self.attention_arch = AttentionArch.MLA
            self.kv_lora_rank = self.hf_text_config.kv_lora_rank
            self.qk_rope_head_dim = self.hf_text_config.qk_rope_head_dim
            self.v_head_dim = self.hf_text_config.v_head_dim
            self.qk_nope_head_dim = self.hf_text_config.qk_nope_head_dim
        elif "KimiLinearForCausalLM" in self.hf_config.architectures:
            self.head_dim = 72
            self.attention_arch = AttentionArch.MLA
            self.kv_lora_rank = self.hf_config.kv_lora_rank
            self.qk_rope_head_dim = self.hf_config.qk_rope_head_dim
            self.v_head_dim = self.hf_config.v_head_dim
            self.qk_nope_head_dim = self.hf_config.qk_nope_head_dim
        else:
            if (
                "MistralModel" in self.hf_config.architectures
                or "MixtralForCausalLM" in self.hf_config.architectures
                or "MistralForCausalLM" in self.hf_config.architectures
            ):
                if getattr(self, "head_dim", None) is None:
                    self.head_dim = (
                        self.hf_config.hidden_size // self.hf_config.num_attention_heads
                    )
                    # In transformers==4.52.3, the head_dim is null in MistralConfig
                    if (
                        not hasattr(self.hf_text_config, "head_dim")
                        or self.hf_text_config.head_dim is None
                    ):
                        setattr(self.hf_text_config, "head_dim", self.head_dim)

            self.attention_arch = AttentionArch.MHA

        self.num_attention_heads = self.hf_text_config.num_attention_heads
        self.num_key_value_heads = getattr(
            self.hf_text_config, "num_key_value_heads", None
        )

        # for Dbrx and MPT models
        if self.hf_config.model_type in ["dbrx", "mpt"]:
            self.num_key_value_heads = getattr(
                self.hf_config.attn_config, "kv_n_heads", None
            )

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.hidden_size = self.hf_text_config.hidden_size
        self.num_hidden_layers = self.hf_text_config.num_hidden_layers
        self.num_attention_layers = self.num_hidden_layers
        if "LongcatFlashForCausalLM" in self.hf_config.architectures:
            self.num_attention_layers = self.num_hidden_layers * 2
        self.num_nextn_predict_layers = getattr(
            self.hf_text_config, "num_nextn_predict_layers", None
        )
        self.vocab_size = self.hf_text_config.vocab_size

    def get_total_num_attention_heads(self) -> int:
        return self.num_attention_heads

    def get_num_attention_heads(self, tensor_parallel_size) -> int:
        total_num_attention_heads = self.num_attention_heads
        return max(1, total_num_attention_heads // tensor_parallel_size)

    # adapted from https://github.com/vllm-project/vllm/blob/main/vllm/config.py#L289
    def get_total_num_kv_heads(self) -> int:
        """Returns the total number of KV heads."""
        # For GPTBigCode & Falcon:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False)
        )
        if not new_decoder_arch_falcon and getattr(
            self.hf_text_config, "multi_query", False
        ):
            # Multi-query attention, only one KV head.
            # Currently, tensor parallelism is not supported in this case.
            return 1

        # For DBRX and MPT
        if self.hf_config.model_type in ["mpt"]:
            if "kv_n_heads" in self.hf_config.attn_config:
                return self.hf_config.attn_config["kv_n_heads"]
            return self.hf_config.num_attention_heads
        if self.hf_config.model_type in ["dbrx"]:
            return getattr(
                self.hf_config.attn_config,
                "kv_n_heads",
                self.hf_config.num_attention_heads,
            )
        if self.hf_config.model_type in ["nemotron-nas"]:
            nkvh = {
                self.hf_config.num_attention_heads // block.attention.n_heads_in_group
                for block in self.hf_config.block_configs
                if not block.attention.no_op
            }
            if len(nkvh) == 0:
                raise RuntimeError("Couldn't determine number of kv heads")
            if len(nkvh) > 1:
                raise ValueError(
                    "Variable GQA (VGQA) is not yet supported for nemotron-nas in sglang"
                )
            return next(iter(nkvh))

        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
            # For Step3
            "num_attention_groups",
        ]
        for attr in attributes:
            num_kv_heads = getattr(self.hf_text_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads

        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        return self.hf_text_config.num_attention_heads

    def get_num_kv_heads(self, tensor_parallel_size) -> int:
        """Returns the number of KV heads per GPU."""
        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1, total_num_kv_heads // tensor_parallel_size)

    # adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/config.py
    def _parse_quant_hf_config(self):
        quant_cfg = getattr(self.hf_config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(self.hf_config, "compression_config", None)
        if quant_cfg is None:
            # check if is modelopt or mixed-precision model -- Both of them don't have corresponding field
            # in hf `config.json` but has a standalone `hf_quant_config.json` in the root directory
            # example: https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8/tree/main
            # example: https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8/tree/main
            is_local = os.path.exists(self.model_path)
            if not is_local:
                import huggingface_hub

                try:
                    from huggingface_hub import HfApi, hf_hub_download

                    hf_api = HfApi()
                    # Retry HF API call up to 3 times
                    file_exists = retry(
                        lambda: hf_api.file_exists(
                            self.model_path, "hf_quant_config.json"
                        ),
                        max_retry=2,
                        initial_delay=1.0,
                        max_delay=5.0,
                    )
                    if file_exists:
                        # Download and parse the quantization config for remote models
                        quant_config_file = hf_hub_download(
                            repo_id=self.model_path,
                            filename="hf_quant_config.json",
                            revision=self.revision,
                        )
                        with open(quant_config_file) as f:
                            quant_config_dict = json.load(f)
                        quant_cfg = self._parse_modelopt_quant_config(quant_config_dict)
                except huggingface_hub.errors.OfflineModeIsEnabled:
                    logger.warning(
                        "Offline mode is enabled, skipping hf_quant_config.json check"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to check hf_quant_config.json: {self.model_path} {e}"
                    )
            elif os.path.exists(os.path.join(self.model_path, "hf_quant_config.json")):
                quant_config_file = os.path.join(
                    self.model_path, "hf_quant_config.json"
                )
                with open(quant_config_file) as f:
                    quant_config_dict = json.load(f)
                quant_cfg = self._parse_modelopt_quant_config(quant_config_dict)
        return quant_cfg

    def _parse_modelopt_quant_config(self, quant_config_dict: dict) -> Optional[dict]:
        """Parse ModelOpt quantization config and return the appropriate quant_method."""
        json_quant_configs = quant_config_dict["quantization"]
        quant_algo = json_quant_configs.get("quant_algo", None)

        if quant_algo == "MIXED_PRECISION":
            return {"quant_method": "w4afp8"}
        elif quant_algo and ("FP4" in quant_algo or "NVFP4" in quant_algo):
            return {"quant_method": "modelopt_fp4"}
        elif quant_algo and "FP8" in quant_algo:
            return {"quant_method": "modelopt_fp8"}
        else:
            return None

    def _is_already_quantized(self) -> bool:
        """Check if the model is already quantized based on config files."""
        # Check for HuggingFace quantization config
        from sglang.srt.utils import has_hf_quant_config

        return has_hf_quant_config(self.model_path)

    def _get_modelopt_quant_type(self) -> str:
        """Extract ModelOpt quantization type from unified quantization flag."""
        if self.quantization == "modelopt_fp8":
            return "fp8"
        elif self.quantization == "modelopt_fp4":
            return "nvfp4"
        elif self.quantization == "modelopt":
            # Auto-detect from model config
            quant_cfg = self._parse_quant_hf_config()
            if quant_cfg:
                quant_method = quant_cfg.get("quant_method", "").lower()
                if "fp4" in quant_method:
                    return "fp4"
                elif "fp8" in quant_method:
                    return "fp8"
            # Default to fp8 if can't detect
            return "fp8"
        else:
            return "fp8"  # Default fallback

    def _validate_quantize_and_serve_config(self):
        """Validate quantize_and_serve configuration."""
        if not self.quantize_and_serve:
            return

        # Check if ModelOpt quantization is specified
        _MODELOPT_QUANTIZATION_METHODS = [
            "modelopt",
            "modelopt_fp8",
            "modelopt_fp4",
        ]
        modelopt_quantization_specified = (
            self.quantization in _MODELOPT_QUANTIZATION_METHODS
        )

        if not modelopt_quantization_specified:
            raise ValueError(
                "quantize_and_serve requires ModelOpt quantization (set with --quantization "
                f"{{{', '.join(sorted(_MODELOPT_QUANTIZATION_METHODS))}}})"
            )

        # quantize_and_serve is disabled due to compatibility issues
        raise NotImplementedError(
            "quantize_and_serve functionality is currently disabled due to compatibility issues. "
            "Please use the separate quantize-then-deploy workflow instead. "
            "Step 1: Quantize and export model. "
            "Step 2: Deploy the exported model."
        )

    # adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/config.py
    def _verify_quantization(self) -> None:
        supported_quantization = [*QUANTIZATION_METHODS]
        rocm_supported_quantization = [
            "awq",
            "gptq",
            "fp8",
            "compressed_tensors",
            "compressed-tensors",
            "fbgemm_fp8",
            "w8a8_fp8",
            "petit_nvfp4",
            "quark",
            "mxfp4",
            "auto-round",
        ]
        optimized_quantization_methods = [
            "fp8",
            "marlin",
            "modelopt_fp8",
            "modelopt_fp4",
            "gptq_marlin_24",
            "gptq_marlin",
            "awq_marlin",
            "fbgemm_fp8",
            "compressed_tensors",
            "compressed-tensors",
            "experts_int8",
            "w8a8_int8",
            "w8a8_fp8",
            "moe_wna16",
            "qoq",
            "w4afp8",
            "petit_nvfp4",
        ]
        compatible_quantization_methods = {
            "modelopt_fp8": ["modelopt"],
            "modelopt_fp4": ["modelopt"],
            "petit_nvfp4": ["modelopt"],
            "w8a8_int8": ["compressed-tensors", "compressed_tensors"],
            "w8a8_fp8": ["compressed-tensors", "compressed_tensors"],
        }
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self._parse_quant_hf_config()

        if quant_cfg is not None:
            quant_method = quant_cfg.get(
                "quant_method", "" if not self.quantization else self.quantization
            ).lower()

            # Detect which checkpoint is it
            for _, method in QUANTIZATION_METHODS.items():
                quantization_override = method.override_quantization_method(
                    quant_cfg, self.quantization
                )
                if quantization_override:
                    quant_method = quantization_override
                    self.quantization = quantization_override
                    break

            # Verify quantization configurations.
            if self.quantization is None:
                self.quantization = quant_method
            elif self.quantization != quant_method:
                if (
                    self.quantization not in compatible_quantization_methods
                    or quant_method
                    not in compatible_quantization_methods[self.quantization]
                ):
                    raise ValueError(
                        "Quantization method specified in the model config "
                        f"({quant_method}) does not match the quantization "
                        f"method specified in the `quantization` argument "
                        f"({self.quantization})."
                    )

        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. Must "
                    f"be one of {supported_quantization}."
                )
            if is_hip() and self.quantization not in rocm_supported_quantization:
                raise ValueError(
                    f"{self.quantization} quantization is currently not "
                    f"supported in ROCm."
                )
            if self.quantization not in optimized_quantization_methods:
                logger.warning(
                    "%s quantization is not fully "
                    "optimized yet. The speed can be slower than "
                    "non-quantized models.",
                    self.quantization,
                )

    def _verify_dual_chunk_attention_config(self) -> None:
        if hasattr(self.hf_config, "dual_chunk_attention_config"):
            # Try loading the sparse attention config
            sparse_attn_config = get_sparse_attention_config(self.model_path)
            if not sparse_attn_config:
                return
            self.hf_config.dual_chunk_attention_config["sparse_attention_config"] = (
                sparse_attn_config
            )
            if (
                "sparse_attention_enabled"
                not in self.hf_config.dual_chunk_attention_config
            ):
                self.hf_config.dual_chunk_attention_config[
                    "sparse_attention_enabled"
                ] = True

    def _get_hf_eos_token_id(self) -> Optional[Set[int]]:
        eos_ids = getattr(self.hf_config, "eos_token_id", None)
        if eos_ids is not None:
            # it can be either int or list of int
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
        if eos_ids is None:
            eos_ids = set()
        if self.hf_generation_config:
            generation_eos_ids = getattr(
                self.hf_generation_config, "eos_token_id", None
            )
            if generation_eos_ids:
                generation_eos_ids = (
                    {generation_eos_ids}
                    if isinstance(generation_eos_ids, int)
                    else set(generation_eos_ids)
                )
                eos_ids = eos_ids | generation_eos_ids
        return eos_ids

    def get_default_sampling_params(self) -> dict[str, Any]:
        """
        Get default sampling parameters from the model's generation config.

        This method returns non-default sampling parameters from the model's
        generation_config.json when sampling_defaults is set to "model".

        Returns:
            A dictionary containing the non-default sampling parameters.
        """
        if self.sampling_defaults != "model":
            return {}

        if self.hf_generation_config is None:
            return {}

        config = self.hf_generation_config.to_dict()

        available_params = [
            "repetition_penalty",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
        ]

        default_sampling_params = {
            p: config.get(p) for p in available_params if config.get(p) is not None
        }

        return default_sampling_params

    def _maybe_pull_model_tokenizer_from_remote(self) -> None:
        """
        Pull the model config files to a temporary
        directory in case of remote.

        Args:
            model: The model name or path.

        """
        from sglang.srt.connector import create_remote_connector
        from sglang.srt.utils import is_remote_url

        if is_remote_url(self.model_path):
            logger.info("Pulling model configs from remote...")
            # BaseConnector implements __del__() to clean up the local dir.
            # Since config files need to exist all the time, so we DO NOT use
            # with statement to avoid closing the client.
            client = create_remote_connector(self.model_path)
            if is_remote_url(self.model_path):
                client.pull_files(allow_pattern=["*config.json"])
                self.model_weights = self.model_path
                self.model_path = client.get_local_dir()


# adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/config.py
_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


# adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/config.py
def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: Union[str, torch.dtype],
) -> torch.dtype:
    # NOTE: getattr(config, "torch_dtype", torch.float32) is not correct
    # because config.torch_dtype can be None.
    config_dtype = getattr(config, "dtype", None)
    if isinstance(config_dtype, str):
        config_dtype = _STR_DTYPE_TO_TORCH_DTYPE.get(config_dtype, None)
    if config_dtype is None:
        config_dtype = torch.float32

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            if config_dtype == torch.float32:
                if config.model_type.startswith("gemma"):
                    if config.model_type == "gemma":
                        gemma_version = ""
                    else:
                        gemma_version = config.model_type[5]
                    logger.info(
                        f"For Gemma {gemma_version}, we downcast float32 to bfloat16 instead "
                        "of float16 by default. Please specify `dtype` if you "
                        "want to use float16."
                    )
                    torch_dtype = torch.bfloat16
                else:
                    # Following the common practice, we use float16 for float32
                    # models.
                    torch_dtype = torch.float16
            else:
                torch_dtype = config_dtype
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    # Verify the dtype.
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            logger.info("Upcasting %s to %s.", config_dtype, torch_dtype)
            pass
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            logger.info("Downcasting %s to %s.", config_dtype, torch_dtype)
            pass
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning("Casting %s to %s.", config_dtype, torch_dtype)

    return torch_dtype


def is_generation_model(model_architectures: List[str], is_embedding: bool = False):
    # We have two ways to determine whether a model is a generative model.
    # 1. Check the model architecture
    # 2. check the `is_embedding` server args

    if (
        "LlamaEmbeddingModel" in model_architectures
        or "MistralModel" in model_architectures
        or "LlamaForSequenceClassification" in model_architectures
        or "LlamaForSequenceClassificationWithNormal_Weights" in model_architectures
        or "InternLM2ForRewardModel" in model_architectures
        or "Qwen2ForRewardModel" in model_architectures
        or "Qwen2ForSequenceClassification" in model_architectures
        or "Qwen3ForSequenceClassification" in model_architectures
        or "CLIPModel" in model_architectures
        or "BertModel" in model_architectures
        or "Contriever" in model_architectures
        or "BertForSequenceClassification" in model_architectures
        or "XLMRobertaModel" in model_architectures
        or "XLMRobertaForSequenceClassification" in model_architectures
    ):
        return False
    else:
        return not is_embedding


multimodal_model_archs = [
    "CLIPModel",
    "DeepseekVL2ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Gemma3nForConditionalGeneration",
    "Glm4vForConditionalGeneration",
    "Glm4vMoeForConditionalGeneration",
    "Grok1VForCausalLM",
    "Grok1AForCausalLM",
    "LlavaLlamaForCausalLM",
    "Llama4ForConditionalGeneration",
    "LlavaMistralForCausalLM",
    "LlavaQwenForCausalLM",
    "LlavaForConditionalGeneration",
    "LlavaVidForCausalLM",
    "MiniCPMO",
    "MiniCPMV",
    "Mistral3ForConditionalGeneration",
    "MultiModalityCausalLM",
    "MllamaForConditionalGeneration",
    "Qwen2AudioForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3OmniMoeForConditionalGeneration",
    "KimiVLForConditionalGeneration",
    "InternVLChatModel",
    "InternS1ForConditionalGeneration",
    "Phi4MMForCausalLM",
    "Step3VLForConditionalGeneration",
    "POINTSV15ChatModel",
    "DotsVLMForCausalLM",
    "DotsOCRForCausalLM",
    "Sarashina2VisionForCausalLM",
    "NVILAForConditionalGeneration",
    "NVILALiteForConditionalGeneration",
    "DeepseekOCRForCausalLM",
]


def is_multimodal_model(model_architectures: List[str]):
    if any(
        multi_model_arch in model_architectures
        for multi_model_arch in multimodal_model_archs
    ):
        return True
    else:
        return False


def is_multimodal_gen_model(model_architectures: List[str]):
    return False


def is_image_gen_model(model_architectures: List[str]):
    return False


def is_audio_model(model_architectures: List[str]):
    return False


def is_encoder_decoder_model(model_architectures: List[str]):
    return "MllamaForConditionalGeneration" in model_architectures


def is_multimodal_chunked_prefill_supported(model_architectures: List[str]):
    """Check if chunked prefill is supported for a MultiModal model."""
    unsupported = [
        "Grok1VForCausalLM",
        "Grok1AForCausalLM",
        "LlavaLlamaForCausalLM",
        "MllamaForConditionalGeneration",
        "CLIPModel",
    ]
    if any(multi_model_arch in unsupported for multi_model_arch in model_architectures):
        return False
    else:
        return True


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def is_hybrid_model(
    model_architectures: List[str],
    hybrid_kvcache_ratio: Optional[float],
    context_length: Optional[int],
    attention_chunk_size: Optional[int],
):
    if hybrid_kvcache_ratio is None:
        return None
    elif (
        hybrid_kvcache_ratio > 0
        and model_architectures[0] == "Llama4ForConditionalGeneration"
        and context_length > attention_chunk_size
    ):
        return hybrid_kvcache_ratio
    else:
        return None


def get_hybrid_layer_ids(model_architectures: List[str], num_hidden_layers: int):
    if "Llama4ForConditionalGeneration" in model_architectures:
        swa_attention_layer_ids = [
            i for i in range(num_hidden_layers) if (i + 1) % 4 != 0
        ]
        full_attention_layer_ids = [
            i for i in range(num_hidden_layers) if (i + 1) % 4 == 0
        ]
    else:
        swa_attention_layer_ids = None
        full_attention_layer_ids = None
    return swa_attention_layer_ids, full_attention_layer_ids
