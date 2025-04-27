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
from enum import IntEnum, auto
from typing import List, Optional, Set, Union

import torch
from transformers import PretrainedConfig

from sglang.srt.hf_transformers_utils import get_config, get_context_length
from sglang.srt.layers.quantization import QUANTIZATION_METHODS
from sglang.srt.utils import get_bool_env_var, is_hip

logger = logging.getLogger(__name__)


class AttentionArch(IntEnum):
    MLA = auto()
    MHA = auto()


class ModelConfig:
    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = True,
        revision: Optional[str] = None,
        context_length: Optional[int] = None,
        model_override_args: Optional[str] = None,
        is_embedding: Optional[bool] = None,
        enable_multimodal: Optional[bool] = None,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        override_config_file: Optional[str] = None,
    ) -> None:

        self.model_path = model_path
        self.revision = revision
        self.quantization = quantization

        # Parse args
        self.maybe_pull_model_tokenizer_from_remote()
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
        self.attention_chunk_size = getattr(
            self.hf_text_config, "attention_chunk_size", None
        )

        if enable_multimodal is None:
            mm_disabled_models = [
                "Gemma3ForConditionalGeneration",
                "Llama4ForConditionalGeneration",
            ]
            if self.hf_config.architectures[0] in mm_disabled_models:
                enable_multimodal = False
                logger.info(
                    f"Multimodal is disabled for {self.hf_config.model_type}. To enable it, set --enable-multimodal."
                )
            else:
                enable_multimodal = True

        # Check model type
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
        self.is_encoder_decoder = is_encoder_decoder_model(self.hf_config.architectures)
        self.dtype = _get_and_verify_dtype(self.hf_text_config, dtype)

        # Derive context length
        derived_context_len = get_context_length(self.hf_text_config)
        if context_length is not None:
            if context_length > derived_context_len:
                if get_bool_env_var(
                    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", default="True"
                ):
                    logger.warning(
                        f"Warning: User-specified context_length ({context_length}) is greater than the derived context_length ({derived_context_len}). "
                        f"This may lead to incorrect model outputs or CUDA errors."
                    )
                    self.context_len = context_length
                else:
                    raise ValueError(
                        f"User-specified context_length ({context_length}) is greater than the derived context_length ({derived_context_len}). "
                        f"This may lead to incorrect model outputs or CUDA errors. Note that the derived context_length may differ from max_position_embeddings in the model's config. "
                        f"To allow overriding this maximum, set the env var SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"
                    )
            else:
                self.context_len = context_length
        else:
            self.context_len = derived_context_len

        # Unify the config keys for hf_text_config
        self.head_dim = getattr(
            self.hf_text_config,
            "head_dim",
            self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads,
        )

        # FIXME: temporary special judge for MLA architecture
        if (
            "DeepseekV2ForCausalLM" in self.hf_config.architectures
            or "DeepseekV3ForCausalLM" in self.hf_config.architectures
            or "DeepseekV3ForCausalLMNextN" in self.hf_config.architectures
        ):
            self.head_dim = 256
            self.attention_arch = AttentionArch.MLA
            self.kv_lora_rank = self.hf_config.kv_lora_rank
            self.qk_nope_head_dim = self.hf_config.qk_nope_head_dim
            self.qk_rope_head_dim = self.hf_config.qk_rope_head_dim
            self.v_head_dim = self.hf_config.v_head_dim

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
        else:
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
        self.vocab_size = self.hf_text_config.vocab_size

        # Verify quantization
        self._verify_quantization()

        # Cache attributes
        self.hf_eos_token_id = self.get_hf_eos_token_id()
        self.image_token_id = getattr(self.hf_config, "image_token_id", None)

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

        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
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
            # check if is modelopt model -- modelopt doesn't have corresponding field
            # in hf `config.json` but has a standalone `hf_quant_config.json` in the root directory
            # example: https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8/tree/main
            is_local = os.path.exists(self.model_path)
            modelopt_quant_config = {"quant_method": "modelopt"}
            if not is_local:
                from huggingface_hub import HfApi

                hf_api = HfApi()
                if hf_api.file_exists(self.model_path, "hf_quant_config.json"):
                    quant_cfg = modelopt_quant_config
            elif os.path.exists(os.path.join(self.model_path, "hf_quant_config.json")):
                quant_cfg = modelopt_quant_config
        return quant_cfg

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
        ]
        optimized_quantization_methods = [
            "fp8",
            "marlin",
            "modelopt",
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
        ]
        compatible_quantization_methods = {
            "modelopt_fp4": ["modelopt"],
            "w8a8_int8": ["compressed-tensors", "compressed_tensors"],
            "w8a8_fp8": ["compressed-tensors", "compressed_tensors"],
        }
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self._parse_quant_hf_config()

        if quant_cfg is not None:
            quant_method = quant_cfg.get("quant_method", "").lower()

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

    def get_hf_eos_token_id(self) -> Optional[Set[int]]:
        eos_ids = getattr(self.hf_config, "eos_token_id", None)
        if eos_ids:
            # it can be either int or list of int
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
        return eos_ids

    def maybe_pull_model_tokenizer_from_remote(self) -> None:
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


def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    class_name = config.architectures[0]
    if class_name.startswith("Llava") and class_name.endswith("ForCausalLM"):
        # We support non-hf version of llava models, so we do not want to
        # read the wrong values from the unused default text_config.
        # NOTE(HandH1998): We set `torch_dtype` of config to `torch.float16` for the weights, as
        # `torch.float16` is default used for image features in `python/sglang/srt/models/llava.py`.
        setattr(config, "torch_dtype", torch.float16)
        return config

    if hasattr(config, "text_config"):
        # The code operates under the assumption that text_config should have
        # `num_attention_heads` (among others). Assert here to fail early
        # if transformers config doesn't align with this assumption.
        assert hasattr(config.text_config, "num_attention_heads")
        return config.text_config
    if hasattr(config, "language_config"):
        return config.language_config
    else:
        return config


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
    config_dtype = getattr(config, "torch_dtype", None)
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
        or "CLIPModel" in model_architectures
    ):
        return False
    else:
        return not is_embedding


multimodal_model_archs = [
    "DeepseekVL2ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Grok1VForCausalLM",
    "Grok1AForCausalLM",
    "LlavaLlamaForCausalLM",
    "Llama4ForConditionalGeneration",
    "LlavaMistralForCausalLM",
    "LlavaQwenForCausalLM",
    "LlavaVidForCausalLM",
    "MiniCPMO",
    "MiniCPMV",
    "MultiModalityCausalLM",
    "MllamaForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "CLIPModel",
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


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0
