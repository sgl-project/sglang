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
"""Utilities for Huggingface Transformers."""

import contextlib
import json
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from numba import njit
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from sglang.srt.configs import (
    ChatGLMConfig,
    DbrxConfig,
    DeepseekVL2Config,
    DotsOCRConfig,
    DotsVLMConfig,
    ExaoneConfig,
    FalconH1Config,
    JetNemotronConfig,
    JetVLMConfig,
    KimiLinearConfig,
    KimiVLConfig,
    LongcatFlashConfig,
    MultiModalityConfig,
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    Olmo3Config,
    Qwen3NextConfig,
    Step3VLConfig,
)
from sglang.srt.configs.deepseek_ocr import DeepseekVLV2Config
from sglang.srt.configs.internvl import InternVLChatConfig
from sglang.srt.connector import create_remote_connector
from sglang.srt.multimodal.customized_mm_processor_utils import _CUSTOMIZED_MM_PROCESSOR
from sglang.srt.utils import is_remote_url, logger, lru_cache_frozenset, mistral_utils

_CONFIG_REGISTRY: List[Type[PretrainedConfig]] = [
    ChatGLMConfig,
    DbrxConfig,
    ExaoneConfig,
    DeepseekVL2Config,
    MultiModalityConfig,
    KimiVLConfig,
    InternVLChatConfig,
    Step3VLConfig,
    LongcatFlashConfig,
    Olmo3Config,
    KimiLinearConfig,
    Qwen3NextConfig,
    FalconH1Config,
    DotsVLMConfig,
    DotsOCRConfig,
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    DeepseekVLV2Config,
    JetNemotronConfig,
    JetVLMConfig,
]

_CONFIG_REGISTRY = {
    config_cls.model_type: config_cls for config_cls in _CONFIG_REGISTRY
}

for name, cls in _CONFIG_REGISTRY.items():
    with contextlib.suppress(ValueError):
        AutoConfig.register(name, cls)


def download_from_hf(
    model_path: str,
    allow_patterns: Optional[Union[str, list]] = None,
):
    if os.path.exists(model_path):
        return model_path

    if not allow_patterns:
        allow_patterns = ["*.json", "*.bin", "*.model"]

    return snapshot_download(model_path, allow_patterns=allow_patterns)


def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    if config.architectures is not None:
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

    if hasattr(config, "llm_config"):
        # PointsV1.5 Chat Model
        assert hasattr(config.llm_config, "num_attention_heads")
        return config.llm_config

    if hasattr(config, "language_config"):
        return config.language_config
    if hasattr(config, "thinker_config"):
        # qwen2.5 omni
        thinker_config = config.thinker_config
        if hasattr(thinker_config, "text_config"):
            setattr(
                thinker_config.text_config,
                "torch_dtype",
                getattr(thinker_config, "torch_dtype", None),
            )
            return thinker_config.text_config
        return thinker_config
    if hasattr(config, "llm_config"):
        return config.llm_config
    else:
        return config


# Temporary hack for DeepSeek-V3.2 model
def _load_deepseek_v32_model(
    model_path: str,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    **kwargs,
):
    # first get the local path
    local_path = download_from_hf(model_path)
    # then load the config file in json
    config_file = os.path.join(local_path, "config.json")
    if not os.path.exists(config_file):
        raise RuntimeError(f"Can't find config file in {local_path}.")

    with open(config_file, "r") as f:
        config_json = json.load(f)

    config_json["architectures"] = ["DeepseekV3ForCausalLM"]
    config_json["model_type"] = "deepseek_v3"

    tmp_path = os.path.join(tempfile.gettempdir(), "_tmp_config_folder")
    os.makedirs(tmp_path, exist_ok=True)

    unique_path = os.path.join(tmp_path, f"deepseek_v32_{os.getpid()}")
    with open(unique_path, "w") as f:
        json.dump(config_json, f)

    return AutoConfig.from_pretrained(
        unique_path, trust_remote_code=trust_remote_code, revision=revision, **kwargs
    )


# Temporary hack for Mistral Large
def _load_mistral_large_3_for_causal_LM(
    model_path: str,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    **kwargs,
):
    # first get the local path
    local_path = download_from_hf(model_path)
    # then load the config file in json
    parser = mistral_utils.MistralConfigParser()
    config_dict, _ = parser.parse(local_path)

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as f:
        json.dump(config_dict, f)
        f.flush()
        loaded_config = AutoConfig.from_pretrained(
            f.name, trust_remote_code=trust_remote_code, revision=revision, **kwargs
        )
    text_config = getattr(loaded_config, "text_config", None)
    if text_config is not None and isinstance(text_config, dict):
        text_config = AutoConfig.for_model(**text_config)
        setattr(loaded_config, "text_config", text_config)
    vision_config = getattr(loaded_config, "vision_config", None)
    if vision_config is not None and isinstance(vision_config, dict):
        vision_config = AutoConfig.for_model(**vision_config)
        setattr(loaded_config, "vision_config", vision_config)

    return loaded_config


def _is_deepseek_ocr_model(config: PretrainedConfig) -> bool:
    # TODO: Remove this workaround related when AutoConfig correctly identifies deepseek-ocr.
    # Hugging Face's AutoConfig currently misidentifies it as deepseekvl2.
    return (
        getattr(config, "auto_map", None) is not None
        and config.auto_map.get("AutoModel")
        == "modeling_deepseekocr.DeepseekOCRForCausalLM"
    )


@lru_cache_frozenset(maxsize=32)
def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_override_args: Optional[dict] = None,
    **kwargs,
):
    is_gguf = check_gguf_file(model)
    if is_gguf:
        kwargs["gguf_file"] = model
        model = Path(model).parent

    if is_remote_url(model):
        # BaseConnector implements __del__() to clean up the local dir.
        # Since config files need to exist all the time, so we DO NOT use
        # with statement to avoid closing the client.
        client = create_remote_connector(model)
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        model = client.get_local_dir()

    if "mistral-large-3" in str(model).lower():
        config = _load_mistral_large_3_for_causal_LM(
            model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
        )
    else:
        try:
            config = AutoConfig.from_pretrained(
                model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
            )
        except ValueError as e:
            if not "deepseek_v32" in str(e):
                raise e
            config = _load_deepseek_v32_model(
                model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
            )

    if (
        config.architectures is not None
        and config.architectures[0] == "Phi4MMForCausalLM"
    ):
        # Phi4MMForCausalLM uses a hard-coded vision_config. See:
        # https://github.com/vllm-project/vllm/blob/6071e989df1531b59ef35568f83f7351afb0b51e/vllm/model_executor/models/phi4mm.py#L71
        # We set it here to support cases where num_attention_heads is not divisible by the TP size.
        from transformers import SiglipVisionConfig

        vision_config = {
            "hidden_size": 1152,
            "image_size": 448,
            "intermediate_size": 4304,
            "model_type": "siglip_vision_model",
            "num_attention_heads": 16,
            "num_hidden_layers": 26,
            # Model is originally 27-layer, we only need the first 26 layers for feature extraction.
            "patch_size": 14,
        }
        config.vision_config = SiglipVisionConfig(**vision_config)
    text_config = get_hf_text_config(config=config)

    if isinstance(model, str) and text_config is not None:
        for key, val in text_config.__dict__.items():
            if not hasattr(config, key) and getattr(text_config, key, None) is not None:
                setattr(config, key, val)

    if config.model_type in _CONFIG_REGISTRY:
        model_type = config.model_type
        if model_type == "deepseek_vl_v2":
            if _is_deepseek_ocr_model(config):
                model_type = "deepseek-ocr"
        config_class = _CONFIG_REGISTRY[model_type]
        config = config_class.from_pretrained(model, revision=revision)
        # NOTE(HandH1998): Qwen2VL requires `_name_or_path` attribute in `config`.
        setattr(config, "_name_or_path", model)

    if isinstance(model, str) and config.model_type == "internvl_chat":
        for key, val in config.llm_config.__dict__.items():
            if not hasattr(config, key):
                setattr(config, key, val)

    if config.model_type == "multi_modality":
        config.update({"architectures": ["MultiModalityCausalLM"]})

    if model_override_args:
        config.update(model_override_args)

    # Special architecture mapping check for GGUF models
    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})

    return config


@lru_cache_frozenset(maxsize=32)
def get_generation_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    **kwargs,
):
    try:
        return GenerationConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
        )
    except OSError as e:
        return None


# Qwen-1M related
def get_sparse_attention_config(
    model: str,
    sparse_attention_config_filename: str = "sparse_attention_config.json",
) -> Dict[str, Any]:
    is_local = os.path.isdir(model)
    if not is_local:
        # Download the config files.
        model = download_from_hf(model, allow_patterns=["*.json"])

    config_file = os.path.join(model, sparse_attention_config_filename)
    if not os.path.exists(config_file):
        return {}

    # Load the sparse attention config.
    with open(config_file) as f:
        config = json.load(f)
    return config


# Models don't use the same configuration key for determining the maximum
# context length.  Store them here so we can sanely check them.
# NOTE: The ordering here is important. Some models have two of these and we
# have a preference for which value gets used.
CONTEXT_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_position_embeddings",
]


def get_context_length(config):
    """Get the context length of a model from a huggingface model configs."""
    text_config = config
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if rope_scaling:
        rope_scaling_factor = rope_scaling.get("factor", 1)
        if "original_max_position_embeddings" in rope_scaling:
            rope_scaling_factor = 1
        if rope_scaling.get("rope_type", None) == "llama3":
            rope_scaling_factor = 1
    else:
        rope_scaling_factor = 1

    for key in CONTEXT_LENGTH_KEYS:
        val = getattr(text_config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048


# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"
# Parallel tokenizer related constants
_SGLANG_MAX_PARALLEL_TOKENIZER_OVERLAP_LEN = int(
    os.environ.get("SGLANG_MAX_PARALLEL_TOKENIZER_OVERLAP_LEN", 256)
)
_SGLANG_MAX_PARALLEL_TOKENIZER_CHUNK_SIZE = int(
    os.environ.get("SGLANG_MAX_PARALLEL_TOKENIZER_CHUNK_SIZE", 4096)
)


@njit(cache=True)
def compute_hash_array_fixed(arr, actual_len, length, base, mod):
    h = 0
    power = 1
    output_len = actual_len - length + 1
    hash_vals = np.full(output_len, -1, dtype=np.int64)

    for i in range(length):
        h = (h * base + arr[i]) % mod
        if i < length - 1:
            power = (power * base) % mod
    hash_vals[0] = h

    for i in range(length, actual_len):
        h = ((h - arr[i - length] * power) * base + arr[i]) % mod
        if h < 0:
            h += mod
        hash_vals[i - length + 1] = h

    return hash_vals


@njit(cache=True)
def check_lcs_fixed(arr1, len1, arr2, len2, length, base, mod):
    if length == 0:
        return -1, -1

    hashes1 = compute_hash_array_fixed(arr1, len1, length, base, mod)
    hashes2 = compute_hash_array_fixed(arr2, len2, length, base, mod)

    for j in range(len2 - length + 1):
        h2 = hashes2[j]
        for i in range(len1 - length + 1):
            if h2 == hashes1[i]:
                match = True
                for k in range(length):
                    if arr1[i + k] != arr2[j + k]:
                        match = False
                        break
                if match:
                    return i, j
    return -1, -1


def lcs_solver(arr1, arr2):
    base = 911
    mod = 10**9 + 7

    arr1_buf = np.zeros(_SGLANG_MAX_PARALLEL_TOKENIZER_OVERLAP_LEN, dtype=np.int64)
    arr2_buf = np.zeros(_SGLANG_MAX_PARALLEL_TOKENIZER_OVERLAP_LEN, dtype=np.int64)
    len1, len2 = len(arr1), len(arr2)
    arr1_buf[:len1] = arr1
    arr2_buf[:len2] = arr2

    left, right = 0, min(len1, len2)
    best_i, best_j, best_len = -1, -1, 0

    while left <= right:
        mid = (left + right) // 2
        i, j = check_lcs_fixed(arr1_buf, len1, arr2_buf, len2, mid, base, mod)
        if i != -1:
            best_i, best_j, best_len = i, j, mid
            left = mid + 1
        else:
            right = mid - 1

    return [best_i, best_j], best_len


class ParallelTokenizer:
    """Parallel tokenizer wrapper over a HF PreTrainedTokenizer.

    - Splits long texts into chunks, tokenizes in parallel, then merges.
    - Return values only include input_ids (and token_type_ids when available).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chunk_size: Optional[int] = None,
        overlap_length: Optional[int] = None,
    ):
        """Initialize the ParallelTokenizer."""
        self.tokenizer_ = tokenizer
        self.chunk_size = chunk_size or _SGLANG_MAX_PARALLEL_TOKENIZER_CHUNK_SIZE
        self.overlap_length = (
            overlap_length or _SGLANG_MAX_PARALLEL_TOKENIZER_OVERLAP_LEN
        )
        # warmup for numba
        self.encode("Hello, world!" * self.chunk_size)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped tokenizer if not found in this class."""
        return getattr(self.tokenizer_, name)

    def _split_text_into_chunks(self, text: str) -> Tuple[List[str], List[str]]:
        """Split text into overlapping chunks and return (chunks, overlap_texts).

        - Text overlap is taken as the last self.overlap_length characters of each chunk.
        - The token-level overlap bound is computed later by encoding these overlap_texts
          with add_special_tokens=False.
        """
        if len(text) <= self.chunk_size:
            return [text], []

        split_text: List[str] = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i : i + self.chunk_size + self.overlap_length]
            split_text.append(chunk)
        if len(split_text) > 1 and len(split_text[-1]) <= self.overlap_length:
            split_text = split_text[:-1]

        overlap_texts: List[str] = []
        for i in range(len(split_text) - 1):
            prev_chunk = split_text[i]
            overlap_texts.append(
                prev_chunk[-self.overlap_length :] if self.overlap_length > 0 else ""
            )

        return split_text, overlap_texts

    def apply_chat_template(self, *args, **kwargs):
        do_tokenize = kwargs.get("tokenize", True)
        if do_tokenize:
            kwargs["tokenize"] = False
        chat_text = self.tokenizer_.apply_chat_template(*args, **kwargs)
        if do_tokenize:
            return self.encode(chat_text)
        else:
            return chat_text

    def __call__(self, *args, **kwargs):
        return self._tokenize_parallel(*args, **kwargs)

    def encode(self, text: Union[str, List[str]], **kwargs):
        """Encode text with parallel processing for long texts."""
        results = self._tokenize_parallel(text, **kwargs)
        return results["input_ids"]

    def _tokenize_parallel(self, text: Union[str, List[str]], **kwargs):
        """Tokenize and return a dict compatible with HF tokenizer output."""
        # Batch inputs: split each text into chunks, batch encode chunks and overlaps, then merge per text
        if isinstance(text, list):
            num_texts = len(text)
            if num_texts > 8:
                # fallback to hf tokenization for large batch
                return self.tokenizer_(text, **kwargs)

            flat_chunks: List[str] = []
            chunk_spans: List[tuple[int, int]] = []  # (text_idx, chunk_idx)
            flat_overlap_texts: List[str] = []
            overlap_spans: List[tuple[int, int]] = []  # (text_idx, pair_idx)
            for ti, t in enumerate(text):
                chunks, overlaps = self._split_text_into_chunks(t)
                for ci, ch in enumerate(chunks):
                    flat_chunks.append(ch)
                    chunk_spans.append((ti, ci))
                for pi, ov in enumerate(overlaps):
                    flat_overlap_texts.append(ov)
                    overlap_spans.append((ti, pi))

            if not flat_chunks:
                return {"input_ids": [[] for _ in range(num_texts)]}

            encoded_chunks = self.tokenizer_(flat_chunks, **kwargs)
            chunk_ids_list: List[List[int]] = encoded_chunks["input_ids"]
            chunk_type_list = encoded_chunks.get("token_type_ids")

            # Encode overlap texts to get per-pair token overlap lengths
            per_text_ov_lens: List[List[int]] = [[] for _ in range(num_texts)]
            if flat_overlap_texts:
                ov_encoded = self.tokenizer_(
                    flat_overlap_texts, add_special_tokens=False
                )
                ov_ids_list: List[List[int]] = ov_encoded["input_ids"]
                for idx, (ti, pi) in enumerate(overlap_spans):
                    per_text_ov_lens[ti].append(len(ov_ids_list[idx]))

            # group per text
            per_text_ids_lists: List[List[List[int]]] = [[] for _ in range(num_texts)]
            per_text_type_lists: Optional[List[Optional[List[List[int]]]]] = (
                [[] for _ in range(num_texts)] if chunk_type_list is not None else None
            )
            for enc_i, (ti, _ci) in enumerate(chunk_spans):
                per_text_ids_lists[ti].append(chunk_ids_list[enc_i])
                if per_text_type_lists is not None:
                    per_text_type_lists[ti].append(chunk_type_list[enc_i])

            final_input_ids: List[List[int]] = []
            final_token_type_ids: Optional[List[List[int]]] = (
                [] if per_text_type_lists is not None else None
            )

            for ti in range(num_texts):
                ids_lists = per_text_ids_lists[ti]
                type_lists = (
                    per_text_type_lists[ti] if per_text_type_lists is not None else None
                )
                ov_lens = per_text_ov_lens[ti]

                if len(ids_lists) == 1:
                    merged_ids = ids_lists[0]
                    merged_types = type_lists[0] if type_lists is not None else None
                else:
                    merged_ids, merged_types = self._merge_results(
                        ids_lists, ov_lens, type_lists
                    )
                    if merged_ids is None:
                        encoded = self.tokenizer_(text[ti], **kwargs)
                        merged_ids = encoded["input_ids"]
                        merged_types = encoded.get("token_type_ids")

                # each chunk's special tokens should be handled by LCS
                final_input_ids.append(merged_ids)
                if final_token_type_ids is not None:
                    if merged_types is not None:
                        final_token_type_ids.append(merged_types)
                    else:
                        final_token_type_ids = None

            result: Dict[str, Any] = {"input_ids": final_input_ids}
            if final_token_type_ids is not None:
                result["token_type_ids"] = final_token_type_ids
            return result

        # Split text into overlapping chunks and collect overlap texts
        split_text, overlap_texts = self._split_text_into_chunks(text)

        # Batch encode chunks
        encoded = self.tokenizer_(split_text, **kwargs)
        ids_lists = encoded["input_ids"]
        type_lists = encoded.get("token_type_ids")

        # Encode overlap texts for dynamic token overlap bounds
        ov_lens: List[int] = []
        if overlap_texts:
            ov_encoded = self.tokenizer_(overlap_texts, add_special_tokens=False)
            ov_ids_list: List[List[int]] = ov_encoded["input_ids"]
            ov_lens = [len(x) for x in ov_ids_list]

        merged_ids, merged_types = self._merge_results(ids_lists, ov_lens, type_lists)
        if merged_ids is None:
            logger.warning(
                f"[ParallelTokenizer] Failed to merge results, fallback to hf tokenization"
            )
            return self.tokenizer_(text, **kwargs)

        if merged_types is not None:
            return {"input_ids": merged_ids, "token_type_ids": merged_types}
        else:
            return {"input_ids": merged_ids}

    def _merge_results(
        self, token_ids_lists, overlap_token_lens, token_type_ids_lists=None
    ):
        """Merge tokenized chunks using LCS algorithm to handle overlaps.
        If token_type_ids_lists is provided, it will be merged with the same
        boundaries derived from input_ids.
        """
        merged_ids = []
        merged_types = [] if token_type_ids_lists is not None else None
        if not token_ids_lists:
            return merged_ids, merged_types
        start_idx = 0

        for idx, cur_ids in enumerate(token_ids_lists):
            if idx < len(token_ids_lists) - 1:
                nxt_ids = token_ids_lists[idx + 1]
                cur_overlap_len = (
                    overlap_token_lens[idx] if idx < len(overlap_token_lens) else 0
                )
                if cur_overlap_len > _SGLANG_MAX_PARALLEL_TOKENIZER_OVERLAP_LEN:
                    logger.warning(
                        f"[ParallelTokenizer] Overlap length {cur_overlap_len} is greater than the maximum allowed {_SGLANG_MAX_PARALLEL_TOKENIZER_OVERLAP_LEN}, fallback to hf tokenization"
                    )
                    return None, None
                lcs_start_list, lcs_len = lcs_solver(
                    cur_ids[-cur_overlap_len:],
                    nxt_ids[:cur_overlap_len],
                )
                if lcs_len <= 0:
                    # this should not happen
                    logger.warning(
                        f"[ParallelTokenizer] Failed to merge results, fallback to hf tokenization"
                    )
                    return None, None
                end_idx = len(cur_ids) - cur_overlap_len + lcs_start_list[0] + lcs_len
                merged_ids.extend(cur_ids[start_idx:end_idx])
                if merged_types is not None:
                    cur_types = token_type_ids_lists[idx]
                    merged_types.extend(cur_types[start_idx:end_idx])
                start_idx = lcs_start_list[1] + lcs_len
            else:
                merged_ids.extend(cur_ids[start_idx:])
                if merged_types is not None:
                    cur_types = token_type_ids_lists[idx]
                    merged_types.extend(cur_types[start_idx:])
        return merged_ids, merged_types


# Filter warnings like: https://github.com/sgl-project/sglang/issues/8082
class TokenizerWarningsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Calling super().encode with" not in record.getMessage()


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_name.endswith(".json"):
        from sglang.srt.tokenizer.tiktoken_tokenizer import TiktokenTokenizer

        return TiktokenTokenizer(tokenizer_name)

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    # TODO(Xinyuan): Remove this once we have a proper tokenizer for Devstral
    if tokenizer_name == "mistralai/Devstral-Small-2505":
        tokenizer_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    is_gguf = check_gguf_file(tokenizer_name)
    if is_gguf:
        kwargs["gguf_file"] = tokenizer_name
        tokenizer_name = Path(tokenizer_name).parent

    if is_remote_url(tokenizer_name):
        # BaseConnector implements __del__() to clean up the local dir.
        # Since config files need to exist all the time, so we DO NOT use
        # with statement to avoid closing the client.
        client = create_remote_connector(tokenizer_name)
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        tokenizer_name = client.get_local_dir()

    # Do not pass custom flags to HF APIs
    enable_parallel_tokenizer = kwargs.pop("enable_parallel_tokenizer", False)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            clean_up_tokenization_spaces=False,
            **kwargs,
        )
        # Filter tokenizer warnings
        logging.getLogger(tokenizer.__class__.__module__).addFilter(
            TokenizerWarningsFilter()
        )
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA V1 model "
            f"consider using '{_FAST_LLAMA_TOKENIZER}' instead of the "
            "original tokenizer."
        )
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if not trust_remote_code and (
            "does not exist or is not currently imported." in str(e)
            or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        warnings.warn(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    attach_additional_stop_token_ids(tokenizer)
    if enable_parallel_tokenizer:
        tokenizer = ParallelTokenizer(tokenizer)
    return tokenizer


# Some models doesn't have an available processor, e.g.: InternVL
def get_tokenizer_from_processor(processor):
    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


def get_processor(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    use_fast: Optional[bool] = True,
    **kwargs,
):
    # pop 'revision' from kwargs if present.
    revision = kwargs.pop("revision", tokenizer_revision)
    if "mistral-large-3" in str(tokenizer_name).lower():
        config = _load_mistral_large_3_for_causal_LM(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )
    else:
        config = AutoConfig.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )
    if _is_deepseek_ocr_model(config):
        # Temporary hack for load deepseek-ocr
        config.model_type = "deepseek-ocr"

    # fix: for Qwen2-VL and Sarashina2Vision models, inject default 'size' if not provided.
    if config.model_type in {"qwen2_vl", "sarashina2_vision"}:
        if "size" not in kwargs:
            kwargs["size"] = {"shortest_edge": 3136, "longest_edge": 1003520}

    if config.model_type not in {"llava", "clip"}:
        kwargs["use_fast"] = use_fast
    try:
        if "InternVL3_5" in tokenizer_name:
            processor = AutoTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
        else:
            if config.model_type in _CUSTOMIZED_MM_PROCESSOR:
                processor = _CUSTOMIZED_MM_PROCESSOR[config.model_type].from_pretrained(
                    tokenizer_name,
                    *args,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    **kwargs,
                )
            else:
                processor = AutoProcessor.from_pretrained(
                    tokenizer_name,
                    *args,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    **kwargs,
                )

    except ValueError as e:
        error_message = str(e)
        if "does not have a slow version" in error_message:
            logger.info(
                f"Processor {tokenizer_name} does not have a slow version. Automatically use fast version"
            )
            kwargs["use_fast"] = True
            processor = AutoProcessor.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
        else:
            raise e
    tokenizer = get_tokenizer_from_processor(processor)

    attach_additional_stop_token_ids(tokenizer)
    return processor


def attach_additional_stop_token_ids(tokenizer):
    # Special handling for stop token <|eom_id|> generated by llama 3 tool use.
    if "<|eom_id|>" in tokenizer.get_added_vocab():
        tokenizer.additional_stop_token_ids = set(
            [tokenizer.get_added_vocab()["<|eom_id|>"]]
        )
    else:
        tokenizer.additional_stop_token_ids = None


def check_gguf_file(model: Union[str, os.PathLike]) -> bool:
    """Check if the file is a GGUF model."""
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    with open(model, "rb") as f:
        header = f.read(4)
    return header == b"GGUF"
