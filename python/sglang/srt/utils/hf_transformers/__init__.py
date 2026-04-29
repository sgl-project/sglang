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
"""Hugging Face Transformers utilities.

This package provides HF Transformers helpers, split into submodules
(common, compat, config, tokenizer, processor, mistral_utils).
All public symbols are re-exported here for convenience.  The old import
path ``sglang.srt.utils.hf_transformers_utils`` is preserved by a
separate shim module.
"""

from .compat import apply_all as _apply_compat

_apply_compat()

from .common import (  # noqa: E402
    CONTEXT_LENGTH_KEYS,
    AutoConfig,
    attach_additional_stop_token_ids,
    check_gguf_file,
    download_from_hf,
    get_context_length,
    get_generation_config,
    get_hf_text_config,
    get_rope_config,
    get_sparse_attention_config,
    get_tokenizer_from_processor,
)
from .compat import normalize_rope_scaling_compat  # noqa: E402
from .config import get_config  # noqa: E402
from .processor import get_processor  # noqa: E402
from .tokenizer import (  # noqa: E402
    _fix_added_tokens_encoding,
    _fix_v5_add_bos_eos_token,
    get_tokenizer,
)

__all__ = [
    "AutoConfig",
    "CONTEXT_LENGTH_KEYS",
    "_fix_added_tokens_encoding",
    "_fix_v5_add_bos_eos_token",
    "attach_additional_stop_token_ids",
    "check_gguf_file",
    "download_from_hf",
    "get_config",
    "get_context_length",
    "get_generation_config",
    "get_hf_text_config",
    "get_processor",
    "get_rope_config",
    "get_sparse_attention_config",
    "get_tokenizer",
    "get_tokenizer_from_processor",
    "normalize_rope_scaling_compat",
]
