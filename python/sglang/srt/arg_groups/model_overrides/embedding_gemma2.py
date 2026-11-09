# Copyright 2025 SGLang Team
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

"""Config-time override declarations for EmbeddingGemma2.

Architectures: EmbeddingGemma2Model.
Sets disable_radix_cache=True and default attention_backend="triton" for EmbeddingGemma2.
"""

import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    resolving_view,
)

logger = logging.getLogger(__name__)


@_register_for("EmbeddingGemma2Model")
def _embedding_gemma2_overrides(server_args: Any, hf_config: Any) -> dict[str, Any]:
    cfg = resolving_view(server_args)
    overrides: dict[str, Any] = {
        "disable_radix_cache": True,
    }
    if is_attention_backend_not_set(cfg) or cfg.attention_backend is None:
        logger.info("Use triton as default attention backend for EmbeddingGemma2Model")
        overrides["attention_backend"] = "triton"

    return overrides
