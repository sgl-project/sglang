# Copyright 2026 SGLang Team
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
"""Configuration checks for the supported Nemotron 3.5 reranker layout."""

from collections.abc import Mapping
from typing import Any


def is_nemotron35_reranker_config(config: Mapping[str, Any]) -> bool:
    if config.get("model_type") != "mistral3":
        return False
    if "Mistral3ForSequenceClassification" not in (config.get("architectures") or ()):
        return False

    text_config = config.get("text_config") or {}
    vision_config = config.get("vision_config") or {}
    # HF serializes the label map instead of num_labels.
    num_labels = config.get("num_labels", len(config.get("id2label") or {}))
    return (
        text_config.get("model_type") == "ministral3"
        and vision_config.get("model_type") == "pixtral"
        and config.get("is_causal") is False
        and text_config.get("is_causal") is False
        and config.get("pooling") == "avg"
        and text_config.get("pooling") == "avg"
        and num_labels == 1
    )
