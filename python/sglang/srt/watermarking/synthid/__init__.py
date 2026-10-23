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
"""SynthID Text watermarking runtime."""

from sglang.srt.watermarking.synthid.processor import (
    REFERENCE_SYNTHID_KEYS,
    SynthIDConfig,
    SynthIDSparseOutput,
    SynthIDState,
    SynthIDStatePool,
    SynthIDWatermarkProcessor,
    compute_context_hash_torch,
    compute_g_values_torch,
    compute_synthid_key_iv,
    get_synthid_config,
    load_synthid_keys,
    watermark_topk_scores_torch,
)

__all__ = [
    "REFERENCE_SYNTHID_KEYS",
    "SynthIDConfig",
    "SynthIDSparseOutput",
    "SynthIDState",
    "SynthIDStatePool",
    "SynthIDWatermarkProcessor",
    "compute_context_hash_torch",
    "compute_g_values_torch",
    "compute_synthid_key_iv",
    "get_synthid_config",
    "load_synthid_keys",
    "watermark_topk_scores_torch",
]
