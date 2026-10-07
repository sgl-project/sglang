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
"""Publish the token rows consumed by the selected stage to LoRA kernels."""

from sglang.srt.layers.layer_boundary.layout import TokenAxis
from sglang.srt.runtime_context import LoRABatchLayout, get_forward


def publish_attn_lora_rows(enabled):
    if enabled:
        get_forward().set("lora_batch_layout", LoRABatchLayout.DP_LOCAL)


def publish_ffn_lora_rows(enabled, input_rows):
    if enabled:
        get_forward().set(
            "lora_batch_layout",
            LoRABatchLayout.DP_LOCAL
            if TokenAxis.ATTN_DP in input_rows.sharded
            else LoRABatchLayout.TP_GLOBAL,
        )
