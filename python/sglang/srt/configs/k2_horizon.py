# Copyright 2023-2026 SGLang Team
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

"""Native configuration shells for the xLLM and K2 Horizon model families.

The released checkpoints persist the complete architecture in ``config.json``.
Keeping these classes intentionally thin lets SGLang select its native runtime
without importing checkpoint-provided Python code, while preserving every
model-specific field for validation in ``sglang.srt.models.xllm``.
"""

from transformers import PretrainedConfig


class XllmConfig(PretrainedConfig):
    model_type = "xllm"


class K2HorizonConfig(PretrainedConfig):
    model_type = "k2_horizon"
