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

from typing import Optional


def compose_prefix_cache_extra_key(
    extra_key: Optional[str], lora_id: Optional[str]
) -> Optional[str]:
    if lora_id is None:
        return extra_key

    parts = []
    if extra_key is not None:
        parts.append(f"extra_key:{len(extra_key)}:{extra_key}")
    parts.append(f"lora_id:{len(lora_id)}:{lora_id}")
    return "|".join(parts)
