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
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ShapeKey:
    # Tokens for prefill/ragged verify; requests for ordinary decode.
    size: int
    # PDMux stream, or None for a single stream.
    stream_idx: Optional[int] = None
    # LoRA or prefill-prefix variant; None selects the default.
    variant_label: Optional[str] = None
    # Independent attention variant; None selects the default.
    attention_variant: Optional[str] = None
