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

"""Per-forward metadata for decode context parallel (DCP)."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DecodeContextParallelMetadata:
    # For decode context parallel
    dcp_kv_indptr: Optional[torch.Tensor] = None
    dcp_kv_buffer: Optional[torch.Tensor] = None
    dcp_kv_indices: Optional[torch.Tensor] = None
    dcp_local_prefix_kv_indices: Optional[torch.Tensor] = None
    dcp_extend_prefix_lens_sum: Optional[int] = None
