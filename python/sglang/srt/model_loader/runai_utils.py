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

import torch

RUNAI_STREAMER_TENSOR_ATTR = "_sglang_runai_streamer_tensor"


def _clone_if_runai_streamed_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Own tensors retained beyond the current RunAI streamer iteration."""
    if getattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, False):
        return tensor.clone().detach()
    return tensor
