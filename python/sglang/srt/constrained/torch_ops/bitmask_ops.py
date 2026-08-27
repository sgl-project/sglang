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

import torch


def apply_token_bitmask_inplace_torch(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
) -> None:
    """Backend-agnostic torch fallback for packed-bitmask application.

    This path is currently used as a fallback on NPU in xgrammar backend.
    """
    vocab_size = logits.shape[-1]
    bitmask_cpu = bitmask.detach().cpu()
    token_ids = torch.arange(vocab_size, device="cpu", dtype=torch.int32)
    word_idx = token_ids // 32
    bit_idx = token_ids % 32
    words = bitmask_cpu[:, word_idx].to(torch.int32)
    allowed = ((words >> bit_idx) & 1).to(torch.bool)
    allowed = allowed.to(logits.device, non_blocking=True)
    logits.masked_fill_(~allowed, float("-inf"))
