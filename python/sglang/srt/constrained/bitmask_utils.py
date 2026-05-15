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

import torch

from sglang.srt.constrained.torch_ops.bitmask_ops import (
    apply_token_bitmask_inplace_torch,
)
from sglang.srt.utils import is_hip

_is_hip = is_hip()
if _is_hip:
    from sgl_kernel import apply_token_bitmask_inplace_cuda
else:
    from sglang.srt.constrained.triton_ops.bitmask_ops import (
        apply_token_bitmask_inplace_triton,
    )


def apply_packed_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
    """Apply a packed int32 vocab mask to logits in-place.

    The packed mask uses one bit per token, where 1 means allowed and 0 means
    masked. Logits beyond the mask coverage are masked out.
    """
    cutoff = vocab_mask.shape[-1] * 32
    if logits.device.type in {"cuda", "xpu", "musa"}:
        if _is_hip:
            apply_token_bitmask_inplace_cuda(logits, vocab_mask)
        else:
            apply_token_bitmask_inplace_triton(logits, vocab_mask)
    elif logits.device.type in {"cpu", "npu"}:
        apply_token_bitmask_inplace_torch(logits[..., :cutoff], vocab_mask)
    else:
        raise RuntimeError(f"Unsupported device: {logits.device.type}")

    if logits.shape[-1] > cutoff:
        logits[..., cutoff:] = float("-inf")
