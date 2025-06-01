# Copyright 2025 SGLang Team. All Rights Reserved.
#
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

from typing import Dict, Tuple

import torch


def get_cuda_stream() -> int:
    return torch.cuda.current_stream().cuda_stream


_cache_buf: Dict[Tuple[str, torch.device], torch.Tensor] = {}


def _get_cache_buf(name: str, bytes: int, device: torch.device) -> torch.Tensor:
    key = (name, device)
    buf = _cache_buf.get(key)
    if buf is None:
        buf = torch.empty(bytes, dtype=torch.uint8, device=device)
        _cache_buf[key] = buf
    return buf


def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)


_IS_HOPPER_ARCH = None


def is_hopper_arch() -> bool:
    global _IS_HOPPER_ARCH
    if _IS_HOPPER_ARCH is None:
        # Hopper arch's compute capability == 9.0
        device = torch.cuda.current_device()
        major, _ = torch.cuda.get_device_capability(device)
        _IS_HOPPER_ARCH = major == 9
    return _IS_HOPPER_ARCH
