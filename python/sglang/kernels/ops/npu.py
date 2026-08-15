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
"""Small runtime helpers shared by Ascend Triton kernels.

Device properties are intentionally queried lazily. Importing the kernel package
must remain safe on CPU-only hosts used by kernel inventory tests.
"""

import functools

import torch
import triton


def _device_properties() -> dict:
    # Importing the Ascend runtime installs the active Triton driver when it is
    # available. The import is deliberately local so CPU imports stay harmless.
    try:
        import triton.backends.ascend.runtime  # noqa: F401
    except ImportError:
        pass
    device = torch.npu.current_device()
    return triton.runtime.driver.active.utils.get_device_properties(device)


@functools.lru_cache(maxsize=1)
def get_npu_vector_core_count() -> int:
    try:
        from sgl_kernel_npu.utils.triton_utils import get_device_properties

        _, num_vector_cores = get_device_properties()
        count = int(num_vector_cores)
    except (ImportError, RuntimeError):
        count = int(_device_properties().get("num_vectorcore", 0))
    if count <= 0:
        raise RuntimeError("Failed to detect the Ascend vector-core count")
    return count


@functools.lru_cache(maxsize=1)
def get_npu_ai_core_count() -> int:
    count = int(_device_properties().get("num_aicore", 0))
    if count <= 0:
        raise RuntimeError("Failed to detect the Ascend AI-core count")
    return count
