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
"""cuBLAS workspace preparation at non-capturing graph boundaries."""

import torch


def prepare_cublas_for_graph_capture(device_module) -> None:
    """Allocate fresh workspaces inside the next CUDA graph's private pool.

    Eager warmup or an eager graph break can cache default-pool workspaces.
    Capturing their pointers does not retain that storage: a later graph reset
    clears cuBLAS's cache, allowing empty_cache to release it. Clear the cache
    before each capture, including resumed breakable-graph segments.

    Call outside capture, with serial captures and one GPU per process.
    The clear is process-wide; current-device synchronization alone does not
    make this safe against concurrent GEMMs on another device or host thread.
    """
    if torch.version.cuda is not None:
        device_module.synchronize()
        torch._C._cuda_clearCublasWorkspaces()
