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

from sglang.srt.utils import is_cpu

_is_cpu = is_cpu()

# Device dispatch happens here (not inside the triton implementations):
# `rotate_input_ids` resolves to the CPU C++ kernel or the Triton kernel once
# at import time. Both mutate input_ids in place; callers ignore the return.
if _is_cpu:
    from sgl_kernel import rotate_input_ids_cpu as rotate_input_ids
else:
    from sglang.srt.speculative.triton_ops.multi_layer_eagle import (
        rotate_input_ids_triton as rotate_input_ids,
    )

__all__ = [
    "rotate_input_ids",
]
