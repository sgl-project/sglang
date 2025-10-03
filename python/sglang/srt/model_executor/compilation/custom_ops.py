# Copyright 2025 SGLang Team
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
from typing import Optional
from sglang.srt.layers.dp_attention import _DpGatheredBufferWrapper

@torch.library.custom_op("mylib::_set_dp_buffer_len", mutates_args=())
def _set_dp_buffer_len(global_dp_buffer_len: Optional[int], num_tokens: Optional[int]) -> None:
    _DpGatheredBufferWrapper._global_dp_buffer_len = global_dp_buffer_len
    _DpGatheredBufferWrapper._local_dp_buffer_len = num_tokens


@_set_dp_buffer_len.register_fake
def _set_dp_buffer_len_register_fake(global_dp_buffer_len, num_tokens) -> None:
    pass
