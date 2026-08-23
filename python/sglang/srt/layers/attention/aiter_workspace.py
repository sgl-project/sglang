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
"""Size of the AITER paged-decode split workspace.

The buffer is allocated by ``AiterAttnBackend`` *after* the KV pool exists, so
``KVCacheConfigurator`` has to charge it against the pool budget. Both callers
share this formula rather than inlining it at the ``torch.empty``: the charge
and the allocation must be the same number, and any gap between them is lost
memory, linear in the gap.
"""

AITER_PARTITION_SIZE_ROCM = 256

_FP32_BYTES = 4


def aiter_attn_workspace_bytes(
    *, max_num_reqs: int, num_head: int, head_dim: int, context_len: int
) -> int:
    """Bytes of ``AiterAttnBackend.workspace_buffer``: fp32 partial outputs plus
    the two fp32 (max_logit, exp_sum) reduction planes, one set per partition."""
    num_partitions = (
        context_len + AITER_PARTITION_SIZE_ROCM - 1
    ) // AITER_PARTITION_SIZE_ROCM
    per_partition = max_num_reqs * num_head * num_partitions
    return per_partition * head_dim * _FP32_BYTES + 2 * per_partition * _FP32_BYTES
