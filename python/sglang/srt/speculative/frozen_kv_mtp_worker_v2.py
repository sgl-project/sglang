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
# ==============================================================================
"""Overlap-scheduling placeholder for frozen-KV MTP (raises until implemented)."""

from __future__ import annotations

from typing import Optional

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.frozen_kv_mtp_worker import FrozenKVMTPWorker


class FrozenKVMTPWorkerV2(FrozenKVMTPWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        raise NotImplementedError(
            "FrozenKVMTPWorkerV2 (overlap scheduling for Frozen-KV MTP) is "
            "not yet implemented. Pass --disable-overlap-schedule to use "
            "FrozenKVMTPWorker."
        )
