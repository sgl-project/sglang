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
"""Elastic EP rank status publishing."""

from typing import Any

import torch

from sglang.srt.managers.io_struct import ActiveRanksOutput
from sglang.srt.server_args import ServerArgs


class ElasticEPStatusPublisher:
    def publish_committed_active_ranks(
        self,
        committed_active_ranks: torch.Tensor,
    ) -> None:
        pass


class ControllerElasticEPStatusPublisher(ElasticEPStatusPublisher):
    def __init__(self, send_to_controller: Any, dp_size: int):
        self.send_to_controller = send_to_controller
        self.dp_size = dp_size
        self.last_status = [True] * dp_size

    def publish_committed_active_ranks(
        self,
        committed_active_ranks: torch.Tensor,
    ) -> None:
        assert committed_active_ranks.numel() % self.dp_size == 0
        dp_active_ranks = committed_active_ranks.reshape(self.dp_size, -1).prod(dim=1)
        status = dp_active_ranks.bool().tolist()

        if status == self.last_status:
            return
        self.last_status = status
        self.send_to_controller.send_output(ActiveRanksOutput(status=status))


def create_elastic_ep_status_publisher(
    server_args: ServerArgs, send_to_controller: Any
) -> ElasticEPStatusPublisher:
    dp_worker_capacity = server_args.max_ep_size or server_args.dp_size
    if (
        server_args.enable_dp_attention
        and server_args.elastic_ep_backend is not None
        and dp_worker_capacity > 1
    ):
        return ControllerElasticEPStatusPublisher(
            send_to_controller, dp_worker_capacity
        )
    return ElasticEPStatusPublisher()
