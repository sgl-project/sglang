# Copyright (C) 2025 Intel Corporation
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
"""Run the model with torch.compile for XPU devices."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)
from typing import TYPE_CHECKING

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class XPUGraphRunner:
    """A XPUGraphRunner wraps the model with torch.compile for XPU devices."""

    def __init__(self, model_runner: ModelRunner):
        if model_runner.server_args.enable_torch_compile:
            model_runner.model = torch.compile(model_runner.model)

    def can_run(self, forward_batch: ForwardBatch):
        """
        Because XPUGraphRunner is only utilizing the torch.compile feature,
        we'll never capture and replay a graph.
        """
        return False
