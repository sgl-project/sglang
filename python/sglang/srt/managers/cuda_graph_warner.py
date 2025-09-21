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
"""NoCudaGraphWarner is a class that warns when CUDA graph is disabled for too many consecutive decode steps."""

import logging

logger = logging.getLogger(__name__)


class NoCudaGraphWarner:
    def __init__(self):
        self.consecutive_no_cuda_graph_count = 0

    def on_step(self, can_run_cuda_graph: bool):
        if can_run_cuda_graph:
            self.consecutive_no_cuda_graph_count = 0
        else:
            self.consecutive_no_cuda_graph_count += 1
            if self.consecutive_no_cuda_graph_count % 5 == 0:
                logger.warning(
                    f"CUDA graph disabled for {self.consecutive_no_cuda_graph_count} consecutive decode steps. "
                    f"This may cause significant throughput degradation. "
                    f"Check if batch size is within CUDA graph capture range."
                )
