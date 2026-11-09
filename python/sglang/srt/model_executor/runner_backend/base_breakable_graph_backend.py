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
"""Semantic interface for segmented device-graph backends."""

from sglang.srt.model_executor.runner_backend.base_graph_backend import (
    BaseGraphBackend,
)

__all__ = ["BaseBreakableGraphBackend"]


class BaseBreakableGraphBackend(BaseGraphBackend):
    """A device graph captured as ordered segments separated by eager calls.

    Implementations provide the :class:`BaseGraphBackend` lifecycle. During
    replay, captured segments and their intervening eager calls must execute in
    capture order, with captured tensor addresses kept valid until cleanup.

    This semantic base intentionally contains no CUDA capture machinery. It lets
    runners recognize breakable behavior without requiring another platform to
    inherit from :class:`BreakableCudaGraphBackend`.
    """
