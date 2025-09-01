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

from dataclasses import dataclass
from typing import Optional


@dataclass
class CompilationConfig:
    splitting_ops: Optional[list[str]] = None
    replay_index: int = 1
    page_size: int = 0

    @classmethod
    def from_cli(cls, cli_value: str) -> "CompilationConfig":
        return CompilationConfig()
