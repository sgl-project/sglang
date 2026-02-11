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
"""Intel Habana HPU platform implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from sglang.srt.platforms.interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from sglang.srt.ops.base import OpProxy
    from sglang.srt.server_args import ServerArgs


class HpuPlatform(Platform):
    """Intel Habana HPU platform implementation."""

    _enum = PlatformEnum.HPU
    device_name = "hpu"
    device_type = "hpu"

    # Lazy-loaded op registry
    _ops: dict[str, Callable] | None = None

    @classmethod
    def _init_ops(cls) -> dict[str, Callable]:
        """Initialize op registry. Called once on first op access.

        HPU currently uses native PyTorch fallbacks for activation ops.
        """
        # HPU doesn't have specialized activation kernels yet
        # Will use fallbacks defined in ops/activation.py
        return {}

    def get_op(self, op: "OpProxy") -> Callable | None:
        """Get the HPU implementation of an operation."""
        if HpuPlatform._ops is None:
            HpuPlatform._ops = self._init_ops()
        return HpuPlatform._ops.get(op.name)

    def postprocess_server_args(self, args: "ServerArgs") -> None:
        """Apply HPU-specific server argument defaults.

        Sets attention and sampling backends to use native PyTorch
        implementations which are compatible with HPU.
        """
        args.attention_backend = "torch_native"
        args.sampling_backend = "pytorch"
