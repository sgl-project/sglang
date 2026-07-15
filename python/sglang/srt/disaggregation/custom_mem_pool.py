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
"""Resolve transport-specific memory pools for PD disaggregation."""

from typing import Any, Optional, Tuple

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_server_args


def _get_server_args_or_none():
    # Standalone pool tests may run before ModelRunner publishes ServerArgs.
    try:
        return get_server_args()
    except ValueError:
        return None


def maybe_init_custom_mem_pool(
    device: str,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """Initialize the custom memory pool required by the active transport."""
    server_args = _get_server_args_or_none()
    # Scope Mori's inner backend knob to processes actually using Mori PD.
    if (
        server_args is not None
        and getattr(server_args, "disaggregation_mode", "null") in ("prefill", "decode")
        and getattr(server_args, "disaggregation_transfer_backend", None) == "mori"
    ):
        from sglang.srt.disaggregation.mori.custom_mem_pool import (
            init_mori_custom_mem_pool,
        )

        return init_mori_custom_mem_pool(device)

    # Preserve the existing Mooncake env-driven behavior. Restricting this to
    # the Mooncake PD backend would be a separate behavior change.
    if envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() is not None:
        from sglang.srt.disaggregation.mooncake.utils import (
            init_mooncake_custom_mem_pool,
        )

        return init_mooncake_custom_mem_pool(device)

    return False, None, None
