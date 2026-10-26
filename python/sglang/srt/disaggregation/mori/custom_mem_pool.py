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
"""Mori-specific custom memory pool management."""

from typing import Any, Optional, Tuple

from sglang.srt.environ import envs

_MORI_FABRIC_MEM_POOL: Optional[Any] = None


def get_mori_fabric_mem_pool() -> Any:
    global _MORI_FABRIC_MEM_POOL
    if _MORI_FABRIC_MEM_POOL is None:
        try:
            from mori.io import make_fabric_mem_pool
        except (AttributeError, ImportError) as exc:
            raise RuntimeError(
                "SGLANG_MORI_BACKEND=fabric requires a Mori build with "
                "fabric memory-pool support"
            ) from exc

        try:
            _MORI_FABRIC_MEM_POOL = make_fabric_mem_pool()
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize the Mori fabric memory pool"
            ) from exc
    return _MORI_FABRIC_MEM_POOL


def init_mori_custom_mem_pool(
    device: str,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """Place KV/state pools in Fabric memory when Fabric reads them directly.

    With staging on, only the staging slots need Fabric memory.
    """
    if (
        envs.SGLANG_MORI_BACKEND.get().strip().lower() != "fabric"
        or envs.SGLANG_MORI_FABRIC_STAGING.get()
    ):
        return False, None, None
    return True, get_mori_fabric_mem_pool(), "mori_fabric"
