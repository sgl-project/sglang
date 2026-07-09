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

"""Kind enum + process-wide singleton for the decode context parallel (DCP) strategy.

Deliberately mirrors the prefill CP-v2 "head" in ``layers/cp/base.py`` (kind enum +
``init_/get_/is_`` singletons with lazy worker recovery) so a developer who knows CP
navigates DCP the same way. But decode-CP is a SEPARATE parallelism on the ``_DCP``
group (owner rule ``pos % dcp_size == dcp_rank``); it does NOT inherit the prefill
``ContextParallelStrategy`` ABC. The concrete strategy lives in
``layers/dcp/strategy.py``; the runtime facade (None-safe op wrappers) in
``layers/dcp/utils.py``.

Keep this module ``server_args``-free at import time: ``strategy`` (which transitively
imports ``planner`` -> ``server_args``) is imported lazily inside ``init_dcp_strategy``.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.layers.dcp.strategy import DecodeContextParallelStrategy
    from sglang.srt.server_args import ServerArgs


class DecodeContextParallelStrategyKind(IntEnum):
    """Decode context parallel strategy identifiers.

    Distinct from ``ContextParallelStrategyKind`` (prefill): a DCP strategy reports
    ``DECODE`` here, so ``strategy.kind`` is never confused with the prefill layout.
    """

    NONE = 0
    DECODE = 1


# _UNSET distinguishes "not yet resolved" from "resolved to disabled" (dcp_size<=1
# -> None), so get_dcp_strategy() memoizes the disabled state instead of re-running
# the server-args lookup on every call. This matters because the DCP facade
# (layers/dcp/utils.py) is invoked from ungated hot-path sites (per attention layer /
# per decode step), where the DCP-off case must stay as cheap as the free functions
# it replaced.
_UNSET = object()
_DCP_STRATEGY = _UNSET


def init_dcp_strategy(server_args: ServerArgs) -> None:
    """Bind the decode-context-parallel strategy for this process.

    Independent of the prefill CP strategy: DCP is configured by ``dcp_size`` (not
    ``attn_cp_size``/``cp_strategy``) and runs on the ``_DCP`` group. Platform-agnostic
    (``dcp_size > 1`` on both CUDA-MLA and AMD-HIP-MHA).
    """
    global _DCP_STRATEGY

    if getattr(server_args, "dcp_size", 1) > 1:
        from sglang.srt.layers.dcp.strategy import DecodeContextParallelStrategy

        _DCP_STRATEGY = DecodeContextParallelStrategy(dcp_size=server_args.dcp_size)
    else:
        _DCP_STRATEGY = None


def get_dcp_strategy() -> Optional[DecodeContextParallelStrategy]:
    """Return the decode-CP strategy, lazily initializing from global server args.

    Mirrors ``get_cp_strategy``'s lazy pattern so pickled worker processes (which
    bypass ``ServerArgs.__post_init__``) recover the singleton. Once resolved (to a
    strategy or, when disabled, None) the result is memoized. Returns None when DCP
    is not configured (``dcp_size <= 1``).
    """
    global _DCP_STRATEGY

    if _DCP_STRATEGY is _UNSET:
        from sglang.srt.server_args import get_global_server_args

        try:
            server_args = get_global_server_args()
        except ValueError:
            return None  # global args not set yet; retry on the next call
        if server_args is None:
            return None
        init_dcp_strategy(server_args)  # resolves _DCP_STRATEGY to a strategy or None
    return None if _DCP_STRATEGY is _UNSET else _DCP_STRATEGY


def is_dcp_enabled() -> bool:
    """True if decode context parallel is configured (``dcp_size > 1``).

    This is the *configuration* gate (strategy existence). It is broader than
    ``layers.dcp.comm.dcp_enabled()`` — the latter additionally requires CUDA and is
    the MLA-path runtime gate. Use ``dcp_enabled()`` where the CUDA gate is intended.
    """
    return get_dcp_strategy() is not None
