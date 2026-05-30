"""ScriptedContext: the object a test script drives.

Passed to the caller-provided script generator as its single argument
(``def my_script(t: ScriptedContext)``). Exposes the script-facing verbs
— submit requests, inject pressure, query scheduler state — and reaches
the live ``Scheduler`` through its :class:`ScriptedSchedulerHook`. The
hook owns the generator stepping and the scheduler-side lookups; this
object owns everything the script itself calls.

This class is a thin FACADE: every public method delegates to a free
function grouped by category in a sibling module (``start_req``,
``lifecycle``, ``radix``). The flat script-author-facing API is unchanged
— the logic and docstrings live on the free functions, which take this
context as their first argument.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Literal, Optional

from sglang.test.scripted_runtime.context import (
    lifecycle,
    radix,
    start_req,
)

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.background_http_poster import BackgroundHttpPoster
    from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle
    from sglang.test.scripted_runtime.scheduler_hook import ScriptedSchedulerHook
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )

logger = logging.getLogger(__name__)


class ScriptedContext:
    """Script-facing control surface, created by :class:`ScriptedSchedulerHook`.

    Holds a back-reference to the hook (and through it the live
    ``Scheduler``) plus the script-side request bookkeeping. Every method
    here is called from the test script on the driver rank and forwards to
    a free function that holds the real logic.
    """

    def __init__(
        self,
        *,
        scheduler_hook: "ScriptedSchedulerHook",
        tokenizer_recv_proxy: Optional["ScriptedTokenizerRecvProxy"],
        http_poster: "BackgroundHttpPoster",
    ) -> None:
        self._scheduler_hook = scheduler_hook
        self._scheduler = scheduler_hook._scheduler
        self._is_driver = scheduler_hook._is_driver
        self._tokenizer_recv_proxy = tokenizer_recv_proxy
        self._http_poster = http_poster

        self._req_handles: dict[str, "ScriptedReqHandle"] = {}
        self._req_counter = 0

    # ============================================================
    # Request submission.
    # ============================================================
    def start_req(
        self,
        *,
        prompt_len: int,
        max_new_tokens: int = 8,
        rid: Optional[str] = None,
    ) -> "ScriptedReqHandle":
        return start_req.start_req(
            self,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            rid=rid,
        )

    # ============================================================
    # Lifecycle control.
    # ============================================================
    def pause_generation(self, *, mode: Literal["retract", "in_place"]) -> None:
        return lifecycle.pause_generation(self, mode=mode)

    def continue_generation(self, *, torch_empty_cache: bool = False) -> None:
        return lifecycle.continue_generation(self, torch_empty_cache=torch_empty_cache)

    def abort_all(self) -> None:
        return lifecycle.abort_all(self)

    def flush_cache(self) -> None:
        return lifecycle.flush_cache(self)

    # ============================================================
    # Radix operations.
    # ============================================================
    def get_all_node_hit_counts(self) -> Dict[int, int]:
        return radix.get_all_node_hit_counts(self)

    def get_all_node_lock_refs(self) -> Dict[int, int]:
        return radix.get_all_node_lock_refs(self)
