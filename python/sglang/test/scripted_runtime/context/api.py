"""ScriptedContext: the object a test script drives.

Passed to the caller-provided script generator as its single argument
(``def my_script(t: ScriptedContext)``). Exposes the script-facing verbs
— submit requests, inject pressure, query scheduler state — and reaches
the live ``Scheduler`` through its :class:`ScriptedSchedulerHook`. The
hook owns the generator stepping and the scheduler-side lookups; this
object owns everything the script itself calls.

This class is a thin FACADE: every public method delegates to a free
function grouped by category in a sibling module (``start_req``,
``lifecycle``, ``pressure``, ``radix``, ``queries``). The flat
script-author-facing API is unchanged — the logic and docstrings live on
the free functions, which take this context as their first argument.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Union

from sglang.test.scripted_runtime.context import (
    lifecycle,
    pressure,
    queries,
    radix,
    start_req,
)

if TYPE_CHECKING:
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
    ) -> None:
        self._scheduler_hook = scheduler_hook
        self._scheduler = scheduler_hook._scheduler
        self._is_driver = scheduler_hook._is_driver
        self._tokenizer_recv_proxy = tokenizer_recv_proxy

        self._req_handles: dict[str, "ScriptedReqHandle"] = {}
        self._req_counter = 0
        self._http_threads: List[threading.Thread] = []

    # ============================================================
    # Request submission.
    # ============================================================
    def start_req(
        self,
        *,
        prompt_len: int,
        max_new_tokens: int = 8,
        rid: Optional[str] = None,
        prompt_tokens: Optional[List[int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        ignore_eos: bool = False,
        min_new_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        return_logprob: bool = False,
        top_logprobs_num: Optional[int] = None,
        logprob_start_len: Optional[int] = None,
        priority: Optional[int] = None,
        lora_path: Optional[str] = None,
        session_id: Optional[str] = None,
        dp_rank: Optional[int] = None,
        return_hidden_states: bool = False,
        grammar: Optional[str] = None,
        stream: bool = False,
    ) -> "ScriptedReqHandle":
        return start_req.start_req(
            self,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            rid=rid,
            prompt_tokens=prompt_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            ignore_eos=ignore_eos,
            min_new_tokens=min_new_tokens,
            stop=stop,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            return_logprob=return_logprob,
            top_logprobs_num=top_logprobs_num,
            logprob_start_len=logprob_start_len,
            priority=priority,
            lora_path=lora_path,
            session_id=session_id,
            dp_rank=dp_rank,
            return_hidden_states=return_hidden_states,
            grammar=grammar,
            stream=stream,
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

    def trigger_abort_on_waiting_timeout(self) -> None:
        return lifecycle.trigger_abort_on_waiting_timeout(self)

    def shutdown(self) -> None:
        return lifecycle.shutdown(self)

    def abort(self, r: "ScriptedReqHandle") -> None:
        return lifecycle.abort(self, r)

    def force_retract(self, r: "ScriptedReqHandle") -> None:
        return lifecycle.force_retract(self, r)

    def retract_all(self) -> None:
        return lifecycle.retract_all(self)

    def pause_retract_all(self) -> None:
        return lifecycle.pause_retract_all(self)

    def force_preempt(
        self, *, req: "ScriptedReqHandle", by: "ScriptedReqHandle"
    ) -> None:
        return lifecycle.force_preempt(self, req=req, by=by)

    def force_lora_drainer_reject(self, *, adapter: str) -> None:
        return lifecycle.force_lora_drainer_reject(self, adapter=adapter)

    # ============================================================
    # KV / pool pressure injection.
    # ============================================================
    def exhaust_kv(self, *, leave_pages: int) -> None:
        return pressure.exhaust_kv(self, leave_pages=leave_pages)

    def exhaust_row_pool(self, *, leave_rows: int) -> None:
        return pressure.exhaust_row_pool(self, leave_rows=leave_rows)

    def exhaust_lock_refs(self, *, leave_refs: int) -> None:
        return pressure.exhaust_lock_refs(self, leave_refs=leave_refs)

    def row_pool_used(self) -> int:
        return pressure.row_pool_used(self)

    # ============================================================
    # Radix operations.
    # ============================================================
    def get_all_node_hit_counts(self) -> Dict[int, int]:
        return radix.get_all_node_hit_counts(self)

    def get_all_node_lock_refs(self) -> Dict[int, int]:
        return radix.get_all_node_lock_refs(self)

    def warmup_radix(self, *, prompt_tokens: List[int]) -> None:
        return radix.warmup_radix(self, prompt_tokens=prompt_tokens)

    def evict_radix(self, *, prompt_tokens: List[int]) -> None:
        return radix.evict_radix(self, prompt_tokens=prompt_tokens)

    # ============================================================
    # State queries (read-only).
    # ============================================================
    def chunked_in_flight_count(self) -> int:
        return queries.chunked_in_flight_count(self)

    def get_chunked_req_rid(self) -> Optional[str]:
        return queries.get_chunked_req_rid(self)

    def is_idle(self) -> bool:
        return queries.is_idle(self)

    def is_fully_idle(self) -> bool:
        return queries.is_fully_idle(self)

    def batch_size(self) -> int:
        return queries.batch_size(self)

    def batch_composition(self) -> Dict[str, List[str]]:
        return queries.batch_composition(self)

    def batch_rids(self) -> Set[str]:
        return queries.batch_rids(self)

    def waiting_rids(self) -> Set[str]:
        return queries.waiting_rids(self)

    def running_rids(self) -> Set[str]:
        return queries.running_rids(self)

    def in_flight_other_mb_rids(self) -> Set[str]:
        return queries.in_flight_other_mb_rids(self)

    def list_active_reqs(self) -> List[str]:
        return queries.list_active_reqs(self)

    def forward_mode(self) -> str:
        return queries.forward_mode(self)

    def engine_stats(self) -> Dict[str, Any]:
        return queries.engine_stats(self)

    def kv_pool_underflow_count(self) -> int:
        return queries.kv_pool_underflow_count(self)

    def lock_refs_snapshot(self) -> int:
        return queries.lock_refs_snapshot(self)

    def load_inquirer_num_pending_tokens(self) -> int:
        return queries.load_inquirer_num_pending_tokens(self)

    def load_inquirer_snapshot(self) -> Dict[str, int]:
        return queries.load_inquirer_snapshot(self)

    def last_admission_path(self) -> Optional[str]:
        return queries.last_admission_path(self)

    def last_scheduler_path(self) -> Optional[str]:
        return queries.last_scheduler_path(self)

    def last_chunked_exclude_set_source(self) -> Optional[str]:
        return queries.last_chunked_exclude_set_source(self)

    def dp_rank_max_pending(self, rank: int) -> int:
        return queries.dp_rank_max_pending(self, rank)

    def dp_rank_is_idle(self, rank: int) -> bool:
        return queries.dp_rank_is_idle(self, rank)
