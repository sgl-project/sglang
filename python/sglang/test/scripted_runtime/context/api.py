from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

from sglang.test.scripted_runtime.context import (
    engine,
    lifecycle,
    queries,
    radix,
)
from sglang.test.scripted_runtime.context.kv_pool_exhauster import (
    ScriptedKvPoolExhauster,
)
from sglang.test.scripted_runtime.context.lock_ref_exhauster import (
    ScriptedLockRefExhauster,
)
from sglang.test.scripted_runtime.context.req_starter import ScriptedContextReqStarter

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.test.scripted_runtime.background_http_poster import BackgroundHttpPoster
    from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle
    from sglang.test.scripted_runtime.scheduler_hook import ScriptedSchedulerHook
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )

logger = logging.getLogger(__name__)


class ScriptedContext:

    def __init__(
        self,
        *,
        scheduler_hook: ScriptedSchedulerHook,
        tokenizer_recv_proxy: Optional[ScriptedTokenizerRecvProxy],
        http_poster: BackgroundHttpPoster,
    ) -> None:
        assert (
            scheduler_hook._is_driver
        ), "ScriptedContext only exists on the driver rank"
        self.scheduler = scheduler_hook.scheduler
        self._scheduler_hook = scheduler_hook
        self._tokenizer_recv_proxy = tokenizer_recv_proxy
        self._http_poster = http_poster

        self._seen_rids: set[str] = set()
        self._kv_exhauster = ScriptedKvPoolExhauster(self.scheduler)
        self._lock_ref_exhauster = ScriptedLockRefExhauster(self.scheduler)
        self._req_starter = ScriptedContextReqStarter(self)

    def start_req(
        self,
        *,
        prompt_len: int,
        max_new_tokens: int = 8,
        rid: Optional[str] = None,
        ignore_eos: bool = False,
        priority: Optional[int] = None,
        dp_rank: Optional[int] = None,
        prompt_token: int = 1,
        return_logprob: bool = False,
        logprob_start_len: Optional[int] = None,
        top_logprobs_num: Optional[int] = None,
        stop_token_ids: Optional[List[int]] = None,
        temperature: Optional[float] = None,
        lora_path: Optional[str] = None,
    ) -> ScriptedReqHandle:
        return self._req_starter.start_req(
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            rid=rid,
            ignore_eos=ignore_eos,
            priority=priority,
            dp_rank=dp_rank,
            prompt_token=prompt_token,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            lora_path=lora_path,
        )

    def pause_generation(self, *, mode: Literal["retract", "in_place"]) -> None:
        return lifecycle.pause_generation(self, mode=mode)

    def continue_generation(self, *, torch_empty_cache: bool = False) -> None:
        return lifecycle.continue_generation(self, torch_empty_cache=torch_empty_cache)

    def abort_all(self) -> None:
        return lifecycle.abort_all(self)

    def abort(self, handle: ScriptedReqHandle, *, await_arrival: bool = True) -> None:
        return lifecycle.abort(self, rid=handle.rid, await_arrival=await_arrival)

    def flush_cache(self) -> None:
        return lifecycle.flush_cache(self)

    def evict_radix(self, *, prefix_tokens: Optional[List[int]]) -> None:
        assert (
            prefix_tokens is None
        ), "evict_radix currently supports only full eviction (prefix_tokens=None)"
        return lifecycle.flush_cache(self)

    def exhaust_kv(self, *, leave_pages: int) -> None:
        return self._kv_exhauster.exhaust(leave_pages=leave_pages)

    def exhaust_lock_refs(self, *, leave_refs: int) -> None:
        return self._lock_ref_exhauster.exhaust(leave_refs=leave_refs)

    def _release_exhausted_pools(self) -> None:
        self._kv_exhauster.release()
        self._lock_ref_exhauster.release()

    def get_all_node_hit_counts(self) -> Dict[int, int]:
        return radix.get_all_node_hit_counts(self)

    def get_all_node_lock_refs(self) -> Dict[int, int]:
        return radix.get_all_node_lock_refs(self)

    @property
    def is_idle(self) -> bool:
        return queries.is_idle(self)

    @property
    def is_fully_idle(self) -> bool:
        return queries.is_fully_idle(self)

    @property
    def last_batch_forward_mode(self) -> Optional[str]:
        return queries.last_batch_forward_mode(self)

    def find_req_by_rid(self, rid: str) -> Optional[Req]:
        return queries.find_req_by_rid(self, rid)

    def is_finished(self, rid: str) -> bool:
        return queries.is_finished(self, rid)

    def is_chunking(self, rid: str) -> bool:
        return queries.is_chunking(self, rid)

    def status(self, rid: str) -> str:
        return queries.status(self, rid)

    def remaining_prompt_tokens(self, rid: str) -> int:
        return queries.remaining_prompt_tokens(self, rid)

    def list_active_reqs(self) -> List[Req]:
        return queries.list_active_reqs(self)

    def chunks_done(self, rid: str) -> int:
        return queries.chunks_done(self, rid)

    def chunked_parks(self, rid: str) -> int:
        return queries.chunked_parks(self, rid)

    def batch_composition(self) -> Dict[str, List[str]]:
        return queries.batch_composition(self)

    def engine_stats(self) -> Dict[str, int]:
        return engine.engine_stats(self)
