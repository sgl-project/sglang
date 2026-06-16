from __future__ import annotations

import logging
import time
from concurrent import futures
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.constrained.base_grammar_backend import (
    InvalidGrammarObject,
    create_grammar_backend,
)
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class GrammarManager:
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.server_args = scheduler.server_args
        self.grammar_queue: List[Req] = []
        if not self.server_args.skip_tokenizer_init:
            self.grammar_backend = create_grammar_backend(
                self.server_args,
                scheduler.tokenizer,
                scheduler.model_config.vocab_size,
                scheduler.model_config.hf_eos_token_id,
                think_end_id=scheduler.model_config.think_end_id,
            )
        else:
            self.grammar_backend = None

        self._enable_strict_thinking = (
            self.grammar_backend.enable_strict_thinking
            if self.grammar_backend is not None
            else False
        )

        self.grammar_sync_group = scheduler.dp_tp_cpu_group
        self.grammar_sync_size = scheduler.dp_tp_group.world_size
        self.grammar_sync_entry = scheduler.dp_tp_group.first_rank
        self.is_grammar_sync_entry = scheduler.dp_tp_group.is_first_rank

        self.SGLANG_GRAMMAR_POLL_INTERVAL = envs.SGLANG_GRAMMAR_POLL_INTERVAL.get()
        self.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = (
            envs.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS.get()
        )

    def __len__(self):
        return len(self.grammar_queue)

    def clear(self):
        if self.grammar_backend:
            self.grammar_backend.reset()

    def has_waiting_grammars(self) -> bool:
        return len(self.grammar_queue) > 0

    def abort_requests(self, recv_req: AbortReq):
        for req in self.grammar_queue:
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                logger.debug(f"Abort grammar queue request. {req.rid=}")
                if isinstance(req.grammar, futures.Future) and req.grammar:
                    req.grammar.cancel()
                req.set_finish_with_abort("Aborted by AbortReq.")

    def _get_request_thinking_budget(self, req: Req) -> int | None:
        custom_params = req.sampling_params.custom_params
        if not isinstance(custom_params, dict):
            return None
        thinking_budget = custom_params.get("thinking_budget")
        return thinking_budget if isinstance(thinking_budget, int) else None

    def _apply_request_reasoning_budget(self, req: Req) -> None:
        thinking_budget = self._get_request_thinking_budget(req)
        if thinking_budget is None:
            return
        if isinstance(req.grammar, ReasonerGrammarObject):
            req.grammar.max_think_tokens = thinking_budget

    def process_req_with_grammar(self, req: Req) -> bool:
        # Init grammar cache for this request
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ):
            if self.grammar_backend is None:
                error_msg = "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not supported when the server is launched with --grammar-backend none"
                req.set_finish_with_abort(error_msg)
            else:
                if req.sampling_params.json_schema is not None:
                    key = ("json", req.sampling_params.json_schema)
                elif req.sampling_params.regex is not None:
                    key = ("regex", req.sampling_params.regex)
                elif req.sampling_params.ebnf is not None:
                    key = ("ebnf", req.sampling_params.ebnf)
                elif req.sampling_params.structural_tag:
                    key = ("structural_tag", req.sampling_params.structural_tag)

                value, cache_hit = self.grammar_backend.get_cached_or_future_value(
                    key, req.require_reasoning
                )
                req.grammar = value

                if not cache_hit:
                    req.grammar_key = key
                    add_to_grammar_queue = True
                else:
                    if isinstance(
                        value, InvalidGrammarObject
                    ):  # We hit a cached invalid grammar.
                        error_msg = (
                            f"Failed to compile {key[0]} grammar: {value.error_message}"
                        )
                        req.set_finish_with_abort(error_msg)
                    else:
                        self._apply_request_reasoning_budget(req)
        elif self._enable_strict_thinking:
            grammar_obj = self.grammar_backend.init_strict_reasoning_grammar(
                req.require_reasoning
            )
            if grammar_obj is not None:
                req.grammar = grammar_obj
                self._apply_request_reasoning_budget(req)

        if add_to_grammar_queue:
            self.grammar_queue.append(req)

        return add_to_grammar_queue

    def poll_ready_grammar_request_rids(self) -> List[List[str]]:
        """
        Poll grammar_queue and return [ready_rids, failed_rids].

        Each rank first computes local ready/failed grammar queue indices.
        These indices are synchronized within the TP/DP grammar sync group:
        ready_idxs_all = all_gather(ready_idxs_i)
        failed_idxs_all = all_gather(failed_idxs_i)

        synced_ready_idxs = intersect(ready_idxs_all)
        synced_failed_idxs = union(failed_idxs_all)

        The synchronized indices are converted to request RIDs before returning,
        so PP consensus can merge grammar readiness by stable request identity.
        """
        assert self.grammar_backend
        ready_req_idxs: set[int] = set()
        failed_req_idxs: set[int] = set()

        # Poll for ready requests
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < self.SGLANG_GRAMMAR_POLL_INTERVAL:
            for i, req in enumerate(self.grammar_queue):
                if i in ready_req_idxs:
                    continue

                if req.finished() or req.grammar is None:  # It is aborted by AbortReq
                    ready_req_idxs.add(i)
                    continue

                assert isinstance(req.grammar, futures.Future), f"{req=}"
                if req.grammar.done():
                    ready_req_idxs.add(i)

            # Sleep a bit to avoid busy waiting
            time.sleep(self.SGLANG_GRAMMAR_POLL_INTERVAL / 10)

        # Check failed requests
        for i, req in enumerate(self.grammar_queue):
            if i not in ready_req_idxs:
                self.grammar_queue[i].grammar_wait_ct += 1
                if (
                    self.grammar_queue[i].grammar_wait_ct
                    >= self.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS
                ):
                    # Timeout after max poll iterations
                    # The actual waiting time is SGLANG_GRAMMAR_MAX_POLL_ITERATIONS * max(SGLANG_GRAMMAR_POLL_INTERVAL, GPU_forward_batch_latency)
                    failed_req_idxs.add(i)

        # Sync ready and failed requests across all ranks
        if self.grammar_sync_size == 1:
            synced_ready_req_idxs = ready_req_idxs
            synced_failed_req_idxs = failed_req_idxs
        else:
            all_gather_output = [None] * self.grammar_sync_size
            torch.distributed.all_gather_object(
                all_gather_output,
                (ready_req_idxs, failed_req_idxs),
                group=self.grammar_sync_group,
            )
            synced_ready_req_idxs = set.intersection(*[x[0] for x in all_gather_output])
            synced_failed_req_idxs = set.union(*[x[1] for x in all_gather_output])

        ready_rids = [
            self.grammar_queue[i].rid
            for i in range(len(self.grammar_queue))
            if i in synced_ready_req_idxs
        ]
        failed_rids = [
            self.grammar_queue[i].rid
            for i in range(len(self.grammar_queue))
            if i in synced_failed_req_idxs
        ]
        return [ready_rids, failed_rids]

    def pop_ready_grammar_requests_by_rids(
        self, ready_rids: List[str], failed_rids: List[str]
    ) -> List[Req]:
        """Move consensus-ready grammar requests out of grammar_queue."""
        assert self.grammar_backend
        ready_rid_set = set(ready_rids)
        failed_rid_set = set(failed_rids)
        # A timeout/failure from any rank should win over readiness.
        ready_rid_set -= failed_rid_set

        return_reqs: List[Req] = []
        popped_req_idxs: set[int] = set()
        for i, req in enumerate(self.grammar_queue):
            is_failed = req.rid in failed_rid_set
            is_ready = req.rid in ready_rid_set
            if not is_failed and not is_ready:
                continue

            popped_req_idxs.add(i)
            return_reqs.append(req)
            if req.finished() or req.grammar is None:  # It is aborted by AbortReq
                continue

            assert req.grammar_key
            assert isinstance(req.grammar, futures.Future), f"{req=}"
            if is_failed:
                req.grammar.cancel()
                self.grammar_backend.set_cache(
                    req.grammar_key,
                    InvalidGrammarObject("Grammar preprocessing timed out"),
                )
                error_msg = f"Grammar preprocessing timed out: {req.grammar_key=}"
                req.set_finish_with_abort(error_msg)
            else:
                try:
                    req.grammar = req.grammar.result()
                except Exception as e:
                    logger.error(
                        f"Grammar compilation raised an exception: {e}, "
                        f"grammar_key={req.grammar_key}"
                    )
                    req.grammar = InvalidGrammarObject(
                        f"Grammar compilation failed: {e}"
                    )
                self.grammar_backend.set_cache(req.grammar_key, req.grammar.copy())
                self._apply_request_reasoning_budget(req)
                if isinstance(req.grammar, InvalidGrammarObject):
                    error_msg = f"Failed to compile {req.grammar_key[0]} grammar: {req.grammar.error_message}"
                    req.set_finish_with_abort(error_msg)

        # Remove finished requests from grammar_queue
        self.grammar_queue = [
            req for i, req in enumerate(self.grammar_queue) if i not in popped_req_idxs
        ]
        return return_reqs

    def get_ready_grammar_requests(self) -> List[Req]:
        """Move requests whose grammar objects are ready from grammar_queue."""
        ready_rids, failed_rids = self.poll_ready_grammar_request_rids()
        return self.pop_ready_grammar_requests_by_rids(ready_rids, failed_rids)
