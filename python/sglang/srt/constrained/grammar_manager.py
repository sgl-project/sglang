from __future__ import annotations

import logging
import time
from concurrent import futures
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.constrained.base_grammar_backend import (
    InvalidGrammarObject,
    PlaceholderGrammarObject,
    create_grammar_backend,
)
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_serving, get_spec
from sglang.srt.sampling.sampling_params import (
    get_request_reasoning_end_token_ids,
)

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
        if not get_serving().skip_tokenizer_init:
            self.grammar_backend = create_grammar_backend(
                self.server_args,
                scheduler.tokenizer,
                scheduler.model_config.vocab_size,
                scheduler.model_config.hf_eos_token_id,
                think_end_ids=scheduler.model_config.think_end_ids,
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
        # With TP > 1 and no speculative decoding, only the entry rank compiles
        # grammars and applies the vocab mask; the sampled token ids are
        # broadcast in the sampler. Speculative decoding drafts from per-rank
        # grammar bitmasks, so it keeps the compile-everywhere behavior.
        self.tp_grammar_entry_only = (
            self.grammar_sync_size > 1 and get_spec().speculative_algorithm is None
        )
        self.pp_rank = scheduler.ps.pp_rank
        self.pp_size = scheduler.ps.pp_size
        self.pp_group = scheduler.pp_group
        self.grammar_pp_sync_work_list = []

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

    def _drain_pp_sync_work(self):
        for p2p_work in self.grammar_pp_sync_work_list:
            p2p_work.work.wait()
        self.grammar_pp_sync_work_list.clear()

    def _pp_sync_ready_failed(
        self,
        ready_req_idxs: set[int],
        failed_req_idxs: set[int],
        failed_reasons: dict[int, str],
    ) -> tuple[set[int], set[int], dict[int, str]]:
        """
        Synchronize ready/failed grammar request indexes across the PP pipeline.

        PP0 provides the data. Each later PP rank receives it from the previous
        rank and asynchronously forwards it to the next rank.
        """
        if self.pp_size <= 1 or self.pp_group is None:
            return ready_req_idxs, failed_req_idxs, failed_reasons

        self._drain_pp_sync_work()
        data = (ready_req_idxs, failed_req_idxs, failed_reasons)
        if self.pp_rank > 0:
            data = self.pp_group.recv_object(
                src=self.pp_rank - 1,
                tag=P2PTag.GRAMMAR_PP_SYNC,
            )
        if self.pp_rank + 1 < self.pp_size:
            self.grammar_pp_sync_work_list.extend(
                self.pp_group.send_object(
                    data,
                    dst=self.pp_rank + 1,
                    async_send=True,
                    tag=P2PTag.GRAMMAR_PP_SYNC,
                )
            )
        return data

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

    def _apply_request_reasoning_config(self, req: Req) -> None:
        if not isinstance(req.grammar, ReasonerGrammarObject):
            return
        think_end_ids = get_request_reasoning_end_token_ids(
            req.sampling_params.custom_params,
            allowed_sequences=getattr(
                self.scheduler.model_config,
                "request_selectable_think_end_id_sequences",
                None,
            ),
        )
        if think_end_ids is not None:
            req.grammar.set_request_think_end_ids(think_end_ids)
        thinking_budget = self._get_request_thinking_budget(req)
        if thinking_budget is not None:
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
                elif req.sampling_params.structural_tag is not None:
                    key = ("structural_tag", req.sampling_params.structural_tag)

                if self.tp_grammar_entry_only and not self.is_grammar_sync_entry:
                    # Non-entry ranks never compile; they carry a placeholder so
                    # batch-level grammar checks stay rank-consistent and wait
                    # for the entry rank's readiness broadcast below.
                    req.grammar = PlaceholderGrammarObject()
                    req.grammar_key = key
                    add_to_grammar_queue = True
                else:
                    value, cache_hit = self.grammar_backend.get_cached_or_future_value(
                        key, req.require_reasoning
                    )
                    req.grammar = value

                    if not cache_hit or self.tp_grammar_entry_only:
                        # With entry-only compilation every rank queues the
                        # request so the ready/failed sync moves them together;
                        # the abort state of cached-invalid grammars is also
                        # propagated through that sync instead of aborting
                        # here on the entry rank only.
                        req.grammar_key = key
                        add_to_grammar_queue = True
                    if cache_hit:
                        if isinstance(
                            value, InvalidGrammarObject
                        ):  # We hit a cached invalid grammar.
                            if not self.tp_grammar_entry_only:
                                error_msg = f"Failed to compile {key[0]} grammar: {value.error_message}"
                                req.set_finish_with_abort(error_msg)
                            # With entry-only compilation the abort is applied
                            # on every rank after the sync below.
                        else:
                            self._apply_request_reasoning_config(req)
        elif self._enable_strict_thinking:
            grammar_obj = self.grammar_backend.init_strict_reasoning_grammar(
                req.require_reasoning
            )
            if grammar_obj is not None:
                req.grammar = grammar_obj
                self._apply_request_reasoning_config(req)

        if add_to_grammar_queue:
            self.grammar_queue.append(req)

        return add_to_grammar_queue

    def get_ready_grammar_requests(self) -> List[Req]:
        """
        Move requests whose grammar objects are ready from grammar_queue to waiting_queue.

        For PP0, DP/TP group rank i returns two sets ready_reqs_i,
        failed_reqs_i. ready_reqs_all = all_gather(ready_reqs_i) within
        PP0's DP/TP group. failed_reqs_all = all_gather(failed_reqs_i)
        within PP0's DP/TP group.

        ready_reqs = intersect(ready_reqs_all)
        failed_reqs = union(failed_reqs_all)

        Compile outcomes (invalid grammar, compile exception, timeout) are
        reported per index in failure messages that travel with the same
        sync, so every rank applies an identical abort state even though only
        the entry rank compiles.

        PP0 then propagates the synced result to later PP ranks. Later PP
        ranks receive and apply the propagated ready/failed decision.
        """
        assert self.grammar_backend
        ready_req_idxs: set[int] = set()
        failed_req_idxs: set[int] = set()
        failed_reasons: dict[int, str] = {}

        if self.pp_rank == 0:
            # Poll for ready requests
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < self.SGLANG_GRAMMAR_POLL_INTERVAL:
                for i, req in enumerate(self.grammar_queue):
                    if i in ready_req_idxs or i in failed_req_idxs:
                        continue

                    if req.finished() or req.grammar is None:
                        # It is aborted by AbortReq
                        ready_req_idxs.add(i)
                        continue

                    if isinstance(req.grammar, PlaceholderGrammarObject):
                        # Non-entry ranks never compile; they are ready as
                        # soon as the entry rank's decision arrives.
                        ready_req_idxs.add(i)
                        continue

                    if isinstance(req.grammar, InvalidGrammarObject):
                        # A cached-invalid grammar on the compiling rank.
                        # Fail through the sync so every rank aborts the
                        # request with the same message.
                        failed_req_idxs.add(i)
                        failed_reasons[i] = (
                            f"Failed to compile {req.grammar_key[0]} grammar: "
                            f"{req.grammar.error_message}"
                        )
                        continue

                    if not isinstance(req.grammar, futures.Future):
                        # A cache hit on the compiling rank: already resolved.
                        ready_req_idxs.add(i)
                        continue

                    if req.grammar.done():
                        try:
                            result = req.grammar.result()
                        except Exception as e:
                            logger.error(
                                f"Grammar compilation raised an exception: {e}, "
                                f"grammar_key={req.grammar_key}"
                            )
                            failed_req_idxs.add(i)
                            failed_reasons[i] = (
                                f"Failed to compile {req.grammar_key[0]} grammar: "
                                f"Grammar compilation failed: {e}"
                            )
                            continue
                        if isinstance(result, InvalidGrammarObject):
                            failed_req_idxs.add(i)
                            failed_reasons[i] = (
                                f"Failed to compile {req.grammar_key[0]} grammar: "
                                f"{result.error_message}"
                            )
                            continue
                        ready_req_idxs.add(i)

                if len(ready_req_idxs) + len(failed_req_idxs) == len(
                    self.grammar_queue
                ):
                    break

                # Sleep a bit to avoid busy waiting
                time.sleep(self.SGLANG_GRAMMAR_POLL_INTERVAL / 10)

            # Check failed requests
            for i, req in enumerate(self.grammar_queue):
                if i not in ready_req_idxs and i not in failed_req_idxs:
                    # grammar_wait_ct is only updated on PP0; later PP ranks
                    # receive PP0's ready/failed decision through PP sync.
                    req.grammar_wait_ct += 1
                    if req.grammar_wait_ct >= self.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS:
                        # Timeout after max poll iterations
                        # The actual waiting time is SGLANG_GRAMMAR_MAX_POLL_ITERATIONS * max(SGLANG_GRAMMAR_POLL_INTERVAL, GPU_forward_batch_latency)
                        failed_req_idxs.add(i)

            # Sync ready and failed requests across all TP ranks in PP0.
            if self.grammar_sync_size == 1:
                synced_ready_req_idxs = ready_req_idxs
                synced_failed_req_idxs = failed_req_idxs
            else:
                all_gather_output = [None] * self.grammar_sync_size
                torch.distributed.all_gather_object(
                    all_gather_output,
                    (ready_req_idxs, failed_req_idxs, failed_reasons),
                    group=self.grammar_sync_group,
                )
                synced_ready_req_idxs = set.intersection(
                    *[x[0] for x in all_gather_output]
                )
                synced_failed_req_idxs = set.union(*[x[1] for x in all_gather_output])
                for x in all_gather_output:
                    failed_reasons.update(x[2])
        else:
            synced_ready_req_idxs = ready_req_idxs
            synced_failed_req_idxs = failed_req_idxs

        # Propagate PP0's grammar queue decision to later PP ranks.
        (
            synced_ready_req_idxs,
            synced_failed_req_idxs,
            failed_reasons,
        ) = self._pp_sync_ready_failed(
            synced_ready_req_idxs,
            synced_failed_req_idxs,
            failed_reasons,
        )

        # Return ready requests
        return_reqs: List[Req] = []
        for i in synced_ready_req_idxs:
            req = self.grammar_queue[i]
            return_reqs.append(req)
            if (
                req.finished()
                or req.grammar is None
                or isinstance(req.grammar, PlaceholderGrammarObject)
            ):
                # Aborted by AbortReq, or a non-entry-rank placeholder whose
                # grammar state lives on the entry rank.
                continue

            if not isinstance(req.grammar, futures.Future):
                # A cache hit resolved at arrival; the reasoning config was
                # already applied when the cached value was fetched.
                continue

            assert req.grammar_key
            try:
                req.grammar = req.grammar.result()
            except Exception as e:
                logger.error(
                    f"Grammar compilation raised an exception: {e}, "
                    f"grammar_key={req.grammar_key}"
                )
                req.grammar = InvalidGrammarObject(f"Grammar compilation failed: {e}")
            self.grammar_backend.set_cache(req.grammar_key, req.grammar.copy())
            self._apply_request_reasoning_config(req)
            if isinstance(req.grammar, InvalidGrammarObject):
                error_msg = f"Failed to compile {req.grammar_key[0]} grammar: {req.grammar.error_message}"
                req.set_finish_with_abort(error_msg)

        # Return failed requests
        for i in synced_failed_req_idxs:
            req = self.grammar_queue[i]
            return_reqs.append(req)
            if req.finished():
                continue

            if isinstance(req.grammar, futures.Future):
                assert req.grammar_key
                req.grammar.cancel()
                self.grammar_backend.set_cache(
                    req.grammar_key,
                    InvalidGrammarObject("Grammar preprocessing timed out"),
                )
            error_msg = failed_reasons.get(
                i, f"Grammar preprocessing timed out: {req.grammar_key=}"
            )
            req.set_finish_with_abort(error_msg)

        # Remove finished requests from grammar_queue
        self.grammar_queue = [
            req
            for i, req in enumerate(self.grammar_queue)
            if i not in synced_ready_req_idxs and i not in synced_failed_req_idxs
        ]
        return return_reqs
