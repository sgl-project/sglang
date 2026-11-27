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
from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.distributed.pp_sync_window import PPSyncSendWindow
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
    move,
)
from sglang.srt.runtime_context import get_parallel, get_serving
from sglang.srt.sampling.sampling_params import (
    get_request_reasoning_end_token_ids,
    get_request_reasoning_in_grammar,
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
        self.grammar_queue = IndexedReqList(ListKind.GRAMMAR, abort=self._abort)
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
        # With attention DP, dp_tp_group excludes CP peers of the same batch.
        # Reuse their existing CPU group to keep grammar admission in lockstep.
        self.grammar_sync_cp_group = (
            scheduler.attn_cp_group
            if get_parallel().attn_dp_enabled and get_parallel().attn_cp_size > 1
            else None
        )
        self.pp_rank = get_parallel().pp_rank
        self.pp_size = get_parallel().pp_size
        self.pp_group = scheduler.pp_group
        # The PP loop runs pp_size + pp_async_batch_depth rounds deep; the
        # forwards must stay un-waited that long or they lock the ranks together.
        self.pp_sync_send_window = PPSyncSendWindow(
            self.pp_size + get_parallel().pp_async_batch_depth
        )

        self.SGLANG_GRAMMAR_POLL_INTERVAL = envs.SGLANG_GRAMMAR_POLL_INTERVAL.get()
        self.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = (
            envs.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS.get()
        )
        # Cache local Future outcomes across poll rounds so each completed
        # Future is resolved at most once while waiting for slower peers.
        self._resolved_futures: dict[
            futures.Future, tuple[object, str | None, bool]
        ] = {}

    def __len__(self):
        return len(self.grammar_queue)

    def clear(self):
        self._resolved_futures.clear()
        if self.grammar_backend:
            self.grammar_backend.reset()

    def _active_sync_groups(self) -> list[tuple[object, int]]:
        groups: list[tuple[object, int]] = []
        if self.grammar_sync_size > 1:
            groups.append((self.grammar_sync_group, self.grammar_sync_size))
        if (
            self.grammar_sync_cp_group is not None
            and self.grammar_sync_cp_group.world_size > 1
        ):
            groups.append(
                (
                    self.grammar_sync_cp_group.cpu_group,
                    self.grammar_sync_cp_group.world_size,
                )
            )
        return groups

    def _resolve_future(self, req: Req) -> tuple[object, str | None, bool]:
        assert isinstance(req.grammar, futures.Future) and req.grammar_key
        cached = self._resolved_futures.get(req.grammar)
        if cached is not None:
            return cached
        try:
            grammar_obj = req.grammar.result()
        except Exception as e:
            logger.error(
                f"Grammar compilation raised an exception: {e}, "
                f"grammar_key={req.grammar_key}"
            )
            grammar_obj = InvalidGrammarObject(f"Grammar compilation failed: {e}")
            outcome = (grammar_obj, grammar_obj.error_message, False)
        else:
            if isinstance(grammar_obj, InvalidGrammarObject):
                outcome = (grammar_obj, grammar_obj.error_message, True)
            else:
                outcome = (grammar_obj, None, False)
        self._resolved_futures[req.grammar] = outcome
        return outcome

    def _sync_compile_outcomes(
        self,
        sync_groups: list[tuple[object, int]],
        compile_errors: dict[int, str],
        cacheable_invalid: set[int],
    ) -> tuple[dict[int, str], set[int]]:
        synced_errors = compile_errors
        synced_cacheable = cacheable_invalid
        for sync_group, sync_size in sync_groups:
            all_gather_output = [None] * sync_size
            torch.distributed.all_gather_object(
                all_gather_output,
                (synced_errors, synced_cacheable),
                group=sync_group,
            )
            merged_errors: dict[int, str] = {}
            for x in all_gather_output:
                for idx, msg in x[0].items():
                    merged_errors.setdefault(idx, msg)
            synced_errors = merged_errors
            synced_cacheable = set.intersection(*[x[1] for x in all_gather_output])
        return synced_errors, synced_cacheable

    def has_waiting_grammars(self) -> bool:
        return len(self.grammar_queue) > 0

    def _pp_sync_ready_failed(
        self,
        ready_req_idxs: set[int],
        failed_req_idxs: set[int],
    ) -> tuple[set[int], set[int]]:
        """
        Synchronize ready/failed grammar request indexes across the PP pipeline.

        PP0 provides the data. Each later PP rank receives it from the previous
        rank and asynchronously forwards it to the next rank. The forwards stay
        un-waited for the window's depth so this rank's rounds are not tied to
        the next rank's.
        """
        if self.pp_size <= 1 or self.pp_group is None:
            return ready_req_idxs, failed_req_idxs

        self.pp_sync_send_window.begin_round()
        data = (ready_req_idxs, failed_req_idxs)
        if self.pp_rank > 0:
            data = self.pp_group.recv_object(
                src=self.pp_rank - 1,
                tag=P2PTag.GRAMMAR_PP_SYNC,
            )
        if self.pp_rank + 1 < self.pp_size:
            self.pp_sync_send_window.add(
                self.pp_group.send_object(
                    data,
                    dst=self.pp_rank + 1,
                    async_send=True,
                    tag=P2PTag.GRAMMAR_PP_SYNC,
                )
            )
        return data

    def _abort(self, recv_req: AbortReq) -> None:
        # An aborted request still runs one prefill pass, which
        # `set_finish_with_abort` cuts to one token.
        for req in self.grammar_queue:
            if recv_req.matches(req.rid):
                logger.debug(f"Abort grammar queue request. {req.rid=}")
                if isinstance(req.grammar, futures.Future) and req.grammar:
                    self._resolved_futures.pop(req.grammar, None)
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
        if get_request_reasoning_in_grammar(req.sampling_params.custom_params):
            req.grammar.set_reasoning_in_grammar()
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
                    key_type = (
                        "full_assistant_ebnf"
                        if req.sampling_params.ebnf_full_assistant
                        else "ebnf"
                    )
                    key = (key_type, req.sampling_params.ebnf)
                elif req.sampling_params.structural_tag is not None:
                    key = ("structural_tag", req.sampling_params.structural_tag)
                key = self.grammar_backend.get_request_grammar_key(
                    key,
                    input_ids=req.origin_input_ids,
                    thinking_budget=self._get_request_thinking_budget(req),
                )

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
                        self._apply_request_reasoning_config(req)
        elif self._enable_strict_thinking:
            grammar_obj = self.grammar_backend.init_strict_reasoning_grammar(
                req.require_reasoning
            )
            if grammar_obj is not None:
                req.grammar = grammar_obj
                self._apply_request_reasoning_config(req)

        if add_to_grammar_queue:
            move([req], self.grammar_queue)

        return add_to_grammar_queue

    def get_ready_grammar_requests(self) -> List[Req]:
        """
        Return the requests whose grammar is ready, then those that timed out.
        They stay in grammar_queue for the caller to move into their queue.

        For PP0, DP/TP group rank i returns two sets ready_reqs_i,
        failed_reqs_i. ready_reqs_all = all_gather(ready_reqs_i) within
        PP0's DP/TP group. failed_reqs_all = all_gather(failed_reqs_i)
        within PP0's DP/TP group.

        ready_reqs = intersect(ready_reqs_all)
        failed_reqs = union(failed_reqs_all)

        PP0 then propagates the synced result to later PP ranks. Later PP
        ranks receive and apply the propagated ready/failed decision.
        """
        assert self.grammar_backend
        # Indices into this snapshot are synced across ranks.
        grammar_queue = list(self.grammar_queue)
        ready_req_idxs: set[int] = set()
        failed_req_idxs: set[int] = set()
        sync_groups = self._active_sync_groups()
        compile_error_msgs: dict[int, str] = {}
        cacheable_invalid_idxs: set[int] = set()

        if self.pp_rank == 0:
            # Poll for ready requests
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < self.SGLANG_GRAMMAR_POLL_INTERVAL:
                for i, req in enumerate(grammar_queue):
                    if i in ready_req_idxs:
                        continue

                    if (
                        req.finished() or req.grammar is None
                    ):  # It is aborted by AbortReq
                        ready_req_idxs.add(i)
                        continue

                    assert isinstance(req.grammar, futures.Future), f"{req=}"
                    if req.grammar.done():
                        ready_req_idxs.add(i)
                        if sync_groups:
                            _, err_msg, is_cacheable = self._resolve_future(req)
                            if err_msg is not None:
                                compile_error_msgs[i] = err_msg
                                if is_cacheable:
                                    cacheable_invalid_idxs.add(i)
                                else:
                                    failed_req_idxs.add(i)

                if len(ready_req_idxs) == len(grammar_queue):
                    break

                # Sleep a bit to avoid busy waiting
                time.sleep(self.SGLANG_GRAMMAR_POLL_INTERVAL / 10)

            # Check failed requests
            for i, req in enumerate(grammar_queue):
                if i not in ready_req_idxs:
                    # grammar_wait_ct is only updated on PP0; later PP ranks
                    # receive PP0's ready/failed decision through PP sync.
                    grammar_queue[i].grammar_wait_ct += 1
                    if (
                        grammar_queue[i].grammar_wait_ct
                        >= self.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS
                    ):
                        # Timeout after max poll iterations
                        # The actual waiting time is SGLANG_GRAMMAR_MAX_POLL_ITERATIONS * max(SGLANG_GRAMMAR_POLL_INTERVAL, GPU_forward_batch_latency)
                        failed_req_idxs.add(i)

            # Sync along TP, then CP: their Cartesian product is one DP batch.
            synced_ready_req_idxs = ready_req_idxs
            synced_failed_req_idxs = failed_req_idxs
            synced_compile_errors = compile_error_msgs
            synced_cacheable_invalid = cacheable_invalid_idxs
            for sync_group, sync_size in sync_groups:
                all_gather_output = [None] * sync_size
                torch.distributed.all_gather_object(
                    all_gather_output,
                    (
                        synced_ready_req_idxs,
                        synced_failed_req_idxs,
                        synced_compile_errors,
                        synced_cacheable_invalid,
                    ),
                    group=sync_group,
                )
                synced_ready_req_idxs = set.intersection(
                    *[x[0] for x in all_gather_output]
                )
                synced_failed_req_idxs = set.union(*[x[1] for x in all_gather_output])
                merged_compile_errors: dict[int, str] = {}
                for x in all_gather_output:
                    if len(x) > 2:
                        for idx, msg in x[2].items():
                            merged_compile_errors.setdefault(idx, msg)
                synced_compile_errors = merged_compile_errors
                if all(len(x) > 3 for x in all_gather_output):
                    synced_cacheable_invalid = set.intersection(
                        *[x[3] for x in all_gather_output]
                    )
                else:
                    synced_cacheable_invalid = set()
                synced_ready_req_idxs -= synced_failed_req_idxs

            # Route rank-divergent invalid outcomes through the failed path so
            # every rank (including PP peers) aborts in lockstep without
            # polluting the grammar cache, while staggered cacheable invalid
            # grammars wait for slower peers to finish.
            if sync_groups and synced_compile_errors:
                divergent_errors = (
                    set(synced_compile_errors) & synced_ready_req_idxs
                ) - synced_cacheable_invalid
                synced_failed_req_idxs |= divergent_errors
                synced_ready_req_idxs -= synced_failed_req_idxs
        else:
            synced_ready_req_idxs = ready_req_idxs
            synced_failed_req_idxs = failed_req_idxs
            synced_compile_errors = {}

        # Propagate PP0's grammar queue decision to later PP ranks.
        (
            synced_ready_req_idxs,
            synced_failed_req_idxs,
        ) = self._pp_sync_ready_failed(
            synced_ready_req_idxs,
            synced_failed_req_idxs,
        )

        # On later PP stages, peers resolve their own futures after PP0 admits
        # the request. Agree on compilation outcomes across TP/CP before caching
        # or starting prefill so a rank-local exception cannot split the batch.
        if self.pp_rank > 0 and sync_groups and synced_ready_req_idxs:
            pp_compile_errors: dict[int, str] = {}
            pp_cacheable_invalid: set[int] = set()
            for i in synced_ready_req_idxs:
                req = grammar_queue[i]
                if req.finished() or req.grammar is None:
                    continue
                _, err_msg, is_cacheable = self._resolve_future(req)
                if err_msg is not None:
                    pp_compile_errors[i] = err_msg
                    if is_cacheable:
                        pp_cacheable_invalid.add(i)
            synced_compile_errors, synced_cacheable_invalid = (
                self._sync_compile_outcomes(
                    sync_groups, pp_compile_errors, pp_cacheable_invalid
                )
            )
            if synced_compile_errors:
                non_cacheable_errors = (
                    set(synced_compile_errors) - synced_cacheable_invalid
                )
                synced_failed_req_idxs = synced_failed_req_idxs | non_cacheable_errors
                synced_ready_req_idxs = synced_ready_req_idxs - synced_failed_req_idxs

        # Return ready requests
        return_reqs: List[Req] = []
        for i in sorted(synced_ready_req_idxs):
            req = grammar_queue[i]
            return_reqs.append(req)
            if req.finished() or req.grammar is None:  # It is aborted by AbortReq
                if isinstance(req.grammar, futures.Future):
                    self._resolved_futures.pop(req.grammar, None)
                continue

            assert isinstance(req.grammar, futures.Future) and req.grammar_key
            resolved = self._resolved_futures.pop(req.grammar, None)
            if resolved is not None:
                req.grammar = resolved[0]
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
            self._apply_request_reasoning_config(req)
            if isinstance(req.grammar, InvalidGrammarObject):
                error_msg = f"Failed to compile {req.grammar_key[0]} grammar: {req.grammar.error_message}"
                req.set_finish_with_abort(error_msg)

        # Return failed requests
        for i in sorted(synced_failed_req_idxs):
            req = grammar_queue[i]
            return_reqs.append(req)
            if req.finished() or req.grammar is None:  # It is aborted by AbortReq
                if isinstance(req.grammar, futures.Future):
                    self._resolved_futures.pop(req.grammar, None)
                continue

            assert isinstance(req.grammar, futures.Future) and req.grammar_key
            resolved = self._resolved_futures.pop(req.grammar, None)
            req.grammar.cancel()
            # Not cached: timeouts and rank-local exceptions do not prove the
            # grammar is deterministically invalid across peers.
            if i in synced_compile_errors:
                local_obj = resolved[0] if resolved is not None else None
                req.grammar = (
                    local_obj
                    if isinstance(local_obj, InvalidGrammarObject)
                    else InvalidGrammarObject(synced_compile_errors[i])
                )
                error_msg = f"Failed to compile {req.grammar_key[0]} grammar: {req.grammar.error_message}"
            else:
                error_msg = f"Grammar preprocessing timed out: {req.grammar_key=}"
            req.set_finish_with_abort(error_msg)

        return return_reqs
