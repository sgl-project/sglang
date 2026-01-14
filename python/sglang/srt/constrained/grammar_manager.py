from __future__ import annotations

import logging
import random
from concurrent import futures
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.constrained.base_grammar_backend import (
    INVALID_GRAMMAR_OBJ,
    create_grammar_backend,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

GRAMMAR_TIMEOUT = envs.SGLANG_GRAMMAR_TIMEOUT.get()
GRAMMAR_POLL_INTERVAL = envs.SGLANG_GRAMMAR_POLL_INTERVAL.get()
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
            )
        else:
            self.grammar_backend = None

        self.grammar_sync_group = scheduler.dp_tp_cpu_group
        self.grammar_sync_size = scheduler.dp_tp_group.world_size
        self.grammar_sync_entry = scheduler.dp_tp_group.first_rank
        self.is_grammar_sync_entry = scheduler.dp_tp_group.is_first_rank

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
                if req.grammar:
                    req.grammar.cancel()
                req.set_finish_with_abort("Aborted by AbortReq.")

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
                    if value is INVALID_GRAMMAR_OBJ:  # We hit a cached invalid grammar.
                        error_msg = f"Invalid grammar request with cache hit: {key=}"
                        req.set_finish_with_abort(error_msg)

        if add_to_grammar_queue:
            self.grammar_queue.append(req)

        return add_to_grammar_queue

    def finalize_grammar(self, req: Req):
        self.grammar_backend.set_cache(req.grammar_key, req.grammar.copy())
        if req.grammar is INVALID_GRAMMAR_OBJ:
            error_msg = f"Invalid grammar request: {req.grammar_key=}"
            req.set_finish_with_abort(error_msg)

    def get_ready_grammar_requests(self) -> List[Req]:
        """Move requests whose grammar objects are ready from grammar_queue to waiting_queue."""
        num_ready_reqs = 0
        num_timeout_pt = 0
        sim_timeout_prob = (
            -1
            if self.is_grammar_sync_entry
            else envs.SGLANG_GRAMMAR_SIMULATE_TIMEOUT.get()
        )  # Entry rank never simulates timeout
        timeout_ct = GRAMMAR_TIMEOUT / GRAMMAR_POLL_INTERVAL

        for req in self.grammar_queue:
            try:
                if req.finished():  # It is aborted by AbortReq
                    num_ready_reqs += 1
                    continue

                if sim_timeout_prob > 0 and random.random() < sim_timeout_prob:
                    # Simulate timeout for non-entry ranks in TP sync group for testing
                    logger.warning(
                        f"Simulating grammar timeout on {self.scheduler.tp_rank=}"
                    )
                    raise futures._base.TimeoutError()

                req.grammar = req.grammar.result(timeout=GRAMMAR_POLL_INTERVAL)
                self.finalize_grammar(req)
                num_ready_reqs += 1
                num_timeout_pt = num_ready_reqs
            except futures._base.TimeoutError:
                req.grammar_wait_ct += 1
                # NOTE(lianmin): this timeout is the waiting time of the above line. It is
                # not the waiting time from it enters the grammar queue.
                if sim_timeout_prob > 0 or req.grammar_wait_ct > timeout_ct:
                    num_timeout_pt = num_ready_reqs + 1
                break

        if self.grammar_sync_size > 1:
            # Sync across TP ranks to make sure they have the same number of ready requests
            tensor = torch.tensor(
                [num_ready_reqs, -num_ready_reqs, -num_timeout_pt], dtype=torch.int32
            )
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MIN, group=self.grammar_sync_group
            )
            num_ready_reqs_min = tensor[0].item()
            num_ready_reqs_max = -tensor[1].item()
            num_timeout_pt_max = -tensor[2].item()
        else:
            num_ready_reqs_min = num_ready_reqs
            num_ready_reqs_max = num_ready_reqs
            num_timeout_pt_max = num_timeout_pt

        if envs.SGLANG_GRAMMAR_SIMULATE_TIMEOUT.get() > 0:
            # NOTE: in simulation timeout mode, if only some TP ranks report a timeout,
            # we still treat those requests as ready instead of canceling them.
            for i in range(num_ready_reqs_min, num_ready_reqs_max):
                req = self.grammar_queue[i]
                if req.finished():
                    continue
                if isinstance(req.grammar, futures.Future):
                    req.grammar = req.grammar.result()
                    self.finalize_grammar(req)

            num_ready_reqs_min = num_ready_reqs_max

        # NOTE: in non-simulation mode, cancel any request that times out on a subset of
        # TP ranks because timeouts can diverge across ranks.
        for i in range(num_ready_reqs_min, num_timeout_pt_max):
            req = self.grammar_queue[i]
            if req.finished():
                continue
            if isinstance(req.grammar, futures.Future):
                req.grammar.cancel()
            req.grammar = INVALID_GRAMMAR_OBJ
            self.finalize_grammar(req)

        ready_grammar_reqs = self.grammar_queue[:num_timeout_pt_max]
        self.grammar_queue = self.grammar_queue[num_timeout_pt_max:]

        return ready_grammar_reqs
