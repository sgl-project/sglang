from __future__ import annotations

import asyncio
import copy
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

import fastapi

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.scheduler_input_blocker import input_blocker_guard_region
from sglang.srt.managers.tokenizer_manager_components.request_preparer import (
    RequestPreparer,
)
from sglang.srt.managers.tokenizer_manager_components.request_state import (
    ReqState,
    init_req,
)
from sglang.srt.managers.tokenizer_manager_components.response_emitter import (
    ResponseEmitter,
)
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchRequestDispatcherConfig:
    enable_trace: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchRequestDispatcher:
    request_preparer: RequestPreparer
    get_disaggregation_mode: Callable[[], DisaggregationMode]
    response_emitter: ResponseEmitter
    rid_to_state: Dict[str, ReqState]
    send_to_scheduler: Any
    send_one_request: Callable[..., None]
    send_batch_request: Callable[..., None]
    config: BatchRequestDispatcherConfig

    async def dispatch(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request],
    ) -> Tuple[List[AsyncGenerator], List[str]]:
        batch_size = obj.batch_size

        generators = []
        rids = []
        if getattr(obj, "parallel_sample_num", 1) == 1:
            if self.request_preparer._should_use_batch_tokenization(batch_size, obj):
                tokenized_objs = (
                    await self.request_preparer._batch_tokenize_and_process(
                        batch_size, obj
                    )
                )
                self.send_batch_request(tokenized_objs)

                # Set up generators for each request in the batch
                for i in range(batch_size):
                    tmp_obj = obj[i]
                    state = self.rid_to_state[tmp_obj.rid]
                    if tmp_obj.return_prompt_token_ids:
                        state.prompt_token_ids = list(tokenized_objs[i].input_ids)
                    generators.append(
                        self.response_emitter._wait_one_response(tmp_obj, request)
                    )
                    rids.append(tmp_obj.rid)
            else:
                # Sequential tokenization and processing
                with (
                    input_blocker_guard_region(send_to_scheduler=self.send_to_scheduler)
                    if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
                    else nullcontext()
                ):
                    for i in range(batch_size):
                        tmp_obj = obj[i]
                        tokenized_obj = (
                            await self.request_preparer._tokenize_one_request(tmp_obj)
                        )
                        state = self.rid_to_state[tmp_obj.rid]
                        if tmp_obj.return_prompt_token_ids:
                            state.prompt_token_ids = list(tokenized_obj.input_ids)
                        self.send_one_request(tokenized_obj)
                        generators.append(
                            self.response_emitter._wait_one_response(tmp_obj, request)
                        )
                        rids.append(tmp_obj.rid)
        else:
            # FIXME: When using batch and parallel_sample_num together, the perf is not optimal.
            if batch_size > 128:
                logger.warning(
                    "Sending a single large batch with parallel sampling (n > 1) has not been well optimized. "
                    "The performance might be better if you just duplicate the requests n times or use "
                    "many threads to send them one by one with parallel sampling (n > 1)."
                )

            # Tokenize all requests
            objs = [obj[i] for i in range(batch_size)]
            tokenized_objs = await asyncio.gather(
                *(self.request_preparer._tokenize_one_request(obj) for obj in objs)
            )

            # Cache the common prefix for parallel sampling
            for i in range(batch_size):
                tmp_obj = copy.copy(objs[i])
                tokenized_obj = copy.copy(tokenized_objs[i])
                # Ensure independent mm_items so wrap_shm_features won't mutate the original
                if hasattr(tokenized_obj, "mm_inputs") and tokenized_obj.mm_inputs:
                    tokenized_obj.mm_inputs = copy.copy(tokenized_obj.mm_inputs)
                    tokenized_obj.mm_inputs.mm_items = [
                        copy.copy(item) for item in tokenized_obj.mm_inputs.mm_items
                    ]
                tokenized_obj.rid = tmp_obj.regenerate_rid()
                tokenized_obj.sampling_params = copy.copy(tokenized_obj.sampling_params)
                tokenized_obj.sampling_params.max_new_tokens = 0
                tokenized_obj.stream = False
                init_req(
                    self.rid_to_state,
                    obj=tmp_obj,
                    enable_trace=self.config.enable_trace,
                    disagg_mode=self.get_disaggregation_mode(),
                )
                self.send_one_request(tokenized_obj)
                await self.response_emitter._wait_one_response(
                    tmp_obj, request
                ).__anext__()

            # Expand requests, assign new rids for them, and send them
            for i in range(batch_size):
                for _ in range(obj.parallel_sample_num):
                    tmp_obj = copy.copy(objs[i])
                    tokenized_obj = copy.copy(tokenized_objs[i])
                    # Ensure independent mm_items so wrap_shm_features won't mutate the original
                    if hasattr(tokenized_obj, "mm_inputs") and tokenized_obj.mm_inputs:
                        tokenized_obj.mm_inputs = copy.copy(tokenized_obj.mm_inputs)
                        tokenized_obj.mm_inputs.mm_items = [
                            copy.copy(item) for item in tokenized_obj.mm_inputs.mm_items
                        ]
                    tokenized_obj.rid = tmp_obj.regenerate_rid()
                    init_req(
                        self.rid_to_state,
                        obj=tmp_obj,
                        enable_trace=self.config.enable_trace,
                        disagg_mode=self.get_disaggregation_mode(),
                    )
                    state = self.rid_to_state[tmp_obj.rid]
                    tokenized_obj.time_stats = state.time_stats
                    if tmp_obj.return_prompt_token_ids:
                        state.prompt_token_ids = list(tokenized_objs[i].input_ids)
                    self.send_one_request(tokenized_obj)
                    generators.append(
                        self.response_emitter._wait_one_response(tmp_obj, request)
                    )
                    rids.append(tmp_obj.rid)

                self.rid_to_state[objs[i].rid].time_stats.set_finished_time()
                del self.rid_to_state[objs[i].rid]

        return generators, rids
