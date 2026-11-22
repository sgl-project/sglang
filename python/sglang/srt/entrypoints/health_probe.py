"""The one-token generation probe behind /health and the frontend stall watchdog."""

import asyncio
import time
import uuid
from typing import Optional, Union

import fastapi

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.runtime_context import get_disagg


def _build_probe_request(
    *, is_generation: bool, rid: str
) -> Union[GenerateReqInput, EmbeddingReqInput]:
    sampling_params = {"max_new_tokens": 1, "temperature": 0.0}
    if not is_generation:
        return EmbeddingReqInput(
            rid=rid, input_ids=[0], sampling_params=sampling_params, log_metrics=False
        )
    probe = GenerateReqInput(
        rid=rid,
        input_ids=[0],
        sampling_params=sampling_params,
        log_metrics=False,
    )
    if get_disagg().disaggregation_mode != DisaggregationMode.NULL.value:
        probe.bootstrap_host = FAKE_BOOTSTRAP_HOST
        probe.bootstrap_room = 0
    return probe


async def run_generation_probe(
    tokenizer_manager,
    *,
    timeout: float,
    request: Optional[fastapi.Request] = None,
    poll_interval: float = 1.0,
) -> bool:
    """Send a one-token request; return whether the scheduler answered in time.

    Any message from the scheduler counts: a busy scheduler acknowledges the
    probe instead of running it.
    """
    # uuid keeps rids unique across tokenizer workers (a bare time.time() can
    # collide, and colliding rids overwrite each other's detokenize state).
    rid = f"{HEALTH_CHECK_RID_PREFIX}_{uuid.uuid4().hex}"
    probe = _build_probe_request(is_generation=tokenizer_manager.is_generation, rid=rid)

    async def gen():
        async for _ in tokenizer_manager.generate_request(probe, request):
            break

    task = asyncio.create_task(gen())
    tic = time.time()
    try:
        while time.time() < tic + timeout:
            await asyncio.sleep(poll_interval)
            if tokenizer_manager.last_receive_tstamp > tic:
                return True
        return False
    finally:
        task.cancel()
        tokenizer_manager.rid_to_state.pop(rid, None)
