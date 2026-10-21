# SPDX-License-Identifier: Apache-2.0
"""CPU service/HTTP/IPC tests using real radix pools and committed requests."""

import asyncio
from array import array
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest
import test_kv_control_cache as cache_fixtures
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    KVControlReqInput,
    KVControlReqOutput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.kv_control.contracts import Status
from sglang.srt.managers.kv_control.http import install_routes
from sglang.srt.managers.kv_control.service import (
    KVControlService,
    ServiceConfig,
    dispatch,
    runtime_config,
)
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

insert, match_len, target = (
    cache_fixtures.insert,
    cache_fixtures.match_len,
    cache_fixtures.target,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.fixture
def owner():
    fixture = cache_fixtures.TestKVControlCache()
    fixture.setUp()
    try:
        cache, _ = fixture.make(session=False)
        cache.reset()
        service = KVControlService(
            cache,
            ServiceConfig(namespace="ns", engine_id="engine"),
            clock=fixture.clock,
        )
        yield cache, service
    finally:
        fixture.doCleanups()


def envelope(*, attached=False, message="message", opaque=False, invalid=False):
    selection = "request_cached" if attached else msgspec.to_builtins(target())
    actions = [
        KvHintAction(
            action_id="keep",
            action_type="sglang.set_priority",
            action_version="1.0",
            payload={
                "target": selection,
                "location": {"tier": "gpu"},
                "claim_id": "keep",
                "revision": 1,
                "value": 90,
                "ttl_seconds": 10,
            },
        )
    ]
    if invalid:
        actions.append(
            KvHintAction(
                action_id="bad",
                action_type="sglang.set_priority",
                action_version="1.0",
                payload={"bad": "value"},
            )
        )
    if opaque:
        actions.append(
            KvHintAction(
                action_id="other",
                action_type="backend.opaque",
                action_version="8",
                payload={"private": [1, 2]},
            )
        )
    return KvHintsEnvelope(protocol_version="0.1", message_id=message, actions=actions)


def request(rid="same", *, salt=None):
    return Req(
        rid=rid,
        origin_input_text=None,
        origin_input_ids=array("q", [1, 2, 3, 4]),
        sampling_params=SamplingParams(max_new_tokens=1),
        vocab_size=128,
        cache_salt=salt,
    )


def command(service, kind, **fields):
    return dispatch(
        service,
        {
            "kind": kind,
            "context": msgspec.to_builtins(service.control.context("router")),
            **fields,
        },
    )


def test_http_executes_real_priority_and_replay_and_epoch_checks(owner):
    cache, service = owner
    insert(cache, [1, 2, 3, 4])
    insert(cache, [5, 6, 7, 8])

    class Manager:
        async def kv_control(self, command):
            # Exercise the exact IPC schema/serialization boundary in process.
            wire = msgpack_decode(
                msgpack_encode(KVControlReqInput(request_id="rpc", command=command))
            )
            reply = KVControlReqOutput(
                request_id=wire.request_id, result=dispatch(service, wire.command)
            )
            return msgpack_decode(msgpack_encode(reply)).result

    app = FastAPI()
    install_routes(app, lambda: Manager())

    async def call():
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://owner"
        ) as client:
            cap = (await client.get("/v1/kv/capabilities")).json()["data"]
            assert cap["engine_id"] == "engine" and cap["topology"] == "single_owner"
            ctx = msgspec.to_builtins(service.control.context("router"))
            body = {
                "kind": "submit",
                "context": ctx,
                "envelope": msgspec.to_builtins(envelope()),
            }
            result = await client.post("/v1/kv/control", json=body)
            assert result.status_code == 200
            assert (
                result.json()["data"]["results"][0]["update"]["status"]
                == Status.COMPLETED.value
            )
            replayed = await client.post("/v1/kv/control", json=body)
            assert replayed.json()["data"]["replayed"]
            stale = await client.post(
                "/v1/kv/control",
                json={"kind": "capacity", "context": dict(ctx, engine_epoch="stale")},
            )
            assert stale.status_code == 409

    asyncio.run(call())
    cache.evict(EvictParams(num_tokens=4))
    assert match_len(cache, [1, 2, 3, 4]) == 4
    assert match_len(cache, [5, 6, 7, 8]) == 0


def test_attached_fanout_lookup_does_not_mutate_shared_envelope(owner):
    cache, service = owner
    insert(cache, [1, 2, 3, 4])
    hints = envelope(attached=True, opaque=True, invalid=True)
    original = msgspec.json.encode(hints)
    reqs = [request(), request()]
    for req in reqs:
        service.attach(req, hints)
        assert len(req.kv_hints.actions) == 1
        assert req.kv_hints.actions[0] is hints.actions[-1]
    assert msgspec.json.encode(hints) == original
    lookup = dispatch(
        service,
        {
            "kind": "lookup",
            "context": msgspec.to_builtins(service.control.context("request")),
            "message_id": "message",
        },
    )
    refs = [message["results"][0]["ref"] for message in lookup["data"]]
    assert (
        len(refs) == 2
        and refs[0]["context"]["binding_id"] != refs[1]["context"]["binding_id"]
    )
    assert service.has_pending
    service.finish_request(reqs[0], aborted=True)
    reqs[1].finished_reason = FINISH_LENGTH(1)
    reqs[1].refresh_fill_ids()
    service.finish_request(reqs[1], checkpointed=True, committed_len=4)
    results = [
        service.control.registry.query(
            msgspec.convert(
                r["context"], type=type(service.control.context("request"))
            ),
            "message",
        ).results
        for r in refs
    ]
    assert results[0][0].status == Status.CANCELLED
    assert results[1][0].status == Status.COMPLETED
    assert all(batch[1].status == Status.REJECTED for batch in results)
    assert not service.has_pending


def test_real_release_binds_after_checkpoint_before_unlock_and_keeps_salt(owner):
    cache, service = owner
    req = request(salt="tenant")
    req.output_ids = array("q", [5])
    req.finished_reason = FINISH_LENGTH(1)
    cache.req_to_token_pool.alloc([req])
    slots = cache.token_to_kv_pool_allocator.alloc(5)
    cache.req_to_token_pool.write((req.kv.req_pool_idx, slice(0, 5)), slots)
    req.kv.kv_committed_len = req.kv.kv_allocated_len = 5
    service.attach(req, envelope(attached=True))
    context = req.kv_control_context
    observed = []
    start = service.control.executor.start

    def inspect(work):
        observed.append(
            (
                req.kv.holds_kv,
                req.lock is not None,
                tuple(work.target.branches[0].content.token_ids),
            )
        )
        return start(work)

    with patch.object(service.control.executor, "start", side_effect=inspect):
        release_kv_cache(req, cache, checkpoint=True)
    assert observed == [(True, True, (1, 2, 3, 4, 5))]
    result = service.control.registry.query(context, "message").results[0]
    assert result.status == Status.COMPLETED
    assert not req.kv.holds_kv and req.lock is None
    assert match_len(cache, [1, 2, 3, 4, 5], salt="tenant") == 5
    assert match_len(cache, [1, 2, 3, 4, 5]) == 0
    assert cache.token_to_kv_pool_allocator.available_size() == 59
    cache.sanity_check()


def test_checkpoint_and_preemption_do_not_complete_attached_work(owner):
    _, service = owner
    req = request()
    service.attach(req, envelope(attached=True))
    context = req.kv_control_context
    service.finish_request(req, checkpointed=True, committed_len=4)
    assert req.kv_control_context == context and service.has_pending
    service.finish_request(req, aborted=True)
    assert (
        service.control.registry.query(context, "message").results[0].status
        == Status.CANCELLED
    )
    assert not service.has_pending


def test_scheduler_factory_session_reconstruction_and_unallocated_abort(owner):
    from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.session.session_controller import Session

    cache, service = owner
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.kv_control_service, scheduler.tree_cache = service, cache
    for streaming in (False, True):
        session = Session(
            capacity_of_str_len=0, session_id=f"s{streaming}", streaming=streaming
        )
        recv = TokenizedGenerateReqInput(
            rid="same",
            input_text="hi",
            input_ids=array("q", [1, 2, 3, 4]),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(max_new_tokens=1),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            session_params=SessionParams(id=session.session_id),
            kv_hints=envelope(attached=True, opaque=True),
        )
        req = session.create_req(
            recv, None, 128, req_factory=partial(scheduler._create_generation_req, recv)
        )
        assert len(req.kv_hints.actions) == 1
        context = req.kv_control_context
        status = service.control.registry.query(context, "message").results[0].status
        assert status == (Status.UNSUPPORTED if streaming else Status.QUEUED)
        assert not req.kv.holds_kv
        scheduler._release_aborted_request(req)
        assert not service.has_pending
        if not streaming:
            assert (
                service.control.registry.query(context, "message").results[0].status
                == Status.CANCELLED
            )
    scheduler.kv_control_service = None
    req = scheduler._create_generation_req(
        SimpleNamespace(kv_hints=recv.kv_hints, mm_inputs=None),
        rid="off",
        origin_input_text=None,
        origin_input_ids=array("q", [1]),
        sampling_params=SamplingParams(),
        vocab_size=128,
    )
    assert req.kv_hints is recv.kv_hints


def test_flush_rotates_epoch_and_old_context_cannot_modify_new_copies(owner):
    cache, service = owner
    ctx = msgspec.to_builtins(service.control.context("router"))
    cache.reset()
    service.reset_after_cache_flush()
    assert command(service, "capacity")["success"]
    reply = dispatch(
        service,
        {"kind": "submit", "context": ctx, "envelope": msgspec.to_builtins(envelope())},
    )
    assert reply["error"]["code"] == "stale_epoch"


@pytest.mark.parametrize(
    "fields",
    [
        {"tp_size": 2},
        {"pp_size": 2},
        {"dp_size": 2},
        {"attn_cp_size": 2},
        {"disable_radix_cache": True},
    ],
)
def test_runtime_profile_rejects_unsupported_config(fields):
    with get_context().override_server_args(
        kv_control_config={"namespace": "ns"}, **fields
    ):
        with pytest.raises(ValueError):
            runtime_config()


def test_disabled_strict_schema_and_openai_native_roundtrip():
    assert dispatch(None, {"kind": "capabilities"})["data"]["enabled"] is False
    assert dispatch(None, {"kind": "capacity"})["success"] is False
    assert (
        dispatch(None, {"kind": "capabilities", "surprise": True})["success"] is False
    )
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        CompletionRequest,
    )

    hints = envelope(attached=True)
    for cls, prompt in [
        (CompletionRequest, {"prompt": [1, 2]}),
        (ChatCompletionRequest, {"messages": [{"role": "user", "content": "hi"}]}),
    ]:
        wire = cls.model_validate(
            dict(model="dummy", kv_hints=msgspec.to_builtins(hints), **prompt)
        )
        assert wire.kv_hints == hints
    batch = GenerateReqInput(
        input_ids=[[1, 2], [3, 4]], sampling_params={"n": 2}, kv_hints=hints
    )
    batch.normalize_batch_and_arguments()
    assert all(batch[i].kv_hints == hints for i in range(batch.batch_size))
