# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints import diffusion_generator as dg
from sglang.multimodal_gen.runtime.scheduler_client import SchedulerClient
from sglang.multimodal_gen.runtime.utils.device_errors import (
    DiffusionRuntimeError,
    is_unrecoverable_device_error,
)

_DEVICE_ERROR = "CUDA error: an illegal memory access was encountered"


@pytest.mark.parametrize(
    "message",
    [
        _DEVICE_ERROR,
        "CUDA error: device-side assert triggered",
        "CUDA error: misaligned address",
        "CUDA error: unspecified launch failure",
        "CUDA driver error: device not ready",
        "INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:467",
    ],
)
def test_classifies_device_errors_and_wrapped_causes(message):
    assert is_unrecoverable_device_error(message)
    error = ValueError("request failed")
    error.__cause__ = RuntimeError(message)
    assert is_unrecoverable_device_error(error)
    error.__cause__ = None
    error.__context__ = RuntimeError(message)
    assert is_unrecoverable_device_error(error)


@pytest.mark.parametrize(
    "message",
    ["CUDA out of memory", "CUDA error: out of memory", "invalid input image"],
)
def test_request_errors_are_recoverable(message):
    error = RuntimeError(message)
    error.__context__ = error
    assert not is_unrecoverable_device_error(error)


@pytest.fixture
def generator(monkeypatch):
    generator = object.__new__(dg.DiffGenerator)
    generator.server_args = SimpleNamespace(
        model_path="unused",
        prompt_file_path=None,
        batching_max_size=1,
        warmup_mode="off",
    )
    generator.local_scheduler_process = None
    generator.owns_scheduler_client = True
    generator._runtime_error = None
    params = SimpleNamespace(image_path=None)
    monkeypatch.setattr(
        dg.SamplingParams,
        "from_user_sampling_params_args",
        Mock(return_value=params),
    )

    def replace_params(_params, **kwargs):
        return SimpleNamespace(
            **kwargs, data_type=DataType.IMAGE, _set_output_file_name=lambda: None
        )

    monkeypatch.setattr(dg, "_replace_sampling_params_for_prompt", replace_params)
    monkeypatch.setattr(
        dg,
        "prepare_request",
        lambda **kwargs: SimpleNamespace(
            sampling_params=kwargs["sampling_params"],
            prompt=kwargs["sampling_params"].prompt,
            data_type=kwargs["sampling_params"].data_type,
            trace_ctx=None,
        ),
    )
    monkeypatch.setattr(dg, "expand_request_outputs", lambda req, **kwargs: [req])
    monkeypatch.setattr(dg, "map_request_outputs", lambda requests: [object()])
    monkeypatch.setattr(dg, "trace_req", lambda _: nullcontext())
    monkeypatch.setattr(dg, "log_generation_timer", lambda *_: nullcontext())
    client = SimpleNamespace(forward=Mock(), close=Mock())
    monkeypatch.setattr(dg, "sync_scheduler_client", client)
    yield generator, client
    generator.owns_scheduler_client = False
    generator.local_scheduler_process = None


@pytest.mark.parametrize("rpc_raises", [False, True])
def test_fatal_error_stops_prompt_list_and_rejects_later_calls(generator, rpc_raises):
    generator, client = generator
    if rpc_raises:
        client.forward.side_effect = RuntimeError(_DEVICE_ERROR)
    else:
        client.forward.return_value = SimpleNamespace(error=_DEVICE_ERROR)

    with pytest.raises(DiffusionRuntimeError, match="illegal memory access"):
        generator.generate({"prompt": ["first", "second"]})
    with pytest.raises(DiffusionRuntimeError, match="illegal memory access"):
        generator.generate({"prompt": "third"})

    assert client.forward.call_count == 1
    # Remote mode closes this client's connection without sending ShutdownReq.
    client.close.assert_called_once_with()
    assert not generator.owns_scheduler_client


@pytest.mark.parametrize("message", ["invalid image", "CUDA out of memory"])
@pytest.mark.parametrize("rpc_raises", [False, True])
def test_recoverable_error_keeps_processing_prompts(generator, message, rpc_raises):
    generator, client = generator
    if rpc_raises:
        client.forward.side_effect = RuntimeError(message)
    else:
        client.forward.return_value = SimpleNamespace(error=message)

    assert generator.generate({"prompt": ["first", "second"]}) is None
    assert generator.generate({"prompt": "third"}) is None
    assert client.forward.call_count == 3
    client.close.assert_not_called()


def test_dead_local_worker_fails_before_sending_request(generator):
    generator, client = generator
    process = Mock(name="worker", pid=123, exitcode=1)
    process.is_alive.return_value = False
    generator.local_scheduler_process = [process]

    with pytest.raises(DiffusionRuntimeError, match="exited with code 1"):
        generator.generate({"prompt": "first"})
    with pytest.raises(DiffusionRuntimeError, match="exited with code 1"):
        generator.generate({"prompt": "second"})

    client.forward.assert_not_called()
    process.terminate.assert_not_called()
    process.join.assert_called_once_with(timeout=1)
    client.close.assert_called_once_with()


def test_worker_death_while_waiting_stops_remaining_local_workers(generator):
    generator, client = generator
    process = Mock(pid=123, exitcode=1)
    process.is_alive.side_effect = [True, False, False, False]
    generator.local_scheduler_process = [process]

    def forward(batch, *, check_worker_health):
        check_worker_health()

    client.forward.side_effect = forward
    with pytest.raises(DiffusionRuntimeError, match="exited with code 1"):
        generator.generate({"prompt": "first"})

    assert client.forward.call_count == 1
    process.join.assert_called_once_with(timeout=1)
    client.close.assert_called_once_with()


def test_cleanup_failure_does_not_hide_original_device_error(generator):
    generator, client = generator
    client.forward.return_value = SimpleNamespace(error=_DEVICE_ERROR)
    client.close.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(DiffusionRuntimeError, match="illegal memory access"):
        generator.generate({"prompt": "first"})


def test_client_warmup_uses_worker_health_checks(generator, monkeypatch):
    generator, client = generator
    process = Mock(pid=123, exitcode=1)
    process.is_alive.return_value = False
    generator.local_scheduler_process = [process]
    monkeypatch.setattr(dg, "should_run_explicit_client_warmup", lambda _: True)
    monkeypatch.setattr(
        dg, "run_sync_client_warmup", lambda args, forward: forward(object(), replica=0)
    )
    with pytest.raises(DiffusionRuntimeError, match="exited with code 1"):
        generator._run_client_warmup_if_needed()
    client.forward.assert_not_called()


@pytest.mark.parametrize("local_mode", [False, True])
def test_client_warmup_preserves_replica_routing_and_worker_health(
    generator, monkeypatch, local_mode
):
    generator, _ = generator
    endpoints = ["tcp://replica-0", "tcp://replica-1"]
    generator.server_args.scheduler_endpoints = endpoints
    generator.server_args.scheduler_rpc_timeout = None
    if local_mode:
        process = Mock(pid=123, exitcode=None)
        process.is_alive.return_value = True
        generator.local_scheduler_process = [process]

    scheduler = SchedulerClient()
    scheduler.server_args = generator.server_args
    scheduler.request_logger = Mock()
    scheduler._forward_one = Mock(return_value=dg.OutputBatch())
    monkeypatch.setattr(dg, "sync_scheduler_client", scheduler)
    monkeypatch.setattr(dg, "should_run_explicit_client_warmup", lambda _: True)
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.server_warmup.should_include_warmup_image",
        lambda *args, **kwargs: False,
    )
    warmup_req = dg.Req(prompt="warmup", is_warmup=True)
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.server_warmup.build_client_warmup_reqs",
        lambda *args, **kwargs: [warmup_req],
    )

    generator._run_client_warmup_if_needed()

    calls = scheduler._forward_one.call_args_list
    assert sorted(call.args[0] for call in calls) == endpoints
    expected_kwargs = (
        {"check_worker_health": generator._check_worker_health} if local_mode else {}
    )
    for call in calls:
        assert call.args[1:] == (warmup_req, None)
        assert call.kwargs == expected_kwargs
    if local_mode:
        assert process.is_alive.call_count == len(endpoints)

    # Explicit warmup routing must leave round-robin real traffic at replica 0.
    scheduler._forward_one.reset_mock()
    request = dg.Req(prompt="real")
    generator._send_to_scheduler_and_wait_for_response(request)
    scheduler._forward_one.assert_called_once_with(
        endpoints[0], request, None, **expected_kwargs
    )


def test_fatal_lora_error_rejects_later_generation(generator):
    generator, client = generator
    client.forward.return_value = SimpleNamespace(error=_DEVICE_ERROR)
    with pytest.raises(DiffusionRuntimeError, match="illegal memory access"):
        generator._send_lora_request(object(), "success", "failed")
    with pytest.raises(DiffusionRuntimeError, match="illegal memory access"):
        generator.generate({"prompt": "later"})
    assert client.forward.call_count == 1


@pytest.mark.parametrize(
    "fatal_first, timeout_later", [(False, False), (True, False), (True, True)]
)
def test_fatal_lora_fanout_error_rejects_later_calls(
    generator, fatal_first, timeout_later
):
    """Replica errors and timeouts must not hide another replica's device failure."""
    generator, client = generator
    scheduler = SchedulerClient()
    scheduler.server_args = SimpleNamespace(
        scheduler_endpoints=["tcp://replica-0", "tcp://replica-1"],
        scheduler_rpc_timeout=None,
    )
    scheduler.request_logger = Mock()
    replies = [
        dg.OutputBatch(error="CUDA out of memory"),
        dg.OutputBatch(error=_DEVICE_ERROR),
    ]
    if fatal_first:
        replies.reverse()
    if timeout_later:
        replies[1] = TimeoutError("Scheduler did not respond in time.")
    scheduler._forward_one = Mock(side_effect=replies)
    client.forward.side_effect = scheduler.forward

    with pytest.raises(DiffusionRuntimeError, match="illegal memory access"):
        generator.set_lora("adapter")
    assert generator._runtime_error == _DEVICE_ERROR
    with pytest.raises(DiffusionRuntimeError, match="illegal memory access"):
        generator.set_lora("later")
    with pytest.raises(DiffusionRuntimeError, match="illegal memory access"):
        generator.generate({"prompt": "later"})

    assert scheduler._forward_one.call_count == (1 if fatal_first else 2)
    assert client.forward.call_count == 1
    client.close.assert_called_once_with()


def test_remote_lora_fatal_reply_without_deadline_invalidates_generator(generator):
    generator, client = generator
    scheduler = SchedulerClient()
    scheduler.server_args = SimpleNamespace(
        dp_size=2,
        scheduler_endpoints=["tcp://replica-0", "tcp://replica-1"],
        scheduler_rpc_timeout=None,
    )
    scheduler.request_logger = Mock()

    def reply(endpoint, request, timeout_ms):
        assert timeout_ms is None
        if endpoint == "tcp://replica-1":
            pytest.fail("A fatal reply must not be held behind another replica's RPC")
        return dg.OutputBatch(error=_DEVICE_ERROR)

    scheduler._forward_one = Mock(side_effect=reply)
    client.forward.side_effect = scheduler.forward
    assert generator.local_scheduler_process is None

    with pytest.raises(DiffusionRuntimeError, match="illegal memory access"):
        generator.set_lora("adapter")
    for operation in (
        lambda: generator.set_lora("later"),
        lambda: generator.generate({"prompt": "later"}),
    ):
        with pytest.raises(DiffusionRuntimeError, match="illegal memory access"):
            operation()

    assert generator._runtime_error == _DEVICE_ERROR
    scheduler._forward_one.assert_called_once()
    client.forward.assert_called_once()
    client.close.assert_called_once_with()
    assert not generator.owns_scheduler_client


@pytest.mark.parametrize("mapping_raises", [False, True])
def test_fatal_error_cleans_unprocessed_video_requests(
    generator, monkeypatch, mapping_raises
):
    generator, client = generator
    prepared_params = []

    def video_params(_params, **kwargs):
        params = SimpleNamespace(
            **kwargs,
            data_type=DataType.VIDEO,
            _set_output_file_name=lambda: None,
            prepare_video_request_for_queue=Mock(),
            cleanup_video_request=Mock(),
        )
        prepared_params.append(params)
        return params

    monkeypatch.setattr(dg, "_replace_sampling_params_for_prompt", video_params)
    if mapping_raises:
        monkeypatch.setattr(
            dg, "map_request_outputs", Mock(side_effect=RuntimeError(_DEVICE_ERROR))
        )
    else:
        client.forward.return_value = SimpleNamespace(error=_DEVICE_ERROR)

    with pytest.raises(DiffusionRuntimeError, match="illegal memory access"):
        generator.generate({"prompt": ["first", "second"]})

    assert len(prepared_params) == 2
    for params in prepared_params:
        params.prepare_video_request_for_queue.assert_called_once()
        params.cleanup_video_request.assert_called_once()
