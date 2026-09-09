"""Actual SGLang dispatcher/runner, real CUDA Graphs, fake external EP library.

These tests verify host lifecycle and GPU buffer ownership on SM89. They do not
verify NCCL EP communication, its native layout, or cross-rank correctness.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Real CUDA execution is required"
)


def test_eager_dispatcher_restores_tokens_and_closes_each_external_handle():
    from nccl_ep_test.dispatcher import forward_layer
    from nccl_ep_test.fake_ep import dispatcher_environment
    from nccl_ep_test.oracle import (
        RoutingBatch,
        assert_combine_matches,
        assert_dispatch_matches,
    )

    with dispatcher_environment(capacity=16) as environment:
        dispatcher = environment.dispatcher(layer_id=0)
        for reverse in (False, True, False):
            tokens = torch.tensor([1.0, 2.0, 4.0, 8.0]).repeat(8, 512).bfloat16()
            ids = torch.tensor([[1, 0] if reverse else [0, 1]] * 8)
            weights = torch.tensor([[0.25, 0.75]] * 8)
            batch = RoutingBatch((tokens,), (ids,), (weights,), 2)
            actual = forward_layer(
                dispatcher, tokens.cuda(), ids.cuda(), weights.cuda(), 0
            )
            assert_dispatch_matches(batch, 0, actual[0], actual[1])
            assert_combine_matches(batch, 0, actual[2])
        assert environment.events.count("handle_create") == 3
        assert environment.events.count("handle_destroy") == 3
    assert environment.events.count("group_create") == 1
    assert environment.events.count("group_destroy") == 1


def graph_backend(coordinator, capacity):
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    return FullCudaGraphBackend(
        SimpleNamespace(
            device_module=torch.cuda,
            model_runner=SimpleNamespace(tp_group=coordinator),
        ),
        nccl_ep_capacity=capacity,
    )


def test_graph_replays_changed_data_without_python_handle_updates():
    from nccl_ep_test.dispatcher import forward_layer
    from nccl_ep_test.fake_ep import dispatcher_environment
    from nccl_ep_test.oracle import RoutingBatch, assert_combine_matches

    from sglang.srt.model_executor.runner.shape_key import ShapeKey

    with dispatcher_environment(capacity=16) as environment:
        dispatcher = environment.dispatcher(layer_id=0)
        backend = graph_backend(environment.coordinator, 16)
        x = torch.ones(16, 2048, dtype=torch.bfloat16, device="cuda")
        ids = torch.tensor([[0, 1]] * 16, device="cuda")
        weights = torch.tensor([[0.25, 0.75]] * 16, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(stream), backend.capture_session(stream):
                for bucket in (16, 8):
                    backend.capture_one(
                        ShapeKey(bucket),
                        lambda bucket=bucket: forward_layer(
                            dispatcher, x[:bucket], ids[:bucket], weights[:bucket], 0
                        ),
                    )
            torch.cuda.current_stream().wait_stream(stream)
            updates = len(environment.updates)
            for bucket, value, route, weight in (
                (8, 1, [0, 1], [0.25, 0.75]),
                (16, 2, [1, 0], [0.75, 0.25]),
                (8, 4, [0, 1], [0.75, 0.25]),
                (16, 1, [0, 1], [0.25, 0.75]),
            ):
                tokens = torch.full((bucket, 2048), value, dtype=torch.bfloat16)
                routes = torch.tensor([route] * bucket)
                factors = torch.tensor([weight] * bucket)
                batch = RoutingBatch((tokens,), (routes,), (factors,), 2)
                with backend.replay_session():
                    x[:bucket].copy_(tokens)
                    ids[:bucket].copy_(routes)
                    weights[:bucket].copy_(factors)
                    actual = backend.replay(ShapeKey(bucket), None)
                assert_combine_matches(batch, 0, actual[2])
            assert len(environment.updates) == updates
            assert environment.events.count("handle_create") == 1
            assert environment.events.count("handle_destroy") == 0
        finally:
            backend.cleanup()


def test_graph_output_survives_next_layer_shared_scratch():
    from nccl_ep_test.dispatcher import forward_layer
    from nccl_ep_test.fake_ep import dispatcher_environment

    from sglang.srt.model_executor.runner.shape_key import ShapeKey

    with dispatcher_environment(capacity=8) as environment:
        first = environment.dispatcher(layer_id=0)
        second = environment.dispatcher(layer_id=1)
        backend = graph_backend(environment.coordinator, 8)
        x = torch.ones(8, 2048, dtype=torch.bfloat16, device="cuda")
        ids = torch.tensor([[0, 1]] * 8, device="cuda")
        weights = torch.tensor([[0.25, 0.75]] * 8, device="cuda")

        def forward():
            earlier = forward_layer(first, x, ids, weights, 0)[2]
            later = forward_layer(second, x * 2, ids, weights, 0)[2]
            return earlier, later

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(stream), backend.capture_session(stream):
                backend.capture_one(ShapeKey(8), forward)
            torch.cuda.current_stream().wait_stream(stream)
            with backend.replay_session():
                earlier, later = backend.replay(ShapeKey(8), None)
            torch.testing.assert_close(
                earlier, torch.full_like(x, 1.75), rtol=0, atol=0
            )
            torch.testing.assert_close(later, torch.full_like(x, 3.5), rtol=0, atol=0)
            assert environment.events.count("handle_create") == 1
        finally:
            backend.cleanup()


def synthetic_ep_runner(
    environment,
    *,
    buckets=(8, 16),
    forward_hook=None,
    before_forward=None,
    share_inputs=False,
):
    from nccl_ep_test.dispatcher import forward_layer
    from nccl_ep_test.runner_inputs import SyntheticDecodeRunner

    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    dispatcher = environment.dispatcher(layer_id=0)
    ids = torch.tensor([[0, 1]] * max(buckets), device="cuda")
    weights = torch.tensor([[0.25, 0.75]] * max(buckets), device="cuda")

    def forward(batch):
        if before_forward is not None:
            before_forward()
        rows = batch.batch_size
        x = batch.input_ids[:, None].to(torch.bfloat16).expand(-1, 2048)
        routes = torch.where(
            torch.arange(rows, device="cuda")[:, None] < batch.num_token_non_padded,
            ids[:rows],
            -1,
        )
        output = forward_layer(dispatcher, x, routes, weights[:rows], 0)[2]
        if forward_hook is not None:
            forward_hook()
        return LogitsProcessorOutput(next_token_logits=output[:, :1])

    return SyntheticDecodeRunner(
        forward,
        environment.coordinator,
        buckets=buckets,
        share_inputs=share_inputs,
        backend_factory=lambda runner: FullCudaGraphBackend(
            runner, nccl_ep_capacity=max(buckets)
        ),
    )


def test_stream_switch_fences_input_writes_and_output_consumers():
    from nccl_ep_test.fake_ep import dispatcher_environment
    from nccl_ep_test.runner_inputs import input_batch

    with dispatcher_environment(capacity=16) as environment:
        runner = synthetic_ep_runner(environment)
        first_stream, second_stream = torch.cuda.Stream(), torch.cuda.Stream()
        try:
            for second_rows in (5, 9):
                first_batch = input_batch([1] * 5)
                second_batch = input_batch([2] * second_rows)
                first_stream.wait_stream(torch.cuda.current_stream())
                second_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(first_stream):
                    first = runner.execute(first_batch).next_token_logits
                    torch.cuda._sleep(20_000_000)
                    # This consumer is submitted after replay_session closes.
                    # Waiting only for an event recorded inside that session
                    # would still allow the next replay to overwrite its input.
                    consumed = first.clone()
                with torch.cuda.stream(second_stream):
                    second = runner.execute(second_batch).next_token_logits.clone()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    consumed, torch.full_like(consumed, 1.75), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    second, torch.full_like(second, 3.5), rtol=0, atol=0
                )
        finally:
            runner.backend.cleanup()


def test_unprotected_replay_input_updates_are_rejected():
    from nccl_ep_test.fake_ep import dispatcher_environment
    from nccl_ep_test.runner_inputs import input_batch

    from sglang.srt.model_executor.runner.shape_key import ShapeKey

    with dispatcher_environment(capacity=16) as environment:
        runner = synthetic_ep_runner(environment)
        try:
            with pytest.raises(RuntimeError, match="replay_session"):
                runner.load_batch(input_batch([1] * 5))
            with pytest.raises(RuntimeError, match="replay_session"):
                runner.backend.replay(ShapeKey(8), None)
            # Rejection leaves the ordinary execute path usable.
            result = runner.execute(input_batch([2] * 5)).next_token_logits
            torch.testing.assert_close(
                result, torch.full_like(result, 3.5), rtol=0, atol=0
            )
        finally:
            runner.backend.cleanup()


def test_recapture_closes_executables_before_persistent_resources():
    from nccl_ep_test.ep_audit import EpAudit
    from nccl_ep_test.fake_ep import dispatcher_environment
    from nccl_ep_test.runner_inputs import input_batch

    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    with dispatcher_environment(capacity=16) as environment, EpAudit() as audit:
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpBuffer

        runner = synthetic_ep_runner(environment)
        try:
            for mode in (CaptureHiddenMode.NULL, CaptureHiddenMode.FULL):
                batch = input_batch([2] * 5, hidden_mode=mode)
                result = runner.execute(batch).next_token_logits
                torch.testing.assert_close(
                    result, torch.full_like(result, 3.5), rtol=0, atol=0
                )
            assert runner.capture_generations == 2
        finally:
            runner.backend.cleanup()
            NcclEpBuffer.destroy()
        # Audit retains old Python references and checks GPU wait -> explicit
        # graph reset -> persistent handle destroy -> group destroy ordering.
        evidence = audit.assert_closed()
        assert evidence["groups_created"] == 3  # eager + two Graph generations
        assert evidence["handles_created"] == 2
        assert evidence["graphs_created"] == 4


def test_failed_capture_can_start_a_fresh_generation():
    from nccl_ep_test.dispatcher import forward_layer
    from nccl_ep_test.ep_audit import EpAudit
    from nccl_ep_test.fake_ep import dispatcher_environment

    from sglang.srt.model_executor.runner.shape_key import ShapeKey

    with dispatcher_environment(capacity=8) as environment, EpAudit() as audit:
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpBuffer

        dispatcher = environment.dispatcher(layer_id=0)
        backend = graph_backend(environment.coordinator, 8)
        x = torch.ones(8, 2048, dtype=torch.bfloat16, device="cuda")
        ids = torch.tensor([[0, 1]] * 8, device="cuda")
        weights = torch.tensor([[0.25, 0.75]] * 8, device="cuda")
        inject_failure = True

        def forward():
            result = forward_layer(dispatcher, x, ids, weights, 0)
            if inject_failure and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("injected failure after completed combine")
            return result

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with pytest.raises(RuntimeError, match="injected failure") as failure:
                with torch.cuda.stream(stream), backend.capture_session(stream):
                    backend.capture_one(ShapeKey(8), forward)
            # Keep the exception/traceback alive too: relying on local graph GC
            # is insufficient when failed capture frames retain it.
            assert failure.value is not None
            assert all(graph.audit_reset for graph in audit.graphs)
            assert all(
                item["closed"]
                for item in audit.groups.values()
                if item["rdma_buffer_size"] == 0
            )
            inject_failure = False
            with torch.cuda.stream(stream), backend.capture_session(stream):
                backend.capture_one(ShapeKey(8), forward)
            with backend.replay_session():
                result = backend.replay(ShapeKey(8), None)[2]
            torch.testing.assert_close(result, torch.full_like(x, 1.75), rtol=0, atol=0)
        finally:
            # Emergency teardown cannot satisfy the assertions above.
            torch.cuda.synchronize()
            for graph in audit.graphs:
                if not graph.audit_reset:
                    graph.reset()
            backend.cleanup()
            NcclEpBuffer.destroy()
        assert audit.assert_closed()["handles_created"] == 2


def test_eager_graph_alternation_keeps_groups_isolated():
    from nccl_ep_test.dispatcher import forward_layer
    from nccl_ep_test.fake_ep import dispatcher_environment
    from nccl_ep_test.runner_inputs import SyntheticEagerRunner, input_batch

    def delay_graph():
        if torch.cuda.is_current_stream_capturing():
            torch.cuda._sleep(20_000_000)

    with dispatcher_environment(capacity=32) as environment:
        runner = synthetic_ep_runner(
            environment, before_forward=delay_graph, share_inputs=True
        )
        eager = SyntheticEagerRunner(runner.synthetic_forward, 16)
        graph_stream, eager_stream = torch.cuda.Stream(), torch.cuda.Stream()
        try:
            graph_batch, eager_batch = input_batch([1] * 5), input_batch([2] * 5)
            graph_stream.wait_stream(torch.cuda.current_stream())
            eager_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(graph_stream):
                graph_output = runner.execute(graph_batch).next_token_logits.clone()
            with torch.cuda.stream(eager_stream):
                eager_output = eager.execute(eager_batch).next_token_logits.clone()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                graph_output, torch.full_like(graph_output, 1.75), rtol=0, atol=0
            )
            torch.testing.assert_close(
                eager_output, torch.full_like(eager_output, 3.5), rtol=0, atol=0
            )

            # A larger eager LL step must not resize or replace Graph resources.
            dispatcher = environment.dispatcher(layer_id=1)
            x = torch.full((24, 2048), 4, dtype=torch.bfloat16, device="cuda")
            ids = torch.tensor([[0, 1]] * 24, device="cuda")
            weights = torch.tensor([[0.25, 0.75]] * 24, device="cuda")
            with torch.cuda.stream(eager_stream):
                eager_stream.wait_stream(torch.cuda.default_stream())
                large = forward_layer(dispatcher, x, ids, weights, 0)[2].clone()
            with torch.cuda.stream(graph_stream):
                repeated = runner.execute(graph_batch).next_token_logits.clone()
            torch.cuda.synchronize()
            torch.testing.assert_close(large, torch.full_like(large, 7), rtol=0, atol=0)
            torch.testing.assert_close(
                repeated, torch.full_like(repeated, 1.75), rtol=0, atol=0
            )
            assert sorted(
                group.config.max_dispatch_tokens_per_rank
                for group in environment.groups
            ) == [16, 32]
            assert environment.events.count("handle_create") == 3
            assert environment.events.count("handle_destroy") == 2
        finally:
            runner.backend.cleanup()


def test_incompatible_layer_rejected_without_replacing_group():
    from nccl_ep_test.fake_ep import dispatcher_environment
    from nccl_ep_test.runner_inputs import input_batch

    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

    with dispatcher_environment(capacity=16) as environment:
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpDispatcher

        runner = synthetic_ep_runner(environment)
        try:
            with pytest.raises(ValueError, match="Incompatible"):
                NcclEpDispatcher(
                    MoeRunnerConfig(
                        num_experts=2,
                        num_local_experts=2,
                        hidden_size=4096,
                        top_k=2,
                        params_dtype=torch.bfloat16,
                        layer_id=1,
                    ),
                    environment.coordinator,
                )
            assert environment.events.count("group_create") == 2
            assert environment.events.count("group_destroy") == 0
            result = runner.execute(input_batch([1] * 5)).next_token_logits
            torch.testing.assert_close(
                result, torch.full_like(result, 1.75), rtol=0, atol=0
            )
        finally:
            runner.backend.cleanup()


def test_model_parallel_shutdown_closes_registered_ep_resources():
    from nccl_ep_test.ep_audit import EpAudit
    from nccl_ep_test.fake_ep import dispatcher_environment
    from nccl_ep_test.runner_inputs import input_batch

    from sglang.srt.distributed.parallel_state import destroy_model_parallel

    with dispatcher_environment(capacity=16) as environment, EpAudit() as audit:
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpBuffer

        runner = synthetic_ep_runner(environment)
        runner.execute(input_batch([1] * 5))
        try:
            destroy_model_parallel()
            assert audit.assert_closed()["all_explicitly_closed"]
        finally:
            # Preserve test isolation even when the public shutdown is broken.
            runner.backend.cleanup()
            torch.cuda.synchronize()
            NcclEpBuffer.destroy()


def test_concurrent_runner_submissions_fail_before_input_writes():
    from concurrent.futures import ThreadPoolExecutor

    from nccl_ep_test.fake_ep import dispatcher_environment
    from nccl_ep_test.runner_inputs import SyntheticEagerRunner, input_batch

    with dispatcher_environment(capacity=16) as environment:
        runner = synthetic_ep_runner(environment, share_inputs=True)
        eager = SyntheticEagerRunner(runner.synthetic_forward, 16)
        batch = input_batch([1] * 5)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                with runner.backend.replay_session():
                    for submit in (runner.execute, eager.execute):
                        future = pool.submit(submit, batch)
                        with pytest.raises(RuntimeError, match="Concurrent NCCL EP"):
                            future.result(timeout=5)
            result = runner.execute(batch).next_token_logits
            torch.testing.assert_close(
                result, torch.full_like(result, 1.75), rtol=0, atol=0
            )
        finally:
            runner.backend.cleanup()


def test_failed_combine_does_not_allow_another_transaction():
    from unittest.mock import patch

    from nccl_ep_test.dispatcher import forward_layer
    from nccl_ep_test.fake_ep import FakeHandle, dispatcher_environment

    from sglang.srt.layers.moe.topk import StandardTopKOutput

    with dispatcher_environment(capacity=8) as environment:
        dispatcher = environment.dispatcher(layer_id=0)
        backend = graph_backend(environment.coordinator, 8)
        x = torch.ones(8, 2048, dtype=torch.bfloat16, device="cuda")
        ids = torch.tensor([[0, 1]] * 8, device="cuda")
        weights = torch.tensor([[0.25, 0.75]] * 8, device="cuda")
        complete = FakeHandle.complete

        def fail_before_combine_completion(handle, *args, **kwargs):
            if handle.pending == "combine":
                raise RuntimeError("injected external completion error")
            return complete(handle, *args, **kwargs)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with pytest.raises(RuntimeError):
                with torch.cuda.stream(stream), backend.capture_session(stream):
                    with patch.object(
                        FakeHandle, "complete", fail_before_combine_completion
                    ):
                        forward_layer(dispatcher, x, ids, weights, 0)
            with pytest.raises(RuntimeError, match="incomplete"):
                dispatcher.dispatch(x, StandardTopKOutput(weights, ids, None))
        finally:
            # This fake raises before invoking any native operation. Complete
            # its pending work solely for fixture teardown. A real native/GPU
            # fault requires process restart; this test makes no recovery claim.
            with torch.cuda.stream(stream):
                dispatcher.combine_b()
            backend.cleanup()


def test_duplicate_bucket_capture_requires_a_new_generation():
    from nccl_ep_test.ep_audit import EpAudit
    from nccl_ep_test.fake_ep import dispatcher_environment
    from nccl_ep_test.runner_inputs import input_batch

    with dispatcher_environment(capacity=16) as environment, EpAudit() as audit:
        runner = synthetic_ep_runner(environment)
        try:
            with pytest.raises(ValueError, match="already captured"):
                runner.capture()
            # Rejection closes the old generation explicitly, even though the
            # audit still retains every graph object. A fresh capture is usable.
            assert all(graph.audit_reset for graph in audit.graphs)
            runner.capture()
            actual = runner.execute(input_batch([2] * 5)).next_token_logits
            torch.testing.assert_close(
                actual, torch.full_like(actual, 3.5), rtol=0, atol=0
            )
        finally:
            runner.backend.cleanup()


def test_graph_rejects_broadcastable_wrong_hidden_size_before_dispatch():
    from nccl_ep_test.dispatcher import forward_layer
    from nccl_ep_test.fake_ep import dispatcher_environment

    from sglang.srt.layers.moe.topk import StandardTopKOutput

    with dispatcher_environment(capacity=8) as environment:
        dispatcher = environment.dispatcher(layer_id=0)
        backend = graph_backend(environment.coordinator, 8)
        ids = torch.tensor([[0, 1]] * 8, device="cuda")
        weights = torch.tensor([[0.25, 0.75]] * 8, device="cuda")
        x = torch.ones(8, 1, dtype=torch.bfloat16, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(stream), backend.capture_session(stream):
                with pytest.raises(ValueError, match="hidden size"):
                    dispatcher.dispatch(x, StandardTopKOutput(weights, ids, None))
                actual = forward_layer(dispatcher, x.expand(-1, 2048), ids, weights, 0)[
                    2
                ]
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(
                actual, torch.full_like(actual, 1.75), rtol=0, atol=0
            )
        finally:
            backend.cleanup()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
