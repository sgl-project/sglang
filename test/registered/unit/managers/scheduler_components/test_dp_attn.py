import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.communicator import LayerCommunicator  # noqa: E402
from sglang.srt.managers.scheduler_components.dp_attn import (  # noqa: E402
    _wait_for_rank_sync,
)
from sglang.srt.model_executor.forward_context import (  # noqa: E402
    ForwardContext,
    forward_context,
    get_forward_context,
)
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP  # noqa: E402
from sglang.srt.speculative.eagle_worker_v2 import (  # noqa: E402
    _draft_extend_rank_sync_context,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _runner(*, event=None, required=True):
    return SimpleNamespace(
        model=SimpleNamespace(
            requires_dp_attention_rank_sync_ordering=required,
        ),
        rank_sync_done_event=event,
        forward_stream=object(),
    )


class TestDPAttnRankSync(CustomTestCase):
    def test_waits_for_boundary_event(self):
        event = object()
        runner = _runner(event=event)
        stream = MagicMock()
        fallback_stream = object()

        _wait_for_rank_sync(runner, stream, fallback_stream)

        stream.wait_event.assert_called_once_with(event)
        stream.wait_stream.assert_not_called()
        self.assertIsNone(runner.rank_sync_done_event)

    def test_missing_producer_waits_actual_forward_stream(self):
        runner = _runner()
        stream = MagicMock()
        fallback_stream = object()
        self.assertIsNot(fallback_stream, runner.forward_stream)

        _wait_for_rank_sync(runner, stream, fallback_stream)

        stream.wait_stream.assert_called_once_with(fallback_stream)
        stream.wait_event.assert_not_called()

    def test_unrelated_model_is_unchanged(self):
        event = object()
        runner = _runner(event=event, required=False)
        stream = MagicMock()

        _wait_for_rank_sync(runner, stream, object())

        stream.wait_event.assert_not_called()
        stream.wait_stream.assert_not_called()
        self.assertIs(runner.rank_sync_done_event, event)


class TestEagerRankSyncPublish(CustomTestCase):
    def test_publishes_after_draft_extend_scope(self):
        event = object()
        runner = SimpleNamespace(
            attn_backend=object(),
            rank_sync_boundary_event=event,
            rank_sync_done_event=None,
        )

        with _draft_extend_rank_sync_context(runner):
            self.assertIs(get_forward_context().rank_sync_done_event, event)
            self.assertIsNone(runner.rank_sync_done_event)

        self.assertIs(runner.rank_sync_done_event, event)


class TestLayerRankSyncBoundary(CustomTestCase):
    def test_records_after_last_layer_communication(self):
        order = []
        event = MagicMock()
        event.record.side_effect = lambda: order.append("event")
        communicator = object.__new__(LayerCommunicator)
        communicator.is_last_layer = True
        communicator.allow_reduce_scatter = True
        communicator._context = object()
        communicator._communicate_summable_tensor_pair_fn = MagicMock(
            side_effect=lambda **_: order.append("communicate")
        )

        with forward_context(
            ForwardContext(
                attn_backend=object(),
                rank_sync_done_event=event,
            )
        ):
            communicator.postprocess_layer("hidden-in", "residual-in", object())

        self.assertEqual(order, ["communicate", "event"])


class TestQwen35MTPRankSync(CustomTestCase):
    def test_boundary_excludes_logits_collective(self):
        model = object.__new__(Qwen3_5ForCausalLMMTP)
        model.logits_processor = SimpleNamespace(do_tensor_parallel_all_gather=False)
        self.assertTrue(model.rank_sync_boundary_after_last_layer_communication)

        model.logits_processor.do_tensor_parallel_all_gather = True
        self.assertFalse(model.rank_sync_boundary_after_last_layer_communication)


if __name__ == "__main__":
    unittest.main()
