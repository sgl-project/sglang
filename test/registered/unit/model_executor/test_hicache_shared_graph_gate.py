"""CPU coverage for graph selection after a shared-host cache restore."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.cache_controller import LayerDoneCounter
from sglang.srt.model_executor import model_runner
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _CopyEvent:
    def __init__(self, ready):
        self.ready = ready
        self.queries = 0

    def query(self):
        self.queries += 1
        return self.ready


class TestHiCacheSharedGraphGate(CustomTestCase):
    def _forward(
        self,
        mode,
        *,
        slot=0,
        enabled=True,
        restore_mode="shared",
        broadcast_overlap=False,
        broadcast_pending=False,
        copy_ready=False,
    ):
        # Construct no CUDA events, model weights, or actual graph runners.
        counter = LayerDoneCounter.__new__(LayerDoneCounter)
        counter.consumer_index = slot
        counter.pending_layerwise_load = lambda index: broadcast_pending
        event = _CopyEvent(copy_ready)
        counter.events = [SimpleNamespace(finish_event=event) for _ in range(3)]

        runner = model_runner.ModelRunner.__new__(model_runner.ModelRunner)
        runner.device = "cuda"
        runner.attn_backend = None
        runner.token_to_kv_pool = SimpleNamespace(layer_transfer_counter=counter)
        runner.hisparse_coordinator = None
        runner.decode_cuda_graph_runner = SimpleNamespace(
            can_run_graph=lambda batch: True,
            execute=lambda *args, **kwargs: "decode_graph",
        )
        runner.prefill_cuda_graph_runner = SimpleNamespace(
            can_run_graph=lambda batch: True,
            execute=lambda *args, **kwargs: "prefill_graph",
        )
        runner.eager_runner = SimpleNamespace(execute=lambda *args, **kwargs: "eager")
        runner._prepare_eager_forward_batch = lambda batch: None
        runner._maybe_execute_deferred_mamba_cow_and_clear = lambda batch: None
        runner._extend_forward_kwargs = lambda *args: {}
        runner.forward_split_prefill = lambda *args, **kwargs: "split_prefill"
        runner.device_timer = None
        runner.pp_group = SimpleNamespace(is_last_rank=True)
        batch = SimpleNamespace(
            forward_mode=mode,
            token_indices_to_pool=None,
            global_num_tokens_cpu=None,
        )
        memory = SimpleNamespace(
            enable_mla_hicache_host_dedup=enabled,
            mla_hicache_host_dedup_mode=restore_mode,
            enable_mla_hicache_layerwise_overlap=broadcast_overlap,
        )
        with (
            patch.object(model_runner, "get_memory", return_value=memory),
            patch.object(model_runner, "has_forward_context", return_value=True),
            patch.object(model_runner, "get_global_dwdp_manager", return_value=None),
            patch.object(
                model_runner,
                "_prefill_cuda_graph_allows_context_parallel",
                return_value=True,
            ),
        ):
            result = runner._forward_raw(batch, pp_proxy_tensors=None)
        return result, event

    def test_shared_decode_and_idle_keep_graphs_with_retained_restore_slot(self):
        # ScheduleBatch retains the restore slot when prefill becomes decode.
        # A slot, including zero, does not mean decode starts another restore.
        for mode in (ForwardMode.DECODE, ForwardMode.IDLE):
            for slot in (0, 2):
                with self.subTest(mode=mode, slot=slot):
                    output, _ = self._forward(mode, slot=slot)
                    self.assertEqual(output.logits_output, "decode_graph")
                    self.assertTrue(output.can_run_graph)

    def test_shared_restore_prefill_keeps_native_layer_waits_outside_graphs(self):
        for mode in (ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.SPLIT_PREFILL):
            for slot in (0, 2):
                with self.subTest(mode=mode, slot=slot):
                    output, _ = self._forward(mode, slot=slot)
                    self.assertEqual(
                        output.logits_output,
                        "split_prefill" if mode.is_split_prefill() else "eager",
                    )
                    self.assertFalse(output.can_run_graph)

    def test_shared_prefill_without_restore_can_use_graphs(self):
        output, _ = self._forward(ForwardMode.EXTEND, slot=-1)
        self.assertEqual(output.logits_output, "prefill_graph")
        self.assertTrue(output.can_run_graph)

    def test_disabled_dedup_and_plain_broadcast_keep_normal_graph_selection(self):
        for settings in ({"enabled": False}, {"restore_mode": "broadcast"}):
            for mode in (ForwardMode.DECODE, ForwardMode.IDLE, ForwardMode.EXTEND):
                with self.subTest(settings=settings, mode=mode):
                    output, _ = self._forward(mode, **settings)
                    self.assertEqual(
                        output.logits_output,
                        "prefill_graph" if mode.is_extend() else "decode_graph",
                    )
                    self.assertTrue(output.can_run_graph)

    def test_broadcast_overlap_still_respects_pending_layerwise_restore(self):
        for pending in (False, True):
            with self.subTest(pending=pending):
                output, _ = self._forward(
                    ForwardMode.DECODE,
                    restore_mode="broadcast",
                    broadcast_overlap=True,
                    broadcast_pending=pending,
                )
                self.assertEqual(
                    output.logits_output, "eager" if pending else "decode_graph"
                )
                self.assertEqual(output.can_run_graph, not pending)

    def test_rank_local_copy_readiness_does_not_change_graph_selection(self):
        for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
            choices = []
            for ready in (False, True):
                output, event = self._forward(mode, copy_ready=ready)
                choices.append((output.logits_output, output.can_run_graph))
                self.assertEqual(event.queries, 0)
            self.assertEqual(choices[0], choices[1])


if __name__ == "__main__":
    unittest.main()
