import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.sampler import Sampler
from sglang.srt.managers.io_struct import FlushCacheReqInput, FlushCacheReqOutput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.flush_wrapper import (
    SchedulerFlushWrapper,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.watermarking.proposal_relay import RequestProposalRelay

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="stage-b-test-cpu-intel")


class TestSchedulerFlushCache(CustomTestCase):
    def _new_scheduler(self) -> SimpleNamespace:
        """The wrapper plus the three collaborators it was handed."""
        flush_cache = MagicMock(return_value=True)
        is_fully_idle = MagicMock(return_value=False)
        send_output = MagicMock()
        return SimpleNamespace(
            flush_cache=flush_cache,
            is_fully_idle=is_fully_idle,
            send_output=send_output,
            flush_wrapper=SchedulerFlushWrapper(
                flush_cache=flush_cache,
                is_fully_idle=is_fully_idle,
                send_output=send_output,
            ),
        )

    def test_immediate_flush_no_timeout(self):
        """No timeout → flush immediately regardless of idle state."""
        scheduler = self._new_scheduler()
        scheduler.flush_cache.return_value = False

        output = scheduler.flush_wrapper.handle(FlushCacheReqInput(timeout_s=None))

        self.assertFalse(output.success)
        scheduler.flush_cache.assert_called_once()

    def test_immediate_flush_when_idle(self):
        """Positive timeout but already idle → flush immediately."""
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = True

        output = scheduler.flush_wrapper.handle(FlushCacheReqInput(timeout_s=5.0))

        self.assertTrue(output.success)
        scheduler.flush_cache.assert_called_once()

    def test_defers_when_busy(self):
        """Positive timeout + busy → defers, returns None."""
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=3.0)

        with patch(
            "sglang.srt.managers.scheduler_components.flush_wrapper.time.monotonic",
            return_value=10.0,
        ):
            output = scheduler.flush_wrapper.handle(req)

        self.assertIsNone(output)
        pending_req, deadline = scheduler.flush_wrapper._pending
        self.assertIs(pending_req, req)
        self.assertEqual(deadline, 13.0)

    def test_rejects_when_already_pending(self):
        """Any new request is rejected while another is pending."""
        scheduler = self._new_scheduler()
        scheduler.flush_wrapper._pending = (FlushCacheReqInput(timeout_s=10.0), 999.0)

        for timeout in [None, 5.0]:
            output = scheduler.flush_wrapper.handle(
                FlushCacheReqInput(timeout_s=timeout)
            )
            self.assertFalse(output.success)
            self.assertIn("already in progress", output.message)

        scheduler.flush_cache.assert_not_called()

    def test_pending_flush_completes_on_idle(self):
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = True
        req = FlushCacheReqInput(timeout_s=1.0)
        scheduler.flush_wrapper._pending = (req, 111.0)

        scheduler.flush_wrapper.check_pending()

        self.assertIsNone(scheduler.flush_wrapper._pending)
        scheduler.flush_cache.assert_called_once()
        out, answered = scheduler.send_output.call_args.args
        self.assertTrue(out.success)
        self.assertIs(answered, req)

    def test_pending_flush_expires_on_timeout(self):
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=1.0)
        scheduler.flush_wrapper._pending = (req, 99.0)

        with patch(
            "sglang.srt.managers.scheduler_components.flush_wrapper.time.monotonic",
            return_value=100.0,
        ):
            scheduler.flush_wrapper.check_pending()

        self.assertIsNone(scheduler.flush_wrapper._pending)
        scheduler.flush_cache.assert_not_called()
        out = scheduler.send_output.call_args.args[0]
        self.assertFalse(out.success)

    def test_pending_flush_survives_before_deadline(self):
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=5.0)
        scheduler.flush_wrapper._pending = (req, 101.0)

        with patch(
            "sglang.srt.managers.scheduler_components.flush_wrapper.time.monotonic",
            return_value=100.0,
        ):
            scheduler.flush_wrapper.check_pending()

        self.assertIsNotNone(scheduler.flush_wrapper._pending)
        scheduler.send_output.assert_not_called()

    def test_control_output_goes_to_the_embedded_rust_server(self):
        """A control reply, deferred flush included, takes the Rust server's
        egress ring when the scheduler embeds one, else the tokenizer socket."""
        output = FlushCacheReqOutput(success=True)
        req = FlushCacheReqInput(rid="r", timeout_s=1.0)
        rust = SimpleNamespace(rust_server=MagicMock(), ipc_channels=MagicMock())
        Scheduler.send_control_output(rust, output, req)
        rust.rust_server.push_control_output.assert_called_once_with(req, output)
        rust.ipc_channels.send_to_tokenizer.send_output.assert_not_called()


class TestSynthIDCacheFlush(CustomTestCase):
    def test_relay_keeps_beam_tail_slicing_and_request_policy(self):
        """Adding watermark metadata must not relay beam-tail rows as requests."""
        for enabled, beam in ((False, True), (True, False), (True, True)):
            with self.subTest(
                enabled=enabled, beam=beam
            ), get_context().override_server_args(
                enable_watermark=enabled,
                watermark_algorithm="synthid" if enabled else None,
            ):
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.spec_algorithm = SimpleNamespace(is_ngram=lambda: False)
                scheduler.future_map = MagicMock()
                scheduler.beam_coordinator = MagicMock()
                scheduler.chunked_req = None
                slots = torch.tensor([1, 2, 3]) if beam else torch.tensor([1, 2])
                batch = SimpleNamespace(
                    beam_tail=SimpleNamespace(num_base_rows=2) if beam else None,
                    req_pool_indices_cpu=slots,
                    sampling_info=SimpleNamespace(
                        synthid_state=(
                            SimpleNamespace(request_enabled_host=[True, False])
                            if enabled
                            else None
                        )
                    ),
                )
                result = SimpleNamespace(
                    next_draft_input=None,
                    has_sampled_token_ids=True,
                    next_token_ids=torch.tensor([7, 8]),
                )
                indices = torch.tensor([0, 1, 2]) if beam else torch.tensor([0, 1])
                scheduler._relay_forward_payload(batch, indices, result)
                actual_indices, payload = scheduler.future_map.stash.call_args.args
                self.assertEqual(actual_indices.tolist(), [0, 1])
                if enabled:
                    self.assertEqual(payload.synthid_enabled_host, [True, False])
                    relay = RequestProposalRelay(
                        slots=4, candidate_count=2, device="cpu"
                    )
                    relay.store(
                        actual_indices,
                        torch.ones(2, 4) / 4,
                        None,
                        payload.synthid_enabled_host,
                        indices_cpu=payload.synthid_slot_indices_cpu,
                    )
                    self.assertEqual(payload.synthid_slot_indices_cpu.tolist(), [1, 2])
                    if not beam:
                        self.assertIs(payload.synthid_slot_indices_cpu, slots)
                self.assertEqual(slots.tolist(), [1, 2, 3] if beam else [1, 2])
                scheduler.beam_coordinator.maybe_select_and_relay.assert_called_once_with(
                    batch,
                    result,
                    chunked_req=None,
                )

    def test_only_opted_in_server_invalidates_watermark_state(self):
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                get_context().override_server_args(
                    enable_watermark=enabled,
                    watermark_algorithm="synthid" if enabled else None,
                    synthid_watermark_key_file="test-keys.json" if enabled else None,
                ),
            ):
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.is_fully_idle = MagicMock(return_value=True)
                scheduler.tree_cache = MagicMock()
                scheduler.req_to_token_pool = MagicMock()
                scheduler.token_to_kv_pool_allocator = MagicMock()
                scheduler.grammar_manager = MagicMock()
                scheduler.metrics_reporter = MagicMock()
                scheduler.metrics_reporter.is_stats_logging_rank = False
                scheduler.draft_worker = None
                processor = MagicMock()
                sampler = Sampler.__new__(Sampler)
                sampler.synthid_watermark = processor if enabled else None
                sampler.watermark_state = None
                scheduler.tp_worker = MagicMock()
                scheduler.tp_worker.model_runner.sampler = sampler

                self.assertTrue(scheduler.flush_cache(empty_cache=False))
                scheduler.req_to_token_pool.clear.assert_called_once()
                self.assertEqual(
                    processor.invalidate_on_cache_flush.call_count,
                    int(enabled),
                )


if __name__ == "__main__":
    unittest.main()
