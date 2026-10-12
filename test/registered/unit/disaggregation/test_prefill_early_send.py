"""CPU coverage for EarlyKVChunkSender (SGLANG_DISAGG_PREFILL_EARLY_SEND_KV)."""

import threading
import unittest
import weakref
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.disaggregation.prefill_early_send import (
    EarlyKVChunkSender,
    _BatchSend,
    _ChunkSend,
)
from sglang.srt.disaggregation.utils import KVSendOwner, TransferBackend
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

INPUT_LEN = 100


class _Event:
    def __init__(self):
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True


class _Req(SimpleNamespace):
    # Hashable by identity, like Req, so it can sit in the failed WeakSet.
    __hash__ = object.__hash__
    __eq__ = object.__eq__


def _req(**overrides):
    req = _Req(
        rid="r",
        bootstrap_room=1,
        kv_send_owner=KVSendOwner.NONE,
        origin_input_ids=list(range(INPUT_LEN)),
        extend_end=INPUT_LEN,
        output_ids=array("q"),
        return_logprob=False,
        return_sampling_mask=False,
        grammar=None,
        pending_bootstrap=False,
        to_finish=None,
        finished_reason=None,
        kv=SimpleNamespace(holds_kv=True),
        disagg_kv_sender=Mock(),
        time_stats=Mock(),
    )
    for k, v in overrides.items():
        setattr(req, k, v)
    return req


def _sender(send_kv_chunk=None):
    """Build a sender without its CUDA worker thread."""
    sender = EarlyKVChunkSender.__new__(EarlyKVChunkSender)
    sender._scheduler = SimpleNamespace(send_kv_chunk=send_kv_chunk or Mock())
    sender.lock = threading.Lock()
    sender._queue = Mock()
    sender._failed_reqs = weakref.WeakSet()
    return sender


def _scheduler(**overrides):
    scheduler = SimpleNamespace(
        enable_overlap=True,
        enable_staging=False,
        kv_checksum_computer=None,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        transfer_backend=TransferBackend.MOONCAKE,
        enable_unified_memory=False,
        disagg_prefill_bootstrap_queue=SimpleNamespace(
            kv_manager=SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
        ),
    )
    for k, v in overrides.items():
        setattr(scheduler, k, v)
    return scheduler


def _result(token_ids, copy_done=True):
    return SimpleNamespace(
        copy_done=_Event() if copy_done else None,
        next_token_ids=torch.tensor(token_ids),
    )


def _batch_send(token_ids, *chunks):
    return _BatchSend(_Event(), torch.tensor(token_ids), list(chunks))


def _patch_disagg(policy="default"):
    return patch(
        "sglang.srt.disaggregation.prefill_early_send.get_disagg",
        return_value=SimpleNamespace(disaggregation_decode_allocation_policy=policy),
    )


class TestIsSupported(CustomTestCase):
    def _supported(self, policy="default", **overrides):
        with _patch_disagg(policy), patch("torch.cuda.is_available", return_value=True):
            return EarlyKVChunkSender.is_supported(_scheduler(**overrides))

    def test_supported(self):
        self.assertTrue(self._supported())

    def test_unsupported_configs(self):
        self.assertFalse(self._supported(enable_overlap=False))
        self.assertFalse(self._supported(enable_staging=True))
        self.assertFalse(self._supported(kv_checksum_computer=object()))
        self.assertFalse(
            self._supported(spec_algorithm=SimpleNamespace(is_none=lambda: False))
        )
        self.assertFalse(self._supported(policy="prefill_complete"))
        self.assertFalse(self._supported(transfer_backend=TransferBackend.NIXL))
        self.assertFalse(self._supported(enable_unified_memory=True))
        self.assertFalse(
            self._supported(
                disagg_prefill_bootstrap_queue=SimpleNamespace(
                    kv_manager=SimpleNamespace(
                        kv_args=SimpleNamespace(state_types=["mamba"])
                    )
                )
            )
        )


class TestMaybeCreate(CustomTestCase):
    def test_disabled_by_default(self):
        with envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_KV.override(False):
            self.assertIsNone(EarlyKVChunkSender.maybe_create(_scheduler()))

    def test_unsupported_config_warns(self):
        with (
            envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_KV.override(True),
            _patch_disagg(),
            patch("torch.cuda.is_available", return_value=True),
            self.assertLogs(
                "sglang.srt.disaggregation.prefill_early_send", level="WARNING"
            ),
        ):
            self.assertIsNone(
                EarlyKVChunkSender.maybe_create(_scheduler(enable_overlap=False))
            )


class TestOwnership(CustomTestCase):
    def test_owner_fixed_at_first_submit(self):
        sender = _sender()
        plain, logprob = _req(), _req(return_logprob=True)
        pending = _req(pending_bootstrap=True)
        sender.submit(
            SimpleNamespace(reqs=[plain, logprob, pending]), _result([1, 2, 3])
        )
        self.assertEqual(plain.kv_send_owner, KVSendOwner.EARLY)
        self.assertEqual(logprob.kv_send_owner, KVSendOwner.SCHEDULER)
        self.assertEqual(pending.kv_send_owner, KVSendOwner.SCHEDULER)

        # Later eligibility changes do not move the owner.
        pending.pending_bootstrap = False
        sender.submit(SimpleNamespace(reqs=[pending]), _result([4]))
        self.assertEqual(pending.kv_send_owner, KVSendOwner.SCHEDULER)

    def test_no_result_copy_means_scheduler(self):
        sender = _sender()
        req = _req()
        sender.submit(SimpleNamespace(reqs=[req]), _result([1], copy_done=False))
        self.assertEqual(req.kv_send_owner, KVSendOwner.SCHEDULER)
        sender._queue.put.assert_not_called()


class TestSubmit(CustomTestCase):
    def test_queues_middle_and_last_chunks(self):
        sender = _sender()
        middle, last, logprob = _req(extend_end=64), _req(), _req(return_logprob=True)
        result = _result([3, 4, 5])
        sender.submit(SimpleNamespace(reqs=[logprob, middle, last]), result)
        (item,), _ = sender._queue.put.call_args
        self.assertIs(item.copy_done, result.copy_done)
        self.assertEqual(
            item.chunks,
            [
                _ChunkSend(batch_index=1, req=middle, end_idx=64, is_last=False),
                _ChunkSend(batch_index=2, req=last, end_idx=INPUT_LEN, is_last=True),
            ],
        )

    def test_snapshots_end_idx(self):
        sender = _sender()
        req = _req(extend_end=64)
        sender.submit(SimpleNamespace(reqs=[req]), _result([3]))
        req.extend_end = INPUT_LEN  # next chunk scheduled
        (item,), _ = sender._queue.put.call_args
        self.assertEqual(item.chunks[0].end_idx, 64)
        self.assertFalse(item.chunks[0].is_last)


class TestSend(CustomTestCase):
    def test_chunks_sent_in_order(self):
        sender = _sender()
        req = _req(kv_send_owner=KVSendOwner.EARLY)
        first = _batch_send([0], _ChunkSend(0, req, 48, is_last=False))
        second = _batch_send([42], _ChunkSend(0, req, INPUT_LEN, is_last=True))
        sender._send_batch(first)
        sender._send_batch(second)

        self.assertTrue(first.copy_done.synchronized)
        self.assertEqual(
            sender._scheduler.send_kv_chunk.call_args_list,
            [call(req, last_chunk=False, end_idx=48), call(req, last_chunk=True)],
        )
        self.assertEqual(list(req.output_ids), [42])
        req.time_stats.set_prefill_transfer_queue_entry_time.assert_called_once()

    def test_skips_aborted_and_released(self):
        for req in (
            _req(to_finish=FINISH_ABORT("aborted")),
            _req(kv=SimpleNamespace(holds_kv=False)),
        ):
            sender = _sender()
            sender._send_batch(
                _batch_send([42], _ChunkSend(0, req, INPUT_LEN, is_last=True))
            )
            sender._scheduler.send_kv_chunk.assert_not_called()
            self.assertEqual(len(req.output_ids), 0)

    def test_failed_middle_aborts_and_skips_rest(self):
        sender = _sender(send_kv_chunk=Mock(side_effect=RuntimeError("boom")))
        req = _req()
        with self.assertLogs(
            "sglang.srt.disaggregation.prefill_early_send", level="ERROR"
        ):
            sender._send_batch(_batch_send([0], _ChunkSend(0, req, 48, is_last=False)))
        req.disagg_kv_sender.abort.assert_called_once()

        sender._scheduler.send_kv_chunk.reset_mock()
        sender._send_batch(
            _batch_send([42], _ChunkSend(0, req, INPUT_LEN, is_last=True))
        )
        sender._scheduler.send_kv_chunk.assert_not_called()

    def test_failed_last_aborts_transfer(self):
        sender = _sender(send_kv_chunk=Mock(side_effect=RuntimeError("boom")))
        req = _req()
        with self.assertLogs(
            "sglang.srt.disaggregation.prefill_early_send", level="ERROR"
        ):
            sender._send_batch(
                _batch_send([42], _ChunkSend(0, req, INPUT_LEN, is_last=True))
            )
        req.disagg_kv_sender.abort.assert_called_once()


if __name__ == "__main__":
    unittest.main()
