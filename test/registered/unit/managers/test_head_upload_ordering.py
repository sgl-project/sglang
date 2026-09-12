"""Unit tests for the HiCache load-burst ordering fix (round-head H2D pre-upload).

The first ops a prefill forward enqueues are small H2D copies (the deferred
input_ids upload plus init_new's extend / DP-sync metadata). With an
asynchronous HiCache load burst already queued on the load stream, the copy
engine can serve those head copies only after the whole burst, and the entire
forward waits behind them. The fix issues those copies on the schedule stream
BEFORE start_loading() records the burst's start_event, so they are ordered
ahead of the burst by stream contract.

Properties pinned here:
* pre_upload_forward_inputs consumes the prefill staging into device-resident
  input_ids for pure-prefill batches, leaves mixed batches on the deferred
  path, and stages the extend / global metadata as one-shot device tensors;
* StagedDeviceTensor hands its tensor out once, and only while the host list
  it was built from is still the batch's current value;
* resolve_forward_inputs on a pre-uploaded batch touches nothing, and keeps
  its behaviour for batches that were NOT pre-uploaded;
* ForwardBatch.init_mlp_sync_metadata consumes the staged global counts for
  non-spec batches and falls back to a fresh upload otherwise;
* Scheduler._handover_hicache_load pre-uploads and then hands the burst over
  only for new prefill batches, exactly once per batch.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.overlap_utils import (
    StagedDeviceTensor,
    pre_upload_forward_inputs,
    resolve_forward_inputs,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _batch(**over):
    base = dict(
        device="cpu",
        prefill_input_ids_cpu=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        mix_running_indices=None,
        input_ids=None,
        extend_lens=[3, 1],
        prefix_lens=[10, 20],
        global_num_tokens=[4, 4, 4, 4],
        global_num_tokens_for_logprob=[4, 4, 4, 4],
        spec_info=None,
        head_staged_extend_seq_lens=None,
        head_staged_extend_prefix_lens=None,
        head_staged_global_num_tokens=None,
        head_staged_global_num_tokens_for_logprob=None,
        hicache_consumer_index=-1,
        forward_mode=ForwardMode.EXTEND,
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestStagedDeviceTensor(CustomTestCase):
    def test_take_is_one_shot(self):
        src = [1, 2]
        staged = StagedDeviceTensor(source=src, tensor=torch.tensor(src))
        first = staged.take(src)
        self.assertIsNotNone(first)
        self.assertIsNone(staged.take(src))

    def test_take_rejects_reassigned_source(self):
        src = [1, 2]
        staged = StagedDeviceTensor(source=src, tensor=torch.tensor(src))
        # Same values, different object: the host list was rebuilt (draft
        # extend, DP re-gather), so the staged copy must not be used.
        self.assertIsNone(staged.take([1, 2]))
        # The original source still takes.
        self.assertIsNotNone(staged.take(src))


class TestPreUpload(CustomTestCase):
    def test_pure_prefill_consumes_staging_into_input_ids(self):
        b = _batch()
        pre_upload_forward_inputs(b)
        self.assertIsNone(b.prefill_input_ids_cpu)
        self.assertEqual(b.input_ids.tolist(), [1, 2, 3, 4])

    def test_mixed_batch_keeps_the_deferred_path(self):
        b = _batch(mix_running_indices=torch.tensor([0, 1]))
        pre_upload_forward_inputs(b)
        # input_ids needs a forward-time FutureMap gather; staging must survive.
        self.assertIsNotNone(b.prefill_input_ids_cpu)
        self.assertIsNone(b.input_ids)
        # Metadata staging is still safe for a mixed batch.
        self.assertIsNotNone(b.head_staged_extend_seq_lens)

    def test_metadata_is_staged_as_device_tensors(self):
        b = _batch()
        pre_upload_forward_inputs(b)
        seq = b.head_staged_extend_seq_lens
        self.assertIs(seq.source, b.extend_lens)
        self.assertEqual(seq.tensor.tolist(), [3, 1])
        self.assertEqual(seq.tensor.dtype, torch.int32)
        pre = b.head_staged_extend_prefix_lens
        self.assertIs(pre.source, b.prefix_lens)
        self.assertEqual(pre.tensor.tolist(), [10, 20])
        glob = b.head_staged_global_num_tokens
        self.assertIs(glob.source, b.global_num_tokens)
        self.assertEqual(glob.tensor.tolist(), [4, 4, 4, 4])
        self.assertEqual(glob.tensor.dtype, torch.int64)
        self.assertEqual(
            b.head_staged_global_num_tokens_for_logprob.tensor.tolist(), [4, 4, 4, 4]
        )

    def test_spec_batch_skips_global_counts(self):
        b = _batch(spec_info=object())  # init_new rescales the counts
        pre_upload_forward_inputs(b)
        self.assertIsNotNone(b.head_staged_extend_seq_lens)
        self.assertIsNone(b.head_staged_global_num_tokens)
        self.assertIsNone(b.head_staged_global_num_tokens_for_logprob)

    def test_no_dp_attention_stages_no_globals(self):
        b = _batch(global_num_tokens=None, global_num_tokens_for_logprob=None)
        pre_upload_forward_inputs(b)
        self.assertIsNone(b.head_staged_global_num_tokens)
        self.assertIsNone(b.head_staged_global_num_tokens_for_logprob)

    def test_tensor_form_extend_lens_not_restaged(self):
        # gpu_only deployments hand device tensors in directly; staging must
        # not clobber them with a second copy.
        t = torch.tensor([3, 1], dtype=torch.int32)
        b = _batch(extend_lens=t, prefix_lens=t)
        pre_upload_forward_inputs(b)
        self.assertIsNone(b.head_staged_extend_seq_lens)
        self.assertIsNone(b.head_staged_extend_prefix_lens)


class _FakeFutureMap:
    def __init__(self):
        self.output_tokens_buf = torch.arange(100, dtype=torch.int64)
        self.spec_algo = SimpleNamespace(is_none=lambda: True)


class TestResolveInterplay(CustomTestCase):
    def _resolve_batch(self, **over):
        b = _batch(**over)
        b.req_pool_indices = torch.tensor([5, 6])
        b.enable_overlap = True
        b.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        return b

    def test_resolve_is_a_noop_after_pre_upload(self):
        b = self._resolve_batch()
        pre_upload_forward_inputs(b)
        uploaded = b.input_ids
        resolve_forward_inputs(b, _FakeFutureMap())
        self.assertIs(b.input_ids, uploaded)

    def test_resolve_unchanged_without_pre_upload(self):
        b = self._resolve_batch()
        resolve_forward_inputs(b, _FakeFutureMap())
        self.assertEqual(b.input_ids.tolist(), [1, 2, 3, 4])
        self.assertIsNone(b.prefill_input_ids_cpu)

    def test_resolve_decode_gather_untouched(self):
        b = self._resolve_batch(prefill_input_ids_cpu=None)
        pre_upload_forward_inputs(b)
        resolve_forward_inputs(b, _FakeFutureMap())
        self.assertEqual(b.input_ids.tolist(), [5, 6])


class TestInitNewConsumesStagedGlobals(CustomTestCase):
    def _forward_batch(self, spec_info=None):
        fb = ForwardBatch.__new__(ForwardBatch)
        fb.spec_info = spec_info
        return fb

    def test_non_spec_takes_staged_tensors_once(self):
        b = _batch(can_run_decode_cuda_graph=False)
        pre_upload_forward_inputs(b)
        staged = b.head_staged_global_num_tokens.tensor
        staged_logprob = b.head_staged_global_num_tokens_for_logprob.tensor

        fb = self._forward_batch()
        fb.init_mlp_sync_metadata(b, "cpu")
        self.assertIs(fb.global_num_tokens_gpu, staged)
        self.assertIs(fb.global_num_tokens_for_logprob_gpu, staged_logprob)
        self.assertEqual(fb.global_num_tokens_cpu, [4, 4, 4, 4])

        # A second init_new on the same batch gets a fresh upload, never the
        # tensor the first forward already owns.
        fb2 = self._forward_batch()
        fb2.init_mlp_sync_metadata(b, "cpu")
        self.assertIsNot(fb2.global_num_tokens_gpu, staged)
        self.assertEqual(fb2.global_num_tokens_gpu.tolist(), [4, 4, 4, 4])

    def test_regathered_counts_fall_back(self):
        b = _batch(can_run_decode_cuda_graph=False)
        pre_upload_forward_inputs(b)
        staged = b.head_staged_global_num_tokens.tensor
        b.global_num_tokens = [8, 8, 8, 8]  # DP step re-gathered the counts
        fb = self._forward_batch()
        fb.init_mlp_sync_metadata(b, "cpu")
        self.assertIsNot(fb.global_num_tokens_gpu, staged)
        self.assertEqual(fb.global_num_tokens_gpu.tolist(), [8, 8, 8, 8])

    def test_without_staging_uploads_as_before(self):
        b = _batch(can_run_decode_cuda_graph=False)
        fb = self._forward_batch()
        fb.init_mlp_sync_metadata(b, "cpu")
        self.assertEqual(fb.global_num_tokens_gpu.tolist(), [4, 4, 4, 4])
        self.assertEqual(fb.global_num_tokens_for_logprob_gpu.dtype, torch.int64)


class TestHandover(CustomTestCase):
    def _scheduler(self, calls, hicache=True, linker=False):
        return SimpleNamespace(
            enable_hierarchical_cache=hicache,
            enable_unified_cache_external_linker=linker,
            tree_cache=SimpleNamespace(
                ready_to_load_host_cache=lambda: calls.append("ready") or 7
            ),
        )

    def test_prefill_batch_pre_uploads_then_hands_over(self):
        calls = []
        b = _batch()
        Scheduler._handover_hicache_load(self._scheduler(calls), b)
        self.assertEqual(calls, ["ready"])
        self.assertEqual(b.hicache_consumer_index, 7)
        # Every round-head copy was issued before the hand-over.
        self.assertIsNone(b.prefill_input_ids_cpu)
        self.assertEqual(b.input_ids.tolist(), [1, 2, 3, 4])
        self.assertIsNotNone(b.head_staged_extend_seq_lens)
        self.assertIsNotNone(b.head_staged_global_num_tokens)

    def test_hand_over_order_is_upload_then_record(self):
        order = []
        b = _batch()
        b.prefill_input_ids_cpu = SimpleNamespace(
            to=lambda *a, **k: order.append("upload") or torch.tensor([1, 2, 3, 4])
        )
        sched = SimpleNamespace(
            enable_hierarchical_cache=True,
            enable_unified_cache_external_linker=False,
            tree_cache=SimpleNamespace(
                ready_to_load_host_cache=lambda: order.append("record") or 0
            ),
        )
        Scheduler._handover_hicache_load(sched, b)
        self.assertEqual(order, ["upload", "record"])

    def test_external_linker_alone_hands_over(self):
        calls = []
        b = _batch()
        Scheduler._handover_hicache_load(
            self._scheduler(calls, hicache=False, linker=True), b
        )
        self.assertEqual(calls, ["ready"])

    def test_disabled_hicache_is_a_noop(self):
        calls = []
        b = _batch()
        Scheduler._handover_hicache_load(self._scheduler(calls, hicache=False), b)
        self.assertEqual(calls, [])
        self.assertEqual(b.hicache_consumer_index, -1)
        # Nothing is pre-uploaded either: no burst, no ordering to enforce.
        self.assertIsNotNone(b.prefill_input_ids_cpu)
        self.assertIsNone(b.head_staged_extend_seq_lens)

    def test_non_prefill_batches_do_not_hand_over(self):
        for mode in (ForwardMode.DECODE, ForwardMode.IDLE, ForwardMode.TARGET_VERIFY):
            calls = []
            b = _batch(forward_mode=mode)
            Scheduler._handover_hicache_load(self._scheduler(calls), b)
            self.assertEqual(calls, [], mode)
            self.assertEqual(b.hicache_consumer_index, -1, mode)

    def test_all_prefill_modes_hand_over(self):
        for mode in (
            ForwardMode.EXTEND,
            ForwardMode.MIXED,
            ForwardMode.SPLIT_PREFILL,
            ForwardMode.DLLM_EXTEND,
        ):
            calls = []
            b = _batch(forward_mode=mode)
            Scheduler._handover_hicache_load(self._scheduler(calls), b)
            self.assertEqual(calls, ["ready"], mode)

    def test_already_handed_over_is_idempotent(self):
        calls = []
        b = _batch(hicache_consumer_index=3)
        Scheduler._handover_hicache_load(self._scheduler(calls), b)
        self.assertEqual(calls, [])
        self.assertEqual(b.hicache_consumer_index, 3)
        self.assertIsNotNone(b.prefill_input_ids_cpu)


if __name__ == "__main__":
    unittest.main()
