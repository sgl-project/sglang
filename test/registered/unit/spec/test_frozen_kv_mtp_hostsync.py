"""Unit tests for Frozen-KV MTP spec_v2 host-sync removal when
needs_cpu_seq_lens is False (seq_lens_cpu=None, seq_lens_sum=None)."""

import contextlib
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.verify_mask import VerifyMask
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode
from sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner import (
    FrozenKVMTPCudaGraphRunner,
)
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPDraftInput
from sglang.srt.speculative.frozen_kv_mtp_utils import (
    expand_for_topk_draft,
    position_for_batch,
    set_frozen_kv_positions,
)
from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import FrozenKVMTPDraftWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@contextlib.contextmanager
def _forbid_tensor_item():
    with patch.object(
        torch.Tensor,
        "item",
        side_effect=AssertionError("Unexpected blocking Tensor.item() host sync"),
    ):
        yield


class TestExpandForTopkDraftNoHostSync(CustomTestCase):
    def _make_batch(self, seq_lens_cpu, seq_lens_sum):
        seq_lens = torch.tensor([10, 20], dtype=torch.int32)
        return ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=2,
            input_ids=None,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=None,
            seq_lens_sum=seq_lens_sum,
            positions=torch.tensor([9, 19], dtype=torch.int64),
            spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP,
            spec_info=None,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

    def test_gpu_only_batch_skips_item_sync(self):
        fb = self._make_batch(seq_lens_cpu=None, seq_lens_sum=None)
        with _forbid_tensor_item():
            expand_for_topk_draft(fb, topk=3)
        self.assertEqual(fb.batch_size, 6)
        self.assertIsNone(fb.seq_lens_cpu)
        self.assertIsNone(fb.seq_lens_sum)
        self.assertEqual(fb.seq_lens.tolist(), [10, 10, 10, 20, 20, 20])

    def test_prepopulated_seq_lens_sum_scales_without_item_sync(self):
        fb = self._make_batch(seq_lens_cpu=None, seq_lens_sum=30)
        with _forbid_tensor_item():
            expand_for_topk_draft(fb, topk=3)
        self.assertEqual(fb.seq_lens_sum, 90)

    def test_cpu_seq_lens_updates_sum_from_cpu_tensor(self):
        fb = self._make_batch(
            seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_sum=30,
        )
        expand_for_topk_draft(fb, topk=3)
        self.assertEqual(fb.seq_lens_cpu.tolist(), [10, 10, 10, 20, 20, 20])
        self.assertEqual(fb.seq_lens_sum, 90)


class TestInitFrozenKVMetadataNoHostSync(CustomTestCase):
    def _make_worker(self, needs_cpu_seq_lens: bool):
        backend = SimpleNamespace(
            needs_cpu_seq_lens=needs_cpu_seq_lens,
            init_forward_metadata=lambda fb: None,
        )
        worker = SimpleNamespace(
            draft_attn_backend=backend,
            _frozen_kv_target_view=lambda fb: contextlib.nullcontext(),
        )
        worker._init_frozen_kv_metadata = MethodType(
            FrozenKVMTPDraftWorker._init_frozen_kv_metadata, worker
        )
        return worker

    def _make_batch(self, seq_lens_cpu=None, seq_lens_sum=None):
        return ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=2,
            input_ids=None,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([12, 18], dtype=torch.int32),
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=None,
            seq_lens_sum=seq_lens_sum,
            positions=torch.tensor([11, 17], dtype=torch.int64),
            spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP,
            spec_info=None,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

    def test_sync_free_backend_leaves_seq_lens_sum_none(self):
        worker = self._make_worker(needs_cpu_seq_lens=False)
        fb = self._make_batch(seq_lens_cpu=None, seq_lens_sum=None)
        with _forbid_tensor_item():
            worker._init_frozen_kv_metadata(fb)
        self.assertIsNone(fb.seq_lens_sum)
        self.assertTrue(fb.forward_metadata_ready)

    def test_cpu_requiring_backend_falls_back_when_seq_lens_cpu_missing(self):
        worker = self._make_worker(needs_cpu_seq_lens=True)
        fb = self._make_batch(seq_lens_cpu=None, seq_lens_sum=None)
        worker._init_frozen_kv_metadata(fb)
        self.assertEqual(fb.seq_lens_sum, 30)
        self.assertTrue(fb.forward_metadata_ready)


class TestFrozenKVMTPCudaGraphRunnerExecuteNoHostSync(CustomTestCase):
    def _make_runner(self, topk: int = 2, max_bs: int = 4):
        recorded = {}

        def _record_replay(fb, expanded_bs, seq_lens_sum):
            recorded["expanded_bs"] = expanded_bs
            recorded["seq_lens_sum"] = seq_lens_sum

        runner = object.__new__(FrozenKVMTPCudaGraphRunner)
        runner.topk = topk
        runner.captured_req_width = topk
        runner.capture_bs = [2, max_bs]
        runner.max_bs = max_bs
        runner.seq_len_fill_value = 1
        runner.require_mlp_tp_gather = False
        runner.require_gathered_buffer = False
        runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
        runner._make_graph_key = lambda bs: bs
        runner.model_runner = SimpleNamespace(device_timer=None)
        runner.frozen_kv_mtp_worker = SimpleNamespace(
            _init_frozen_kv_metadata_replay_cuda_graph=_record_replay
        )
        max_expanded = max_bs * topk
        runner.buffers = SimpleNamespace(
            req_pool_indices=torch.zeros((max_expanded,), dtype=torch.int32),
            seq_lens=torch.ones((max_expanded,), dtype=torch.int32),
            seq_lens_cpu=torch.ones((max_expanded,), dtype=torch.int32),
            positions=torch.zeros((max_expanded,), dtype=torch.int64),
            mrope_positions=torch.zeros((3, max_expanded), dtype=torch.int64),
            hidden_states=torch.zeros((max_bs, 8), dtype=torch.float32),
            bonus_tokens=torch.zeros((max_bs,), dtype=torch.int64),
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_gpu=None,
        )
        runner._replay_graph = lambda shape_key, fb: (
            torch.zeros((max_bs, 1), dtype=torch.int64),
            torch.zeros((max_bs, 2), dtype=torch.int64),
            torch.zeros((max_bs, 2), dtype=torch.int64),
        )
        return runner, recorded

    def test_execute_with_none_seq_lens_sum_passes_none(self):
        runner, recorded = self._make_runner(topk=2, max_bs=4)
        spec_info = FrozenKVMTPDraftInput(
            hidden_states=torch.zeros((3, 8), dtype=torch.float32),
            bonus_tokens=torch.tensor([1, 2, 3], dtype=torch.int64),
            capture_hidden_mode=CaptureHiddenMode.LAST,
            num_tokens_per_req=2,
            num_tokens_for_logprob_per_req=2,
        )
        # Pre-expanded batch for raw_bs=3, topk=2 -> raw_expanded_bs=6, padded to bs=4 (expanded_bs=8)
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=6,
            input_ids=None,
            req_pool_indices=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32),
            seq_lens=torch.tensor([5, 5, 6, 6, 7, 7], dtype=torch.int32),
            seq_lens_cpu=None,
            out_cache_loc=None,
            seq_lens_sum=None,
            positions=torch.tensor([4, 4, 5, 5, 6, 6], dtype=torch.int64),
            spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        with _forbid_tensor_item():
            out = runner.execute(fb)
        self.assertEqual(recorded["expanded_bs"], 8)
        self.assertIsNone(recorded["seq_lens_sum"])
        self.assertEqual(out[0].shape[0], 3)


class TestFrozenKVMTPDraftNoHostSync(CustomTestCase):
    def _run_draft(self, verify_mask):
        worker = SimpleNamespace(
            topk=2,
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            draft_model_runner=SimpleNamespace(),
            cuda_graph_runner=None,
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    attn_backend=SimpleNamespace(
                        verify_mask=verify_mask,
                        max_context_len=1024,
                    )
                )
            ),
            _position_for_batch=position_for_batch,
        )
        worker._set_positions = lambda fb: set_frozen_kv_positions(fb, worker.topk)
        worker._expand_for_topk_draft = lambda fb: expand_for_topk_draft(
            fb, worker.topk
        )
        worker.draft_forward = lambda fb: (
            torch.zeros((2, 1), dtype=torch.int64),
            torch.zeros((2, 2), dtype=torch.int64),
            torch.zeros((2, 2), dtype=torch.int64),
        )
        worker.draft = MethodType(FrozenKVMTPDraftWorker.draft, worker)

        spec_info = FrozenKVMTPDraftInput(
            hidden_states=torch.zeros((2, 8), dtype=torch.float32),
            bonus_tokens=torch.tensor([4, 5], dtype=torch.int64),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            spec_info=spec_info,
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_cpu=None,
            seq_lens_sum=None,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            return_hidden_states=False,
        )

        def _fake_init_new(b, runner, return_hidden_states_before_norm=False):
            return ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=b.seq_lens.shape[0],
                input_ids=None,
                req_pool_indices=b.req_pool_indices.clone(),
                seq_lens=b.seq_lens.clone(),
                seq_lens_cpu=b.seq_lens_cpu,
                out_cache_loc=None,
                seq_lens_sum=b.seq_lens_sum,
                positions=b.spec_info.positions.clone(),
                spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP,
                spec_info=b.spec_info,
                capture_hidden_mode=b.spec_info.capture_hidden_mode,
            )

        captured_tree_args = {}

        def _fake_build_tree(
            bonus_tokens,
            parent_list,
            top_scores_index,
            draft_tokens,
            seq_lens,
            seq_lens_sum,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode=TreeMaskMode.FULL_MASK,
            tree_mask_buf=None,
            fill_prefix_mask=True,
        ):
            captured_tree_args["seq_lens_sum"] = seq_lens_sum
            captured_tree_args["tree_mask_mode"] = tree_mask_mode
            captured_tree_args["tree_mask_buf"] = tree_mask_buf
            captured_tree_args["fill_prefix_mask"] = fill_prefix_mask
            return (
                torch.ones((18,), dtype=torch.bool),
                torch.zeros((6,), dtype=torch.int64),
                torch.zeros((2, 3), dtype=torch.int64),
                torch.zeros((2, 3), dtype=torch.int64),
                torch.zeros((2, 3), dtype=torch.int64),
                torch.zeros((6,), dtype=torch.int64),
            )

        with (
            patch.object(ForwardBatch, "init_new", side_effect=_fake_init_new),
            patch(
                "sglang.srt.speculative.frozen_kv_mtp_worker_v2.build_tree_kernel_efficient",
                side_effect=_fake_build_tree,
            ),
            _forbid_tensor_item(),
        ):
            verify_input = worker.draft(batch)

        return verify_input, captured_tree_args

    def test_draft_with_verify_mask_buffer_uses_zero_seq_lens_sum(self):
        buf = torch.zeros((36,), dtype=torch.bool)
        vm = VerifyMask(
            buffer=buf,
            mode=TreeMaskMode.QLEN_ONLY,
            max_bs=4,
            is_read=False,
        )
        verify_input, tree_args = self._run_draft(verify_mask=vm)
        self.assertIsNone(verify_input.seq_lens_sum)
        self.assertIsNone(verify_input.seq_lens_cpu)
        self.assertEqual(tree_args["seq_lens_sum"], 0)
        self.assertEqual(tree_args["tree_mask_mode"], TreeMaskMode.QLEN_ONLY)
        self.assertIs(tree_args["tree_mask_buf"], buf)
        self.assertFalse(tree_args["fill_prefix_mask"])

    def test_draft_without_verify_mask_uses_context_len_upper_bound(self):
        verify_input, tree_args = self._run_draft(verify_mask=None)
        self.assertIsNone(verify_input.seq_lens_sum)
        self.assertEqual(tree_args["seq_lens_sum"], 2 * 1024)
        self.assertIsNone(tree_args["tree_mask_buf"])
        self.assertTrue(tree_args["fill_prefix_mask"])


if __name__ == "__main__":
    unittest.main()
