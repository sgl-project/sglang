"""CPU contracts for shared field views and each preparer's distinct policies."""

import dataclasses
import unittest

import torch

from sglang.srt.afd.contracts import AFDError
from sglang.srt.afd.model_adapters.base import _slice_forward_batch, _token_lengths
from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_batch_view import slice_batch_field
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Review new ForwardBatch fields before admitting them to either preparer. This
# test-side inventory does not add schema introspection to the forward hot path.
# "inherit" records the current shallow-copy behavior, not support for new model
# families (e.g. multimodal or Mamba) or speculative modes.
_AFD_FIELD_POLICY = {
    "token_view": "input_ids out_cache_loc input_embeds token_type_ids",
    "request_view": """
        req_pool_indices req_pool_indices_cpu seq_lens seq_lens_cpu extend_seq_lens extend_prefix_lens
        extend_seq_lens_cpu extend_prefix_lens_cpu extend_logprob_start_lens_cpu
        lora_ids rids
    """,
    "positions_view": "positions",
    "derived_or_reset": """
        batch_size seq_lens_sum extend_num_tokens extend_start_loc
        forward_metadata_ready forward_metadata_planned_bs
        forward_metadata_planned_num_tokens
    """,
    "inherit": """
        forward_mode orig_seq_lens out_cache_loc_virtual out_cache_loc_dsv4
        mamba_track_indices mamba_track_mask mamba_track_seqlens
        mamba_cow_src_indices mamba_cow_dst_indices mamba_clear_indices
        replace_embeds replace_positions encoder_lens encoder_out_cache_loc
        return_logprob is_prefill_only spec_algorithm dimensions
        return_pooled_hidden_states is_extend_in_batch can_run_decode_cuda_graph
        can_run_dp_prefill_cuda_graph dp_prefill_cuda_graph_max_prefix_len
        global_forward_mode tbo_split_seq_index top_logprobs_nums token_ids_logprobs
        mm_inputs encoder_cached encoder_lens_cpu multi_item_delimiter_indices
        sampling_info spec_info capture_hidden_mode return_hidden_states_before_norm
        reuse_dsa_topk_indices minimax_m3_precached_sparse_layers
        extend_input_logprob_token_ids_gpu original_global_num_tokens_cpu
        _original_batch_size _original_forward_mode _original_num_tokens
        global_num_tokens_cpu global_num_tokens_gpu global_num_tokens_for_logprob_cpu
        global_num_tokens_for_logprob_gpu global_num_token_non_padded
        global_num_token_non_padded_cpu num_token_non_padded attn_tp_sequence_sharded
        _attn_output next_token_logits_buffer temperature top_p hidden_states residual
        model_specific_states split_index mm_input_embeds cross_attention_custom_mask
        dp_padding_mode dp_local_start_pos dp_local_num_tokens global_dp_buffer_len
        mrope_positions tbo_parent_token_range tbo_padded_len tbo_children
        attn_cp_metadata attn_dcp_metadata dcp_kv_mask ngram_embedding_info rids_int
        bootstrap_room_ids_int req_all_ids_flat req_all_ids_lens
        forward_metadata_replan_equivalent max_seq_len_override
        mamba_track_seqlens_cpu mamba_prefill_track_mask_cpu
        dp_spec_prefill_coordination_applied global_num_tokens_padded_cpu
        multi_gate_indices engram_history mm_token_modalities token_indices_to_pool
        encoder_swa_replay defer_logits_to_eager
    """,
}


def _batch(mode=ForwardMode.DECODE):
    extending = mode == ForwardMode.EXTEND
    rows, requests = (6, 3) if extending else (4, 4)
    lengths = torch.arange(7, 7 + requests, dtype=torch.int32)
    batch = ForwardBatch(
        forward_mode=mode,
        batch_size=requests,
        input_ids=torch.arange(rows, dtype=torch.int64),
        positions=torch.arange(rows, dtype=torch.int64) + 20,
        out_cache_loc=torch.arange(rows, dtype=torch.int64) + 100,
        req_pool_indices=torch.arange(requests, dtype=torch.int32) + 10,
        req_pool_indices_cpu=torch.arange(requests, dtype=torch.int32) + 10,
        seq_lens=lengths,
        seq_lens_cpu=lengths.clone(),
        seq_lens_sum=int(lengths.sum()),
        lora_ids=[f"lora-{i}" for i in range(requests)],
        rids=[f"request-{i}" for i in range(requests)],
    )
    if extending:
        batch.extend_num_tokens = rows
        batch.extend_seq_lens_cpu = [2, 1, 3]
        batch.extend_prefix_lens_cpu = [5, 7, 6]
        batch.extend_logprob_start_lens_cpu = [0, 1, 2]
        batch.extend_seq_lens = torch.tensor([2, 1, 3], dtype=torch.int32)
        batch.extend_prefix_lens = torch.tensor([5, 7, 6], dtype=torch.int32)
        batch.extend_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    return batch


def _afd(batch, request_slice=slice(1, 3), token_slice=slice(1, 3)):
    return _slice_forward_batch(
        forward_batch=batch, request_slice=request_slice, token_slice=token_slice
    )


def _tbo(batch, request_slice=slice(1, 3), token_slice=slice(1, 3), real_rows=2):
    with (
        get_context().override_server_args(
            attention_backend="fa3", moe_dense_tp_size=1
        ),
        get_parallel().override(attn_tp_size=1),
    ):
        return TboForwardBatchPreparer.filter_batch(
            batch,
            start_seq_index=request_slice.start,
            end_seq_index=request_slice.stop,
            start_token_index=token_slice.start,
            end_token_index=token_slice.stop,
            out_num_token_non_padded=torch.tensor(real_rows, dtype=torch.int32),
            out_num_token_non_padded_cpu=real_rows,
        )


class TestForwardBatchFieldViews(CustomTestCase):
    def assert_view(self, actual, expected, parent):
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, parent.dtype)
        self.assertEqual(actual.device, parent.device)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(
            actual.untyped_storage().data_ptr(), parent.untyped_storage().data_ptr()
        )

    def test_forward_batch_schema_has_an_explicit_afd_field_policy(self):
        classified = [
            name for names in _AFD_FIELD_POLICY.values() for name in names.split()
        ]
        self.assertEqual(len(classified), len(set(classified)))
        self.assertEqual(
            set(classified),
            {field.name for field in dataclasses.fields(ForwardBatch)},
            "Review slicing and inheritance in both AFD and TBO for schema changes",
        )
        parent = _batch()
        child = _afd(parent)
        for name in _AFD_FIELD_POLICY["inherit"].split():
            self.assertIs(getattr(child, name), getattr(parent, name), name)

    def test_tensor_views_keep_storage_and_host_lists_keep_slice_semantics(self):
        batch = _batch()
        batch.input_embeds = torch.arange(32, dtype=torch.float32).reshape(4, 8)[:, ::2]
        for start, stop in [(0, 4), (1, 3), (4, 4)]:
            with self.subTest(start=start, stop=stop):
                selection = slice(start, stop)
                for name in (
                    "input_ids",
                    "input_embeds",
                    "req_pool_indices",
                    "req_pool_indices_cpu",
                    "seq_lens_cpu",
                ):
                    parent = getattr(batch, name)
                    view = slice_batch_field(parent, selection)
                    self.assert_view(view, parent[selection], parent)
                self.assertIsNone(slice_batch_field(batch.token_type_ids, selection))
                rids = slice_batch_field(batch.rids, selection)
                self.assertEqual(rids, batch.rids[selection])
                self.assertIsNot(rids, batch.rids)
        view = slice_batch_field(batch.input_ids, slice(1, 3))
        view[0] = 99
        self.assertEqual(batch.input_ids[1], 99)

    def test_declared_missing_field_is_not_silently_ignored(self):
        parent = _batch()
        del parent.input_ids
        for prepare in (_afd, _tbo):
            with self.assertRaises(AttributeError):
                prepare(parent)

    def test_both_preparers_preserve_common_views(self):
        for mode in (ForwardMode.DECODE, ForwardMode.IDLE, ForwardMode.EXTEND):
            with self.subTest(mode=mode):
                parent = _batch(mode)
                req_slice = slice(1, 3)
                tok_slice = slice(2, 6) if mode == ForwardMode.EXTEND else req_slice
                for prepare in (_afd, _tbo):
                    child = prepare(parent, req_slice, tok_slice)
                    self.assertIsNot(child, parent)
                    self.assertEqual(child.batch_size, 2)
                    for name in ("input_ids", "positions", "out_cache_loc"):
                        value = getattr(parent, name)
                        self.assert_view(getattr(child, name), value[tok_slice], value)
                    for name in ("req_pool_indices", "seq_lens", "seq_lens_cpu"):
                        value = getattr(parent, name)
                        self.assert_view(getattr(child, name), value[req_slice], value)
                    self.assertEqual(child.rids, parent.rids[req_slice])
                    self.assertEqual(
                        child.seq_lens_sum, parent.seq_lens_cpu[req_slice].sum()
                    )
                self.assertEqual(parent.batch_size, len(parent.seq_lens))

    def test_empty_tail_preserves_empty_tensor_views(self):
        parent = _batch()
        for prepare in (_afd, _tbo):
            child = prepare(parent, slice(4, 4), slice(4, 4))
            self.assertEqual(child.batch_size, 0)
            self.assert_view(child.input_ids, parent.input_ids[4:4], parent.input_ids)
            self.assertEqual(child.seq_lens_sum, 0)

    def test_extend_offsets_and_decode_token_counts_remain_caller_owned(self):
        parent = _batch(ForwardMode.EXTEND)
        afd = _afd(parent, slice(1, 3), slice(2, 6))
        tbo = _tbo(parent, slice(1, 3), slice(2, 6))
        self.assertEqual(afd.extend_start_loc.tolist(), [0, 1])
        self.assertNotEqual(
            afd.extend_start_loc.data_ptr(), parent.extend_start_loc.data_ptr()
        )
        self.assert_view(
            tbo.extend_start_loc, parent.extend_start_loc[1:3], parent.extend_start_loc
        )
        self.assertEqual(afd.extend_num_tokens, 4)
        self.assertEqual(tbo.extend_num_tokens, 4)
        for mode in (ForwardMode.DECODE, ForwardMode.IDLE):
            with self.subTest(mode=mode):
                parent = _batch(mode)
                self.assertEqual(_afd(parent).extend_num_tokens, 2)
                self.assertIsNone(_tbo(parent).extend_num_tokens)

    def test_virtual_cache_and_mrope_are_not_new_afd_support(self):
        parent = _batch()
        parent.out_cache_loc_virtual = torch.arange(4) + 200
        parent.mrope_positions = torch.arange(12).reshape(3, 4)
        afd, tbo = _afd(parent), _tbo(parent)
        self.assertIs(afd.out_cache_loc_virtual, parent.out_cache_loc_virtual)
        self.assertIs(afd.mrope_positions, parent.mrope_positions)
        self.assert_view(
            tbo.out_cache_loc_virtual,
            parent.out_cache_loc_virtual[1:3],
            parent.out_cache_loc_virtual,
        )
        self.assert_view(
            tbo.mrope_positions, parent.mrope_positions[:, 1:3], parent.mrope_positions
        )

    def test_afd_positions_use_last_axis_without_expanding_tbo_layout(self):
        parent = _batch()
        parent.positions = torch.arange(12).reshape(3, 4)
        child = _afd(parent)
        self.assert_view(child.positions, parent.positions[:, 1:3], parent.positions)
        with self.assertRaises(AssertionError):
            _tbo(parent)

    def test_dp_counts_masks_and_planning_state_remain_caller_owned(self):
        parent = _batch()
        parent.global_num_tokens_cpu = [4, 2]
        parent.global_num_tokens_gpu = torch.tensor([4, 2])
        parent.original_global_num_tokens_cpu = [3, 2]
        parent.global_num_tokens_for_logprob_cpu = [4, 2]
        parent.global_num_tokens_for_logprob_gpu = torch.tensor([4, 2])
        parent.global_dp_buffer_len = 6
        parent.num_token_non_padded = torch.tensor(0, dtype=torch.int32)
        parent.global_num_token_non_padded = torch.tensor(0, dtype=torch.int32)
        parent.global_num_token_non_padded_cpu = 0
        parent.mark_forward_metadata_ready(replan_equivalent=True)
        afd, tbo = _afd(parent), _tbo(parent, real_rows=0)
        for name in (
            "global_num_tokens_cpu",
            "global_num_tokens_gpu",
            "original_global_num_tokens_cpu",
            "global_num_tokens_for_logprob_cpu",
            "global_num_tokens_for_logprob_gpu",
        ):
            self.assertIs(getattr(afd, name), getattr(parent, name))
            self.assertIsNone(getattr(tbo, name))
        self.assertEqual(afd.global_dp_buffer_len, 6)
        self.assertEqual(tbo.global_dp_buffer_len, 2)
        self.assertIs(afd.num_token_non_padded, parent.num_token_non_padded)
        self.assertEqual(tbo.num_token_non_padded.item(), 0)
        self.assertEqual(tbo.global_num_token_non_padded_cpu, 0)
        self.assertIsNone(tbo.global_num_token_non_padded)
        for child in (afd, tbo):
            self.assertFalse(child.forward_metadata_ready)
            self.assertIsNone(child.forward_metadata_planned_bs)
            self.assertIsNone(child.forward_metadata_planned_num_tokens)
        self.assertTrue(afd.forward_metadata_replan_equivalent)
        self.assertFalse(tbo.forward_metadata_replan_equivalent)
        self.assertTrue(parent.forward_metadata_ready)

    def test_afd_embedding_fields_do_not_bypass_tbo_completeness_guard(self):
        for name in ("input_embeds", "token_type_ids"):
            with self.subTest(field=name):
                parent = _batch()
                value = torch.arange(8).reshape(4, 2)
                setattr(parent, name, value)
                self.assert_view(getattr(_afd(parent), name), value[1:3], value)
                with self.assertRaisesRegex(Exception, f"Field {name} has value"):
                    _tbo(parent)

    def test_tbo_still_rejects_unclassified_fields(self):
        parent = _batch()
        parent.dimensions = [128]
        with self.assertRaisesRegex(Exception, "Field dimensions has value"):
            _tbo(parent)

    def test_tbo_retains_input_shape_and_extend_count_checks(self):
        for name in ("input_ids", "positions", "out_cache_loc", "seq_lens", "lora_ids"):
            with self.subTest(field=name):
                parent = _batch()
                setattr(parent, name, getattr(parent, name)[:1])
                with self.assertRaises(AssertionError):
                    _tbo(parent)
        parent = _batch(ForwardMode.EXTEND)
        parent.extend_num_tokens = 99
        with self.assertRaises(AssertionError):
            _tbo(parent, slice(1, 3), slice(2, 6))
        with self.assertRaises(AssertionError):
            _tbo(_batch(), slice(1, 3), slice(3, 1))
        parent = _batch()
        parent.positions = None
        with self.assertRaises(AttributeError):
            _tbo(parent)

    def test_tbo_rids_exception_remains_available(self):
        parent = _batch()
        parent.rids = ["real-0", "real-1"]
        self.assertEqual(_tbo(parent).rids, ["real-1"])

    def test_target_verify_keeps_native_spec_slicing_and_afd_rejection(self):
        parent = _batch(ForwardMode.TARGET_VERIFY)
        parent.input_ids = torch.arange(8)
        parent.positions = torch.arange(8) + 10
        parent.out_cache_loc = torch.arange(8) + 100
        parent.extend_seq_lens = torch.arange(3)  # ignored in target verify
        parent.extend_prefix_lens = torch.arange(3)
        parent.extend_start_loc = torch.arange(3)
        parent.extend_seq_lens_cpu = [1, 2, 3]
        parent.extend_prefix_lens_cpu = [1, 2, 3]
        parent.extend_logprob_start_lens_cpu = [1, 2, 3]
        spec_lens = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        parent.spec_info = EagleVerifyInput(
            draft_token=torch.arange(8),
            custom_mask=torch.arange(36) % 2 == 0,
            positions=torch.arange(8) + 10,
            retrieve_index=torch.arange(8).reshape(4, 2),
            retrieve_next_token=torch.arange(8).reshape(4, 2) + 20,
            retrieve_next_sibling=torch.arange(8).reshape(4, 2) + 40,
            retrieve_cum_len=torch.arange(4),
            spec_steps=1,
            topk=1,
            draft_token_num=2,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=int(spec_lens.sum()),
            seq_lens_cpu=spec_lens,
        )
        child = _tbo(parent, slice(1, 3), slice(2, 6), real_rows=4)
        self.assert_view(
            child.spec_info.draft_token,
            parent.spec_info.draft_token[2:6],
            parent.spec_info.draft_token,
        )
        self.assert_view(
            child.spec_info.retrieve_index,
            parent.spec_info.retrieve_index[1:3],
            parent.spec_info.retrieve_index,
        )
        self.assert_view(
            child.spec_info.custom_mask,
            parent.spec_info.custom_mask[6:24],
            parent.spec_info.custom_mask,
        )
        self.assertEqual(child.spec_info.seq_lens_sum, 5)
        for name in (
            "extend_seq_lens",
            "extend_prefix_lens",
            "extend_start_loc",
            "extend_seq_lens_cpu",
            "extend_prefix_lens_cpu",
            "extend_logprob_start_lens_cpu",
        ):
            self.assertIsNone(getattr(child, name))
        self.assertIsNone(child.extend_num_tokens)
        with self.assertRaisesRegex(AFDError, "AFD_MTP_SPECULATIVE_UNSUPPORTED"):
            _token_lengths(parent)


if __name__ == "__main__":
    unittest.main()
