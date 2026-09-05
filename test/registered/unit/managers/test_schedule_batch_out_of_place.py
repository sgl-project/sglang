import dataclasses
import types
import unittest
from array import array
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import (  # noqa: E402
    ScheduleBatch,
    split_cached_prefix_by_tier,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402
from sglang.srt.utils.common import Range  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

AUTO_FILL_EXCLUDED_FIELDS = ["reqs"]


class TestCachedPrefixTierAttribution(unittest.TestCase):
    def test_split_cached_prefix_by_tier(self):
        cases = [
            (
                {"prefix_len": 100, "host_hit_len": 70, "storage_hit_len": 20},
                (30, 50, 20),
            ),
            (
                {
                    "prefix_len": 80,
                    "host_hit_len": 50,
                    "storage_hit_len": 20,
                    "storage_hit_start": 80,
                },
                (30, 50, 0),
            ),
            (
                {
                    "prefix_len": 90,
                    "host_hit_len": 60,
                    "storage_hit_len": 20,
                    "storage_hit_start": 80,
                },
                (30, 50, 10),
            ),
            (
                {
                    "prefix_len": 100,
                    "host_hit_len": 70,
                    "storage_hit_len": 70,
                    "host_hit_is_storage": True,
                },
                (30, 0, 70),
            ),
            (
                {
                    "prefix_len": 100,
                    "host_hit_len": 0,
                    "storage_hit_len": 20,
                    "storage_hit_start": 80,
                },
                (80, 0, 20),
            ),
            (
                {
                    "prefix_len": 100,
                    "host_hit_len": 0,
                    "storage_hit_len": 20,
                    "storage_hit_start": 80,
                    "host_hit_is_storage": True,
                },
                (80, 0, 20),
            ),
        ]
        for kwargs, expected in cases:
            with self.subTest(**kwargs):
                self.assertEqual(split_cached_prefix_by_tier(**kwargs), expected)


def make_schedule_batch(bs: int, **overrides) -> ScheduleBatch:
    batch = ScheduleBatch(reqs=overrides.pop("reqs"))
    # init_new always sets a SpeculativeAlgorithm enum, never None.
    batch.spec_algorithm = SpeculativeAlgorithm.NONE
    for field in dataclasses.fields(ScheduleBatch):
        name = field.name
        if name in overrides or name in AUTO_FILL_EXCLUDED_FIELDS:
            continue
        annotation = str(field.type)
        if "List" in annotation or "list[" in annotation:
            setattr(batch, name, [f"{name}-{i}" for i in range(bs)])
        elif "Tensor" in annotation:
            setattr(batch, name, torch.arange(bs, dtype=torch.int64))
    for name, value in overrides.items():
        setattr(batch, name, value)
    return batch


def _snapshot_mutable_fields(batch):
    snapshot = []
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name, None)
        if isinstance(value, MagicMock):
            continue
        if isinstance(value, list):
            snapshot.append((field.name, value, list(value)))
        elif isinstance(value, torch.Tensor):
            snapshot.append((field.name, value, value.clone()))
    return snapshot


def _assert_snapshot_not_mutated(test_case, snapshot):
    for name, obj, value_copy in snapshot:
        if isinstance(obj, list):
            test_case.assertEqual(obj, value_copy, f"field {name} was mutated in place")
        else:
            test_case.assertTrue(
                torch.equal(obj, value_copy), f"field {name} was mutated in place"
            )


class _FakeReq:
    def __init__(self, rid, origin_len, output_len):
        self.rid = rid
        self.origin_input_ids = list(range(origin_len))
        self.output_ids = list(range(output_len))
        self.full_untruncated_fill_ids = list(range(origin_len + output_len))
        self.extend_range = None

    def _refresh_fill_ids(self):
        self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)

    def set_extend_range(self, start, end):
        self.extend_range = Range(start, end)


class TestMergeBatchOutOfPlace(unittest.TestCase):
    def test_merge_batch_rebinds_lists_without_mutating_either_side(self):
        """merge_batch must build new list objects; no field of either side may be mutated in place."""
        self_batch = make_schedule_batch(
            2,
            reqs=[types.SimpleNamespace(rid="a"), types.SimpleNamespace(rid="b")],
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            sampling_info=MagicMock(),
            return_logprob=True,
            top_logprobs_nums=[1, 2],
            token_ids_logprobs=[[10], [20]],
        )
        other_batch = make_schedule_batch(
            1,
            reqs=[types.SimpleNamespace(rid="c")],
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            sampling_info=MagicMock(),
            return_logprob=True,
            top_logprobs_nums=[3],
            token_ids_logprobs=[[30]],
        )

        self_reqs_before = self_batch.reqs
        self_top_before = self_batch.top_logprobs_nums
        self_snapshot = _snapshot_mutable_fields(self_batch)
        other_snapshot = _snapshot_mutable_fields(other_batch)

        self_batch.merge_batch(other_batch)

        self.assertEqual([r.rid for r in self_batch.reqs], ["a", "b", "c"])
        self.assertEqual(self_batch.top_logprobs_nums, [1, 2, 3])
        self.assertEqual(self_batch.token_ids_logprobs, [[10], [20], [30]])
        self.assertIsNot(self_batch.reqs, self_reqs_before)
        self.assertIsNot(self_batch.top_logprobs_nums, self_top_before)
        _assert_snapshot_not_mutated(self, self_snapshot)
        _assert_snapshot_not_mutated(self, other_snapshot)


class TestMixWithRunningOutOfPlace(unittest.TestCase):
    def test_mix_with_running_rebinds_extend_fields_without_mutating_either_side(self):
        """mix_with_running must append via rebound lists; no field of either side may be mutated in place."""
        extend_batch = make_schedule_batch(
            2,
            reqs=[types.SimpleNamespace(rid="e1"), types.SimpleNamespace(rid="e2")],
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            sampling_info=MagicMock(),
            return_logprob=False,
            forward_mode=ForwardMode.EXTEND,
            enable_overlap=False,
            is_prefill_only=True,
            out_cache_loc=torch.arange(6, dtype=torch.int64),
            prefix_lens=[0, 0],
            extend_lens=[3, 3],
            extend_num_tokens=6,
            extend_logprob_start_lens=[0, 0],
        )
        running_batch = make_schedule_batch(
            1,
            reqs=[_FakeReq("r1", origin_len=4, output_len=2)],
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            sampling_info=MagicMock(),
            return_logprob=False,
            forward_mode=ForwardMode.DECODE,
            out_cache_loc=torch.arange(6, 7, dtype=torch.int64),
            # prepare_for_decode ran: the row already counts this step's token.
            seq_lens_cpu=torch.tensor([6], dtype=torch.int64),
        )

        extend_prefix_before = extend_batch.prefix_lens
        extend_lens_before = extend_batch.extend_lens
        extend_snapshot = _snapshot_mutable_fields(extend_batch)
        running_snapshot = _snapshot_mutable_fields(running_batch)

        extend_batch.mix_with_running(running_batch)

        self.assertEqual(extend_batch.forward_mode, ForwardMode.MIXED)
        self.assertIs(extend_batch.mix_running_indices, running_batch.req_pool_indices)
        self.assertEqual([r.rid for r in extend_batch.reqs], ["e1", "e2", "r1"])
        self.assertTrue(
            torch.equal(extend_batch.out_cache_loc, torch.arange(7, dtype=torch.int64))
        )
        # The decode tail's prefix is its row length minus this step's token.
        self.assertEqual(extend_batch.prefix_lens, [0, 0, 5])
        self.assertEqual(extend_batch.extend_lens, [3, 3, 1])
        self.assertEqual(extend_batch.extend_num_tokens, 7)
        self.assertEqual(extend_batch.extend_logprob_start_lens, [0, 0, 0])
        self.assertFalse(extend_batch.is_prefill_only)
        self.assertIsNot(extend_batch.prefix_lens, extend_prefix_before)
        self.assertIsNot(extend_batch.extend_lens, extend_lens_before)
        _assert_snapshot_not_mutated(self, extend_snapshot)
        _assert_snapshot_not_mutated(self, running_snapshot)


class TestPrepareEncoderInfoExtendOutOfPlace(unittest.TestCase):
    def test_incremental_encoder_cache_remaps_request_row(self):
        old_encoder_locs = torch.tensor([100, 101], dtype=torch.int64)
        old_decoder_locs = torch.tensor([200, 201, 202], dtype=torch.int64)
        fresh_encoder_locs = torch.tensor([300, 301], dtype=torch.int64)
        fresh_decoder_locs = torch.tensor([400, 401], dtype=torch.int64)
        prefix_indices = torch.cat([old_encoder_locs, old_decoder_locs])
        req_to_token = torch.full((1, 16), -1, dtype=torch.int64)
        req_to_token[0, :9] = torch.cat(
            [
                prefix_indices,
                fresh_encoder_locs,
                fresh_decoder_locs,
            ]
        )
        fill_ids = array("q", [90, 91, 92, 93, 10, 11, 12, 13, 14])
        mm_input = types.SimpleNamespace(
            num_image_tokens=4,
            incremental_encoder_cache=True,
            encoder_cached_len=2,
            encoder_append_len=2,
        )
        req = types.SimpleNamespace(
            rid="incremental",
            multimodal_inputs=mm_input,
            prefix_indices=prefix_indices,
            extend_range=Range(5, 9),
            logprob_start_len=0,
            incremental_encoder_cache_remapped=False,
            incremental_encoder_cache_prefix_indices=None,
            kv=types.SimpleNamespace(req_pool_idx=0),
            get_fill_ids=lambda: fill_ids,
        )
        batch = make_schedule_batch(
            1,
            reqs=[req],
            device="cpu",
            forward_mode=ForwardMode.EXTEND,
            out_cache_loc=torch.cat([fresh_encoder_locs, fresh_decoder_locs]),
            prefix_lens=[5],
            extend_lens=[4],
            extend_num_tokens=4,
            extend_logprob_start_lens=[0],
            multimodal_inputs=[mm_input],
            tree_cache=types.SimpleNamespace(page_size=1),
            token_to_kv_pool_allocator=types.SimpleNamespace(page_size=1),
            req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token),
        )

        batch.prepare_encoder_info_extend(
            input_ids=[array("q", fill_ids[5:])],
            seq_lens=[9],
        )

        self.assertTrue(
            torch.equal(
                req_to_token[0, :9],
                torch.cat(
                    [
                        old_encoder_locs,
                        fresh_encoder_locs,
                        old_decoder_locs,
                        fresh_decoder_locs,
                    ]
                ),
            )
        )
        self.assertTrue(torch.equal(batch.encoder_out_cache_loc, fresh_encoder_locs))
        self.assertTrue(torch.equal(batch.out_cache_loc, fresh_decoder_locs))
        self.assertEqual(batch.encoder_lens_cpu, [4])
        self.assertEqual(batch.encoder_cached, [False])
        self.assertEqual(batch.prefix_lens, [3])
        self.assertEqual(batch.extend_lens, [2])
        self.assertEqual(batch.extend_num_tokens, 2)
        self.assertTrue(
            torch.equal(batch.seq_lens_cpu, torch.tensor([5], dtype=torch.int64))
        )
        self.assertEqual(batch.prefill_input_ids_cpu.tolist(), [13, 14])
        self.assertEqual(req.extend_range, Range(7, 9))
        self.assertTrue(req.incremental_encoder_cache_remapped)
        self.assertTrue(
            torch.equal(
                req.incremental_encoder_cache_prefix_indices,
                prefix_indices,
            )
        )

    def test_incremental_cache_without_new_encoder_tokens_keeps_mixed_request(self):
        old_encoder_locs = torch.tensor([110, 111], dtype=torch.int64)
        old_decoder_locs = torch.tensor([210, 211], dtype=torch.int64)
        incremental_decoder_loc = torch.tensor([410], dtype=torch.int64)
        text_decoder_locs = torch.tensor([510, 511], dtype=torch.int64)
        incremental_prefix = torch.cat([old_encoder_locs, old_decoder_locs])
        req_to_token = torch.full((2, 12), -1, dtype=torch.int64)
        req_to_token[0, :5] = torch.cat([incremental_prefix, incremental_decoder_loc])
        req_to_token[1, :3] = torch.tensor([500, 510, 511], dtype=torch.int64)
        fill_ids = array("q", [90, 91, 20, 21, 22])
        mm_input = types.SimpleNamespace(
            num_image_tokens=2,
            incremental_encoder_cache=True,
            encoder_cached_len=2,
            encoder_append_len=0,
        )
        incremental_req = types.SimpleNamespace(
            rid="incremental",
            multimodal_inputs=mm_input,
            prefix_indices=incremental_prefix,
            extend_range=Range(4, 5),
            logprob_start_len=0,
            incremental_encoder_cache_remapped=False,
            incremental_encoder_cache_prefix_indices=None,
            kv=types.SimpleNamespace(req_pool_idx=0),
            get_fill_ids=lambda: fill_ids,
        )
        text_req = types.SimpleNamespace(
            rid="text",
            multimodal_inputs=None,
            prefix_indices=torch.tensor([500], dtype=torch.int64),
            extend_range=Range(1, 3),
            logprob_start_len=0,
        )
        batch = make_schedule_batch(
            2,
            reqs=[incremental_req, text_req],
            device="cpu",
            forward_mode=ForwardMode.EXTEND,
            out_cache_loc=torch.cat([incremental_decoder_loc, text_decoder_locs]),
            prefix_lens=[4, 1],
            extend_lens=[1, 2],
            extend_num_tokens=3,
            extend_logprob_start_lens=[0, 0],
            multimodal_inputs=[mm_input, None],
            tree_cache=types.SimpleNamespace(page_size=1),
            token_to_kv_pool_allocator=types.SimpleNamespace(page_size=1),
            req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token),
        )

        batch.prepare_encoder_info_extend(
            input_ids=[array("q", fill_ids[4:]), array("q", [31, 32])],
            seq_lens=[5, 3],
        )

        self.assertTrue(
            torch.equal(
                req_to_token[0, :5],
                torch.cat(
                    [old_encoder_locs, old_decoder_locs, incremental_decoder_loc]
                ),
            )
        )
        self.assertTrue(torch.equal(req_to_token[1, :3], torch.tensor([500, 510, 511])))
        self.assertEqual(batch.encoder_lens_cpu, [2, 0])
        self.assertEqual(batch.encoder_cached, [True, True])
        self.assertEqual(batch.prefix_lens, [2, 1])
        self.assertEqual(batch.extend_lens, [1, 2])
        self.assertEqual(batch.extend_num_tokens, 3)
        self.assertTrue(
            torch.equal(batch.seq_lens_cpu, torch.tensor([3, 3], dtype=torch.int64))
        )
        self.assertEqual(batch.prefill_input_ids_cpu.tolist(), [22, 31, 32])
        self.assertTrue(
            torch.equal(
                batch.out_cache_loc,
                torch.cat([incremental_decoder_loc, text_decoder_locs]),
            )
        )
        self.assertEqual(batch.encoder_out_cache_loc.numel(), 0)
        self.assertTrue(incremental_req.incremental_encoder_cache_remapped)
        self.assertEqual(incremental_req.extend_range, Range(4, 5))

    def test_prepare_encoder_info_extend_rebinds_lens_without_mutating_old_lists(self):
        """prepare_encoder_info_extend must strip encoder tokens via rebound lists; old list objects stay intact."""
        req_with_image = types.SimpleNamespace(
            rid="img",
            multimodal_inputs=types.SimpleNamespace(num_image_tokens=2),
            prefix_indices=[],
            extend_range=Range(0, 5),
            logprob_start_len=0,
        )
        req_text_only = types.SimpleNamespace(
            rid="txt",
            multimodal_inputs=None,
            prefix_indices=[],
            extend_range=Range(0, 4),
            logprob_start_len=0,
        )
        batch = make_schedule_batch(
            2,
            reqs=[req_with_image, req_text_only],
            device="cpu",
            forward_mode=ForwardMode.EXTEND,
            out_cache_loc=torch.arange(9, dtype=torch.int64),
            prefix_lens=[0, 0],
            extend_lens=[5, 4],
            extend_num_tokens=9,
            extend_logprob_start_lens=[0, 0],
            extend_input_logprob_token_ids=torch.arange(9, dtype=torch.int64),
        )

        prefix_before = batch.prefix_lens
        extend_before = batch.extend_lens
        logprob_start_before = batch.extend_logprob_start_lens
        snapshot = _snapshot_mutable_fields(batch)

        batch.prepare_encoder_info_extend(
            input_ids=[array("q", range(5)), array("q", range(4))],
            seq_lens=[5, 4],
        )

        self.assertEqual(batch.encoder_lens_cpu, [2, 0])
        self.assertEqual(batch.encoder_cached, [False, True])
        self.assertEqual(batch.extend_lens, [3, 4])
        self.assertEqual(batch.prefix_lens, [0, 0])
        self.assertEqual(batch.extend_num_tokens, 7)
        self.assertTrue(
            torch.equal(batch.out_cache_loc, torch.arange(2, 9, dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(batch.encoder_out_cache_loc, torch.arange(2, dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(batch.seq_lens_cpu, torch.tensor([3, 4], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                batch.extend_input_logprob_token_ids,
                torch.arange(2, 9, dtype=torch.int64),
            )
        )
        self.assertEqual(batch.extend_logprob_start_lens, [0, 0])
        self.assertIsNot(batch.prefix_lens, prefix_before)
        self.assertIsNot(batch.extend_lens, extend_before)
        self.assertIsNot(batch.extend_logprob_start_lens, logprob_start_before)
        _assert_snapshot_not_mutated(self, snapshot)


if __name__ == "__main__":
    unittest.main()
