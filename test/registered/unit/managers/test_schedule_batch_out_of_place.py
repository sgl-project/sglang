import dataclasses
import types
import unittest
from array import array
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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

    def set_extend_range(self, start, end):
        self.extend_range = types.SimpleNamespace(
            start=start, end=end, length=end - start
        )


def _make_batch(req_names, top_logprobs_nums, token_ids_logprobs):
    reqs = [types.SimpleNamespace(rid=name) for name in req_names]
    bs = len(reqs)
    batch = ScheduleBatch(reqs=reqs)
    batch.model_config = types.SimpleNamespace(is_encoder_decoder=False)
    batch.sampling_info = MagicMock()
    batch.req_pool_indices = torch.arange(bs, dtype=torch.int64)
    batch.req_pool_indices_cpu = batch.req_pool_indices.clone()
    batch.seq_lens = torch.full((bs,), 8, dtype=torch.int64)
    batch.seq_lens_cpu = batch.seq_lens.clone()
    batch.orig_seq_lens = batch.seq_lens.to(torch.int32)
    batch.input_ids = torch.arange(bs, dtype=torch.int64)
    batch.return_logprob = True
    batch.top_logprobs_nums = top_logprobs_nums
    batch.token_ids_logprobs = token_ids_logprobs
    batch.multimodal_inputs = [None] * bs
    batch.spec_info = None
    return batch


class TestMergeBatchOutOfPlace(unittest.TestCase):
    def test_merge_batch_rebinds_lists_without_mutating_either_side(self):
        """merge_batch must build new list objects; no field of either side may be mutated in place."""
        self_batch = _make_batch(["a", "b"], [1, 2], [[10], [20]])
        other_batch = _make_batch(["c"], [3], [[30]])

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
        extend_batch = _make_batch(["e1", "e2"], None, None)
        extend_batch.return_logprob = False
        extend_batch.forward_mode = ForwardMode.EXTEND
        extend_batch.enable_overlap = False
        extend_batch.is_prefill_only = True
        extend_batch.out_cache_loc = torch.arange(6, dtype=torch.int64)
        extend_batch.prefix_lens = [0, 0]
        extend_batch.extend_lens = [3, 3]
        extend_batch.extend_num_tokens = 6
        extend_batch.extend_logprob_start_lens = [0, 0]

        running_batch = _make_batch(["r1"], None, None)
        running_batch.reqs = [_FakeReq("r1", origin_len=4, output_len=2)]
        running_batch.return_logprob = False
        running_batch.forward_mode = ForwardMode.DECODE
        running_batch.out_cache_loc = torch.arange(6, 7, dtype=torch.int64)

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
        # delta is -1 without overlap: 4 origin + 2 output - 1
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
    def test_prepare_encoder_info_extend_rebinds_lens_without_mutating_old_lists(self):
        """prepare_encoder_info_extend must strip encoder tokens via rebound lists; old list objects stay intact."""
        req_with_image = types.SimpleNamespace(
            rid="img",
            multimodal_inputs=types.SimpleNamespace(num_image_tokens=2),
            prefix_indices=[],
            extend_range=types.SimpleNamespace(length=5),
        )
        req_text_only = types.SimpleNamespace(
            rid="txt",
            multimodal_inputs=None,
            prefix_indices=[],
            extend_range=types.SimpleNamespace(length=4),
        )
        batch = ScheduleBatch(reqs=[req_with_image, req_text_only])
        batch.device = "cpu"
        batch.forward_mode = ForwardMode.EXTEND
        batch.out_cache_loc = torch.arange(9, dtype=torch.int64)
        batch.prefix_lens = [0, 0]
        batch.extend_lens = [5, 4]
        batch.extend_num_tokens = 9
        batch.extend_logprob_start_lens = [0, 0]
        batch.extend_input_logprob_token_ids = torch.arange(9, dtype=torch.int64)

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
