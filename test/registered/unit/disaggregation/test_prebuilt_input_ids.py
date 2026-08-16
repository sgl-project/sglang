import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode_schedule_batch_mixin import (  # noqa: E402
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _PrebuiltReq:
    def __init__(
        self,
        *,
        req_pool_idx: int,
        fill_ids: list[int],
        prefix_len: int,
        extend_len: int,
        output_ids: list[int],
        already_computed: int = 0,
    ):
        self.req_pool_idx = req_pool_idx
        self._fill_ids = array("q", fill_ids)
        self.prefix_indices = torch.arange(prefix_len, dtype=torch.int64)
        self.extend_range = SimpleNamespace(length=extend_len)
        self.origin_input_ids = fill_ids[: max(0, len(fill_ids) - len(output_ids))]
        self.output_ids = output_ids
        self.retracted_stain = False
        self.already_computed = already_computed
        self.cached_tokens = 11
        self.cached_tokens_device = 7
        self.is_retracted = True
        self.pd_rebootstrap_in_progress = False
        self.multimodal_inputs = None
        self.grammar = None

    def get_fill_ids(self):
        return self._fill_ids


def _make_batch(reqs):
    req_to_token = torch.arange(4 * 64, dtype=torch.int64).reshape(4, 64)
    return SimpleNamespace(
        reqs=reqs,
        device="cpu",
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        tree_cache=object(),
        return_logprob=False,
        return_hidden_states_mode=SimpleNamespace(need_capture=lambda: False),
        model_config=SimpleNamespace(vocab_size=128),
    )


def _legacy_metadata(batch):
    reqs = batch.reqs
    input_ids = [r.get_fill_ids()[len(r.prefix_indices) :] for r in reqs]
    total_size = sum(req.extend_range.length for req in reqs)
    out_cache_loc = torch.empty(total_size, dtype=torch.int64)
    offset = 0
    for req in reqs:
        pre_len = len(req.prefix_indices)
        length = req.extend_range.length
        out_cache_loc[offset : offset + length] = batch.req_to_token_pool.req_to_token[
            req.req_pool_idx, pre_len : pre_len + length
        ]
        offset += length
    seq_lens = [
        len(req.origin_input_ids) + max(0, len(req.output_ids) - 1) for req in reqs
    ]
    return {
        "input_ids": torch.tensor(sum(input_ids, array("q")), dtype=torch.int32),
        "extend_num_tokens": sum(len(ids) for ids in input_ids),
        "out_cache_loc": out_cache_loc,
        "req_pool_indices": torch.tensor(
            [req.req_pool_idx for req in reqs], dtype=torch.int64
        ),
        "seq_lens": torch.tensor(seq_lens, dtype=torch.int64),
        "orig_seq_lens": torch.tensor(seq_lens, dtype=torch.int32),
    }


class TestPrebuiltInputIds(unittest.TestCase):
    def _prepare(self, reqs):
        batch = _make_batch(reqs)
        expected = _legacy_metadata(batch)
        sampling_info = object()
        with patch(
            "sglang.srt.disaggregation.decode_schedule_batch_mixin."
            "SamplingBatchInfo.from_schedule_batch",
            return_value=sampling_info,
        ):
            ScheduleBatchDisaggregationDecodeMixin.prepare_for_prebuilt(batch)
        return batch, expected, sampling_info

    def test_empty_prebuilt_keeps_legacy_metadata_without_input_tensor(self):
        batch, expected, sampling_info = self._prepare([])

        self.assertEqual(batch.forward_mode, ForwardMode.PREBUILT)
        self.assertIsNone(batch.input_ids)
        self.assertEqual(expected["input_ids"].numel(), 0)
        self.assertEqual(batch.extend_num_tokens, expected["extend_num_tokens"])
        torch.testing.assert_close(batch.out_cache_loc, expected["out_cache_loc"])
        torch.testing.assert_close(batch.req_pool_indices, expected["req_pool_indices"])
        torch.testing.assert_close(batch.seq_lens, expected["seq_lens"])
        torch.testing.assert_close(batch.orig_seq_lens, expected["orig_seq_lens"])
        self.assertIs(batch.sampling_info, sampling_info)

    def test_multi_request_metadata_matches_legacy_bitwise(self):
        reqs = [
            _PrebuiltReq(
                req_pool_idx=1,
                fill_ids=list(range(17)),
                prefix_len=5,
                extend_len=12,
                output_ids=[101],
                already_computed=2,
            ),
            _PrebuiltReq(
                req_pool_idx=3,
                fill_ids=list(range(31)),
                prefix_len=11,
                extend_len=20,
                output_ids=[102, 103],
                already_computed=14,
            ),
        ]
        batch, expected, sampling_info = self._prepare(reqs)

        self.assertIsNone(batch.input_ids)
        self.assertGreater(expected["input_ids"].numel(), 0)
        self.assertEqual(batch.extend_num_tokens, expected["extend_num_tokens"])
        torch.testing.assert_close(batch.out_cache_loc, expected["out_cache_loc"])
        torch.testing.assert_close(batch.req_pool_indices, expected["req_pool_indices"])
        torch.testing.assert_close(
            batch.req_pool_indices_cpu, expected["req_pool_indices"]
        )
        torch.testing.assert_close(batch.seq_lens, expected["seq_lens"])
        torch.testing.assert_close(batch.seq_lens_cpu, expected["seq_lens"])
        torch.testing.assert_close(batch.orig_seq_lens, expected["orig_seq_lens"])
        self.assertEqual(batch.seq_lens_sum, int(expected["seq_lens"].sum()))
        self.assertEqual(batch.prefix_lens, [5, 11])
        self.assertEqual(batch.extend_lens, [12, 20])
        self.assertIs(batch.sampling_info, sampling_info)
        self.assertEqual(reqs[0].cached_tokens, 14)
        self.assertEqual(reqs[0].cached_tokens_device, 10)
        self.assertEqual(reqs[1].cached_tokens, 11)
        self.assertEqual(reqs[1].cached_tokens_device, 7)
        self.assertFalse(reqs[0].is_retracted)
        self.assertFalse(reqs[1].is_retracted)

    def test_spec_relay_consumes_metadata_without_input_ids(self):
        req = _PrebuiltReq(
            req_pool_idx=2,
            fill_ids=list(range(20)),
            prefix_len=4,
            extend_len=16,
            output_ids=[117],
        )
        batch, _, _ = self._prepare([req])
        sentinel = object()
        batch.spec_algorithm = MagicMock()
        batch.spec_algorithm.build_disagg_draft_input.return_value = sentinel
        future_map = MagicMock()

        with patch(
            "sglang.srt.disaggregation.decode_schedule_batch_mixin."
            "maybe_cache_unfinished_req"
        ):
            ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(
                batch, SimpleNamespace(), future_map
            )

        self.assertIsNone(batch.input_ids)
        self.assertIs(batch.spec_info, sentinel)
        args = batch.spec_algorithm.build_disagg_draft_input.call_args.args
        self.assertIs(args[0], batch)
        torch.testing.assert_close(args[2], torch.tensor([117], dtype=torch.int64))
        self.assertIs(args[3], future_map)
        future_map.stash.assert_not_called()

    def test_non_spec_relay_stashes_bonus_token(self):
        req = _PrebuiltReq(
            req_pool_idx=0,
            fill_ids=list(range(10)),
            prefix_len=2,
            extend_len=8,
            output_ids=[91],
        )
        batch, _, _ = self._prepare([req])
        batch.spec_algorithm = MagicMock()
        batch.spec_algorithm.build_disagg_draft_input.return_value = None
        future_map = MagicMock()

        with patch(
            "sglang.srt.disaggregation.decode_schedule_batch_mixin."
            "maybe_cache_unfinished_req"
        ):
            ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(
                batch, SimpleNamespace(), future_map
            )

        self.assertIsNone(batch.input_ids)
        indices, payload = future_map.stash.call_args.args
        torch.testing.assert_close(indices, batch.req_pool_indices)
        torch.testing.assert_close(
            payload.bonus_tokens, torch.tensor([91], dtype=torch.int64)
        )

    def test_merge_preserves_supported_none_rebuild_contract(self):
        def make_schedule_batch(input_ids, value):
            sampling_info = MagicMock()
            return ScheduleBatch(
                reqs=[SimpleNamespace()],
                model_config=SimpleNamespace(is_encoder_decoder=False),
                input_ids=input_ids,
                req_pool_indices=torch.tensor([value], dtype=torch.int64),
                req_pool_indices_cpu=torch.tensor([value], dtype=torch.int64),
                seq_lens=torch.tensor([value + 10], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([value + 10], dtype=torch.int64),
                orig_seq_lens=torch.tensor([value + 10], dtype=torch.int32),
                out_cache_loc=torch.tensor([value + 20], dtype=torch.int64),
                sampling_info=sampling_info,
            )

        running = make_schedule_batch(torch.tensor([7]), 1)
        prebuilt = make_schedule_batch(None, 2)

        running.merge_batch(prebuilt)

        self.assertIsNone(running.input_ids)
        torch.testing.assert_close(running.req_pool_indices, torch.tensor([1, 2]))
        torch.testing.assert_close(running.seq_lens, torch.tensor([11, 12]))
        torch.testing.assert_close(
            running.orig_seq_lens, torch.tensor([11, 12], dtype=torch.int32)
        )
        running.sampling_info.merge_batch.assert_called_once_with(
            prebuilt.sampling_info
        )


if __name__ == "__main__":
    unittest.main()
