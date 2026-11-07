from __future__ import annotations

import unittest
from types import SimpleNamespace

import msgspec
import torch

from sglang.srt.kv_canary.perturb.real_kv_unused_cache import (
    _translate_full_slots_to_swa_slots,
)
from sglang.srt.kv_canary.perturb.slot_picker import (
    ReqToTokenEntry,
    collect_active_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPerturbSlotPickerAndSwaTranslation(CustomTestCase):
    def test_req_to_token_entry_is_frozen_msgspec_struct(self) -> None:
        entry = ReqToTokenEntry(req_pool_idx=2, position=1, value=42, seq_len=8)
        self.assertIsInstance(entry, msgspec.Struct)
        self.assertEqual(entry.req_pool_idx, 2)
        self.assertEqual(entry.position, 1)
        self.assertEqual(entry.value, 42)
        self.assertEqual(entry.seq_len, 8)
        with self.assertRaises(AttributeError):
            entry.value = 99  # type: ignore[misc]

    def test_collect_active_slots_batches_heterogeneous_seq_lens_and_filters(
        self,
    ) -> None:
        req_to_token = torch.full((5, 6), -1, dtype=torch.int32)
        req_to_token[1, :3] = torch.tensor([10, 11, 12], dtype=torch.int32)
        req_to_token[3, :5] = torch.tensor([30, -1, 32, 33, 34], dtype=torch.int32)
        req_to_token[3, 5] = 999  # beyond seq_len=5, must be ignored

        pool = SimpleNamespace(req_to_token=req_to_token)
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([-1, 1, 99, 3, 2], dtype=torch.int64),
            seq_lens=torch.tensor([4, 3, 2, 5, 0], dtype=torch.int64),
            out_cache_loc=torch.tensor([11, 33, 10], dtype=torch.int32),
            global_num_token_non_padded_cpu=2,
        )

        entries = collect_active_slots(
            maybe_inaccurate_forward_batch=forward_batch,
            req_to_token_pool=pool,
            exclude_out_cache_loc=True,
        )

        expected = [
            ReqToTokenEntry(req_pool_idx=1, position=0, value=10, seq_len=3),
            ReqToTokenEntry(req_pool_idx=1, position=2, value=12, seq_len=3),
            ReqToTokenEntry(req_pool_idx=3, position=0, value=30, seq_len=5),
            ReqToTokenEntry(req_pool_idx=3, position=2, value=32, seq_len=5),
            ReqToTokenEntry(req_pool_idx=3, position=4, value=34, seq_len=5),
        ]
        self.assertEqual(entries, expected)

    def test_collect_active_slots_returns_empty_when_no_valid_requests(self) -> None:
        req_to_token = torch.zeros((2, 4), dtype=torch.int32)
        pool = SimpleNamespace(req_to_token=req_to_token)
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([-1, 5], dtype=torch.int64),
            seq_lens=torch.tensor([2, 0], dtype=torch.int64),
            out_cache_loc=None,
            global_num_token_non_padded_cpu=None,
        )

        entries = collect_active_slots(
            maybe_inaccurate_forward_batch=forward_batch,
            req_to_token_pool=pool,
        )
        self.assertEqual(entries, [])

    def test_translate_full_slots_to_swa_slots(self) -> None:
        lut = torch.tensor([100, -1, 200, 300, -5], dtype=torch.int32)
        translated = _translate_full_slots_to_swa_slots(
            slots=[-1, 0, 1, 2, 3, 4, 5, 99],
            full_to_swa_index_mapping=lut,
        )
        self.assertEqual(translated, [100, 200, 300])
        self.assertEqual(
            _translate_full_slots_to_swa_slots(
                slots=[0, 2], full_to_swa_index_mapping=None
            ),
            [],
        )
        self.assertEqual(
            _translate_full_slots_to_swa_slots(
                slots=[], full_to_swa_index_mapping=lut
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
