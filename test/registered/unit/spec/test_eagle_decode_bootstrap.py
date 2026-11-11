"""PD bootstrap reserves speculative KV without counting a decode step."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.speculative import eagle_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEagleDecodeBootstrap(CustomTestCase):
    def _prepare(self, *, page_size, is_bootstrap):
        reqs = [
            SimpleNamespace(
                kv=SimpleNamespace(kv_allocated_len=8, kv_committed_len=7),
                decode_batch_idx=0,
            ),
            SimpleNamespace(
                kv=SimpleNamespace(kv_allocated_len=16, kv_committed_len=8),
                decode_batch_idx=3,
            ),
        ]
        batch = SimpleNamespace(
            reqs=reqs,
            maybe_evict_swa=Mock(),
            cumulate_penalty_output_tokens=Mock(),
            sampling_info=SimpleNamespace(
                penalizer_orchestrator=SimpleNamespace(is_required=True)
            ),
            token_to_kv_pool_allocator=SimpleNamespace(page_size=page_size),
            seq_lens_cpu=torch.tensor([7, 8], dtype=torch.int32),
            req_to_token_pool=object(),
            req_pool_indices=torch.tensor([1, 2]),
            tree_cache=object(),
            device="cpu",
        )
        with (
            patch.object(eagle_utils, "get_alloc_reserve_per_decode", return_value=6),
            patch.object(
                eagle_utils,
                "get_spec",
                return_value=SimpleNamespace(speculative_eagle_topk=1),
            ),
            patch.object(eagle_utils, "alloc_for_spec_decode") as allocate,
        ):
            if is_bootstrap:
                eagle_utils.eagle_prepare_for_decode(batch, is_bootstrap=True)
            else:
                eagle_utils.eagle_prepare_for_decode(batch)

        allocate.assert_called_once()
        kwargs = allocate.call_args.kwargs
        expected_next = [13, 16] if page_size == 1 else [16, 16]
        torch.testing.assert_close(
            kwargs["cur_kv_lens_cpu"], torch.tensor([8, 16], dtype=torch.int32)
        )
        torch.testing.assert_close(
            kwargs["nxt_kv_lens_cpu"], torch.tensor(expected_next, dtype=torch.int32)
        )
        torch.testing.assert_close(kwargs["cur_kv_lens"], kwargs["cur_kv_lens_cpu"])
        torch.testing.assert_close(kwargs["nxt_kv_lens"], kwargs["nxt_kv_lens_cpu"])
        self.assertEqual(kwargs["num_needed_tokens"], expected_next[0] - 8)
        self.assertIs(kwargs["batch"], batch)
        self.assertEqual([req.kv.kv_committed_len for req in reqs], [7, 8])
        return batch

    def test_bootstrap_reserves_without_decode_side_effects(self):
        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                batch = self._prepare(page_size=page_size, is_bootstrap=True)
                batch.maybe_evict_swa.assert_not_called()
                batch.cumulate_penalty_output_tokens.assert_not_called()
                self.assertEqual([req.decode_batch_idx for req in batch.reqs], [0, 3])

    def test_normal_decode_keeps_existing_side_effects(self):
        for page_size in (1, 4):
            with self.subTest(page_size=page_size):
                batch = self._prepare(page_size=page_size, is_bootstrap=False)
                batch.maybe_evict_swa.assert_called_once_with()
                batch.cumulate_penalty_output_tokens.assert_called_once_with()
                self.assertEqual([req.decode_batch_idx for req in batch.reqs], [1, 4])


if __name__ == "__main__":
    unittest.main()
