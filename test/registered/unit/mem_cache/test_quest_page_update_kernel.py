import unittest

import torch

from sglang.srt.mem_cache.sparsity.kernels.quest_page_update import (
    quest_update_page_representations_,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


def _reference_update(
    req_pool_indices,
    seq_lens,
    req_to_token,
    k_buffer,
    repr_constructed,
    last_constructed_page,
    page_k_min,
    page_k_max,
    page_valid,
    page_size,
    *,
    advance_trackers,
):
    for batch_idx, req_idx_tensor in enumerate(req_pool_indices):
        req_idx = int(req_idx_tensor.item())
        end_page = int(seq_lens[batch_idx].item()) // page_size
        start_page = (
            int(last_constructed_page[req_idx].item())
            if bool(repr_constructed[req_idx].item())
            else 0
        )
        for logical_page in range(start_page, end_page):
            logical_start = logical_page * page_size
            physical_tokens = req_to_token[
                req_idx, logical_start : logical_start + page_size
            ].clamp(0, k_buffer.shape[0] - 1)
            keys = k_buffer[physical_tokens.to(torch.long)].to(torch.float32)
            target_page = int(
                (req_to_token[req_idx, logical_start] // page_size)
                .clamp(0, page_k_min.shape[0] - 1)
                .item()
            )
            page_k_min[target_page] = keys.amin(dim=0)
            page_k_max[target_page] = keys.amax(dim=0)
            page_valid[target_page] = True

        if advance_trackers and start_page < end_page:
            repr_constructed[req_idx] = True
            last_constructed_page[req_idx] = end_page


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None,
    "NVIDIA CUDA is required",
)
class TestQuestPageUpdateKernel(unittest.TestCase):
    page_size = 4

    @staticmethod
    def _make_inputs(
        *,
        k_dtype: torch.dtype = torch.float16,
    ):
        device = torch.device("cuda")
        req_to_token = torch.tensor(
            [
                [8, 13, 10, 15, 24, 31, 26, 29, 40, 47, 42, 45],
                [48, 55, 50, 53, 60, 67, 62, 65, 72, 79, 74, 77],
                [56, 63, 58, 61, 80, 87, 82, 85, 96, 103, 98, 101],
                [88, 95, 90, 93, 104, 111, 106, 109, 112, 127, 114, 125],
            ],
            dtype=torch.int64,
            device=device,
        )
        torch.manual_seed(17)
        k_buffer = torch.randn(128, 3, 33, dtype=k_dtype, device=device)
        return {
            "req_pool_indices": torch.tensor(
                [3, 0, 2], dtype=torch.int64, device=device
            ),
            "seq_lens": torch.tensor([12, 8, 7], dtype=torch.int64, device=device),
            "req_to_token": req_to_token,
            "k_buffer": k_buffer,
            "repr_constructed": torch.tensor(
                [False, False, True, True], dtype=torch.bool, device=device
            ),
            # Slot 0's stale value must be ignored because it is unconstructed.
            "last_constructed_page": torch.tensor(
                [99, 0, 1, 2], dtype=torch.int64, device=device
            ),
            "page_k_min": torch.zeros(32, 3, 33, dtype=torch.float32, device=device),
            "page_k_max": torch.zeros(32, 3, 33, dtype=torch.float32, device=device),
            "page_valid": torch.zeros(32, dtype=torch.bool, device=device),
        }

    def _assert_matches_reference(self, *, advance_trackers):
        actual = self._make_inputs()
        expected = {key: value.clone() for key, value in actual.items()}

        _reference_update(
            **expected,
            page_size=self.page_size,
            advance_trackers=advance_trackers,
        )
        quest_update_page_representations_(
            **actual,
            page_size=self.page_size,
            advance_trackers=advance_trackers,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(actual["page_valid"], expected["page_valid"]))
        torch.testing.assert_close(actual["page_k_min"], expected["page_k_min"])
        torch.testing.assert_close(actual["page_k_max"], expected["page_k_max"])
        self.assertTrue(
            torch.equal(actual["repr_constructed"], expected["repr_constructed"])
        )
        self.assertTrue(
            torch.equal(
                actual["last_constructed_page"],
                expected["last_constructed_page"],
            )
        )

    def test_matches_reference_for_mapped_tokens_and_tracker_modes(self):
        for advance_trackers in (False, True):
            with self.subTest(advance_trackers=advance_trackers):
                self._assert_matches_reference(advance_trackers=advance_trackers)

    def test_rejects_noncontiguous_trackers(self):
        inputs = self._make_inputs()
        tracker_storage = torch.empty((4, 2), dtype=torch.bool, device="cuda")
        page_storage = torch.empty((4, 2), dtype=torch.int64, device="cuda")
        invalid_values = {
            "repr_constructed": tracker_storage[:, 0],
            "last_constructed_page": page_storage[:, 0],
        }
        for name, value in invalid_values.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                quest_update_page_representations_(
                    **{**inputs, name: value},
                    page_size=self.page_size,
                    advance_trackers=True,
                )


if __name__ == "__main__":
    unittest.main()
