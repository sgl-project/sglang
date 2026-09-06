"""CPU-checkable shape policy tests for the FlashKDA prefill backend."""

import unittest

import torch

from sglang.srt.layers.attention.linear.kernels.kda_flashkda import (
    _FLASHKDA_H200_MAX_TOTAL_MEMORY,
    _FLASHKDA_H200_MIN_TOTAL_MEMORY,
    _flashkda_supported,
    _is_h200_profile,
    _prefer_flashkda_for_shape,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _prefer(
    heads: int,
    seq_len: int,
    *,
    capability: tuple[int, int] = (9, 0),
    sm_count: int = 132,
    total_memory: int = 140 * (1 << 30),
    sequences: int = 1,
    min_seq_len: int | None = None,
    state_dtype: torch.dtype = torch.float32,
) -> bool:
    return _prefer_flashkda_for_shape(
        compute_capability=capability,
        multi_processor_count=sm_count,
        total_memory=total_memory,
        state_dtype=state_dtype,
        num_heads=heads,
        num_sequences=sequences,
        min_seq_len=seq_len if min_seq_len is None else min_seq_len,
        max_seq_len=seq_len,
    )


class TestFlashKdaDispatch(CustomTestCase):
    def test_h200_narrow_shards_fall_back_at_q2048(self):
        for heads in (8, 12, 16, 32):
            with self.subTest(heads=heads):
                self.assertFalse(_prefer(heads, 2048))

    def test_h200_wide_shard_extends_the_long_window(self):
        for seq_len in (2048, 4096, 8192):
            with self.subTest(seq_len=seq_len):
                self.assertTrue(_prefer(64, seq_len))
        self.assertFalse(_prefer(64, 8193))

    def test_short_request_forces_whole_batch_fallback(self):
        self.assertFalse(_prefer(64, 8192, min_seq_len=32))

    def test_unmeasured_multi_request_shapes_keep_existing_policy(self):
        self.assertTrue(_prefer(8, 2048, sequences=2))
        self.assertFalse(_prefer(64, 4096, sequences=2))

    def test_unmeasured_state_dtype_keeps_existing_policy(self):
        self.assertTrue(_prefer(16, 2048, state_dtype=torch.bfloat16))
        self.assertFalse(_prefer(64, 4096, state_dtype=torch.bfloat16))

    def test_h100_profiles_keep_existing_policy(self):
        profiles = (
            ((9, 0), 132, 80 * (1 << 30)),  # H100 SXM 80 GB
            ((9, 0), 114, 80 * (1 << 30)),  # H100 PCIe 80 GB
            ((9, 0), 132, 94 * (1 << 30)),  # H100 NVL 94 GB
        )
        for capability, sm_count, total_memory in profiles:
            with self.subTest(
                capability=capability,
                sm_count=sm_count,
                total_memory=total_memory,
            ):
                self.assertTrue(
                    _prefer(
                        16,
                        2048,
                        capability=capability,
                        sm_count=sm_count,
                        total_memory=total_memory,
                    )
                )
                self.assertFalse(
                    _prefer(
                        64,
                        4096,
                        capability=capability,
                        sm_count=sm_count,
                        total_memory=total_memory,
                    )
                )

    def test_unmeasured_architecture_and_sm_count_keep_existing_policy(self):
        profiles = (((8, 0), 132), ((10, 0), 132), ((9, 0), 120))
        for capability, sm_count in profiles:
            with self.subTest(capability=capability, sm_count=sm_count):
                self.assertTrue(
                    _prefer(16, 2048, capability=capability, sm_count=sm_count)
                )
                self.assertFalse(
                    _prefer(64, 4096, capability=capability, sm_count=sm_count)
                )

    def test_h200_memory_profile_boundaries_are_inclusive(self):
        for total_memory in (
            _FLASHKDA_H200_MIN_TOTAL_MEMORY,
            140 * (1 << 30),
            _FLASHKDA_H200_MAX_TOTAL_MEMORY,
        ):
            with self.subTest(total_memory=total_memory):
                self.assertTrue(
                    _is_h200_profile(
                        compute_capability=(9, 0),
                        multi_processor_count=132,
                        total_memory=total_memory,
                    )
                )

    def test_memory_outside_h200_profile_keeps_existing_policy(self):
        for total_memory in (
            _FLASHKDA_H200_MIN_TOTAL_MEMORY - 1,
            _FLASHKDA_H200_MAX_TOTAL_MEMORY + 1,
        ):
            with self.subTest(total_memory=total_memory):
                self.assertFalse(
                    _is_h200_profile(
                        compute_capability=(9, 0),
                        multi_processor_count=132,
                        total_memory=total_memory,
                    )
                )
                self.assertTrue(_prefer(16, 2048, total_memory=total_memory))
                self.assertFalse(_prefer(64, 4096, total_memory=total_memory))

    def test_unmeasured_head_count_is_not_extrapolated(self):
        self.assertTrue(_prefer(96, 2048))
        self.assertFalse(_prefer(96, 4096))

    def test_static_flashkda_contract(self):
        heads, tokens, slots = 2, 64, 3
        q = torch.empty(1, tokens, heads, 128, dtype=torch.bfloat16)
        tensors = dict(
            q=q,
            k=torch.empty_like(q),
            v=torch.empty_like(q),
            g=torch.empty_like(q),
            beta=torch.empty(1, tokens, heads, dtype=torch.bfloat16),
            ssm_states=torch.empty(slots, heads, 128, 128, dtype=torch.float32),
            cache_indices=torch.tensor([1], dtype=torch.int64),
            query_start_loc=torch.tensor([0, tokens], dtype=torch.int32),
            A_log=torch.empty(1, 1, heads, 1),
            dt_bias=torch.empty(heads * 128),
        )
        self.assertTrue(_flashkda_supported(**tensors))

        invalid = (
            {"v": torch.empty(1, tokens, heads, 64, dtype=torch.bfloat16)},
            {"v": torch.empty(1, tokens, heads + 1, 128, dtype=torch.bfloat16)},
            {"q": q.float()},
            {"A_log": None},
            {"dt_bias": None},
        )
        for override in invalid:
            with self.subTest(override=tuple(override)):
                args = tensors | override
                self.assertFalse(_flashkda_supported(**args))


if __name__ == "__main__":
    unittest.main()
