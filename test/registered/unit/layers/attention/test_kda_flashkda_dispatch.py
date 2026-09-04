"""CPU-checkable shape policy tests for the FlashKDA prefill backend."""

import unittest

import torch

from sglang.srt.layers.attention.linear.kernels.kda_flashkda import (
    _flashkda_supported,
    _prefer_flashkda_for_shape,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _prefer(
    heads: int,
    seq_len: int,
    *,
    device_name: str = "NVIDIA L20X",
    capability: tuple[int, int] = (9, 0),
    sm_count: int = 132,
    sequences: int = 1,
    min_seq_len: int | None = None,
    state_dtype: torch.dtype = torch.float32,
) -> bool:
    return _prefer_flashkda_for_shape(
        device_name=device_name,
        compute_capability=capability,
        multi_processor_count=sm_count,
        state_dtype=state_dtype,
        num_heads=heads,
        num_sequences=sequences,
        min_seq_len=seq_len if min_seq_len is None else min_seq_len,
        max_seq_len=seq_len,
    )


class TestFlashKdaDispatch(CustomTestCase):
    def test_l20x_narrow_shards_fall_back_at_q2048(self):
        for heads in (8, 12, 16, 32):
            with self.subTest(heads=heads):
                self.assertFalse(_prefer(heads, 2048))

    def test_l20x_wide_shard_extends_the_long_window(self):
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

    def test_unmeasured_hardware_keeps_existing_policy(self):
        profiles = (
            ("NVIDIA L20X", (8, 0), 132),
            ("NVIDIA L20X", (10, 0), 132),
            ("NVIDIA H200", (9, 0), 132),
            ("NVIDIA L20X", (9, 0), 120),
        )
        for name, capability, sm_count in profiles:
            with self.subTest(name=name, capability=capability, sm_count=sm_count):
                self.assertTrue(
                    _prefer(
                        16,
                        2048,
                        device_name=name,
                        capability=capability,
                        sm_count=sm_count,
                    )
                )
                self.assertFalse(
                    _prefer(
                        64,
                        4096,
                        device_name=name,
                        capability=capability,
                        sm_count=sm_count,
                    )
                )

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
