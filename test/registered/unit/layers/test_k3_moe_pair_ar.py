import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.communication.k3_moe_pair_ar import (
    MOE_LATENT_WIDTH,
    MOE_SHARED_WIDTH,
    all_reduce_moe_latent_shared,
    moe_pair_nbytes,
    should_split_oversized_moe_pair,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_QR_CAP = 256 * 1024 * 1024


class TestK3MoePairAr(CustomTestCase):
    def test_16k_concat_misses_qr_cap_but_slices_fit(self):
        concat = moe_pair_nbytes(16384, torch.bfloat16)
        latent = 16384 * MOE_LATENT_WIDTH * 2
        shared = 16384 * MOE_SHARED_WIDTH * 2
        self.assertEqual(concat, 336 * 1024 * 1024)
        self.assertGreater(concat, _QR_CAP)
        self.assertLessEqual(latent, _QR_CAP)
        self.assertLessEqual(shared, _QR_CAP)
        self.assertTrue(should_split_oversized_moe_pair(concat, _QR_CAP, True))
        self.assertFalse(should_split_oversized_moe_pair(concat, _QR_CAP, False))
        self.assertFalse(should_split_oversized_moe_pair(shared, _QR_CAP, True))
        self.assertFalse(should_split_oversized_moe_pair(concat, None, True))

    def test_decode_concat_stays_on_one_collective(self):
        concat = moe_pair_nbytes(64, torch.bfloat16)
        self.assertLess(concat, _QR_CAP)
        self.assertFalse(should_split_oversized_moe_pair(concat, _QR_CAP, True))

    def test_all_reduce_splits_when_predicate_hits(self):
        num_tokens = 4
        moe_hidden = 8
        hidden = 16
        buf = torch.arange(
            num_tokens * (moe_hidden + hidden), dtype=torch.bfloat16
        ).reshape(-1)

        def _fake_ar(x):
            return x + 1

        with (
            patch(
                "sglang.srt.layers.communication.k3_moe_pair_ar.qr_max_size_bytes",
                return_value=buf.numel() * buf.element_size() - 1,
            ),
            patch(
                "sglang.srt.layers.communication.k3_moe_pair_ar.envs.SGLANG_ROCM_K3_SPLIT_OVERSIZED_MOE_AR.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.k3_moe_pair_ar.tensor_model_parallel_all_reduce",
                side_effect=_fake_ar,
            ) as ar,
        ):
            latent, shared = all_reduce_moe_latent_shared(
                buf,
                num_tokens=num_tokens,
                moe_hidden_size=moe_hidden,
                hidden_size=hidden,
            )
        self.assertEqual(ar.call_count, 2)
        self.assertEqual(tuple(latent.shape), (num_tokens, moe_hidden))
        self.assertEqual(tuple(shared.shape), (num_tokens, hidden))
        self.assertTrue(
            torch.equal(latent.reshape(-1), buf[: num_tokens * moe_hidden] + 1)
        )

    def test_all_reduce_keeps_single_collective_under_cap(self):
        num_tokens = 4
        moe_hidden = 8
        hidden = 16
        buf = torch.zeros(num_tokens * (moe_hidden + hidden), dtype=torch.bfloat16)
        reduced = torch.ones_like(buf)

        with (
            patch(
                "sglang.srt.layers.communication.k3_moe_pair_ar.qr_max_size_bytes",
                return_value=buf.numel() * buf.element_size(),
            ),
            patch(
                "sglang.srt.layers.communication.k3_moe_pair_ar.envs.SGLANG_ROCM_K3_SPLIT_OVERSIZED_MOE_AR.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communication.k3_moe_pair_ar.tensor_model_parallel_all_reduce",
                return_value=reduced,
            ) as ar,
        ):
            latent, shared = all_reduce_moe_latent_shared(
                buf,
                num_tokens=num_tokens,
                moe_hidden_size=moe_hidden,
                hidden_size=hidden,
            )
        ar.assert_called_once()
        self.assertTrue(
            torch.equal(
                latent, reduced[: num_tokens * moe_hidden].view(num_tokens, moe_hidden)
            )
        )
        self.assertTrue(
            torch.equal(
                shared,
                reduced[num_tokens * moe_hidden :].view(num_tokens, hidden),
            )
        )


if __name__ == "__main__":
    unittest.main()
