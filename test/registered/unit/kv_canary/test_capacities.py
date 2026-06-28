"""Tests for sglang.srt.kv_canary.capacities: CanaryLaunchCapacities arithmetic and validation."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from sglang.srt.kv_canary.capacities import CanaryLaunchCapacities
from sglang.test.test_utils import CustomTestCase


def _sa(
    cuda_graph_max_bs=0,
    spec_num_draft_tokens=None,
    max_prefill_tokens=8192,
    chunked_prefill_size=None,
):
    """Build a minimal server_args SimpleNamespace."""
    if cuda_graph_max_bs == 0:
        cuda_graph_config = None
    else:
        cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(max_bs=cuda_graph_max_bs)
        )
    return SimpleNamespace(
        cuda_graph_config=cuda_graph_config,
        speculative_num_draft_tokens=spec_num_draft_tokens,
        max_prefill_tokens=max_prefill_tokens,
        chunked_prefill_size=chunked_prefill_size,
    )


class TestVerifyCapacityArithmetic(CustomTestCase):
    def test_verify_capacity_is_three_times_pool_slot_count(self):
        caps = CanaryLaunchCapacities.from_args(
            server_args=_sa(),
            req_to_token_pool_size=512,
            max_seq_len_per_req=2048,
            pool_slot_count=100,
        )
        self.assertEqual(caps.per_forward_verify_capacity, 300)

    def test_verify_capacity_with_small_pool(self):
        caps = CanaryLaunchCapacities.from_args(
            server_args=_sa(),
            req_to_token_pool_size=4,
            max_seq_len_per_req=64,
            pool_slot_count=7,
        )
        self.assertEqual(caps.per_forward_verify_capacity, 21)

    def test_verify_capacity_with_large_pool(self):
        caps = CanaryLaunchCapacities.from_args(
            server_args=_sa(),
            req_to_token_pool_size=4096,
            max_seq_len_per_req=4096,
            pool_slot_count=500,
        )
        self.assertEqual(caps.per_forward_verify_capacity, 1500)


class TestWriteReqCapacityArithmetic(CustomTestCase):
    def test_req_capacity_equals_pool_when_no_cuda_graph(self):
        # max_bs = max(0, req_to_token_pool_size)
        caps = CanaryLaunchCapacities.from_args(
            server_args=_sa(cuda_graph_max_bs=0),
            req_to_token_pool_size=128,
            max_seq_len_per_req=1024,
            pool_slot_count=50,
        )
        self.assertEqual(caps.per_forward_write_req_capacity, 128)

    def test_req_capacity_uses_cuda_graph_max_when_larger(self):
        # max_bs = max(256, 128) = 256
        caps = CanaryLaunchCapacities.from_args(
            server_args=_sa(cuda_graph_max_bs=256),
            req_to_token_pool_size=128,
            max_seq_len_per_req=1024,
            pool_slot_count=50,
        )
        self.assertEqual(caps.per_forward_write_req_capacity, 256)

    def test_req_capacity_uses_pool_when_larger_than_cuda_graph(self):
        # max_bs = max(32, 512) = 512
        caps = CanaryLaunchCapacities.from_args(
            server_args=_sa(cuda_graph_max_bs=32),
            req_to_token_pool_size=512,
            max_seq_len_per_req=1024,
            pool_slot_count=50,
        )
        self.assertEqual(caps.per_forward_write_req_capacity, 512)


class TestWriteEntryCapacityArithmetic(CustomTestCase):
    def test_chunked_none_gives_inf_limit(self):
        # chunked_prefill_size=None -> chunked_limit=math.inf
        # max_extend_tokens_per_forward = min(8192, inf) = 8192
        # write_entry = max(100*1, 8192) = 8192
        caps = CanaryLaunchCapacities.from_args(
            server_args=_sa(chunked_prefill_size=None, max_prefill_tokens=8192),
            req_to_token_pool_size=100,
            max_seq_len_per_req=2048,
            pool_slot_count=50,
        )
        self.assertEqual(caps.per_forward_write_entry_capacity, 8192)

    def test_chunked_size_limits_extend_tokens(self):
        # chunked_prefill_size=512, max_prefill_tokens=8192
        # max_extend_tokens_per_forward = min(8192, 512) = 512
        # write_entry = max(100*1, 512) = 512
        caps = CanaryLaunchCapacities.from_args(
            server_args=_sa(chunked_prefill_size=512, max_prefill_tokens=8192),
            req_to_token_pool_size=100,
            max_seq_len_per_req=2048,
            pool_slot_count=50,
        )
        self.assertEqual(caps.per_forward_write_entry_capacity, 512)

    def test_spec_tokens_scale_write_entry(self):
        # spec_num_draft_tokens=4 -> num_tokens_per_bs = max(1, 4) = 4
        # max_bs = max(256, 128) = 256
        # chunked_limit = 512
        # max_extend_tokens_per_forward = min(8192, 512) = 512
        # write_entry = max(256*4=1024, 512) = 1024
        caps = CanaryLaunchCapacities.from_args(
            server_args=_sa(
                cuda_graph_max_bs=256,
                spec_num_draft_tokens=4,
                chunked_prefill_size=512,
                max_prefill_tokens=8192,
            ),
            req_to_token_pool_size=128,
            max_seq_len_per_req=2048,
            pool_slot_count=50,
        )
        self.assertEqual(caps.per_forward_write_entry_capacity, 1024)

    def test_no_spec_tokens_uses_one_per_bs(self):
        # spec_num_draft_tokens=None -> num_tokens_per_bs=1
        # max_bs = 64, chunked_prefill_size=32
        # max_extend = min(8192, 32) = 32
        # write_entry = max(64*1, 32) = 64
        caps = CanaryLaunchCapacities.from_args(
            server_args=_sa(
                cuda_graph_max_bs=64,
                spec_num_draft_tokens=None,
                chunked_prefill_size=32,
                max_prefill_tokens=8192,
            ),
            req_to_token_pool_size=1,
            max_seq_len_per_req=256,
            pool_slot_count=1,
        )
        self.assertEqual(caps.per_forward_write_entry_capacity, 64)

    def test_spec_zero_draft_tokens_treated_as_no_spec(self):
        # spec_num_draft_tokens=0 -> falsy, num_tokens_per_bs=1
        caps = CanaryLaunchCapacities.from_args(
            server_args=_sa(
                spec_num_draft_tokens=0,
                chunked_prefill_size=32,
                max_prefill_tokens=8192,
            ),
            req_to_token_pool_size=64,
            max_seq_len_per_req=256,
            pool_slot_count=1,
        )
        # write_entry = max(64*1, 32) = 64
        self.assertEqual(caps.per_forward_write_entry_capacity, 64)


class TestValidationErrors(CustomTestCase):
    def test_nonpositive_req_to_token_pool_size_raises(self):
        with self.assertRaises(ValueError) as ctx:
            CanaryLaunchCapacities.from_args(
                server_args=_sa(),
                req_to_token_pool_size=0,
                max_seq_len_per_req=2048,
                pool_slot_count=10,
            )
        self.assertIn("req_to_token_pool_size", str(ctx.exception))

    def test_negative_req_to_token_pool_size_raises(self):
        with self.assertRaises(ValueError):
            CanaryLaunchCapacities.from_args(
                server_args=_sa(),
                req_to_token_pool_size=-1,
                max_seq_len_per_req=2048,
                pool_slot_count=10,
            )

    def test_nonpositive_max_seq_len_raises(self):
        with self.assertRaises(ValueError) as ctx:
            CanaryLaunchCapacities.from_args(
                server_args=_sa(),
                req_to_token_pool_size=10,
                max_seq_len_per_req=0,
                pool_slot_count=10,
            )
        self.assertIn("max_seq_len_per_req", str(ctx.exception))

    def test_nonpositive_pool_slot_count_raises(self):
        with self.assertRaises(ValueError) as ctx:
            CanaryLaunchCapacities.from_args(
                server_args=_sa(),
                req_to_token_pool_size=10,
                max_seq_len_per_req=2048,
                pool_slot_count=0,
            )
        self.assertIn("pool_slot_count", str(ctx.exception))

    def test_negative_cuda_graph_max_bs_raises(self):
        sa = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=-1)),
            speculative_num_draft_tokens=None,
            max_prefill_tokens=8192,
            chunked_prefill_size=None,
        )
        with self.assertRaises(ValueError) as ctx:
            CanaryLaunchCapacities.from_args(
                server_args=sa,
                req_to_token_pool_size=10,
                max_seq_len_per_req=2048,
                pool_slot_count=10,
            )
        self.assertIn("cuda_graph_max_bs", str(ctx.exception))

    def test_negative_spec_draft_tokens_raises(self):
        sa = _sa()
        sa.speculative_num_draft_tokens = -1
        with self.assertRaises(ValueError) as ctx:
            CanaryLaunchCapacities.from_args(
                server_args=sa,
                req_to_token_pool_size=10,
                max_seq_len_per_req=2048,
                pool_slot_count=10,
            )
        self.assertIn("speculative_num_draft_tokens", str(ctx.exception))

    def test_nonpositive_max_prefill_tokens_raises(self):
        with self.assertRaises(ValueError) as ctx:
            CanaryLaunchCapacities.from_args(
                server_args=_sa(max_prefill_tokens=0),
                req_to_token_pool_size=10,
                max_seq_len_per_req=2048,
                pool_slot_count=10,
            )
        self.assertIn("max_prefill_tokens", str(ctx.exception))


class TestPostInitValidation(CustomTestCase):
    def test_zero_verify_capacity_raises(self):
        with self.assertRaises(ValueError) as ctx:
            CanaryLaunchCapacities(
                per_forward_verify_capacity=0,
                per_forward_write_req_capacity=10,
                per_forward_write_entry_capacity=10,
            )
        self.assertIn("per_forward_verify_capacity", str(ctx.exception))

    def test_zero_write_req_capacity_raises(self):
        with self.assertRaises(ValueError) as ctx:
            CanaryLaunchCapacities(
                per_forward_verify_capacity=10,
                per_forward_write_req_capacity=0,
                per_forward_write_entry_capacity=10,
            )
        self.assertIn("per_forward_write_req_capacity", str(ctx.exception))

    def test_zero_write_entry_capacity_raises(self):
        with self.assertRaises(ValueError) as ctx:
            CanaryLaunchCapacities(
                per_forward_verify_capacity=10,
                per_forward_write_req_capacity=10,
                per_forward_write_entry_capacity=0,
            )
        self.assertIn("per_forward_write_entry_capacity", str(ctx.exception))

    def test_negative_any_field_raises(self):
        with self.assertRaises(ValueError):
            CanaryLaunchCapacities(
                per_forward_verify_capacity=-1,
                per_forward_write_req_capacity=10,
                per_forward_write_entry_capacity=10,
            )


class TestFrozenDataclass(CustomTestCase):
    def test_is_frozen(self):
        caps = CanaryLaunchCapacities.from_args(
            server_args=_sa(),
            req_to_token_pool_size=10,
            max_seq_len_per_req=128,
            pool_slot_count=5,
        )
        with self.assertRaises((AttributeError, TypeError)):
            caps.per_forward_verify_capacity = 999


if __name__ == "__main__":
    unittest.main(verbosity=3)
