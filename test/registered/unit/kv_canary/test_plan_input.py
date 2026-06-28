"""Tests for sglang.srt.kv_canary.plan_input: PlanInput allocation, zeroing, and dispatch."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.kv_canary.plan_input import PlanInput
from sglang.test.test_utils import CustomTestCase

_DEVICE = torch.device("cpu")


def _mode(name):
    """Return a SimpleNamespace that responds to exactly one ForwardMode predicate."""

    class _M:
        def is_decode_or_idle(self):
            return name == "decode"

        def is_target_verify(self):
            return name == "target_verify"

        def is_draft_extend_v2(self):
            return name == "draft_extend_v2"

        def is_extend(self):
            return name == "extend"

        def __repr__(self):
            return f"MockForwardMode({name!r})"

    return _M()


def _unknown_mode():
    class _U:
        def is_decode_or_idle(self):
            return False

        def is_target_verify(self):
            return False

        def is_draft_extend_v2(self):
            return False

        def is_extend(self):
            return False

        def __repr__(self):
            return "UnknownMode"

    return _U()


class TestPlanInputAllocate(CustomTestCase):
    def test_req_pool_indices_shape(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        self.assertEqual(pi.req_pool_indices.shape, torch.Size([4]))

    def test_req_pool_indices_dtype(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        self.assertEqual(pi.req_pool_indices.dtype, torch.int64)

    def test_prefix_lens_shape(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        self.assertEqual(pi.prefix_lens.shape, torch.Size([4]))

    def test_prefix_lens_dtype(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        self.assertEqual(pi.prefix_lens.dtype, torch.int64)

    def test_extend_seq_lens_shape(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        self.assertEqual(pi.extend_seq_lens.shape, torch.Size([4]))

    def test_extend_seq_lens_dtype(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        self.assertEqual(pi.extend_seq_lens.dtype, torch.int64)

    def test_valid_lens_shape(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        self.assertEqual(
            pi.req_to_verify_expected_tokens_valid_lens.shape, torch.Size([4])
        )

    def test_all_tensors_initialised_to_zero(self):
        pi = PlanInput.allocate(bs_capacity=3, device=_DEVICE)
        self.assertTrue(pi.req_pool_indices.eq(0).all())
        self.assertTrue(pi.prefix_lens.eq(0).all())
        self.assertTrue(pi.extend_seq_lens.eq(0).all())
        self.assertTrue(pi.req_to_verify_expected_tokens_valid_lens.eq(0).all())

    def test_frozen_dataclass_rejects_mutation(self):
        pi = PlanInput.allocate(bs_capacity=2, device=_DEVICE)
        with self.assertRaises((AttributeError, TypeError)):
            pi.req_pool_indices = torch.zeros(2, dtype=torch.int64)


class TestPlanInputZero(CustomTestCase):
    def test_zero_clears_all_tensors(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        pi.req_pool_indices.fill_(7)
        pi.prefix_lens.fill_(8)
        pi.extend_seq_lens.fill_(9)
        pi.req_to_verify_expected_tokens_valid_lens.fill_(10)
        pi.zero_()
        self.assertTrue(pi.req_pool_indices.eq(0).all())
        self.assertTrue(pi.prefix_lens.eq(0).all())
        self.assertTrue(pi.extend_seq_lens.eq(0).all())
        self.assertTrue(pi.req_to_verify_expected_tokens_valid_lens.eq(0).all())


class TestPlanInputBatchSizeGuard(CustomTestCase):
    def test_batch_larger_than_capacity_raises_runtime_error(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        fb = SimpleNamespace(
            req_pool_indices=torch.zeros(10, dtype=torch.int32),
            forward_mode=_mode("decode"),
            spec_info=None,
            positions=torch.arange(10, dtype=torch.int64),
            seq_lens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            req_all_ids_lens=None,
        )
        with self.assertRaises(RuntimeError) as ctx:
            pi.fill_from_forward_batch(forward_batch=fb)
        self.assertIn("exceeds static capacity", str(ctx.exception))

    def test_batch_equal_to_capacity_is_accepted(self):
        pi = PlanInput.allocate(bs_capacity=3, device=_DEVICE)
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 2, 3], dtype=torch.int32),
            forward_mode=_mode("decode"),
            spec_info=None,
            positions=torch.tensor([0, 1, 2], dtype=torch.int64),
            seq_lens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            req_all_ids_lens=None,
        )
        pi.fill_from_forward_batch(forward_batch=fb)  # must not raise


class TestFillFromForwardBatchDecode(CustomTestCase):
    def test_decode_prefix_lens_from_positions(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor([10, 20], dtype=torch.int32),
            forward_mode=_mode("decode"),
            spec_info=None,
            positions=torch.tensor([5, 7], dtype=torch.int64),
            seq_lens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            req_all_ids_lens=None,
        )
        pi.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(pi.prefix_lens[:2].tolist(), [5, 7])

    def test_decode_extend_seq_lens_all_ones(self):
        # fill_(1) is applied to the extend_seq_lens[:bs] slice; padding slot stays 0.
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 2, 3], dtype=torch.int32),
            forward_mode=_mode("decode"),
            spec_info=None,
            positions=torch.tensor([0, 1, 2], dtype=torch.int64),
            seq_lens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            req_all_ids_lens=None,
        )
        pi.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(pi.extend_seq_lens[:3].tolist(), [1, 1, 1])
        self.assertEqual(pi.extend_seq_lens[3].item(), 0)

    def test_decode_req_pool_indices_copied(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor([7, 8], dtype=torch.int32),
            forward_mode=_mode("decode"),
            spec_info=None,
            positions=torch.tensor([3, 4], dtype=torch.int64),
            seq_lens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            req_all_ids_lens=None,
        )
        pi.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(pi.req_pool_indices[:2].tolist(), [7, 8])

    def test_decode_padding_tail_zeroed(self):
        # bs=2, capacity=4: indices 2 and 3 must remain zero after zero_() + fill.
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        pi.req_pool_indices.fill_(99)
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
            forward_mode=_mode("decode"),
            spec_info=None,
            positions=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            req_all_ids_lens=None,
        )
        pi.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(pi.req_pool_indices[2:].tolist(), [0, 0])


class TestFillFromForwardBatchTargetVerify(CustomTestCase):
    def test_target_verify_prefix_lens_from_seq_lens(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 2, 3], dtype=torch.int32),
            forward_mode=_mode("target_verify"),
            spec_info=SimpleNamespace(draft_token_num=4),
            positions=None,
            seq_lens=torch.tensor([10, 20, 30], dtype=torch.int32),
            extend_seq_lens=None,
            extend_prefix_lens=None,
            req_all_ids_lens=None,
        )
        pi.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(pi.prefix_lens[:3].tolist(), [10, 20, 30])

    def test_target_verify_extend_seq_lens_is_draft_token_num(self):
        # fill_(draft_token_num) applied to extend_seq_lens[:bs] slice; padding slot stays 0.
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 2, 3], dtype=torch.int32),
            forward_mode=_mode("target_verify"),
            spec_info=SimpleNamespace(draft_token_num=6),
            positions=None,
            seq_lens=torch.tensor([10, 20, 30], dtype=torch.int32),
            extend_seq_lens=None,
            extend_prefix_lens=None,
            req_all_ids_lens=None,
        )
        pi.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(pi.extend_seq_lens[:3].tolist(), [6, 6, 6])
        self.assertEqual(pi.extend_seq_lens[3].item(), 0)


class TestFillFromForwardBatchDraftExtendV2(CustomTestCase):
    def test_draft_extend_v2_prefix_is_seq_minus_extend(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
            forward_mode=_mode("draft_extend_v2"),
            spec_info=None,
            positions=None,
            seq_lens=torch.tensor([15, 25], dtype=torch.int32),
            extend_seq_lens=torch.tensor([3, 5], dtype=torch.int32),
            extend_prefix_lens=None,
            req_all_ids_lens=None,
        )
        pi.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(pi.prefix_lens[:2].tolist(), [12, 20])

    def test_draft_extend_v2_extend_seq_lens_copied(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
            forward_mode=_mode("draft_extend_v2"),
            spec_info=None,
            positions=None,
            seq_lens=torch.tensor([15, 25], dtype=torch.int32),
            extend_seq_lens=torch.tensor([3, 5], dtype=torch.int32),
            extend_prefix_lens=None,
            req_all_ids_lens=None,
        )
        pi.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(pi.extend_seq_lens[:2].tolist(), [3, 5])


class TestFillFromForwardBatchExtend(CustomTestCase):
    def test_extend_prefix_lens_from_extend_prefix_lens(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
            forward_mode=_mode("extend"),
            spec_info=None,
            positions=None,
            seq_lens=None,
            extend_seq_lens=torch.tensor([7, 8], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([100, 200], dtype=torch.int32),
            req_all_ids_lens=None,
        )
        pi.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(pi.prefix_lens[:2].tolist(), [100, 200])

    def test_extend_seq_lens_from_extend_seq_lens(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
            forward_mode=_mode("extend"),
            spec_info=None,
            positions=None,
            seq_lens=None,
            extend_seq_lens=torch.tensor([7, 8], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([100, 200], dtype=torch.int32),
            req_all_ids_lens=None,
        )
        pi.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(pi.extend_seq_lens[:2].tolist(), [7, 8])


class TestFillFromForwardBatchUnknownMode(CustomTestCase):
    def test_unknown_mode_raises_not_implemented(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor([1], dtype=torch.int32),
            forward_mode=_unknown_mode(),
            spec_info=None,
            positions=None,
            seq_lens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            req_all_ids_lens=None,
        )
        with self.assertRaises(NotImplementedError) as ctx:
            pi.fill_from_forward_batch(forward_batch=fb)
        self.assertIn("Unsupported forward mode", str(ctx.exception))


class TestFillFromForwardBatchValidLens(CustomTestCase):
    def test_valid_lens_copied_when_present(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
            forward_mode=_mode("decode"),
            spec_info=None,
            positions=torch.tensor([5, 7], dtype=torch.int64),
            seq_lens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            req_all_ids_lens=torch.tensor([12, 15], dtype=torch.int32),
        )
        pi.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(
            pi.req_to_verify_expected_tokens_valid_lens[:2].tolist(), [12, 15]
        )

    def test_valid_lens_stays_zero_when_none(self):
        pi = PlanInput.allocate(bs_capacity=4, device=_DEVICE)
        fb = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
            forward_mode=_mode("decode"),
            spec_info=None,
            positions=torch.tensor([5, 7], dtype=torch.int64),
            seq_lens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            req_all_ids_lens=None,
        )
        pi.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(
            pi.req_to_verify_expected_tokens_valid_lens.tolist(), [0, 0, 0, 0]
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
