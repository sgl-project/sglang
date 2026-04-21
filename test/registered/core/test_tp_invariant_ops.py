import os
import unittest

import torch
import torch.distributed as dist

from sglang.srt.tp_invariant_ops import (
    disable_tp_invariant_mode,
    enable_tp_invariant_mode,
    is_tp_invariant_mode_enabled,
    matmul_tp_persistent,
    moe_sum_tree_reduce,
    set_tp_invariant_mode,
    tree_all_reduce_sum,
)
from sglang.srt.tp_invariant_ops.tp_invariant_ops import _fixed_tree_sum_tensors
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci


register_cpu_ci(est_time=8, suite="stage-a-test-cpu")
register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-8-gpu-h200")


class TestTPInvariantMode(unittest.TestCase):
    def tearDown(self):
        disable_tp_invariant_mode()

    def test_mode_context_restores_previous_state(self):
        disable_tp_invariant_mode()
        self.assertFalse(is_tp_invariant_mode_enabled())

        with set_tp_invariant_mode(True):
            self.assertTrue(is_tp_invariant_mode_enabled())

        self.assertFalse(is_tp_invariant_mode_enabled())

        enable_tp_invariant_mode()
        with set_tp_invariant_mode(False):
            self.assertFalse(is_tp_invariant_mode_enabled())

        self.assertTrue(is_tp_invariant_mode_enabled())


class TestTPInvariantReferenceOps(unittest.TestCase):
    def test_fixed_tree_sum_order_is_stable(self):
        values = [
            torch.tensor([1.0e20], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([-1.0e20], dtype=torch.float32),
            torch.tensor([3.0], dtype=torch.float32),
        ]

        tree_result = _fixed_tree_sum_tensors(values)
        sequential_result = values[0] + values[1] + values[2] + values[3]

        self.assertEqual(tree_result.item(), 0.0)
        self.assertEqual(sequential_result.item(), 3.0)

    def test_matmul_tp_persistent_matches_torch_matmul(self):
        torch.manual_seed(0)
        A = torch.randn(5, 257, dtype=torch.float32)
        B = torch.randn(257, 7, dtype=torch.float32)
        bias = torch.randn(7, dtype=torch.float32)

        actual = matmul_tp_persistent(A, B, bias=bias)
        expected = A @ B + bias

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_torch_custom_op_dispatches_to_matmul(self):
        A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        B = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        actual = torch.ops.tp_inv_ops.matmul_tp_inv(A, B)
        expected = A @ B

        torch.testing.assert_close(actual, expected)

    def test_moe_sum_tree_reduce_matches_expert_order_reference(self):
        input_tensor = torch.tensor(
            [
                [
                    [1.0e20, 1.0],
                    [1.0, 2.0],
                    [-1.0e20, 4.0],
                    [3.0, 8.0],
                ],
                [
                    [5.0, 7.0],
                    [11.0, 13.0],
                    [17.0, 19.0],
                    [23.0, 29.0],
                ],
            ],
            dtype=torch.float32,
        )
        curr_topk_ids = torch.tensor(
            [
                [0, 1, 2, 3],
                [3, -1, 1, 0],
            ],
            dtype=torch.int64,
        )
        output = torch.empty(2, 2, dtype=torch.float32)

        moe_sum_tree_reduce(
            input=input_tensor,
            output=output,
            curr_topk_ids=curr_topk_ids,
            routed_scaling_factor=0.5,
            E=4,
        )

        expected = torch.tensor(
            [
                [0.0, 7.5],
                [22.5, 27.5],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(output, expected)

    def test_moe_sum_tree_reduce_rejects_non_power_of_two_expert_count(self):
        input_tensor = torch.zeros(1, 1, 2)
        output = torch.zeros(1, 2)
        curr_topk_ids = torch.zeros(1, 1, dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "power of two"):
            moe_sum_tree_reduce(
                input=input_tensor,
                output=output,
                curr_topk_ids=curr_topk_ids,
                routed_scaling_factor=1.0,
                E=3,
            )


class TestTPInvariance(unittest.TestCase):
    """Verify that matmul_tp_inv produces bit-wise identical results
    regardless of how the K dimension is partitioned across TP ranks."""

    def _simulate_tp_matmul(self, A, B, tp_size):
        K = A.shape[1]
        shard = K // tp_size
        partials = []
        for r in range(tp_size):
            start = r * shard
            end = start + shard
            partials.append(matmul_tp_persistent(A[:, start:end], B[start:end, :]))
        return _fixed_tree_sum_tensors(partials)

    def test_matmul_tp1_equals_tp2_equals_tp4_float32(self):
        torch.manual_seed(42)
        A = torch.randn(8, 512, dtype=torch.float32)
        B = torch.randn(512, 16, dtype=torch.float32)

        result_tp1 = matmul_tp_persistent(A, B)
        result_tp2 = self._simulate_tp_matmul(A, B, tp_size=2)
        result_tp4 = self._simulate_tp_matmul(A, B, tp_size=4)

        self.assertTrue(torch.equal(result_tp1, result_tp2))
        self.assertTrue(torch.equal(result_tp1, result_tp4))

    def test_matmul_tp1_equals_tp2_equals_tp4_bfloat16(self):
        torch.manual_seed(42)
        A = torch.randn(8, 512, dtype=torch.bfloat16)
        B = torch.randn(512, 16, dtype=torch.bfloat16)

        result_tp1 = matmul_tp_persistent(A, B)
        result_tp2 = self._simulate_tp_matmul(A, B, tp_size=2)
        result_tp4 = self._simulate_tp_matmul(A, B, tp_size=4)

        self.assertTrue(torch.equal(result_tp1, result_tp2))
        self.assertTrue(torch.equal(result_tp1, result_tp4))

    def test_matmul_tp_invariant_large_k(self):
        torch.manual_seed(7)
        A = torch.randn(4, 1024, dtype=torch.bfloat16)
        B = torch.randn(1024, 32, dtype=torch.bfloat16)

        result_tp1 = matmul_tp_persistent(A, B)
        result_tp8 = self._simulate_tp_matmul(A, B, tp_size=8)

        self.assertTrue(torch.equal(result_tp1, result_tp8))


class TestBFloat16Ops(unittest.TestCase):
    """Verify ops work correctly under bfloat16, the real production dtype."""

    def test_matmul_tp_persistent_bf16_close_to_torch(self):
        torch.manual_seed(0)
        A = torch.randn(6, 256, dtype=torch.bfloat16)
        B = torch.randn(256, 10, dtype=torch.bfloat16)
        bias = torch.randn(10, dtype=torch.bfloat16)

        actual = matmul_tp_persistent(A, B, bias=bias)
        expected = A @ B + bias

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_moe_sum_tree_reduce_bf16(self):
        input_tensor = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]],
            dtype=torch.bfloat16,
        )
        curr_topk_ids = torch.tensor([[0, 1]], dtype=torch.int64)
        output = torch.empty(1, 2, dtype=torch.bfloat16)

        moe_sum_tree_reduce(
            input=input_tensor,
            output=output,
            curr_topk_ids=curr_topk_ids,
            routed_scaling_factor=1.0,
            E=2,
        )

        expected = torch.tensor([[4.0, 6.0]], dtype=torch.bfloat16)
        torch.testing.assert_close(output, expected)


class TestTreeAllReduceNonDistributed(unittest.TestCase):
    """Cover tree_all_reduce_sum branches that run without torchrun."""

    def test_returns_clone_when_dist_not_initialized(self):
        if dist.is_initialized():
            self.skipTest("dist already initialized")
        x = torch.tensor([1.0, 2.0, 3.0])
        result = tree_all_reduce_sum(x)
        self.assertTrue(torch.equal(x, result))
        self.assertFalse(x.data_ptr() == result.data_ptr())


class TestMoeReduceOrderMatters(unittest.TestCase):
    """Prove that expert-id-ordered reduce differs from naive slot-ordered sum,
    demonstrating why moe_sum_tree_reduce exists."""

    def test_expert_order_differs_from_slot_order(self):
        input_tensor = torch.tensor(
            [
                [
                    [1.0e16, 0.0],
                    [1.0, 0.0],
                    [-1.0e16, 0.0],
                    [2.0, 0.0],
                ],
            ],
            dtype=torch.float32,
        )
        curr_topk_ids = torch.tensor([[2, 0, 3, 1]], dtype=torch.int64)
        output = torch.empty(1, 2, dtype=torch.float32)

        moe_sum_tree_reduce(
            input=input_tensor,
            output=output,
            curr_topk_ids=curr_topk_ids,
            routed_scaling_factor=1.0,
            E=4,
        )

        slot_order_sum = (
            input_tensor[0, 0] + input_tensor[0, 1]
            + input_tensor[0, 2] + input_tensor[0, 3]
        )

        self.assertNotEqual(output[0, 0].item(), slot_order_sum[0].item())


class TestEdgeCases(unittest.TestCase):
    """Edge cases for robustness."""

    def test_fixed_tree_sum_single_tensor(self):
        t = torch.tensor([5.0])
        result = _fixed_tree_sum_tensors([t])
        self.assertEqual(result.item(), 5.0)

    def test_fixed_tree_sum_odd_count(self):
        values = [
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([3.0]),
        ]
        result = _fixed_tree_sum_tensors(values)
        self.assertEqual(result.item(), 6.0)

    def test_fixed_tree_sum_empty_raises(self):
        with self.assertRaises(ValueError):
            _fixed_tree_sum_tensors([])

    def test_matmul_tp_persistent_k_less_than_block(self):
        torch.manual_seed(0)
        A = torch.randn(3, 64, dtype=torch.float32)
        B = torch.randn(64, 5, dtype=torch.float32)

        actual = matmul_tp_persistent(A, B)
        expected = A @ B
        torch.testing.assert_close(actual, expected)

    def test_matmul_tp_persistent_k_equals_block(self):
        torch.manual_seed(0)
        A = torch.randn(3, 128, dtype=torch.float32)
        B = torch.randn(128, 5, dtype=torch.float32)

        actual = matmul_tp_persistent(A, B)
        expected = A @ B
        torch.testing.assert_close(actual, expected)

    def test_moe_sum_tree_reduce_single_expert(self):
        input_tensor = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
        curr_topk_ids = torch.tensor([[0]], dtype=torch.int64)
        output = torch.empty(1, 2, dtype=torch.float32)

        moe_sum_tree_reduce(
            input=input_tensor,
            output=output,
            curr_topk_ids=curr_topk_ids,
            routed_scaling_factor=1.0,
            E=1,
        )

        expected = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        torch.testing.assert_close(output, expected)

    def test_moe_sum_tree_reduce_single_topk(self):
        input_tensor = torch.tensor(
            [[[10.0, 20.0]], [[30.0, 40.0]]],
            dtype=torch.float32,
        )
        curr_topk_ids = torch.tensor([[1], [0]], dtype=torch.int64)
        output = torch.empty(2, 2, dtype=torch.float32)

        moe_sum_tree_reduce(
            input=input_tensor,
            output=output,
            curr_topk_ids=curr_topk_ids,
            routed_scaling_factor=0.5,
            E=2,
        )

        expected = torch.tensor(
            [[5.0, 10.0], [15.0, 20.0]], dtype=torch.float32
        )
        torch.testing.assert_close(output, expected)

    def test_moe_sum_tree_reduce_all_remote(self):
        input_tensor = torch.tensor(
            [[[99.0, 99.0], [99.0, 99.0]]],
            dtype=torch.float32,
        )
        curr_topk_ids = torch.tensor([[-1, -1]], dtype=torch.int64)
        output = torch.empty(1, 2, dtype=torch.float32)

        moe_sum_tree_reduce(
            input=input_tensor,
            output=output,
            curr_topk_ids=curr_topk_ids,
            routed_scaling_factor=1.0,
            E=2,
        )

        expected = torch.zeros(1, 2, dtype=torch.float32)
        torch.testing.assert_close(output, expected)


class TestDistributedTreeAllReduce(unittest.TestCase):
    @unittest.skipUnless(
        int(os.environ.get("WORLD_SIZE", "1")) > 1,
        "requires torchrun with WORLD_SIZE > 1",
    )
    def test_tree_all_reduce_sum_distributed(self):
        own_pg = False
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            own_pg = True

        try:
            world_size = dist.get_world_size()
            if world_size & (world_size - 1) != 0:
                self.skipTest(
                    "tree_all_reduce_sum requires a power-of-two world size"
                )

            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                device = torch.device("cuda", local_rank)
            else:
                device = torch.device("cpu")

            rank = dist.get_rank()
            value = torch.full((4,), float(rank + 1), device=device)
            actual = tree_all_reduce_sum(value)
            expected = torch.full(
                (4,),
                float(world_size * (world_size + 1) // 2),
                device=device,
            )

            torch.testing.assert_close(actual, expected)
            dist.barrier()
        finally:
            if own_pg:
                dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
