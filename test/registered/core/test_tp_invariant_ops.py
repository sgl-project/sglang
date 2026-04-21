"""Tests for TP-invariant kernels (PR1).

TP-invariance property:
    Given the same TP degree and the same input data, matmul_tp_persistent
    plus tree_all_reduce_sum produces bitwise identical results across runs.
    When K/BLOCK_K is divisible by tp_size AND each shard yields a power-of-two
    block count, TP=1 and TP=N also agree (isomorphic tree structure).

    For production K values (e.g. 3584, 5120) where block counts per shard are
    not power-of-two, the tree structure differs between TP degrees. The
    invariance guarantee is *determinism for a fixed TP degree*.

All bitwise assertions use torch.equal, never approximate tolerances.
"""

import os
import random
import unittest

import torch
import torch.distributed as dist

from sglang.srt.tp_invariant_ops import (
    disable_tp_invariant_mode,
    enable_tp_invariant_mode,
    is_tp_invariant_mode_enabled,
    matmul_tp_inv,
    matmul_tp_persistent,
    moe_sum_tree_reduce,
    set_tp_invariant_mode,
    tree_all_reduce_sum,
)
from sglang.srt.tp_invariant_ops.tp_invariant_ops import (
    _MATMUL_K_BLOCK,
    _fixed_tree_sum_tensors,
    _is_power_of_two,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=12, suite="stage-a-test-cpu")
register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-8-gpu-h200")

BLOCK_K = _MATMUL_K_BLOCK  # 128


def _simulate_tp_matmul(A, B, tp_size, **kwargs):
    """Simulate what production TP does: each rank runs matmul_tp_persistent
    on its K-shard, then tree_all_reduce_sum gathers and tree-sums."""
    K = A.shape[1]
    shard = K // tp_size
    partials = []
    for r in range(tp_size):
        start = r * shard
        end = start + shard
        partials.append(
            matmul_tp_persistent(A[:, start:end], B[start:end, :], **kwargs)
        )
    return _fixed_tree_sum_tensors(partials)


# ---------------------------------------------------------------------------
# Mode flag
# ---------------------------------------------------------------------------
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

    def test_enable_is_idempotent(self):
        enable_tp_invariant_mode()
        enable_tp_invariant_mode()
        self.assertTrue(is_tp_invariant_mode_enabled())
        disable_tp_invariant_mode()
        self.assertFalse(is_tp_invariant_mode_enabled())

    def test_context_restores_after_exception(self):
        disable_tp_invariant_mode()
        try:
            with set_tp_invariant_mode(True):
                raise RuntimeError("deliberate")
        except RuntimeError:
            pass
        self.assertFalse(is_tp_invariant_mode_enabled())


# ---------------------------------------------------------------------------
# Reference ops: correctness
# ---------------------------------------------------------------------------
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

    def test_matmul_tp_persistent_matches_torch_matmul_fp32(self):
        torch.manual_seed(0)
        A = torch.randn(5, 257, dtype=torch.float32)
        B = torch.randn(257, 7, dtype=torch.float32)
        bias = torch.randn(7, dtype=torch.float32)

        actual = matmul_tp_persistent(A, B, bias=bias)
        expected = A @ B + bias
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_matmul_tp_persistent_bf16_approximate_to_torch(self):
        """BF16 block-tree matmul vs native torch matmul. Different accumulation
        order means results differ by up to a few BF16 ULPs; this test only
        checks that the magnitude is in the same ballpark, not bitwise equality."""
        torch.manual_seed(0)
        A = torch.randn(6, 256, dtype=torch.bfloat16)
        B = torch.randn(256, 10, dtype=torch.bfloat16)
        bias = torch.randn(10, dtype=torch.bfloat16)

        actual = matmul_tp_persistent(A, B, bias=bias)
        expected = A @ B + bias
        torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-1)

    def test_torch_custom_op_dispatches_to_matmul(self):
        A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        B = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        actual = torch.ops.tp_inv_ops.matmul_tp_inv(A, B)
        expected = A @ B
        torch.testing.assert_close(actual, expected)

    def test_torch_custom_op_dispatches_with_bias(self):
        torch.manual_seed(99)
        A = torch.randn(4, 128, dtype=torch.float32)
        B = torch.randn(128, 8, dtype=torch.float32)
        bias = torch.randn(8, dtype=torch.float32)

        actual = torch.ops.tp_inv_ops.matmul_tp_inv(A, B, bias)
        expected = matmul_tp_persistent(A, B, bias=bias)
        self.assertTrue(torch.equal(actual, expected))

    def test_matmul_tp_inv_matches_persistent(self):
        torch.manual_seed(11)
        A = torch.randn(4, 256, dtype=torch.bfloat16)
        B = torch.randn(256, 16, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(matmul_tp_inv(A, B), matmul_tp_persistent(A, B)))

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
            [[0, 1, 2, 3], [3, -1, 1, 0]],
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

        expected = torch.tensor([[0.0, 7.5], [22.5, 27.5]], dtype=torch.float32)
        torch.testing.assert_close(output, expected)

    def test_moe_sum_tree_reduce_rejects_non_power_of_two_expert_count(self):
        with self.assertRaisesRegex(ValueError, "power of two"):
            moe_sum_tree_reduce(
                input=torch.zeros(1, 1, 2),
                output=torch.zeros(1, 2),
                curr_topk_ids=torch.zeros(1, 1, dtype=torch.int64),
                routed_scaling_factor=1.0,
                E=3,
            )


# ---------------------------------------------------------------------------
# Matmul TP invariance: the core contract
#
# Two classes of tests:
#   1. Cross-TP: TP=1 == TP=N, requires isomorphic tree (K/BLOCK_K power-of-two
#      multiple of tp_size). Uses torch.equal.
#   2. Determinism: same TP degree, two runs == bitwise identical. Works for
#      any valid K alignment.
# ---------------------------------------------------------------------------
class TestTPInvarianceCrossTP(unittest.TestCase):
    """TP=1 == TP=N when the binary tree structure is isomorphic."""

    def test_fp32_cross_tp_k512(self):
        torch.manual_seed(42)
        A = torch.randn(8, 512, dtype=torch.float32)
        B = torch.randn(512, 16, dtype=torch.float32)

        result_tp1 = matmul_tp_persistent(A, B)
        result_tp2 = _simulate_tp_matmul(A, B, tp_size=2)
        result_tp4 = _simulate_tp_matmul(A, B, tp_size=4)

        self.assertTrue(torch.equal(result_tp1, result_tp2))
        self.assertTrue(torch.equal(result_tp1, result_tp4))

    def test_bf16_cross_tp_k512(self):
        torch.manual_seed(42)
        A = torch.randn(8, 512, dtype=torch.bfloat16)
        B = torch.randn(512, 16, dtype=torch.bfloat16)

        result_tp1 = matmul_tp_persistent(A, B)
        result_tp2 = _simulate_tp_matmul(A, B, tp_size=2)
        result_tp4 = _simulate_tp_matmul(A, B, tp_size=4)

        self.assertTrue(torch.equal(result_tp1, result_tp2))
        self.assertTrue(torch.equal(result_tp1, result_tp4))

    def test_bf16_cross_tp_k1024(self):
        torch.manual_seed(7)
        A = torch.randn(4, 1024, dtype=torch.bfloat16)
        B = torch.randn(1024, 32, dtype=torch.bfloat16)

        result_tp1 = matmul_tp_persistent(A, B)
        result_tp2 = _simulate_tp_matmul(A, B, tp_size=2)
        result_tp4 = _simulate_tp_matmul(A, B, tp_size=4)
        result_tp8 = _simulate_tp_matmul(A, B, tp_size=8)

        self.assertTrue(torch.equal(result_tp1, result_tp2))
        self.assertTrue(torch.equal(result_tp1, result_tp4))
        self.assertTrue(torch.equal(result_tp1, result_tp8))

    def test_bf16_cross_tp_k2048_all_sizes(self):
        """K=2048: 16 blocks. TP={1,2,4,8,16} all produce isomorphic trees."""
        torch.manual_seed(102)
        A = torch.randn(4, 2048, dtype=torch.bfloat16)
        B = torch.randn(2048, 16, dtype=torch.bfloat16)

        result_tp1 = matmul_tp_persistent(A, B)
        for tp_size in [2, 4, 8, 16]:
            result_tpN = _simulate_tp_matmul(A, B, tp_size=tp_size)
            self.assertTrue(
                torch.equal(result_tp1, result_tpN),
                f"TP=1 != TP={tp_size} for K=2048",
            )

    def test_bf16_cross_tp_k4096(self):
        """K=4096: 32 blocks."""
        torch.manual_seed(103)
        A = torch.randn(2, 4096, dtype=torch.bfloat16)
        B = torch.randn(4096, 8, dtype=torch.bfloat16)

        result_tp1 = matmul_tp_persistent(A, B)
        for tp_size in [2, 4, 8]:
            result_tpN = _simulate_tp_matmul(A, B, tp_size=tp_size)
            self.assertTrue(
                torch.equal(result_tp1, result_tpN),
                f"TP=1 != TP={tp_size} for K=4096",
            )

    def test_fp16_cross_tp_k512(self):
        torch.manual_seed(500)
        A = torch.randn(4, 512, dtype=torch.float16)
        B = torch.randn(512, 16, dtype=torch.float16)

        result_tp1 = matmul_tp_persistent(A, B)
        result_tp4 = _simulate_tp_matmul(A, B, tp_size=4)

        self.assertTrue(torch.equal(result_tp1, result_tp4))

    def test_torch_ops_dispatch_bf16_cross_tp(self):
        """torch.ops.tp_inv_ops.matmul_tp_inv dispatch preserves cross-TP invariance."""
        torch.manual_seed(400)
        A = torch.randn(4, 512, dtype=torch.bfloat16)
        B = torch.randn(512, 16, dtype=torch.bfloat16)

        result_full = torch.ops.tp_inv_ops.matmul_tp_inv(A, B)

        K = A.shape[1]
        shard = K // 4
        partials = []
        for r in range(4):
            s, e = r * shard, (r + 1) * shard
            partials.append(torch.ops.tp_inv_ops.matmul_tp_inv(A[:, s:e], B[s:e, :]))
        result_tp4 = _fixed_tree_sum_tensors(partials)

        self.assertTrue(torch.equal(result_full, result_tp4))


class TestTPInvarianceDeterminism(unittest.TestCase):
    """Same TP degree, two runs -> bitwise identical. Works for all production K."""

    def _assert_deterministic(self, K, tp_size, dtype=torch.bfloat16):
        torch.manual_seed(42)
        A = torch.randn(4, K, dtype=dtype)
        B = torch.randn(K, 16, dtype=dtype)

        result_a = _simulate_tp_matmul(A, B, tp_size=tp_size)
        result_b = _simulate_tp_matmul(A, B, tp_size=tp_size)
        self.assertTrue(
            torch.equal(result_a, result_b),
            f"K={K} TP={tp_size} dtype={dtype} not deterministic",
        )

    def test_bf16_k3584_tp2(self):
        """Qwen3-4B hidden_size. 3584/2=1792 -> 14 blocks per shard."""
        self._assert_deterministic(3584, tp_size=2)

    def test_bf16_k3584_tp4(self):
        self._assert_deterministic(3584, tp_size=4)

    def test_bf16_k5120_tp4(self):
        """Qwen3-30B hidden_size. 5120/4=1280 -> 10 blocks per shard."""
        self._assert_deterministic(5120, tp_size=4)

    def test_bf16_k5120_tp8(self):
        self._assert_deterministic(5120, tp_size=8)

    def test_bf16_k4096_tp8(self):
        self._assert_deterministic(4096, tp_size=8)

    def test_fp32_k3584_tp2(self):
        self._assert_deterministic(3584, tp_size=2, dtype=torch.float32)

    def test_fp32_accum_deterministic(self):
        """fp32_accum=True is deterministic for a fixed TP degree."""
        torch.manual_seed(200)
        A = torch.randn(4, 512, dtype=torch.bfloat16)
        B = torch.randn(512, 16, dtype=torch.bfloat16)

        result_a = _simulate_tp_matmul(A, B, tp_size=2, fp32_accum=True)
        result_b = _simulate_tp_matmul(A, B, tp_size=2, fp32_accum=True)
        self.assertTrue(torch.equal(result_a, result_b))

    def test_fp32_accum_output_dtype_is_input_dtype(self):
        torch.manual_seed(300)
        A = torch.randn(4, 512, dtype=torch.bfloat16)
        B = torch.randn(512, 16, dtype=torch.bfloat16)

        result = matmul_tp_persistent(A, B, fp32_accum=True)
        self.assertEqual(result.dtype, torch.bfloat16)


# ---------------------------------------------------------------------------
# BFloat16 ops: approximate correctness vs torch
# ---------------------------------------------------------------------------
class TestBFloat16Ops(unittest.TestCase):
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


# ---------------------------------------------------------------------------
# MoE tree reduce: EP/slot invariance
# ---------------------------------------------------------------------------
class TestMoeReduceSlotInvariance(unittest.TestCase):
    """moe_sum_tree_reduce must produce bitwise identical results regardless
    of which topk slot an expert appears in. This is the EP invariance
    property: different EP ranks may route the same experts to different
    slot positions, but the tree-reduce result must be bitwise identical."""

    def _run_moe_reduce(self, input_tensor, topk_ids, E, scaling=1.0):
        output = torch.zeros(
            input_tensor.shape[0], input_tensor.shape[2], dtype=input_tensor.dtype
        )
        moe_sum_tree_reduce(
            input=input_tensor,
            output=output,
            curr_topk_ids=topk_ids,
            routed_scaling_factor=scaling,
            E=E,
        )
        return output

    def _permute_slots(self, values, ids_src, ids_dst, topk):
        """Rearrange values so that each expert's data moves to its new slot."""
        tokens = values.shape[0]
        result = torch.zeros_like(values)
        for t in range(tokens):
            for slot_dst in range(topk):
                eid = ids_dst[t, slot_dst].item()
                if eid == -1:
                    continue
                slot_src = (ids_src[t] == eid).nonzero(as_tuple=True)[0].item()
                result[t, slot_dst] = values[t, slot_src]
        return result

    def test_slot_permutation_gives_identical_result_fp32(self):
        torch.manual_seed(10)
        H = 64
        values = torch.randn(1, 4, H, dtype=torch.float32)

        ids_a = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
        ids_b = torch.tensor([[2, 0, 3, 1]], dtype=torch.int64)

        input_b = self._permute_slots(values, ids_a, ids_b, topk=4)

        result_a = self._run_moe_reduce(values, ids_a, E=4)
        result_b = self._run_moe_reduce(input_b, ids_b, E=4)
        self.assertTrue(torch.equal(result_a, result_b))

    def test_slot_permutation_gives_identical_result_bf16(self):
        torch.manual_seed(20)
        H = 128
        values = torch.randn(2, 8, H, dtype=torch.bfloat16)

        expert_ids = list(range(8))
        rng = random.Random(42)
        shuffled = expert_ids.copy()
        rng.shuffle(shuffled)

        ids_a = torch.tensor([expert_ids, expert_ids], dtype=torch.int64)
        ids_b = torch.tensor([shuffled, shuffled], dtype=torch.int64)

        input_b = self._permute_slots(values, ids_a, ids_b, topk=8)

        result_a = self._run_moe_reduce(values, ids_a, E=8)
        result_b = self._run_moe_reduce(input_b, ids_b, E=8)
        self.assertTrue(torch.equal(result_a, result_b))

    def test_slot_invariance_with_remote_experts(self):
        """Remote experts (-1) in different positions must not change the result."""
        torch.manual_seed(30)
        H = 32
        values_a = torch.randn(1, 4, H, dtype=torch.float32)

        ids_a = torch.tensor([[0, -1, 2, -1]], dtype=torch.int64)
        ids_b = torch.tensor([[-1, 2, -1, 0]], dtype=torch.int64)

        values_b = torch.randn(1, 4, H, dtype=torch.float32)
        for slot_b in range(4):
            eid = ids_b[0, slot_b].item()
            if eid == -1:
                continue
            slot_a = (ids_a[0] == eid).nonzero(as_tuple=True)[0].item()
            values_b[0, slot_b] = values_a[0, slot_a]

        result_a = self._run_moe_reduce(values_a, ids_a, E=4)
        result_b = self._run_moe_reduce(values_b, ids_b, E=4)
        self.assertTrue(torch.equal(result_a, result_b))

    def test_moe_bf16_large_hidden_invariance(self):
        """Production-scale hidden dim (H=7168) with E=64 in BF16."""
        torch.manual_seed(40)
        H = 7168
        topk = 8
        E = 64
        tokens = 2

        expert_ids = torch.zeros(tokens, topk, dtype=torch.int64)
        for t in range(tokens):
            chosen = torch.randperm(E)[:topk]
            expert_ids[t] = chosen

        values = torch.randn(tokens, topk, H, dtype=torch.bfloat16)

        output_a = torch.zeros(tokens, H, dtype=torch.bfloat16)
        moe_sum_tree_reduce(
            input=values,
            output=output_a,
            curr_topk_ids=expert_ids,
            routed_scaling_factor=0.25,
            E=E,
        )

        perm = torch.randperm(topk)
        values_b = values[:, perm, :]
        ids_b = expert_ids[:, perm]

        output_b = torch.zeros(tokens, H, dtype=torch.bfloat16)
        moe_sum_tree_reduce(
            input=values_b,
            output=output_b,
            curr_topk_ids=ids_b,
            routed_scaling_factor=0.25,
            E=E,
        )

        self.assertTrue(torch.equal(output_a, output_b))

    def test_moe_deterministic_two_runs(self):
        """Same input, two calls -> bitwise identical."""
        torch.manual_seed(50)
        values = torch.randn(4, 4, 256, dtype=torch.bfloat16)
        ids = torch.tensor(
            [[0, 1, 2, 3], [3, 2, 1, 0], [0, 0, 1, 1], [2, 3, 0, 1]],
            dtype=torch.int64,
        )

        out_a = torch.zeros(4, 256, dtype=torch.bfloat16)
        moe_sum_tree_reduce(
            input=values,
            output=out_a,
            curr_topk_ids=ids,
            routed_scaling_factor=0.5,
            E=4,
        )

        out_b = torch.zeros(4, 256, dtype=torch.bfloat16)
        moe_sum_tree_reduce(
            input=values,
            output=out_b,
            curr_topk_ids=ids,
            routed_scaling_factor=0.5,
            E=4,
        )

        self.assertTrue(torch.equal(out_a, out_b))


# ---------------------------------------------------------------------------
# tree_all_reduce_sum: non-distributed
# ---------------------------------------------------------------------------
class TestTreeAllReduceNonDistributed(unittest.TestCase):
    def test_returns_clone_when_dist_not_initialized(self):
        if dist.is_initialized():
            self.skipTest("dist already initialized")
        x = torch.tensor([1.0, 2.0, 3.0])
        result = tree_all_reduce_sum(x)
        self.assertTrue(torch.equal(x, result))
        self.assertFalse(x.data_ptr() == result.data_ptr())

    def test_fixed_tree_sum_is_order_deterministic(self):
        torch.manual_seed(50)
        world_size = 8
        shards = [torch.randn(16, dtype=torch.bfloat16) for _ in range(world_size)]

        result_a = _fixed_tree_sum_tensors(shards)
        result_b = _fixed_tree_sum_tensors(list(shards))
        self.assertTrue(torch.equal(result_a, result_b))

    def test_tree_sum_power_of_two_sizes(self):
        for n in [1, 2, 4, 8, 16]:
            shards = [torch.tensor([float(i + 1)]) for i in range(n)]
            result = _fixed_tree_sum_tensors(shards)
            self.assertAlmostEqual(result.item(), n * (n + 1) / 2, places=5)


# ---------------------------------------------------------------------------
# MoE reduce: order-matters proof
# ---------------------------------------------------------------------------
class TestMoeReduceOrderMatters(unittest.TestCase):
    def test_expert_order_differs_from_slot_order(self):
        input_tensor = torch.tensor(
            [[[1.0e16, 0.0], [1.0, 0.0], [-1.0e16, 0.0], [2.0, 0.0]]],
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
            input_tensor[0, 0]
            + input_tensor[0, 1]
            + input_tensor[0, 2]
            + input_tensor[0, 3]
        )
        self.assertNotEqual(output[0, 0].item(), slot_order_sum[0].item())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases(unittest.TestCase):
    def test_fixed_tree_sum_single_tensor(self):
        t = torch.tensor([5.0])
        result = _fixed_tree_sum_tensors([t])
        self.assertEqual(result.item(), 5.0)

    def test_fixed_tree_sum_odd_count(self):
        values = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
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

    def test_matmul_tp_persistent_k_multi_block_non_aligned(self):
        """K not divisible by BLOCK_K: reference handles remainder gracefully."""
        torch.manual_seed(0)
        A = torch.randn(3, 300, dtype=torch.float32)
        B = torch.randn(300, 5, dtype=torch.float32)
        actual = matmul_tp_persistent(A, B)
        expected = A @ B
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

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
        expected = torch.tensor([[5.0, 10.0], [15.0, 20.0]], dtype=torch.float32)
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

    def test_moe_sum_tree_reduce_duplicate_expert_ids(self):
        """Same expert in multiple slots: both contributions must be summed."""
        input_tensor = torch.tensor(
            [[[10.0, 20.0], [30.0, 40.0]]],
            dtype=torch.float32,
        )
        curr_topk_ids = torch.tensor([[0, 0]], dtype=torch.int64)
        output = torch.empty(1, 2, dtype=torch.float32)

        moe_sum_tree_reduce(
            input=input_tensor,
            output=output,
            curr_topk_ids=curr_topk_ids,
            routed_scaling_factor=1.0,
            E=2,
        )
        expected = torch.tensor([[40.0, 60.0]], dtype=torch.float32)
        torch.testing.assert_close(output, expected)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
class TestInputValidation(unittest.TestCase):
    def test_matmul_rejects_non_2d_inputs(self):
        with self.assertRaisesRegex(ValueError, "expected 2D"):
            matmul_tp_persistent(torch.randn(2, 3, 4), torch.randn(4, 5))
        with self.assertRaisesRegex(ValueError, "expected 2D"):
            matmul_tp_persistent(torch.randn(2, 3), torch.randn(3))

    def test_matmul_rejects_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            matmul_tp_persistent(torch.randn(2, 3), torch.randn(4, 5))

    def test_moe_rejects_wrong_input_dims(self):
        with self.assertRaisesRegex(ValueError, "tokens, topk, hidden"):
            moe_sum_tree_reduce(
                input=torch.zeros(4, 8),
                output=torch.zeros(4, 8),
                curr_topk_ids=torch.zeros(4, 2, dtype=torch.int64),
                routed_scaling_factor=1.0,
                E=2,
            )

    def test_moe_rejects_wrong_topk_ids_dims(self):
        with self.assertRaisesRegex(ValueError, "tokens, topk"):
            moe_sum_tree_reduce(
                input=torch.zeros(4, 2, 8),
                output=torch.zeros(4, 8),
                curr_topk_ids=torch.zeros(4, dtype=torch.int64),
                routed_scaling_factor=1.0,
                E=2,
            )

    def test_moe_rejects_shape_mismatch_between_input_and_ids(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            moe_sum_tree_reduce(
                input=torch.zeros(4, 2, 8),
                output=torch.zeros(4, 8),
                curr_topk_ids=torch.zeros(4, 3, dtype=torch.int64),
                routed_scaling_factor=1.0,
                E=2,
            )

    def test_moe_rejects_wrong_output_shape(self):
        with self.assertRaisesRegex(ValueError, "output must have shape"):
            moe_sum_tree_reduce(
                input=torch.zeros(4, 2, 8),
                output=torch.zeros(4, 4),
                curr_topk_ids=torch.zeros(4, 2, dtype=torch.int64),
                routed_scaling_factor=1.0,
                E=2,
            )

    def test_is_power_of_two_helper(self):
        self.assertTrue(_is_power_of_two(1))
        self.assertTrue(_is_power_of_two(2))
        self.assertTrue(_is_power_of_two(64))
        self.assertFalse(_is_power_of_two(0))
        self.assertFalse(_is_power_of_two(3))
        self.assertFalse(_is_power_of_two(6))


# ---------------------------------------------------------------------------
# Distributed tree all-reduce (multi-GPU only)
# ---------------------------------------------------------------------------
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
                self.skipTest("requires power-of-two world size")

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

    @unittest.skipUnless(
        int(os.environ.get("WORLD_SIZE", "1")) > 1,
        "requires torchrun with WORLD_SIZE > 1",
    )
    def test_tree_all_reduce_bf16_bitwise_deterministic(self):
        """Run tree all-reduce twice with same inputs, verify bitwise identical."""
        own_pg = False
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            own_pg = True

        try:
            world_size = dist.get_world_size()
            if world_size & (world_size - 1) != 0:
                self.skipTest("requires power-of-two world size")

            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                device = torch.device("cuda", local_rank)
            else:
                device = torch.device("cpu")

            rank = dist.get_rank()
            torch.manual_seed(rank * 1000 + 777)
            value = torch.randn(256, device=device, dtype=torch.bfloat16)

            result_a = tree_all_reduce_sum(value)
            result_b = tree_all_reduce_sum(value)

            self.assertTrue(torch.equal(result_a, result_b))
            dist.barrier()
        finally:
            if own_pg:
                dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
