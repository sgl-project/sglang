import unittest

import torch

from sglang.kernels.ops.attention import (
    can_use_kda_qwen38_qsa_sm121,
    qwen38_qsa_sm121_varlen,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_REAL_SHAPES = (
    (1, 12, 1, (858,)),
    (3, 12, 1, (113, 511, 908)),
    (4, 12, 1, (64, 861, 1307, 2051)),
    (12, 12, 1, (97, 211, 389, 511, 607, 701, 809, 857, 881, 893, 907, 911)),
    (1, 24, 2, (858,)),
    (3, 24, 2, (113, 511, 908)),
    (4, 24, 2, (64, 861, 1307, 2051)),
    (12, 24, 2, (97, 211, 389, 511, 607, 701, 809, 857, 881, 893, 907, 911)),
)


def _make_inputs(num_q_heads, num_kv_heads, lengths, *, capacity=None):
    batch = len(lengths)
    if capacity is None:
        capacity = sum(lengths)
    q = torch.randn(batch, num_q_heads, 256, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(capacity, num_kv_heads, 256, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    cu_k = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )
    return q, k, v, cu_q, cu_k


def _reference(args, scale):
    q, k, v, _, cu_k = args
    queries_per_kv = q.shape[1] // k.shape[1]
    rows = []
    for row in range(q.shape[0]):
        start = int(cu_k[row])
        end = int(cu_k[row + 1])
        keys = k[start:end].repeat_interleave(queries_per_kv, dim=1).float()
        values = v[start:end].repeat_interleave(queries_per_kv, dim=1).float()
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys) * scale
        rows.append(torch.einsum("hk,khd->hd", scores.softmax(-1), values))
    return torch.stack(rows).to(q.dtype)


class TestKdaQwen38QsaSm121(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        if torch.cuda.get_device_capability() != (12, 1):
            raise unittest.SkipTest("requires an SM121 GPU")
        torch.manual_seed(2026)

    @classmethod
    def tearDownClass(cls):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def assert_matches_reference(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        relative_l2 = (
            actual.float() - expected.float()
        ).norm() / expected.float().norm()
        # Synthetic normal inputs can have a much smaller output norm than the
        # captured post-projection tensors. The real replay separately enforces
        # the task's stricter relative-L2 <= 2e-3 acceptance gate.
        self.assertLessEqual(relative_l2.item(), 5e-3)

    def test_real_tp1_tp2_shapes_match_reference(self):
        scale = 256**-0.5
        for _, num_q_heads, num_kv_heads, lengths in _REAL_SHAPES:
            args = _make_inputs(num_q_heads, num_kv_heads, lengths)
            max_seqlen_k = max(lengths)
            self.assertTrue(
                can_use_kda_qwen38_qsa_sm121(*args, max_seqlen_k=max_seqlen_k)
            )
            expected = _reference(args, scale)
            actual = qwen38_qsa_sm121_varlen(
                *args, max_seqlen_k=max_seqlen_k, softmax_scale=scale
            )
            self.assert_matches_reference(actual, expected)

    def test_all_low_concurrency_batches(self):
        scale = 256**-0.5
        for num_q_heads, num_kv_heads in ((12, 1), (24, 2)):
            for batch in range(1, 17):
                lengths = tuple(17 + (row * 37) % 211 for row in range(batch))
                args = _make_inputs(num_q_heads, num_kv_heads, lengths)
                max_seqlen_k = max(lengths)
                self.assertTrue(
                    can_use_kda_qwen38_qsa_sm121(*args, max_seqlen_k=max_seqlen_k)
                )
                actual = qwen38_qsa_sm121_varlen(
                    *args,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=scale,
                )
                self.assert_matches_reference(actual, _reference(args, scale))

    def test_extended_batches_and_contract_guard(self):
        scale = 256**-0.5
        for num_q_heads, num_kv_heads in ((12, 1), (24, 2)):
            for batch in (17, 32, 64, 128):
                lengths = tuple(17 + (row * 37) % 211 for row in range(batch))
                args = _make_inputs(num_q_heads, num_kv_heads, lengths)
                max_seqlen_k = max(lengths)
                self.assertTrue(
                    can_use_kda_qwen38_qsa_sm121(*args, max_seqlen_k=max_seqlen_k)
                )
                actual = qwen38_qsa_sm121_varlen(
                    *args,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=scale,
                )
                self.assert_matches_reference(actual, _reference(args, scale))

        unsupported = _make_inputs(12, 1, tuple(17 for _ in range(129)))
        self.assertFalse(can_use_kda_qwen38_qsa_sm121(*unsupported, max_seqlen_k=17))
        with self.assertRaisesRegex(ValueError, "unsupported SM121 QSA call"):
            qwen38_qsa_sm121_varlen(*unsupported, max_seqlen_k=17, softmax_scale=scale)

    def test_cuda_graph_replays_live_cu_seqlens(self):
        scale = 256**-0.5
        capacity = 4 * 2051
        args = _make_inputs(12, 1, (64, 511, 861, 2051), capacity=capacity)
        q, k, v, cu_q, cu_k = args

        # Allocate scratch and compile before capture.
        qwen38_qsa_sm121_varlen(*args, max_seqlen_k=2051, softmax_scale=scale)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = qwen38_qsa_sm121_varlen(
                *args, max_seqlen_k=2051, softmax_scale=scale
            )

        replay_lengths = (2051, 1307, 511, 64)
        cu_k.copy_(
            torch.tensor(
                [0, *torch.tensor(replay_lengths).cumsum(0).tolist()],
                dtype=torch.int32,
                device="cuda",
            )
        )
        q.normal_()
        k.normal_()
        v.normal_()
        graph.replay()
        torch.cuda.synchronize()
        expected = _reference(args, scale)
        self.assert_matches_reference(captured, expected)

    def test_split_counters_reset_across_repeated_launches(self):
        from sglang.kernels.kda_kernels.qwen38_qsa_sm121.kernel import _get_scratch

        scale = 256**-0.5
        # bs=17 and TP1's two KV heads exercise 34 counter slots, crossing the
        # old bs<=16 scratch boundary while long rows require two active splits.
        args = _make_inputs(24, 2, tuple(1537 for _ in range(17)))
        expected = _reference(args, scale)
        for _ in range(100):
            actual = qwen38_qsa_sm121_varlen(
                *args, max_seqlen_k=2051, softmax_scale=scale
            )
        torch.cuda.synchronize()
        self.assert_matches_reference(actual, expected)
        counters = _get_scratch(torch.device("cuda"))[-1]
        self.assertEqual(torch.count_nonzero(counters).item(), 0)


if __name__ == "__main__":
    unittest.main()
