"""BF16 bidirectional HD512 extend attention, including graph reuse."""

import unittest

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.hd512_bmm_attention import hd512_bmm_attention
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "Exercises the data-center Blackwell HD512 launch configuration",
)
class TestExtendAttentionHD512(CustomTestCase):
    def _check(self, prefix_lens, extend_lens, *, graph=False):
        torch.manual_seed(42)
        hq, hk, dim = 16, 2, 512
        total_q, total_k = sum(extend_lens), sum(prefix_lens)
        # Match the strided Q view from the model's packed QKV projection.
        packed = (
            torch.randn(
                total_q, (hq + 2 * hk) * dim, device="cuda", dtype=torch.bfloat16
            )
            * 0.15
        )
        q = packed[:, : hq * dim].view(total_q, hq, dim)
        k = packed[:, hq * dim : (hq + hk) * dim].contiguous().view(total_q, hk, dim)
        v = packed[:, (hq + hk) * dim :].contiguous().view(total_q, hk, dim)
        pool_k = (
            torch.randn(
                max(1, total_k + 17), hk, dim, device="cuda", dtype=torch.bfloat16
            )
            * 0.15
        )
        pool_v = torch.randn_like(pool_k) * 0.15
        indices = torch.randperm(pool_k.shape[0], device="cuda")[:total_k].int()
        qo = [0]
        kv = [0]
        for n, m in zip(prefix_lens, extend_lens):
            qo.append(qo[-1] + m)
            kv.append(kv[-1] + n)
        qo_gpu = torch.tensor(qo, device="cuda", dtype=torch.int32)
        kv_gpu = torch.tensor(kv, device="cuda", dtype=torch.int32)
        output = torch.empty_like(q, memory_format=torch.contiguous_format)

        def run():
            extend_attention_fwd(
                q,
                k,
                v,
                output,
                pool_k,
                pool_v,
                qo_gpu,
                kv_gpu,
                indices,
                custom_mask=None,
                is_causal=False,
                mask_indptr=None,
                max_len_extend=max(extend_lens),
                k_scale=1.0,
                v_scale=1.0,
                sm_scale=1.0,
                sliding_window_size=-1,
            )

        run()
        if graph:
            torch.cuda.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured):
                run()
            # Replay with changed query values and changed page indirection.
            q.mul_(1.5)
            indices.copy_(indices.flip(0))
            captured.replay()

        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for batch in range(len(extend_lens)):
                start, end = qo[batch : batch + 2]
                slots = indices[kv[batch] : kv[batch + 1]].long()
                keys = torch.cat((pool_k[slots], k[start:end]))
                values = torch.cat((pool_v[slots], v[start:end]))
                keys = keys.repeat_interleave(hq // hk, dim=1).transpose(0, 1).float()
                values = (
                    values.repeat_interleave(hq // hk, dim=1).transpose(0, 1).float()
                )
                queries = q[start:end].transpose(0, 1).float()
                probabilities = torch.softmax(queries @ keys.transpose(1, 2), dim=-1)
                expected = (probabilities @ values).transpose(0, 1)
                actual = output[start:end].float()
                torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.03)
                relative_rms = (
                    ((actual - expected).square().mean() / expected.square().mean())
                    .sqrt()
                    .item()
                )
                self.assertLessEqual(relative_rms, 0.01)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    def test_prefix_lengths_and_partial_tiles(self):
        for prefix, canvas in [
            (0, 1),
            (1, 15),
            (257, 31),
            (905, 255),
            (8000, 256),
            (9024, 257),
        ]:
            with self.subTest(prefix=prefix, canvas=canvas):
                self._check([prefix], [canvas])

    def test_ragged_batch_graph_reuse(self):
        self._check([33, 8000], [17, 256], graph=True)

    def test_packed_bmm_graph_dynamic_prefix(self):
        torch.manual_seed(7)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for capacity, canvas, scale in [
                (0, 32, 1.0),
                (905, 255, 0.125),
                (16384, 256, 1.0),
            ]:
                with self.subTest(capacity=capacity, canvas=canvas, scale=scale):
                    packed = (
                        torch.randn(
                            canvas, 20 * 512, device="cuda", dtype=torch.bfloat16
                        )
                        * 0.15
                    )
                    q = packed[:, : 16 * 512].view(canvas, 16, 512)
                    k = packed[:, 16 * 512 : 18 * 512].contiguous().view(canvas, 2, 512)
                    v = packed[:, 18 * 512 :].contiguous().view(canvas, 2, 512)
                    pool_k = (
                        torch.randn(
                            capacity + 17, 2, 512, device="cuda", dtype=torch.bfloat16
                        )
                        * 0.15
                    )
                    pool_v = torch.randn_like(pool_k) * 0.15
                    # A nonzero indptr start and invalid padding detect stale
                    # length use and speculative reads outside the valid prefix.
                    indices = torch.full(
                        (capacity + 7,), 2**30, device="cuda", dtype=torch.int64
                    )
                    indptr = torch.tensor([7, 7], device="cuda", dtype=torch.int32)

                    def run():
                        return hd512_bmm_attention(
                            q, k, v, pool_k, pool_v, indices, indptr, capacity, scale
                        )

                    run()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = run()
                    for prefix in sorted(
                        {
                            0,
                            min(1, capacity),
                            min(257, capacity),
                            min(8000, capacity),
                            capacity,
                        },
                        reverse=True,
                    ):
                        indices.fill_(2**30)
                        indices[7 : prefix + 7] = torch.randint(
                            pool_k.shape[0], (prefix,), device="cuda"
                        )
                        indptr[1] = prefix + 7
                        q.mul_(1.01)
                        pool_v.mul_(1.02)
                        graph.replay()
                        slots = indices[7 : prefix + 7]
                        keys = (
                            torch.cat((pool_k[slots], k))
                            .repeat_interleave(8, dim=1)
                            .transpose(0, 1)
                            .float()
                        )
                        values = (
                            torch.cat((pool_v[slots], v))
                            .repeat_interleave(8, dim=1)
                            .transpose(0, 1)
                            .float()
                        )
                        scores = (
                            q.transpose(0, 1).float() @ keys.transpose(1, 2) * scale
                        )
                        expected = (scores.softmax(-1) @ values).transpose(0, 1)
                        actual = output.float()
                        torch.testing.assert_close(
                            actual, expected, atol=0.005, rtol=0.03
                        )
                        error = (
                            (
                                (actual - expected).square().mean()
                                / expected.square().mean()
                            )
                            .sqrt()
                            .item()
                        )
                        self.assertLessEqual(error, 0.01)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32


if __name__ == "__main__":
    unittest.main()
