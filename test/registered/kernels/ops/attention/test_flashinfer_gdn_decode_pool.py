import unittest

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="stage-b", runner_config="small-1-gpu")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "Requires SM90",
)
class TestFlashInferGDNDecodePool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = FlashInferGDNKernel()

    def _inputs(self, state_dtype, index_dtype):
        torch.manual_seed(0)
        batch, heads, value_heads, dim, slots = 8, 4, 12, 128, 17
        self.patterns = (
            [1, 3, -1, 6, 9, -1, 12, 14],
            [2, -1, 4, 5, -1, 7, 10, 13],
        )
        self.q, self.k = [
            torch.randn(1, batch, heads, dim, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        self.v = torch.randn(
            1, batch, value_heads, dim, device="cuda", dtype=torch.bfloat16
        )
        self.a, self.b = [
            torch.randn(1, batch, value_heads, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        self.A_log = torch.zeros(value_heads, device="cuda", dtype=torch.float32)
        self.dt_bias = torch.zeros(value_heads, device="cuda", dtype=torch.bfloat16)
        self.query_start_loc = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        self.original = torch.randn(
            slots, value_heads, dim, dim, device="cuda", dtype=state_dtype
        )
        self.pool = self.original.clone()
        self.indices = torch.tensor(self.patterns[0], device="cuda", dtype=index_dtype)

    def _decode(self):
        return self.kernel.decode(
            self.q,
            self.k,
            self.v,
            self.a,
            self.b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            ssm_states=self.pool,
            cache_indices=self.indices,
            query_start_loc=self.query_start_loc,
        )

    def _reference(self, pattern):
        from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

        rows = [i for i, slot in enumerate(pattern) if slot >= 0]
        slots = [slot for slot in pattern if slot >= 0]
        batch = len(rows)
        output, state = gated_delta_rule_decode_pretranspose(
            q=self.q[:, rows].contiguous().view(batch, 1, *self.q.shape[2:]),
            k=self.k[:, rows].contiguous().view(batch, 1, *self.k.shape[2:]),
            v=self.v[:, rows].contiguous().view(batch, 1, *self.v.shape[2:]),
            state=self.original[slots].clone(),
            A_log=self.A_log,
            a=self.a[:, rows].contiguous().view(batch, 1, -1),
            dt_bias=self.dt_bias,
            b=self.b[:, rows].contiguous().view(batch, 1, -1),
            use_qk_l2norm=True,
        )
        return rows, slots, output[:, 0], state

    def _check(self, output, pattern, reference):
        rows, slots, expected_output, expected_state = reference
        untouched = [i for i in range(1, self.pool.shape[0]) if i not in slots]
        # Slot zero is reserved; every other unused slot must remain unchanged.
        # In particular, -1 padding must never alias the last real pool slot.
        self.assertTrue(torch.equal(self.pool[untouched], self.original[untouched]))
        self.assertTrue(torch.equal(self.pool[-1], self.original[-1]))
        torch.testing.assert_close(
            self.pool[slots], expected_state, rtol=0.02, atol=0.02
        )
        torch.testing.assert_close(
            output[0, rows], expected_output, rtol=0.02, atol=0.02
        )
        self.assertTrue(torch.isfinite(output[0, rows]).all().item())
        if self.pool.dtype == torch.float32:
            padding_rows = [i for i, slot in enumerate(pattern) if slot < 0]
            self.assertEqual(torch.count_nonzero(output[0, padding_rows]).item(), 0)

    def test_eager_padding(self):
        for state_dtype in (torch.bfloat16, torch.float32):
            for index_dtype in (torch.int32, torch.int64):
                with self.subTest(state_dtype=state_dtype, index_dtype=index_dtype):
                    self._inputs(state_dtype, index_dtype)
                    reference = self._reference(self.patterns[0])
                    self._check(self._decode(), self.patterns[0], reference)

    def test_graph_replay_updates_indices(self):
        for state_dtype in (torch.bfloat16, torch.float32):
            for index_dtype in (torch.int32, torch.int64):
                with self.subTest(state_dtype=state_dtype, index_dtype=index_dtype):
                    self._inputs(state_dtype, index_dtype)
                    references = [self._reference(p) for p in self.patterns]
                    # Compile and warm the decode kernel before graph capture.
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            self._decode()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = self._decode()
                    for pattern, reference in zip(self.patterns, references):
                        self.pool.copy_(self.original)
                        self.indices.copy_(
                            torch.tensor(pattern, device="cuda", dtype=index_dtype)
                        )
                        graph.replay()
                        self._check(output, pattern, reference)


if __name__ == "__main__":
    unittest.main()
