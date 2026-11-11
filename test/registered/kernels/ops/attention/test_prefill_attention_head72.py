"""Head72 attention: independent reference, graph updates and dispatch guards."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import triton

from sglang.kernels.ops.attention import prefill_attention as attention
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] >= 8,
    "Requires CUDA with BF16 tensor cores",
)
class TestPrefillAttentionHead72(CustomTestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def _case(self, lengths, heads, amplitude=1.0, dtype=torch.bfloat16, dim=72):
        total = sum(lengths)
        packed = torch.randn(total, 3, heads, dim, device="cuda", dtype=dtype)
        packed.mul_(amplitude)
        q, k, v = packed.unbind(1)
        seq_lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        starts = torch.cat(
            [torch.zeros_like(seq_lens[:1]), seq_lens.cumsum(0)[:-1]]
        ).int()
        storage = torch.full(
            (total + 2, heads, dim + 8), 123, device="cuda", dtype=dtype
        )
        out = storage[1:-1, :, :dim]
        return packed, q, k, v, out, storage, starts, seq_lens

    def _launch(
        self, q, k, v, out, starts, lengths, maximum, split, scale=None, causal=False
    ):
        scale = q.shape[-1] ** -0.5 if scale is None else scale
        return attention._fwd_kernel[
            (len(lengths), q.shape[1], triton.cdiv(maximum, 128))
        ](
            q,
            k,
            v,
            scale,
            starts,
            lengths,
            out,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            out.stride(0),
            out.stride(1),
            kv_group_num=q.shape[1] // k.shape[1],
            BLOCK_M=128,
            BLOCK_DMODEL=triton.next_power_of_2(k.shape[-1]),
            BLOCK_N=128,
            IS_CAUSAL=causal,
            Lk=k.shape[-1],
            SPLIT_OUTPUT=split,
            num_warps=8,
            num_stages=1,
        )

    def _reference(self, q, k, v, lengths, scale=None, causal=False):
        # Independent CPU FP64 dense softmax, including the complete key axis.
        q, k, v = [x.detach().cpu().double() for x in (q, k, v)]
        scale = q.shape[-1] ** -0.5 if scale is None else scale
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        outputs = []
        start = 0
        for length in lengths:
            if length:
                end = start + length
                scores = (
                    torch.einsum("qhd,khd->hqk", q[start:end], k[start:end]) * scale
                )
                if causal:
                    mask = torch.ones(length, length, dtype=torch.bool).triu(1)
                    scores.masked_fill_(mask, -torch.inf)
                outputs.append(
                    torch.einsum("hqk,khd->qhd", scores.softmax(-1), v[start:end])
                )
            start += length
        return torch.cat(outputs)

    def _check(self, actual, original, reference):
        output_epsilon = torch.finfo(actual.dtype).eps
        actual = actual.detach().cpu().double()
        original = original.detach().cpu().double()
        self.assertTrue(torch.isfinite(actual).all().item())
        error = actual - reference
        reference_error = original - reference
        norm = max(reference.norm().item(), 1e-20)
        # A fixed absolute bound alone can be smaller than BF16 rounding
        # for large outputs. Keep an independent, magnitude-aware bound,
        # plus both maximum and L2 error regression checks against upstream.
        torch.testing.assert_close(actual, reference, atol=0.02, rtol=output_epsilon)
        self.assertLessEqual(
            error.abs().max().item(),
            1.05 * reference_error.abs().max().item() + 1e-7,
        )
        self.assertLessEqual(error.norm().item() / norm, 0.01)
        self.assertLessEqual(
            error.norm().item() / norm,
            1.05 * reference_error.norm().item() / norm + 1e-7,
        )

    def test_ragged_tails_strides_and_independent_reference(self):
        for heads in (1, 3, 16, 17):
            for lengths, amplitude, scale in (
                ([0, 1, 7, 127, 128, 129, 257], 1.0, None),
                ([63, 64, 65, 195, 196], 0.2, 0.5),
                ([1, 17, 129], 3.0, None),
            ):
                with self.subTest(heads=heads, lengths=lengths, amplitude=amplitude):
                    packed, q, k, v, out, storage, starts, lens = self._case(
                        lengths, heads, amplitude
                    )
                    inputs = packed.clone()
                    original = torch.empty_like(out)
                    self._launch(
                        q, k, v, original, starts, lens, max(lengths), False, scale
                    )
                    self._launch(q, k, v, out, starts, lens, max(lengths), True, scale)
                    reference = self._reference(q, k, v, lengths, scale)
                    self._check(out, original, reference)
                    torch.testing.assert_close(packed, inputs, rtol=0, atol=0)
                    self.assertTrue((storage[0] == 123).all().item())
                    self.assertTrue((storage[-1] == 123).all().item())
                    self.assertTrue((storage[:, :, 72:] == 123).all().item())

    def test_cuda_graph_reuses_updated_inputs(self):
        lengths = [7, 128, 257]
        packed, q, k, v, out, storage, starts, lens = self._case(lengths, 3)
        stream = torch.cuda.Stream(device=q.device)
        self._launch(q, k, v, out, starts, lens, max(lengths), True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self._launch(q, k, v, out, starts, lens, max(lengths), True)
        for seed in (43, 44, None):
            if seed is None:
                packed.zero_()
            else:
                torch.manual_seed(seed)
                packed.normal_()
            inputs = packed.clone()
            original = torch.empty_like(out)
            eager = torch.empty_like(out)
            self._launch(q, k, v, original, starts, lens, max(lengths), False)
            self._launch(q, k, v, eager, starts, lens, max(lengths), True)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(out.view(torch.int16), eager.view(torch.int16)))
            self._check(out, original, self._reference(q, k, v, lengths))
            torch.testing.assert_close(packed, inputs, rtol=0, atol=0)
            self.assertTrue((storage[:, :, 72:] == 123).all().item())

    def test_public_dispatch_and_fallback_reference(self):
        for dtype, dim, causal, kv_heads in (
            (torch.bfloat16, 72, False, 4),
            (torch.bfloat16, 72, True, 4),
            (torch.bfloat16, 72, False, 2),
            (torch.float16, 72, False, 4),
            (torch.bfloat16, 64, False, 4),
            (torch.bfloat16, 80, False, 4),
        ):
            with self.subTest(dtype=dtype, dim=dim, causal=causal, kv_heads=kv_heads):
                lengths = [7, 129]
                packed, q, k, v, out, storage, starts, lens = self._case(
                    lengths, 4, dtype=dtype, dim=dim
                )
                k, v = k[:, :kv_heads], v[:, :kv_heads]
                inputs = packed.clone()
                attention.context_attention_fwd(
                    q, k, v, out, starts, lens, max(lengths), is_causal=causal
                )
                reference = self._reference(q, k, v, lengths, causal=causal)
                error = out.detach().cpu().double() - reference
                self.assertTrue(torch.isfinite(out).all().item())
                self.assertLessEqual(error.abs().max().item(), 0.02)
                self.assertLessEqual(
                    error.norm().item() / reference.norm().item(), 0.01
                )
                torch.testing.assert_close(packed, inputs, rtol=0, atol=0)
                self.assertTrue((storage[:, :, dim:] == 123).all().item())

    def test_dispatch_scope(self):
        class Recorder:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    self.selected = kwargs["SPLIT_OUTPUT"]

                return launch

        cases = [
            (12, 0, torch.bfloat16, 72, False, 4, 4, 1, True),
            (12, 1, torch.bfloat16, 72, False, 4, 4, 1, False),
            (9, 0, torch.bfloat16, 72, False, 4, 4, 1, False),
            (12, 0, torch.float16, 72, False, 4, 4, 1, False),
            (12, 0, torch.bfloat16, 64, False, 4, 4, 1, False),
            (12, 0, torch.bfloat16, 80, False, 4, 4, 1, False),
            (12, 0, torch.bfloat16, 72, True, 4, 4, 1, False),
            (12, 0, torch.bfloat16, 72, False, 4, 2, 1, False),
            (12, 0, torch.bfloat16, 72, False, 4, 4, 2, False),
        ]
        for major, minor, dtype, dim, causal, heads, kv_heads, step, expected in cases:
            with self.subTest(
                case=(major, minor, dtype, dim, causal, heads, kv_heads, step)
            ):
                q = torch.empty((7, heads, dim * step), device="cuda", dtype=dtype)[
                    ..., ::step
                ]
                k = torch.empty((7, kv_heads, dim), device="cuda", dtype=dtype)
                v = torch.empty_like(k)
                out = torch.empty_like(q)
                starts = torch.tensor([0], device="cuda", dtype=torch.int32)
                lens = torch.tensor([7], device="cuda", dtype=torch.int32)
                arch = SimpleNamespace(
                    props=SimpleNamespace(major=major, minor=minor),
                    cuda=SimpleNamespace(is_sm=lambda a, b: (a, b) == (major, minor)),
                )
                recorder = Recorder()
                with (
                    mock.patch.object(attention, "current_arch", return_value=arch),
                    mock.patch.object(attention, "_fwd_kernel", recorder),
                ):
                    attention.context_attention_fwd(
                        q, k, v, out, starts, lens, 7, is_causal=causal
                    )
                self.assertEqual(recorder.selected, expected)


if __name__ == "__main__":
    unittest.main()
