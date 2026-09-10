"""Real SM120 contract test; run with the FlashInfer version pinned by main.

No model weights or distributed service are required. Independent packed KV
fixtures check base-2 LSE and artificial DCP2/4 merges, including empty owners.
"""

import importlib.metadata
import math
import unittest

import torch

from sglang.kernels.ops.attention.flash_mla_sm120 import flashinfer_sparse_mla_forward


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSM120SparseMLALSE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 12:
            raise unittest.SkipTest("FlashInfer sparse MLA requires SM120/SM121")
        torch.manual_seed(42)
        print(
            {
                "gpu": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "flashinfer": importlib.metadata.version("flashinfer-python"),
            },
            flush=True,
        )
        # Independent byte packing, not SGLang's production quantization kernel.
        latent = torch.randn(256, 512, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.rand(256, 4, device="cuda", dtype=torch.float32) * 0.5 + 0.7
        rope = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
        cls.kv = torch.cat(
            (
                latent.view(torch.uint8),
                scales.view(torch.uint8),
                rope.view(torch.uint8),
            ),
            dim=-1,
        ).unsqueeze(1)
        cls.values = latent.float() * scales.repeat_interleave(128, dim=-1)
        cls.keys = torch.cat((cls.values, rope.float()), dim=-1)
        cls.workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        cls.scale = 1 / math.sqrt(576)

    def run_attention(self, query, indices, lengths, return_lse=True):
        return flashinfer_sparse_mla_forward(
            q=query,
            kv_cache=self.kv,
            indices=indices,
            seq_lens=lengths,
            workspace_buffer=self.workspace,
            page_size=64,
            kv_cache_dim=656,
            qk_nope_head_dim=192,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            sm_scale=self.scale,
            skip_softmax_threshold_scale_factor=None,
            return_lse=return_lse,
        )

    def fixture(self, tokens, heads, topk=2048):
        query = torch.randn(tokens, heads, 576, device="cuda", dtype=torch.bfloat16)
        # Every row uses an uneven subset; all selected tokens have owner 0 or
        # 1 in DCP4, leaving owners 2/3 empty without making the full row empty.
        candidates = torch.arange(256, device="cuda", dtype=torch.int32)
        candidates = candidates[candidates % 4 < 2]
        indices = torch.full((tokens, topk), -1, dtype=torch.int32, device="cuda")
        lengths = 65 + torch.arange(tokens, device="cuda", dtype=torch.int32) % 63
        valid = torch.arange(128, device="cuda")[None, :] < lengths[:, None]
        indices[:, :128].copy_(torch.where(valid, candidates[None, :], -1))
        return query, indices, lengths

    def test_lse_and_partition_merge(self):
        # Analytic fixture: only one RoPE component contributes to the score,
        # bypassing Q_nope FP8 quantization. Constant values separate the LSE
        # contract from the kernel's quantized probability-times-value product.
        self.values = torch.full_like(self.values, 0.5)
        self.keys = torch.cat((self.values, self.keys[:, 512:]), dim=-1)
        self.kv = torch.cat(
            (
                self.values.to(torch.float8_e4m3fn).view(torch.uint8),
                torch.ones(256, 4, device="cuda").view(torch.uint8),
                self.keys[:, 512:].to(torch.bfloat16).contiguous().view(torch.uint8),
            ),
            dim=-1,
        ).unsqueeze(1)
        for tokens, heads in ((1, 16), (6, 32), (6, 64), (6, 128), (65, 16)):
            with self.subTest(tokens=tokens, heads=heads):
                query, indices, lengths = self.fixture(tokens, heads)
                query.zero_()
                query[:, :, -1] = 1
                output, lse = self.run_attention(query, indices, lengths)
                plain = self.run_attention(query, indices, lengths, False)
                torch.testing.assert_close(output, plain, rtol=0, atol=0)
                self.assertEqual(lse.shape, (tokens, heads))
                self.assertEqual(lse.dtype, torch.float32)
                selected_keys = self.keys[indices.clamp_min(0).long()]
                selected_values = self.values[indices.clamp_min(0).long()]
                scores = (
                    torch.einsum("bhd,bkd->bhk", query.float(), selected_keys)
                    * self.scale
                )
                scores.masked_fill_((indices < 0)[:, None, :], -torch.inf)
                ref_lse = torch.logsumexp(scores, dim=-1) / math.log(2)
                ref_output = torch.einsum(
                    "bhk,bkd->bhd", scores.softmax(-1), selected_values
                )
                torch.testing.assert_close(lse, ref_lse, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(
                    output.float(), ref_output, rtol=0, atol=0.01
                )
                for size in (2, 4):
                    states = []
                    for rank in range(size):
                        valid = (indices >= 0) & (indices % size == rank)
                        order = (~valid).to(torch.int32).argsort(dim=-1, stable=True)
                        local_lengths = valid.sum(-1, dtype=torch.int32)
                        local_indices = indices.gather(-1, order)
                        local_indices = torch.where(
                            torch.arange(indices.shape[1], device="cuda")[None, :]
                            < local_lengths[:, None],
                            local_indices,
                            -1,
                        )
                        partial, partial_lse = self.run_attention(
                            query, local_indices, local_lengths
                        )
                        empty = local_lengths == 0
                        self.assertTrue((partial_lse[empty] < -1e20).all())
                        self.assertTrue((partial_lse[empty].exp2() == 0).all())
                        self.assertTrue(torch.isfinite(partial).all())
                        states.append((partial.float(), partial_lse))
                    stacked_lse = torch.stack([state[1] for state in states])
                    merged_lse = torch.logsumexp(
                        stacked_lse * math.log(2), dim=0
                    ) / math.log(2)
                    torch.testing.assert_close(merged_lse, lse, rtol=1e-5, atol=1e-5)
                    weights = ((stacked_lse - merged_lse) * math.log(2)).exp()
                    merged = (
                        weights[..., None] * torch.stack([state[0] for state in states])
                    ).sum(0)
                    torch.testing.assert_close(
                        weights.sum(0), torch.ones_like(lse), rtol=1e-5, atol=1e-5
                    )
                    torch.testing.assert_close(
                        merged, output.float(), rtol=0, atol=0.01
                    )

    def test_random_quantization_diagnostics(self):
        # This is a diagnostic, not a model-accuracy gate: FlashInfer internally
        # quantizes Q_nope and P to FP8, while this independent oracle is FP32.
        for tokens, heads in ((1, 16), (6, 32), (6, 64), (6, 128), (65, 16)):
            query, indices, lengths = self.fixture(tokens, heads)
            output, lse = self.run_attention(query, indices, lengths)
            torch.testing.assert_close(
                output,
                self.run_attention(query, indices, lengths, False),
                rtol=0,
                atol=0,
            )
            scores = (
                torch.einsum(
                    "bhd,bkd->bhk",
                    query.float(),
                    self.keys[indices.clamp_min(0).long()],
                )
                * self.scale
            )
            scores.masked_fill_((indices < 0)[:, None, :], -torch.inf)
            ref_output = torch.einsum(
                "bhk,bkd->bhd",
                scores.softmax(-1),
                self.values[indices.clamp_min(0).long()],
            )
            ref_lse = torch.logsumexp(scores, dim=-1) / math.log(2)
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.isfinite(lse).all())
            print(
                {
                    "tokens": tokens,
                    "heads": heads,
                    "output_max_abs": (output.float() - ref_output).abs().max().item(),
                    "output_relative_rms": (
                        (output.float() - ref_output).square().mean()
                        / ref_output.square().mean()
                    )
                    .sqrt()
                    .item(),
                    "lse_max_abs": (lse - ref_lse).abs().max().item(),
                },
                flush=True,
            )

    def test_cuda_graph_replay_with_changed_lengths(self):
        query, indices, lengths = self.fixture(6, 32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.run_attention(query, indices, lengths)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual_output, actual_lse = self.run_attention(query, indices, lengths)
        for length in (65, 1, 0, 93):
            lengths.fill_(length)
            candidates = torch.arange(
                indices.shape[1], device="cuda", dtype=torch.int32
            )
            indices.copy_(
                torch.where(
                    candidates[None, :] < lengths[:, None], candidates[None, :], -1
                )
            )
            expected_output, expected_lse = self.run_attention(query, indices, lengths)
            graph.replay()
            torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
            torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(actual_output).all())
            if length == 0:
                self.assertTrue((actual_lse < -1e20).all())
                self.assertTrue((actual_lse.exp2() == 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
