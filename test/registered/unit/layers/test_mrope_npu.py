"""CPU regressions for multimodal positions in the NPU RoPE dispatch."""

import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.rotary_embedding import mrope
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMRotaryEmbeddingNPU(unittest.TestCase):
    def setUp(self):
        angles = torch.arange(128).float()[:, None] * (
            10000 ** (-torch.arange(32).float() / 32)
        )
        self.rope = SimpleNamespace(
            head_size=256,
            rotary_dim=64,
            is_neox_style=True,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
            cos_sin_cache=torch.cat((angles.cos(), angles.sin()), dim=-1),
        )
        self.rope.forward_native = MethodType(
            mrope.MRotaryEmbedding.forward_native, self.rope
        )

    def test_multimodal_tp_matches_full_rotation(self):
        generator = torch.Generator().manual_seed(42)
        # Qwen3.5-27B: 24 Q heads, 4 KV heads, partial interleaved MRoPE.
        for tokens in (1, 80):
            positions = torch.arange(tokens)[None, :] + torch.tensor([[0], [1], [3]])
            query = torch.randn(tokens, 24, 256, generator=generator)
            key = torch.randn(tokens, 4, 256, generator=generator)
            for interleaved in (False, True):
                self.rope.mrope_interleaved = interleaved
                axes = (
                    torch.arange(32) % 3
                    if interleaved
                    else torch.repeat_interleave(
                        torch.arange(3), torch.tensor([11, 11, 10])
                    )
                )
                cache = self.rope.cos_sin_cache[positions]
                lanes = torch.arange(32)
                cos = cache[axes, :, lanes].T[:, None, :].repeat(1, 1, 2)
                sin = cache[axes, :, lanes + 32].T[:, None, :].repeat(1, 1, 2)
                expected = []
                for x in (query, key):
                    rotated = torch.cat((-x[..., 32:64], x[..., :32]), dim=-1)
                    expected.append(
                        torch.cat(
                            (x[..., :64] * cos + rotated * sin, x[..., 64:]), dim=-1
                        )
                    )
                for tp in (1, 2, 4, 8):
                    for rank in range(tp):
                        with self.subTest(
                            tokens=tokens, interleaved=interleaved, tp=tp, rank=rank
                        ):
                            q_slice = slice(rank * (24 // tp), (rank + 1) * (24 // tp))
                            kv_heads = max(1, 4 // tp)
                            kv_rank = rank // max(1, tp // 4)
                            k_slice = slice(
                                kv_rank * kv_heads, (kv_rank + 1) * kv_heads
                            )
                            q = query[:, q_slice].reshape(tokens, -1)
                            k = key[:, k_slice].reshape(tokens, -1)
                            with mock.patch.object(
                                mrope, "torch_npu", create=True
                            ) as npu:
                                npu.npu_mrope.side_effect = AssertionError(
                                    "3-axis positions must not use ordinary RoPE"
                                )
                                actual_q, actual_k = mrope.MRotaryEmbedding.forward_npu(
                                    self.rope, positions, q, k
                                )
                                npu.npu_mrope.assert_not_called()
                            torch.testing.assert_close(
                                actual_q, expected[0][:, q_slice].reshape_as(q)
                            )
                            torch.testing.assert_close(
                                actual_k, expected[1][:, k_slice].reshape_as(k)
                            )

    def test_ordinary_rope_keeps_npu_dispatch(self):
        positions = torch.arange(2)
        query, key = torch.randn(2, 4096), torch.randn(2, 512)
        for neox, mode in ((True, "half"), (False, "interleave")):
            self.rope.is_neox_style = neox
            with mock.patch.object(mrope, "torch_npu", create=True) as npu:
                npu.npu_mrope.return_value = (query, key)
                result = mrope.MRotaryEmbedding.forward_npu(
                    self.rope, positions, query, key
                )
                self.assertIs(result[0], query)
                self.assertIs(result[1], key)
                npu.npu_mrope.assert_called_once_with(
                    positions,
                    query,
                    key,
                    self.rope.cos_sin_cache,
                    256,
                    mrope_section=[0, 0, 0],
                    rotary_mode=mode,
                )


if __name__ == "__main__":
    unittest.main()
