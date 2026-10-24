import unittest
from unittest.mock import patch

import torch

import sglang.srt.layers.attention.vision as vision_attention


class TestVisionFlash4AttentionKwargs(unittest.TestCase):
    def _run_with_captured_flash_attn(self, **forward_kwargs):
        captured = {}

        def fake_flash_attn_func(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return torch.empty_like(args[0])

        q = torch.randn(4, 2, 8)
        k = torch.randn(4, 2, 8)
        v = torch.randn(4, 2, 8)
        cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)

        with (
            patch.object(vision_attention, "_is_cuda", True),
            patch.object(
                vision_attention,
                "flash_attn_func",
                side_effect=fake_flash_attn_func,
                create=True,
            ),
        ):
            attn = vision_attention.VisionFlash4Attention()
            output = attn(
                q=q,
                k=k,
                v=v,
                cu_seqlens=cu_seqlens,
                bsz=1,
                seq_len=4,
                softmax_scale=0.125,
                **forward_kwargs,
            )

        self.assertEqual(output.shape, q.shape)
        self.assertIs(captured["args"][0], q)
        self.assertIs(captured["args"][1], k)
        self.assertIs(captured["args"][2], v)
        self.assertIs(captured["kwargs"]["cu_seqlens_q"], cu_seqlens)
        self.assertIs(captured["kwargs"]["cu_seqlens_k"], cu_seqlens)
        self.assertEqual(captured["kwargs"]["max_seqlen_q"], 4)
        self.assertEqual(captured["kwargs"]["max_seqlen_k"], 4)
        self.assertEqual(captured["kwargs"]["softmax_scale"], 0.125)
        self.assertEqual(captured["kwargs"]["ver"], 4)
        return captured["kwargs"]

    def test_forwards_window_size_and_sinks_to_fa4_kernel(self):
        s_aux = torch.randn(2)

        kwargs = self._run_with_captured_flash_attn(
            window_size=(32, 32),
            s_aux=s_aux,
        )

        self.assertEqual(kwargs["window_size"], (32, 32))
        self.assertIs(kwargs["sinks"], s_aux)

    def test_defaults_to_full_attention_without_sinks(self):
        kwargs = self._run_with_captured_flash_attn()

        self.assertEqual(kwargs["window_size"], (-1, -1))
        self.assertNotIn("sinks", kwargs)


if __name__ == "__main__":
    unittest.main()
