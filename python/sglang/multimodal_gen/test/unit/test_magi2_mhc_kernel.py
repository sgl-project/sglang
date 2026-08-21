# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.configs.models.dits.magi2 import Magi2PreviewArchConfig

NUM_STREAM = 4
HIDDEN = 128
TOKENS = 37


def _inputs(device):
    torch.manual_seed(0)
    streams = torch.randn(
        TOKENS, NUM_STREAM, HIDDEN, device=device, dtype=torch.bfloat16
    )
    block_out = torch.randn(TOKENS, HIDDEN, device=device, dtype=torch.bfloat16)
    post = torch.rand(TOKENS, NUM_STREAM, device=device) * 2.0
    res = torch.rand(TOKENS, NUM_STREAM, NUM_STREAM, device=device)
    return streams, block_out, post, res / res.sum(-1, keepdim=True)


def _reference(streams, block_out, post, res):
    """fp32 throughout: the kernel accumulates in fp32, the einsum spelling does not."""
    return torch.einsum("tij,tjc->tic", res.float(), streams.float()) + (
        post.float()[..., None] * block_out.float()[:, None, :]
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestMhcMixOutputKernel(unittest.TestCase):
    def test_matches_an_fp32_reference(self):
        from sglang.multimodal_gen.runtime.layers.magi2_mhc_kernel import (
            mhc_mix_output,
        )

        args = _inputs(torch.device("cuda"))
        got = mhc_mix_output(*args)
        self.assertEqual(got.shape, args[0].shape)
        self.assertEqual(got.dtype, args[0].dtype)
        # bf16 output spacing at these magnitudes is ~0.06.
        torch.testing.assert_close(got.float(), _reference(*args), rtol=0, atol=0.07)

    def test_hidden_size_wider_than_one_column_block(self):
        from sglang.multimodal_gen.runtime.layers.magi2_mhc_kernel import (
            mhc_mix_output,
        )

        hidden = Magi2PreviewArchConfig().hidden_size
        dev = torch.device("cuda")
        torch.manual_seed(1)
        streams = torch.randn(8, NUM_STREAM, hidden, device=dev, dtype=torch.bfloat16)
        block_out = torch.randn(8, hidden, device=dev, dtype=torch.bfloat16)
        post = torch.rand(8, NUM_STREAM, device=dev) * 2.0
        res = torch.rand(8, NUM_STREAM, NUM_STREAM, device=dev)
        res = res / res.sum(-1, keepdim=True)

        torch.testing.assert_close(
            mhc_mix_output(streams, block_out, post, res).float(),
            _reference(streams, block_out, post, res),
            rtol=0,
            atol=0.07,
        )

    def test_registered_as_an_op_so_compile_emits_an_opaque_call(self):
        # A raw Triton launch inside a compiled region breaks the graph and pays
        # Triton's JIT per block: measured as +330s on the first clip.
        self.assertTrue(hasattr(torch.ops.sglang, "magi2_mhc_mix_output"))

    def test_module_uses_the_kernel_and_tracks_the_einsum_fallback(self):
        from sglang.multimodal_gen.runtime.models.dits.magi2_mhc import Magi2MHC

        dev = torch.device("cuda")
        mhc = Magi2MHC(num_stream=NUM_STREAM, hidden_size=HIDDEN).to(dev)
        streams, block_out, _, _ = _inputs(dev)
        h_post = torch.randn(TOKENS, NUM_STREAM, device=dev)
        h_res = torch.randn(TOKENS, NUM_STREAM, NUM_STREAM, device=dev)

        got = mhc.mix_output(streams, block_out, h_post, h_res)
        # The CPU branch keeps the einsum spelling, so this pins the fallback too.
        cpu = mhc.cpu().mix_output(
            streams.cpu(), block_out.cpu(), h_post.cpu(), h_res.cpu()
        )
        self.assertEqual(got.shape, streams.shape)
        torch.testing.assert_close(got.cpu().float(), cpu.float(), rtol=0, atol=0.07)


if __name__ == "__main__":
    unittest.main()
