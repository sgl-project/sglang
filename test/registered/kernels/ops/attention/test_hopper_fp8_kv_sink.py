import unittest

import torch

from sglang.kernels.ops.attention.flash_mla_interface import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from sglang.kernels.ops.quantization.dsa_quant_k_cache import quantize_k_cache
from sglang.srt.layers.attention.dsa_backend import _apply_flashmla_kv_sink
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "requires Hopper",
)
class TestHopperFP8KVSink(unittest.TestCase):
    def test_sink_against_dense_attention(self):
        # Constant values within each quantization block survive FP8 round-trip.
        device = "cuda"
        tokens = torch.arange(128, device=device)
        values = (2 * (tokens % 2) - 1).to(torch.bfloat16)
        cache = values[:, None].expand(-1, 576).contiguous().view(2, 64, 1, 576)
        packed = quantize_k_cache(cache)
        q = torch.zeros(2, 1, 64, 576, dtype=torch.bfloat16, device=device)
        q[..., 0] = 1
        lengths = torch.full((2,), 128, dtype=torch.int32, device=device)
        sink = torch.linspace(2, 6, 16, device=device)
        for width in (64, 128):
            with self.subTest(width=width):
                indices = tokens[:width].int().repeat(2, 1, 1)
                indices[1].fill_(-1)
                meta, splits = get_mla_metadata(lengths, 64, 1, 64, True, width)
                out, lse = flash_mla_with_kvcache(
                    q=q,
                    k_cache=packed,
                    block_table=lengths.new_empty((2, 0)),
                    cache_seqlens=lengths,
                    head_dim_v=512,
                    tile_scheduler_metadata=meta,
                    num_splits=splits,
                    softmax_scale=1.0,
                    is_fp8_kvcache=True,
                    indices=indices,
                    kv_format="fp8_ds_mla",
                )
                actual = _apply_flashmla_kv_sink(out[:, :, :16], lse, sink)
                logits = values[:width].float().expand(16, -1)
                probs = torch.cat((logits, sink[:, None]), dim=1).softmax(dim=1)
                expected = (probs[:, :-1] * logits).sum(dim=1)
                torch.testing.assert_close(
                    actual[0, 0].float(),
                    expected[:, None].expand(-1, 512),
                    atol=0.006,
                    rtol=0.006,
                )
                self.assertEqual(actual[1].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
