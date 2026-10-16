"""Blackwell regression for DSv4 TRT-LLM long-query dispatch.

Run directly on SM100/SM103. Compare FP8 attention with a float32 reference
across the short-query boundary, then verify the TP4 prefill tile at 8K.
"""

import unittest

import torch
from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

from sglang.srt.layers.attention.deepseek_v4_trtllm_backend import (
    _install_persistent_trtllm_semaphores,
)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() in ((10, 0), (10, 3)),
    "requires SM100/SM103",
)
class TestTrtllmPrefill(unittest.TestCase):
    def test_prefill_reference_and_dispatch(self):
        torch.manual_seed(41603)
        _install_persistent_trtllm_semaphores(8192)
        device = "cuda"
        dtype = torch.float8_e4m3fn
        heads, dim, topk = 16, 512, 2176
        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        swa = torch.randn(64, 1, 64, dim, device=device).to(dtype)
        compressed = torch.randn(64, 1, 64, dim, device=device).to(dtype)
        sink = torch.randn(heads, device=device)
        for rows in (1, 16, 17, 8192):
            with self.subTest(rows=rows):
                q = torch.randn(rows, heads, dim, device=device).to(dtype)
                indices = torch.randint(
                    0, 4096, (rows, topk), dtype=torch.int32, device=device
                )
                lens = torch.randint(
                    128, topk + 1, (rows,), dtype=torch.int32, device=device
                )
                cu = torch.tensor([0, rows], dtype=torch.int32, device=device)
                # A cached prefix ensures that all 128 SWA slots are valid.
                seq_lens = torch.tensor([rows + 128], dtype=torch.int32, device=device)

                def run():
                    return trtllm_batch_decode_sparse_mla_dsv4(
                        query=q,
                        swa_kv_cache=swa,
                        workspace_buffer=workspace,
                        sparse_indices=indices,
                        compressed_kv_cache=compressed,
                        sparse_topk_lens=lens,
                        seq_lens=seq_lens,
                        bmm1_scale=dim**-0.5,
                        bmm2_scale=1.0,
                        sinks=sink,
                        kv_layout="HND",
                        backend="trtllm-gen",
                        cum_seq_lens_q=cu,
                        max_q_len=rows,
                    )

                out = run()
                torch.cuda.synchronize()
                self.assertTrue(torch.isfinite(out).all().item())
                samples = sorted(set([0, rows // 2, rows - 1]))
                for row in samples:
                    length = lens[row].item()
                    keys = torch.cat(
                        (
                            swa.float().reshape(-1, dim)[indices[row, :128].long()],
                            compressed.float().reshape(-1, dim)[
                                indices[row, 128:length].long()
                            ],
                        )
                    )
                    logits = q[row].float() @ keys.T * dim**-0.5
                    probs = torch.cat((logits, sink[:, None]), dim=-1).softmax(-1)
                    expected = probs[:, :-1] @ keys
                    # Allow FP8 attention rounding relative to the FP32 reference.
                    torch.testing.assert_close(
                        out[row].float(), expected, atol=0.02, rtol=0.03
                    )
                if rows == 8192:
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CUDA]
                    ) as prof:
                        run()
                        torch.cuda.synchronize()
                    names = [
                        event.name
                        for event in prof.events()
                        if "fmhaSm100" in event.name
                    ]
                    self.assertTrue(names, "No TRT-LLM attention kernel recorded")
                    self.assertTrue(
                        all("VarSeqQ16Kv128" in name for name in names), names
                    )


if __name__ == "__main__":
    unittest.main()
