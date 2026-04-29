import unittest

import torch

from sglang.jit_kernel.deepseek_v4 import compress_forward
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=2, suite="stage-b-test-small-1-gpu")


def _cpu_c128_decode_reference(
    kv_score_buffer: torch.Tensor,
    ape: torch.Tensor,
    seq_len: int,
    head_dim: int,
) -> torch.Tensor:
    scores = []
    kvs = []
    for i in range(128):
        j = (seq_len + i) % 128
        src = kv_score_buffer[0, j]
        kv = src[:head_dim]
        score = src[head_dim:]
        kvs.append(kv.float())
        scores.append(score.float())

    kv_tensor = torch.stack(kvs, dim=0)
    score_tensor = torch.stack(scores, dim=0) + ape.float()
    weights = torch.softmax(score_tensor, dim=0)
    return (kv_tensor * weights).sum(dim=0)


class TestDeepseekV4C128Compress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def _run_decode_case(self, buffer_dtype: torch.dtype):
        device = "cuda"
        head_dim = 128
        torch.manual_seed(1)

        kv_score_buffer = torch.zeros(
            (1, 128, head_dim * 2), dtype=buffer_dtype, device=device
        )
        ape = torch.randn((128, head_dim), dtype=torch.float32, device=device)
        indices = torch.zeros((1,), dtype=torch.int32, device=device)

        ref_buffer = torch.zeros((1, 128, head_dim * 2), dtype=buffer_dtype)

        for seq_len in range(1, 257):
            kv_score_input = torch.randn(
                (1, head_dim * 2), dtype=torch.float32, device=device
            )
            write_pos = (seq_len + 127) % 128
            ref_buffer[0, write_pos] = kv_score_input[0].cpu().to(buffer_dtype)

            out = compress_forward(
                kv_score_buffer=kv_score_buffer,
                kv_score_input=kv_score_input,
                ape=ape,
                indices=indices,
                head_dim=head_dim,
                compress_ratio=128,
                seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
            )

            self.assertTrue(
                torch.equal(kv_score_buffer.cpu(), ref_buffer),
                msg=f"buffer mismatch for seq_len={seq_len}, dtype={buffer_dtype}",
            )

            if seq_len % 128 != 0:
                continue

            ref_out = _cpu_c128_decode_reference(
                ref_buffer,
                ape.cpu(),
                seq_len=seq_len,
                head_dim=head_dim,
            )
            atol = 2e-4 if buffer_dtype == torch.float32 else 2e-2
            rtol = 2e-4 if buffer_dtype == torch.float32 else 2e-2
            self.assertTrue(
                torch.allclose(out[0].cpu().float(), ref_out, atol=atol, rtol=rtol),
                msg=(
                    f"output mismatch for seq_len={seq_len}, "
                    f"dtype={buffer_dtype}, "
                    f"max_diff={(out[0].cpu().float() - ref_out).abs().max().item():.6f}"
                ),
            )

    def test_c128_decode_fp32_buffer_fp32_input(self):
        self._run_decode_case(torch.float32)

    def test_c128_decode_bf16_buffer_fp32_input(self):
        self._run_decode_case(torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
