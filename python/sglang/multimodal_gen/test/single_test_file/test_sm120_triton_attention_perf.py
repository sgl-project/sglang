import unittest
import warnings

import torch


def _is_sm120_cuda_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


class TestSM120TritonAttentionPerf(unittest.TestCase):
    @unittest.skipUnless(
        _is_sm120_cuda_available(),
        "sm120_triton_attn is only supported on SM12.x CUDA GPUs",
    )
    def test_sm120_triton_attention_matches_sdpa_and_benchmarks(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
            SDPAImpl,
        )
        from sglang.multimodal_gen.runtime.layers.attention.backends.sm120_triton_attn import (
            SM120TritonAttentionImpl,
        )

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(0)

        batch_size = 1
        seq_len = 1024
        num_heads = 24
        head_size = 128
        softmax_scale = head_size**-0.5
        dtype = torch.bfloat16
        device = torch.device("cuda")

        query = torch.randn(
            batch_size, seq_len, num_heads, head_size, device=device, dtype=dtype
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        sm120_impl = SM120TritonAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=False,
            softmax_scale=softmax_scale,
        )
        sdpa_impl = SDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=False,
            softmax_scale=softmax_scale,
        )

        for _ in range(3):
            sm120_impl.forward(query, key, value, None)
            sdpa_impl.forward(query, key, value, None)
        torch.cuda.synchronize()

        sm120_output = sm120_impl.forward(query, key, value, None)
        sdpa_output = sdpa_impl.forward(query, key, value, None)
        torch.testing.assert_close(
            sm120_output.float(),
            sdpa_output.float(),
            atol=5e-2,
            rtol=5e-2,
        )

        sm120_ms = self._bench_cuda_ms(
            lambda: sm120_impl.forward(query, key, value, None)
        )
        sdpa_ms = self._bench_cuda_ms(
            lambda: sdpa_impl.forward(query, key, value, None)
        )
        bench_message = (
            "SM120_TRITON_ATTN_BENCH "
            f"shape=({batch_size},{seq_len},{num_heads},{head_size}) "
            f"dtype={dtype} "
            f"sm120_triton_attn_ms={sm120_ms:.3f} "
            f"torch_sdpa_ms={sdpa_ms:.3f} "
            f"speedup={sdpa_ms / sm120_ms:.3f}x"
        )
        print(bench_message, flush=True)
        warnings.warn(bench_message, RuntimeWarning, stacklevel=1)

    @staticmethod
    def _bench_cuda_ms(fn, warmup: int = 5, repeats: int = 20) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeats


if __name__ == "__main__":
    unittest.main()
