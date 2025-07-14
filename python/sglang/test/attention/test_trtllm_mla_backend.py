import unittest
import torch

from sglang.srt.layers import dp_attention as _dp_attn

# Patch DP-attention globals **before** importing the backend so that all
# downstream `from … import get_attention_tp_size` statements receive the
# patched version.
_dp_attn.get_attention_tp_size = lambda: 1  # TP size = 1 for unit test

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.test_utils import CustomTestCase
from sglang.srt.utils import is_flashinfer_available


class MockModelRunner:
    """Minimal fake `ModelRunner` for MLA backend unit tests."""

    def __init__(self, page_size: int):
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.kv_cache_dtype = torch.bfloat16
        self.page_size = page_size

        # Model-config stub – only the attributes accessed by the backend.
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": 2048,
                "attention_arch": AttentionArch.MLA,
                "num_attention_heads": 128,
                "kv_lora_rank": 512,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 512,
                "scaling": 1.0 / ((128 + 64) ** 0.5),
                "get_num_kv_heads": staticmethod(lambda _: 1),
            },
        )

        # Req-to-token pool (dummy)
        max_bs = 64
        max_ctx = self.model_config.context_len
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                "size": max_bs,
                "req_to_token": torch.zeros(max_bs, max_ctx, dtype=torch.int32, device=self.device),
            },
        )
        
        # KV-token pool
        self.token_to_kv_pool = MLATokenToKVPool(
            size=max_bs * max_ctx,
            page_size=page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=1,
            device=self.device,
            enable_memory_saver=False,
        )


@unittest.skipIf(not torch.cuda.is_available() or not is_flashinfer_available(), "CUDA + flashinfer required")
class TestTRTLLMMLABackend(CustomTestCase):
    """Structure mirrors `test_flashattn_backend.py` but focuses on MLA decode."""

    def setUp(self):
        self.batch_size = 16
        self.seq_len = 512
        self.page_sizes = [32, 64]
        self.device = "cuda"
        self.dtype = torch.bfloat16
        
    # ‑- helpers ---------------------------------------------------------
    def _init(self, page_size: int):
        self.model_runner = MockModelRunner(page_size)
        self.backend = TRTLLMMLABackend(self.model_runner)
        # Attach num_heads required by RadixAttention convenience
        self.model_runner.model_config.num_attention_heads = 128

    def _alloc_qkv(self):
        head_dim = 512 + 64  # kv_lora_rank + qk_rope_head_dim
        q_shape = (self.batch_size, 128, head_dim)
        q = torch.randn(q_shape, dtype=self.dtype, device=self.device)
        k = torch.randn(self.batch_size, 1, head_dim, dtype=self.dtype, device=self.device)
        v = None  # TRTLLM MLA decode kernel ignores v
        return q, k, v

    def _create_forward_batch(self, seq_lens: torch.Tensor):
        fb = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(0, 100, (self.batch_size, 1), device=self.device),
            out_cache_loc=torch.arange(self.batch_size, device=self.device),
            seq_lens_sum=int(seq_lens.sum().item()),
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(self.batch_size, device=self.device),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            attn_backend=self.backend,
        )
        fb.req_to_token_pool = self.model_runner.req_to_token_pool
        fb.token_to_kv_pool = self.model_runner.token_to_kv_pool
        return fb

    # ‑- actual tests ----------------------------------------------------
    def test_forward_decode(self):
        """Smoke test decode across several page sizes."""
        for ps in self.page_sizes:
            self._init(ps)

            # Random seq lens (ensure one matches max)
            seq_lens = torch.randint(1, self.seq_len, (self.batch_size,), device=self.device)
            seq_lens[0] = self.seq_len

            forward_batch = self._create_forward_batch(seq_lens)
            self.backend.init_forward_metadata(forward_batch)
        
            q, k, v = self._alloc_qkv()
            layer = RadixAttention(
                num_heads=128,
                head_dim=512 + 64,
                scaling=self.model_runner.model_config.scaling,
                num_kv_heads=1,
                layer_id=0,
                v_head_dim=512,
                prefix="attn_mqa",
            )
            out = self.backend.forward_decode(q, k, v, layer, forward_batch)
        
            self.assertEqual(out.shape, (self.batch_size, 128 * 512))
            self.assertEqual(out.dtype, self.dtype)
            self.assertEqual(out.device.type, "cuda")
            self.assertFalse(torch.isnan(out).any())
            self.assertFalse(torch.isinf(out).any())


if __name__ == "__main__":
    unittest.main() 