import unittest
import torch
import math

from sglang.srt.layers import dp_attention as _dp_attn

# Patch DP-attention globals before importing backends
_dp_attn.get_attention_tp_size = lambda: 1  # TP size = 1 for unit test

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.test_utils import CustomTestCase
from sglang.srt.utils import is_flashinfer_available


class MockModelRunner:
    """Minimal fake ModelRunner for comparing both MLA backends."""

    def __init__(self, page_size: int):
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.kv_cache_dtype = torch.bfloat16
        self.page_size = page_size

        # Model-config stub with MLA attributes
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

        # Req-to-token pool
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
        
        # KV-token pool (MLA)
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
class TestTRTLLMvsFlashInferMLA(CustomTestCase):
    """Test numerical equivalence between TRTLLM and FlashInfer MLA backends."""

    def setUp(self):
        self.batch_size = 8
        self.seq_len = 256
        self.page_size = 32
        self.device = "cuda"
        self.dtype = torch.bfloat16
        
        # Create model runner
        self.model_runner = MockModelRunner(self.page_size)
        
        # Initialize both backends
        self.trtllm_backend = TRTLLMMLABackend(self.model_runner)
        self.flashinfer_backend = FlashInferMLAAttnBackend(self.model_runner)
        
        # Create RadixAttention layer for testing
        self.layer = RadixAttention(
            num_heads=128,
            head_dim=512 + 64,  # kv_lora_rank + qk_rope_head_dim
            scaling=self.model_runner.model_config.scaling,
            num_kv_heads=1,
            layer_id=0,
            v_head_dim=512,
            prefix="attn_mqa",
        )

    def _create_qkv_tensors(self):
        """Create Q, K, V tensors for testing."""
        head_dim = 512 + 64  # kv_lora_rank + qk_rope_head_dim
        q = torch.randn((self.batch_size, 128, head_dim), dtype=self.dtype, device=self.device)
        k = torch.randn((self.batch_size, 1, head_dim), dtype=self.dtype, device=self.device)
        # For FlashInfer MLA, if k is provided v must not be None.
        v = torch.randn((self.batch_size, 1, 512), dtype=self.dtype, device=self.device)
        return q, k, v

    def _create_forward_batch(self, backend):
        """Create a forward batch for the given backend."""
        # Random sequence lengths
        seq_lens = torch.randint(1, self.seq_len, (self.batch_size,), device=self.device)
        seq_lens[0] = self.seq_len  # Ensure at least one max length
        
        fb = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(0, 100, (self.batch_size, 1), device=self.device),
            out_cache_loc=torch.arange(self.batch_size, device=self.device),
            seq_lens_sum=int(seq_lens.sum().item()),
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(self.batch_size, device=self.device),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            attn_backend=backend,
        )
        fb.req_to_token_pool = self.model_runner.req_to_token_pool
        fb.token_to_kv_pool = self.model_runner.token_to_kv_pool
        return fb

    def test_decode_output_match(self):
        """Test that TRTLLM and FlashInfer MLA backends produce matching outputs."""
        # Create identical forward batches for both backends
        fb_trtllm = self._create_forward_batch(self.trtllm_backend)
        fb_flashinfer = self._create_forward_batch(self.flashinfer_backend)
        
        # Initialize metadata for both backends
        self.trtllm_backend.init_forward_metadata(fb_trtllm)
        self.flashinfer_backend.init_forward_metadata(fb_flashinfer)
        
        # Create Q, K, V tensors
        q, k, v = self._create_qkv_tensors()
        
        # Run forward decode on both backends
        out_trtllm = self.trtllm_backend.forward_decode(q.clone(), k.clone(), v, self.layer, fb_trtllm)
        out_flashinfer = self.flashinfer_backend.forward_decode(q.clone(), k.clone(), v.clone(), self.layer, fb_flashinfer)

        # Debug: print scale info
        print(f"\n[DEBUG] Scale analysis:")
        print(f"  layer.scaling = {self.layer.scaling}")
        print(f"  qk_nope_head_dim = {self.model_runner.model_config.qk_nope_head_dim}")
        print(f"  qk_rope_head_dim = {self.model_runner.model_config.qk_rope_head_dim}")
        print(f"  kv_lora_rank = {self.model_runner.model_config.kv_lora_rank}")
        print(f"  Expected TRT scale factor = {math.sqrt(128 + 64) / math.sqrt(512 + 64)} = {math.sqrt(192) / math.sqrt(576)}")
        print(f"  Output shapes: TRTLLM {out_trtllm.shape}, FlashInfer {out_flashinfer.shape}")
        print(f"  Output means: TRTLLM {out_trtllm.mean().item():.6f}, FlashInfer {out_flashinfer.mean().item():.6f}")
        print(f"  Output stds: TRTLLM {out_trtllm.std().item():.6f}, FlashInfer {out_flashinfer.std().item():.6f}")
        print(f"  Max diff = {(out_trtllm - out_flashinfer).abs().max().item()}")
        print(f"  Ratio of means = {out_trtllm.mean().item() / out_flashinfer.mean().item() if out_flashinfer.mean().item() != 0 else 'inf'}")
        
        # Additional debug
        print(f"\n[DEBUG] Scale computation:")
        print(f"  layer.scaling = 1/sqrt(192) = {1/math.sqrt(192)}")
        print(f"  TRT scale passed = layer.scaling * sqrt(192)/sqrt(576) = {self.layer.scaling * math.sqrt(192) / math.sqrt(576)}")
        print(f"  TRT kernel will compute: 1 / (sqrt(576) * scale) = {1 / (math.sqrt(576) * self.layer.scaling * math.sqrt(192) / math.sqrt(576))}")
        print(f"  Which equals: 1 / (layer.scaling * sqrt(192)) = {1 / (self.layer.scaling * math.sqrt(192))}")
        print(f"  But FlashInfer uses: layer.scaling = {self.layer.scaling}")
        print(f"  Ratio: {(1 / (self.layer.scaling * math.sqrt(192))) / self.layer.scaling} = sqrt(192) = {math.sqrt(192)}")

        # Check output shapes match
        self.assertEqual(out_trtllm.shape, out_flashinfer.shape,
                         f"Output shapes differ: TRTLLM {out_trtllm.shape} vs FlashInfer {out_flashinfer.shape}")
        
        # Check output dtypes match
        self.assertEqual(out_trtllm.dtype, out_flashinfer.dtype)
        
        # Check numerical equivalence with tolerance
        # Note: Using higher tolerance due to potential numerical differences in implementations
        self.assertTrue(
            torch.allclose(out_trtllm, out_flashinfer, atol=1e-2, rtol=1e-2),
            f"TRTLLM and FlashInfer outputs differ beyond tolerance. "
            f"Max diff: {(out_trtllm - out_flashinfer).abs().max().item()}"
        )
        
        # Additional checks
        self.assertFalse(torch.isnan(out_trtllm).any(), "TRTLLM output contains NaN")
        self.assertFalse(torch.isnan(out_flashinfer).any(), "FlashInfer output contains NaN")
        self.assertFalse(torch.isinf(out_trtllm).any(), "TRTLLM output contains Inf")
        self.assertFalse(torch.isinf(out_flashinfer).any(), "FlashInfer output contains Inf")

    def test_decode_with_different_page_sizes(self):
        """Test output consistency across different page sizes."""
        page_sizes = [32, 64]
        outputs = []
        
        for ps in page_sizes:
            # Reinitialize with new page size
            self.model_runner = MockModelRunner(ps)
            self.trtllm_backend = TRTLLMMLABackend(self.model_runner)
            
            # Create batch and run decode
            fb = self._create_forward_batch(self.trtllm_backend)
            self.trtllm_backend.init_forward_metadata(fb)
            
            q, k, v = self._create_qkv_tensors()
            out = self.trtllm_backend.forward_decode(q, k, v, self.layer, fb)
            outputs.append(out)
        
        # Check that outputs are consistent across page sizes
        # Note: Different page sizes might lead to slightly different numerical results
        for i in range(1, len(outputs)):
            self.assertTrue(
                torch.allclose(outputs[0], outputs[i], atol=5e-2, rtol=5e-2),
                f"Output with page_size={page_sizes[0]} differs from page_size={page_sizes[i]}"
            )


if __name__ == "__main__":
    unittest.main() 