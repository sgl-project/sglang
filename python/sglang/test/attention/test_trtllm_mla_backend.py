import unittest

import pytest
import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.test.test_utils import CustomTestCase


class MockModelRunner:
    def __init__(
        self,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size=16,
    ):
        attention_arch = AttentionArch.MLA
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.kv_cache_dtype = torch.bfloat16
        context_len = 2048
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": context_len,
                "attention_arch": attention_arch,
                "num_attention_heads": 128,
                "kv_lora_rank": kv_lora_rank,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": qk_rope_head_dim,
                "v_head_dim": kv_lora_rank,
                "scaling": 1.0 / ((128 + 64) ** 0.5),
            },
        )
        self.sliding_window_size = None
        self.page_size = page_size

        batch_size = 256
        # Create a proper req_to_token_pool with the req_to_token attribute
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                "size": batch_size,
                "req_to_token": torch.zeros(
                    batch_size, context_len, dtype=torch.int32, device=self.device
                ),
            },
        )
        
        max_total_num_tokens = batch_size * context_len
        self.token_to_kv_pool = MLATokenToKVPool(
            size=max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.dtype,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=1,  # only consider layer=1 for unit test
            device=self.device,
            enable_memory_saver=False,
        )

    def get_num_kv_heads(self, tp_size):
        """MLA uses single KV head."""
        return 1


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    reason="Test requires CUDA and flashinfer"
)
@pytest.mark.parametrize("batch_size", [16, 32, 64])
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("seq_len", [256, 512, 1024])
class TestTRTLLMMLABackend(CustomTestCase):
    def test_trtllm_decode_mla(self, batch_size, page_size, seq_len):
        """Test TRTLLM MLA decode operation with various configurations."""
        # Check if PyTorch supports current GPU
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            archs = torch.cuda.get_arch_list()
            current_arch = f"sm_{capability[0]}{capability[1]}"
            supported = any(current_arch in arch for arch in archs)
            if not supported:
                pytest.skip(f"PyTorch doesn't support {current_arch} - need nightly build")
        
        # DeepSeek MLA configuration
        num_q_heads = 128
        kv_lora_rank = 512
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        device = "cuda"
        dtype = torch.bfloat16
        
        # Initialize model runner and backend
        model_runner = MockModelRunner(
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            page_size=page_size,
        )
        
        # Check if flashinfer has TRTLLM MLA support
        try:
            import flashinfer
            if not hasattr(flashinfer.decode, 'trtllm_batch_decode_with_kv_cache_mla'):
                pytest.skip("flashinfer version does not have TRTLLM MLA support")
        except ImportError:
            pytest.skip("flashinfer not available")
        
        backend = TRTLLMMLABackend(model_runner)
        
        # Create attention layer
        layer = RadixAttention(
            num_heads=num_q_heads,
            head_dim=kv_lora_rank + qk_rope_head_dim,
            scaling=model_runner.model_config.scaling,
            num_kv_heads=1,
            layer_id=0,
            v_head_dim=kv_lora_rank,
            prefix="attn_mqa",
        )
        
        # Generate sequence lengths
        seq_lens = torch.randint(1, seq_len, (batch_size,), device=device)
        seq_lens[-1] = seq_len  # Ensure at least one max length
        max_seq_len = seq_lens.max().item()
        
        # Calculate blocks needed
        blocks_per_seq = (seq_lens + page_size - 1) // page_size
        total_blocks_needed = blocks_per_seq.sum().item()
        
        # Create req_to_token mapping
        req_to_token = torch.zeros(batch_size, seq_len, dtype=torch.int32, device=device)
        token_offset = 0
        for i in range(batch_size):
            seq_len_i = seq_lens[i].item()
            req_to_token[i, :seq_len_i] = torch.arange(
                token_offset, token_offset + seq_len_i, device=device
            )
            token_offset += seq_len_i
        
        model_runner.req_to_token_pool.req_to_token = req_to_token
        
        # Create forward batch for decode
        forward_batch = ForwardBatch(
            batch_size=batch_size,
            input_ids=torch.randint(0, 100, (batch_size, 1), device=device),
            out_cache_loc=torch.arange(batch_size, device=device),
            seq_lens_sum=seq_lens.sum().item(),
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(batch_size, device=device),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            attn_backend=backend,
        )
        
        # Add pools to forward batch
        forward_batch.req_to_token_pool = model_runner.req_to_token_pool
        forward_batch.token_to_kv_pool = model_runner.token_to_kv_pool
        
        # Fill KV cache with some data
        cache_data = torch.randn(
            seq_lens.sum().item(),
            1,  # num_kv_heads
            kv_lora_rank + qk_rope_head_dim,
            dtype=dtype,
            device=device,
        )
        cache_indices = torch.arange(seq_lens.sum().item(), device=device)
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, cache_indices, cache_data, None
        )
        
        # Initialize metadata
        backend.init_forward_metadata(forward_batch)
        
        # Create input tensors
        q_shape = (batch_size, num_q_heads, kv_lora_rank + qk_rope_head_dim)
        q = torch.randn(q_shape, dtype=dtype, device=device)
        
        # For MLA, k contains compressed KV, v is not used
        k = torch.randn(batch_size, 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
        v = None
        
        # Run forward decode
        output = backend.forward_decode(q, k, v, layer, forward_batch)
        
        # Verify output
        expected_shape = (batch_size, num_q_heads * kv_lora_rank)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        assert output.dtype == dtype
        assert output.device.type == "cuda"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"


# Simplified test for quick verification
@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    reason="Test requires CUDA and flashinfer"
)
def test_trtllm_mla_basic():
    """Basic test to verify TRTLLM MLA backend works."""
    # Check if flashinfer has TRTLLM MLA support
    try:
        import flashinfer
        if not hasattr(flashinfer.decode, 'trtllm_batch_decode_with_kv_cache_mla'):
            pytest.skip("flashinfer version does not have TRTLLM MLA support")
    except ImportError:
        pytest.skip("flashinfer not available")
    
    test = TestTRTLLMMLABackend()
    test.test_trtllm_decode_mla(batch_size=32, page_size=32, seq_len=512)
    print("TRTLLM MLA basic test passed!")


if __name__ == "__main__":
    test_trtllm_mla_basic() 