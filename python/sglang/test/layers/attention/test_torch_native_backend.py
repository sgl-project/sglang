import unittest
import torch
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

class MockRadixAttention:
    def __init__(self, q_head_num=8, k_head_num=8, qk_head_dim=64, v_head_dim=64):
        self.tp_q_head_num = q_head_num
        self.tp_k_head_num = k_head_num
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.layer_id = 0
        self.scaling = None
        self.is_cross_attention = False

class MockTokenKVPool:
    def __init__(self, key_buffer, value_buffer):
        self.key_buffer = key_buffer
        self.value_buffer = value_buffer
    
    def set_kv_buffer(self, layer, cache_loc, k, v):
        pass
    
    def get_key_buffer(self, layer_id):
        return self.key_buffer
    
    def get_value_buffer(self, layer_id):
        return self.value_buffer

class MockModelRunner:
    def __init__(self, device='cuda'):
        self.device = device

class TestTorchNativeAttnBackend(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model_runner = MockModelRunner()
        self.backend = TorchNativeAttnBackend(self.model_runner)
        
        # Common test parameters
        self.batch_size = 2
        self.seq_len = 16
        self.num_heads = 8
        self.head_dim = 64

    def test_forward_extend(self):
        """Test the forward extend functionality"""
        # Create mock inputs
        q = torch.randn(self.seq_len, self.num_heads * self.head_dim, device='cuda')
        k = torch.randn(self.seq_len, self.num_heads * self.head_dim, device='cuda')
        v = torch.randn(self.seq_len, self.num_heads * self.head_dim, device='cuda')
        
        # Create mock layer
        layer = MockRadixAttention()
        
        # Create mock forward batch with all required arguments
        forward_batch = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(0, 1000, (self.batch_size, self.seq_len), device='cuda'),
            out_cache_loc=torch.arange(self.batch_size * self.seq_len, device='cuda'),
            seq_lens_sum=self.batch_size * self.seq_len,
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=torch.zeros(self.batch_size, dtype=torch.long, device='cuda'),
            seq_lens=torch.tensor([self.seq_len] * self.batch_size, device='cuda'),
            extend_prefix_lens=torch.tensor([8] * self.batch_size, device='cuda'),
            extend_seq_lens=torch.tensor([8] * self.batch_size, device='cuda')
        )
        
        # Create mock KV cache
        kv_cache_size = self.batch_size * self.seq_len
        forward_batch.token_to_kv_pool = MockTokenKVPool(
            torch.randn(kv_cache_size, self.num_heads, self.head_dim, device='cuda'),
            torch.randn(kv_cache_size, self.num_heads, self.head_dim, device='cuda')
        )
        
        # Create mock token pool
        forward_batch.req_to_token_pool = type('', (), {
            'req_to_token': torch.arange(kv_cache_size, device='cuda').reshape(self.batch_size, -1)
        })()
        
        # Run forward extend
        output = self.backend.forward_extend(q, k, v, layer, forward_batch)
        
        # Assertions
        self.assertEqual(output.shape, q.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_forward_decode(self):
        """Test the forward decode functionality"""
        # Create mock inputs (for decode, we only have one token per sequence)
        q = torch.randn(self.batch_size, self.num_heads * self.head_dim, device='cuda')
        k = torch.randn(self.batch_size, self.num_heads * self.head_dim, device='cuda')
        v = torch.randn(self.batch_size, self.num_heads * self.head_dim, device='cuda')
        
        # Create mock layer
        layer = MockRadixAttention()
        
        # Create mock forward batch with all required arguments
        forward_batch = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(0, 1000, (self.batch_size, 1), device='cuda'),  # Only one token for decode
            out_cache_loc=torch.arange(self.batch_size, device='cuda'),
            seq_lens_sum=self.batch_size,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.zeros(self.batch_size, dtype=torch.long, device='cuda'),
            seq_lens=torch.tensor([self.seq_len] * self.batch_size, device='cuda')
        )
        
        # Create mock KV cache
        kv_cache_size = self.batch_size * self.seq_len
        forward_batch.token_to_kv_pool = MockTokenKVPool(
            torch.randn(kv_cache_size, self.num_heads, self.head_dim, device='cuda'),
            torch.randn(kv_cache_size, self.num_heads, self.head_dim, device='cuda')
        )
        
        # Create mock token pool
        forward_batch.req_to_token_pool = type('', (), {
            'req_to_token': torch.arange(kv_cache_size, device='cuda').reshape(self.batch_size, -1)
        })()
        
        # Run forward decode
        output = self.backend.forward_decode(q, k, v, layer, forward_batch)
        
        # Assertions
        self.assertEqual(output.shape, q.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

if __name__ == '__main__':
    unittest.main(verbosity=2)
