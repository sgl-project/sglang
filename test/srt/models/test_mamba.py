import unittest
import torch
from transformers import MambaConfig

from sglang.srt.models.mamba import MambaForCausalLM, MambaCacheParams


class TestMamba(unittest.TestCase):
    def test_mamba_interfaces(self):
        # Test that Mamba has the correct interfaces
        model = MambaForCausalLM(MambaConfig())
        self.assertTrue(hasattr(model, 'has_inner_state'))
        self.assertTrue(hasattr(model, 'is_attention_free'))
        self.assertEqual(model.has_inner_state, True)
        self.assertEqual(model.is_attention_free, True)
    
    def test_mamba_cache_params(self):
        # Test cache params structure
        batch_size = 2
        num_layers = 4
        dim = 768
        conv_kernel_size = 4
        state_size = 16
        
        conv_state = torch.zeros(num_layers, batch_size, conv_kernel_size, dim)
        ssm_state = torch.zeros(num_layers, batch_size, dim, state_size)
        state_indices = torch.zeros(batch_size, dtype=torch.long)
        
        cache_params = MambaCacheParams(conv_state, ssm_state, state_indices)
        
        # Test at_layer_idx
        layer_cache = cache_params.at_layer_idx(0)
        self.assertEqual(layer_cache.conv_state.shape, (batch_size, conv_kernel_size, dim))
        self.assertEqual(layer_cache.ssm_state.shape, (batch_size, dim, state_size))
    
    def test_no_kv_cache(self):
        # Test that Mamba reports no KV cache
        config = MambaConfig(hidden_size=768, num_hidden_layers=12)
        model = MambaForCausalLM(config)
        
        self.assertEqual(model.get_num_kv_heads(), 0)
        self.assertEqual(model.get_kv_head_dim(), 0)


if __name__ == "__main__":
    unittest.main()