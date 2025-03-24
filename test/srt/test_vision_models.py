import torch
import unittest
from transformers import Qwen2VLConfig
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VisionBlock
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.attention.vision import VisionAttention

class TestQwen2VLAttention(unittest.TestCase):
    def setUp(self):
        self.config = Qwen2VLConfig(
            hidden_size=32,
            vision_config={
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "patch_size": 14,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "in_chans": 3,
                "embed_dim": 32,
                "depth": 2,
                "mlp_ratio": 4.0,
            }
        )
        self.model = Qwen2VLForConditionalGeneration(self.config)

    def test_vision_block_default_attention(self):
        """Test that VisionBlock uses SDPA attention by default"""
        vision_block = Qwen2VisionBlock(
            dim=32,
            num_heads=4,
            mlp_ratio=4.0,
            norm_layer=lambda x: torch.nn.LayerNorm(x, eps=1e-6)
        )
        self.assertIsInstance(vision_block.attn, VisionAttention)

    def test_vision_block_radix_attention(self):
        """Test that VisionBlock can use RadixAttention when specified"""
        vision_block = Qwen2VisionBlock(
            dim=32,
            num_heads=4,
            mlp_ratio=4.0,
            norm_layer=lambda x: torch.nn.LayerNorm(x, eps=1e-6),
            attn_implementation="radix"
        )
        self.assertIsInstance(vision_block.attn, RadixAttention)

    def test_vision_block_forward_default(self):
        """Test forward pass through VisionBlock with default attention"""
        vision_block = Qwen2VisionBlock(
            dim=32,
            num_heads=4,
            mlp_ratio=4.0,
            norm_layer=lambda x: torch.nn.LayerNorm(x, eps=1e-6)
        )
        
        # Create test inputs
        batch_size = 2
        seq_len = 16
        x = torch.randn(seq_len, batch_size, 32)
        
        # Run forward pass
        output = vision_block(x)
        
        # Check output shape
        self.assertEqual(output.shape, (seq_len, batch_size, 32))

    def test_vision_block_forward_radix(self):
        """Test forward pass through VisionBlock with RadixAttention"""
        vision_block = Qwen2VisionBlock(
            dim=32,
            num_heads=4,
            mlp_ratio=4.0,
            norm_layer=lambda x: torch.nn.LayerNorm(x, eps=1e-6),
            attn_implementation="radix"
        )
        
        # Create test inputs
        batch_size = 2
        seq_len = 16
        x = torch.randn(seq_len, batch_size, 32)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)
        
        # Create position embeddings
        freqs = torch.randn(seq_len, 4)  # head_dim=8, so 4 for cos and 4 for sin
        position_embeddings = (freqs.cos(), freqs.sin())
        
        # Run forward pass
        output = vision_block(x, cu_seqlens, position_embeddings)
        
        # Check output shape
        self.assertEqual(output.shape, (seq_len, batch_size, 32))

if __name__ == "__main__":
    unittest.main() 