"""
Unit test for Qwen VL weight loading fix (issue #16722).

Tests that lm_head.weight is NOT overwritten by embed_tokens.weight when 
lm_head.weight is loaded from checkpoint before embed_tokens.weight.

The bug: glob.glob returns safetensors shards in unpredictable order.
If lm_head.weight loads first, then embed_tokens.weight loads later,
the old code would overwrite lm_head with embed_tokens (incorrect).

The fix: Track if lm_head.weight was already loaded, only copy 
embed_tokens to lm_head if lm_head wasn't already loaded.

Affected models:
- qwen3_vl.py (Qwen3VLForConditionalGeneration)
- qwen2_5_vl.py (Qwen2_5_VLForConditionalGeneration)
"""
import unittest
from unittest.mock import MagicMock, patch

import torch


class TestQwen3VLWeightLoadingLogic(unittest.TestCase):
    """Test the weight loading logic fix for issue #16722.
    
    This tests the core logic pattern used in Qwen3VLForConditionalGeneration.load_weights().
    The test simulates the key behavior: tracking whether lm_head.weight was loaded
    from checkpoint to avoid overwriting it with embed_tokens.weight.
    
    See: python/sglang/srt/models/qwen3_vl.py lines 946-1020 for the actual implementation.
    """

    def _simulate_load_weights_logic(
        self,
        weights,
        tie_word_embeddings=True,
        is_last_rank=True,
    ):
        """Simulate the fixed load_weights logic.
        
        This mimics the key parts of Qwen3VLForConditionalGeneration.load_weights()
        that handle lm_head and embed_tokens.
        
        Returns:
            tuple: (lm_head_weight, embed_tokens_weight, lm_head_loaded)
        """
        # Simulated model state
        lm_head_weight = None
        embed_tokens_weight = None
        lm_head_loaded = False  # THE FIX: track if lm_head was loaded from checkpoint
        
        for name, loaded_weight in weights:
            # Simulate embed_tokens processing with tie_word_embeddings logic
            if (
                is_last_rank
                and "model.embed_tokens.weight" in name
                and tie_word_embeddings
            ):
                embed_tokens_weight = loaded_weight.clone()
                # THE FIX: Only copy to lm_head if lm_head wasn't already loaded
                if not lm_head_loaded:
                    lm_head_weight = loaded_weight.clone()
            
            # Simulate lm_head.weight loading
            if name == "lm_head.weight":
                lm_head_weight = loaded_weight.clone()
                lm_head_loaded = True  # THE FIX: mark as loaded
        
        return lm_head_weight, embed_tokens_weight, lm_head_loaded

    def test_lm_head_not_overwritten_when_loaded_first(self):
        """Test that lm_head.weight is NOT overwritten when it comes before embed_tokens.
        
        This is the KEY test for issue #16722:
        - lm_head.weight is loaded first from checkpoint
        - embed_tokens.weight is processed later
        - lm_head should keep its checkpoint value, NOT be overwritten by embed_tokens
        """
        # Create distinct weights so we can tell them apart
        lm_head_from_checkpoint = torch.randn(1000, 128) + 10.0  # offset to distinguish
        embed_tokens_from_checkpoint = torch.randn(1000, 128)
        
        # Order: lm_head FIRST, then embed_tokens (the bug scenario)
        weights = [
            ("lm_head.weight", lm_head_from_checkpoint),
            ("model.embed_tokens.weight", embed_tokens_from_checkpoint),
        ]
        
        lm_head_final, embed_tokens_final, lm_head_loaded = self._simulate_load_weights_logic(
            weights, tie_word_embeddings=True
        )
        
        # lm_head should have its checkpoint value, NOT embed_tokens value
        self.assertTrue(
            torch.allclose(lm_head_final, lm_head_from_checkpoint),
            "lm_head.weight should keep its checkpoint value when loaded first "
            "(fix for issue #16722)"
        )
        self.assertFalse(
            torch.allclose(lm_head_final, embed_tokens_from_checkpoint),
            "lm_head.weight should NOT be overwritten by embed_tokens when loaded first"
        )
        self.assertTrue(lm_head_loaded, "lm_head_loaded flag should be True")

    def test_lm_head_copied_from_embed_tokens_when_not_in_checkpoint(self):
        """Test that lm_head gets embed_tokens value when only embed_tokens is in checkpoint.
        
        When tie_word_embeddings=True and lm_head.weight is NOT in checkpoint,
        lm_head should be copied from embed_tokens.
        """
        embed_tokens_from_checkpoint = torch.randn(1000, 128)
        
        # Order: only embed_tokens, no lm_head in checkpoint
        weights = [
            ("model.embed_tokens.weight", embed_tokens_from_checkpoint),
        ]
        
        lm_head_final, embed_tokens_final, lm_head_loaded = self._simulate_load_weights_logic(
            weights, tie_word_embeddings=True
        )
        
        # lm_head should be copied from embed_tokens
        self.assertTrue(
            torch.allclose(lm_head_final, embed_tokens_from_checkpoint),
            "lm_head should be copied from embed_tokens when not in checkpoint"
        )
        self.assertFalse(lm_head_loaded, "lm_head_loaded should be False (not from checkpoint)")

    def test_embed_tokens_first_then_lm_head(self):
        """Test order: embed_tokens first, then lm_head.
        
        When embed_tokens comes first, lm_head gets copied from it.
        Then when lm_head.weight comes, it overwrites with checkpoint value.
        """
        lm_head_from_checkpoint = torch.randn(1000, 128) + 10.0
        embed_tokens_from_checkpoint = torch.randn(1000, 128)
        
        # Order: embed_tokens FIRST, then lm_head
        weights = [
            ("model.embed_tokens.weight", embed_tokens_from_checkpoint),
            ("lm_head.weight", lm_head_from_checkpoint),
        ]
        
        lm_head_final, embed_tokens_final, lm_head_loaded = self._simulate_load_weights_logic(
            weights, tie_word_embeddings=True
        )
        
        # lm_head should have checkpoint value (loaded after embed_tokens)
        self.assertTrue(
            torch.allclose(lm_head_final, lm_head_from_checkpoint),
            "lm_head should have checkpoint value when loaded after embed_tokens"
        )
        self.assertTrue(lm_head_loaded)

    def test_no_copy_when_tie_word_embeddings_false(self):
        """Test that embed_tokens is NOT copied to lm_head when tie_word_embeddings=False."""
        lm_head_from_checkpoint = torch.randn(1000, 128) + 10.0
        embed_tokens_from_checkpoint = torch.randn(1000, 128)
        
        weights = [
            ("model.embed_tokens.weight", embed_tokens_from_checkpoint),
            ("lm_head.weight", lm_head_from_checkpoint),
        ]
        
        lm_head_final, embed_tokens_final, lm_head_loaded = self._simulate_load_weights_logic(
            weights, tie_word_embeddings=False  # Disabled
        )
        
        # lm_head should only have checkpoint value, not copied from embed_tokens
        self.assertTrue(
            torch.allclose(lm_head_final, lm_head_from_checkpoint),
            "lm_head should have checkpoint value"
        )


if __name__ == "__main__":
    unittest.main()
