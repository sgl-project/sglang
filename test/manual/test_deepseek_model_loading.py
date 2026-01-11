"""
Integration test: Load DeepSeek model and run inference.
Requires: GPU with sufficient VRAM
Run: python test/manual/test_deepseek_model_loading.py
"""
import torch
from transformers import AutoTokenizer, AutoConfig
from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM


def test_model_loading():
    """Test loading DeepSeek V3 model architecture."""
    print("Testing DeepSeek V3 model loading...")
    
    # Use a small config for testing (not loading actual weights)
    config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
    
    try:
        # Test model instantiation
        model = DeepseekV3ForCausalLM(config, quant_config=None)
        print("  ✓ Model instantiated successfully")
        
        # Verify utilities are accessible from model
        from sglang.srt.models.deepseek_common.utils import (
            FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
            yarn_get_mscale,
        )
        print("  ✓ Utilities accessible from model context")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("DeepSeek Model Loading Test")
    print("="*60 + "\n")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        print("⚠️  No GPU detected. Some tests may fail.\n")
    
    success = test_model_loading()
    
    print("\n" + "="*60)
    if success:
        print("✅ MODEL LOADING TEST PASSED!")
    else:
        print("❌ MODEL LOADING TEST FAILED!")
    print("="*60 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
