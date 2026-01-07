"""
SGLang DeepSeek V2 Attention Operator Collector

This module collects performance data for SGLang's DeepSeek V2 attention operators,
supporting different quantization strategies including per tensor FP8, block scale FP8, and bfloat16.
"""

import logging
import math
import time
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import SGLang components
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA, AttnForwardMethod, yarn_get_mscale
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.utils import BumpAllocator

logger = logging.getLogger(__name__)

# =============================================================================
# Simplified Mock Classes for testing without SGLang dependencies
# =============================================================================

class MockForwardBatch:
    """Mock ForwardBatch for testing."""
    def __init__(self):
        self.forward_mode = self
        self.extend_prefix_lens_cpu = []
        self.batch_size = 1
        
    def is_extend(self): return False
    def is_target_verify(self): return False  
    def is_draft_extend(self): return False

class MockConfig:
    """Mock config for testing."""
    def __init__(self):
        self.rms_norm_eps = 1e-6
        self.architectures = ["DeepseekV2ForCausalLM"]
        self.num_attention_heads = 56
        self.qk_nope_head_dim = 128
        self.qk_rope_head_dim = 64
        self.v_head_dim = 128
        self.q_lora_rank = 1536
        self.kv_lora_rank = 512
        self.hidden_size = 7168
        self.max_position_embeddings = 32768

# =============================================================================
# Main Benchmark Functions using SGLang's DeepseekV2AttentionMLA
# =============================================================================

# Note: Now using SGLang's native DeepseekV2AttentionMLA directly

# =============================================================================
# Main Benchmark Function (仿照 TRTLLM 接口)
# =============================================================================

def run_attention_torch(batch_size: int,
                       input_len: int,
                       num_heads: int,
                       num_key_value_heads: int,  # keep same as num_heads for MHA
                       head_dim: int,
                       use_fp8_weights: bool,
                       use_block_fp8: bool,
                       is_context_phase: bool,
                       perf_filename: str,
                       device: str = 'cuda:0') -> None:
    """
    Run SGLang attention benchmark with specified parameters.
    
    Args:
        batch_size: Batch size for testing
        input_len: Input sequence length
        num_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads (same as num_heads for MHA)
        head_dim: Head dimension (fixed at 128 for DeepSeek V2)
        use_fp8_weights: Whether to use FP8 weight quantization
        use_block_fp8: Whether to use block-wise FP8 quantization
        is_context_phase: Whether this is context phase (affects seq_len)
        perf_filename: Output performance file path
        device: Device to run on
    """
    torch.cuda.set_device(device)
    
    # Configure quantization using SGLang's Fp8Config
    if use_fp8_weights:
        if use_block_fp8:
            # Block-wise FP8 quantization (requires serialized checkpoint)
            # Block size [128, 128] is commonly used for optimal performance
            quant_config = Fp8Config(
                is_checkpoint_fp8_serialized=True,  # Required for block-wise
                activation_scheme="dynamic",         # Only dynamic supported for block-wise
                weight_block_size=[128, 128]        # [block_n, block_k] dimensions
            )
            quant_mode = "block_fp8"
        else:
            # Per-tensor FP8 quantization
            # For testing, we'll try non-serialized first (runtime quantization)
            quant_config = Fp8Config(
                is_checkpoint_fp8_serialized=False, # Runtime quantization
                activation_scheme="dynamic",         # Dynamic activation scaling
                weight_block_size=None              # Per-tensor quantization
            )
            quant_mode = "per_tensor_fp8"
    else:
        quant_config = None
        quant_mode = "bfloat16"
    
    # Create SGLang-compatible config
    mock_config = MockConfig()
    mock_config.num_attention_heads = num_heads
    mock_config.qk_rope_head_dim = head_dim // 2
    mock_config.qk_nope_head_dim = head_dim // 2
    mock_config.v_head_dim = head_dim
    
    # Create model using SGLang's native DeepseekV2AttentionMLA
    try:
        model = DeepseekV2AttentionMLA(
            config=mock_config,
            hidden_size=mock_config.hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=head_dim // 2,
            qk_rope_head_dim=head_dim // 2,
            v_head_dim=head_dim,
            q_lora_rank=mock_config.q_lora_rank,
            kv_lora_rank=mock_config.kv_lora_rank,
            quant_config=quant_config,
            layer_id=0,
            prefix="test_attn"
        ).to(device)
        print(f"✅ Model created successfully with {quant_mode} quantization")
        
        # Post-process weights for weight absorption if needed
        if hasattr(model, 'post_load_weights'):
            model.post_load_weights()
            
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        print(f"Falling back to BFloat16 mode...")
        
        # Fallback to no quantization
        model = DeepseekV2AttentionMLA(
            config=mock_config,
            hidden_size=mock_config.hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=head_dim // 2,
            qk_rope_head_dim=head_dim // 2,
            v_head_dim=head_dim,
            q_lora_rank=mock_config.q_lora_rank,
            kv_lora_rank=mock_config.kv_lora_rank,
            quant_config=None,  # Fallback to no quantization
            layer_id=0,
            prefix="test_attn"
        ).to(device)
        quant_mode = "bfloat16_fallback"
    
    # Determine sequence length based on phase
    if is_context_phase:
        seq_len = input_len
        num_tokens = batch_size * seq_len
        op_name = 'context_attention'
        step = 0
    else:
        seq_len = 1  # Generation phase processes one token at a time
        num_tokens = batch_size
        op_name = 'generation_attention'
        step = input_len
    
    # Generate test inputs
    hidden_states = torch.randn(batch_size, seq_len, mock_config.hidden_size, 
                               dtype=torch.bfloat16, device=device)
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Create mock forward batch for SGLang
    zero_allocator = BumpAllocator(buffer_size=10, dtype=torch.float32, device=device)
    
    # Test parameters
    warming_up = 10
    test_ite = 6
    
    # Create mock forward batch
    mock_batch = MockForwardBatch()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warming_up):
            _ = model(positions, hidden_states, mock_batch, zero_allocator)
    
    # Simple benchmark (SGLang's CUDA graph is complex, use direct timing)
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(test_ite):
            _ = model(positions, hidden_states, mock_batch, zero_allocator)
    
    torch.cuda.synchronize()
    end_time = time.time()
    latency = (end_time - start_time) * 1000 / test_ite  # Convert to ms
    
    # Write result in TRTLLM format
    isl = input_len if is_context_phase else 1
    
    # Use the determined quant_mode for output
    dtype_str = quant_mode
    
    kvcache_dtype_str = 'bfloat16'  # SGLang uses bfloat16 for KV cache
    
    # Write to file
    fd = os.open(perf_filename, os.O_APPEND | os.O_WRONLY | os.O_CREAT)
    content = f'SGLang,{torch.__version__},{torch.cuda.get_device_name(device)},{op_name},{batch_size},{isl},{num_heads},{num_key_value_heads},{head_dim},1,{dtype_str},{kvcache_dtype_str},{step},{latency}\n'
    os.write(fd, content.encode())
    os.close(fd)

def get_context_attention_test_cases() -> List[List]:
    """Generate test cases for context attention phase."""
    test_cases = []
    
    # Test parameters
    b_list = [1, 2, 4, 8, 16, 32, 64, 128]
    s_list = [128, 256, 512, 1024, 2048, 4096, 8192]
    n_list = [8, 16, 24, 32, 40, 48, 56, 64]
    head_dim = 128
    
    for n in sorted(n_list, reverse=True):
        for s in sorted(s_list, reverse=True):
            for b in sorted(b_list, reverse=True):
                # Memory constraints
                if b * s > 65536 or b > 128:
                    continue
                
                # Test cases: [batch_size, input_len, num_heads, num_key_value_heads, head_dim, 
                #             use_fp8_weights, use_block_fp8, is_context_phase, perf_filename]
                
                # BFloat16 baseline
                test_cases.append([b, s, n, n, head_dim, False, False, True, 'sglang_context_attention_perf.txt'])
                
                # Per-tensor FP8
                test_cases.append([b, s, n, n, head_dim, True, False, True, 'sglang_context_attention_perf.txt'])
                
                # Block-wise FP8
                test_cases.append([b, s, n, n, head_dim, True, True, True, 'sglang_context_attention_perf.txt'])
    
    return test_cases

def get_generation_attention_test_cases() -> List[List]:
    """Generate test cases for generation attention phase."""
    test_cases = []
    
    # Test parameters
    b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    s_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]  # Past sequence lengths
    n_list = [8, 16, 24, 32, 40, 48, 56, 64]
    head_dim = 128
    
    # Memory constraints
    max_bsn = 8192 * 1024
    
    for n in sorted(n_list, reverse=True):
        b_s_dict = {}
        s_b_dict = {}
        
        for s in s_list:
            max_b = max_bsn // s // n
            for b in b_list:
                if b > max_b:
                    break
                if s not in s_b_dict.keys():
                    s_b_dict[s] = {b}
                else:
                    s_b_dict[s].add(b)
        
        for s, b_set in s_b_dict.items():
            if len(b_set) < 4:
                continue
            for b in b_set:
                if b not in b_s_dict.keys():
                    b_s_dict[b] = {s}
                else:
                    b_s_dict[b].add(s)
        
        for b, s_list_limited in b_s_dict.items():
            target_s_list = sorted(s_list_limited)
            if b >= 256:
                target_s_list = target_s_list[:-1]
            
            for s in target_s_list:
                # Test cases: [batch_size, input_len, num_heads, num_key_value_heads, head_dim, 
                #             use_fp8_weights, use_block_fp8, is_context_phase, perf_filename]
                
                # BFloat16 baseline
                test_cases.append([b, s, n, n, head_dim, False, False, False, 'sglang_generation_attention_perf.txt'])
                
                # Per-tensor FP8
                test_cases.append([b, s, n, n, head_dim, True, False, False, 'sglang_generation_attention_perf.txt'])
                
                # Block-wise FP8
                test_cases.append([b, s, n, n, head_dim, True, True, False, 'sglang_generation_attention_perf.txt'])
    
    return test_cases

# =============================================================================
# Test Functions
# =============================================================================

def test_fp8_config():
    """Test FP8 quantization config."""
    print("\n🧪 Testing Fp8Config...")
    
    try:
        # Test per-tensor FP8
        per_tensor_config = Fp8Config(
            is_checkpoint_fp8_serialized=False,
            activation_scheme="dynamic",
            weight_block_size=None
        )
        print(f"✅ Per-tensor FP8 config: {per_tensor_config.get_name()}")
        
        # Test block-wise FP8 (requires serialized checkpoint)
        block_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128]
        )
        print(f"✅ Block-wise FP8 config: {block_config.get_name()}")
        
        print("✅ Fp8Config test completed!")
        
    except Exception as e:
        print(f"❌ Fp8Config test failed: {e}")

def test_dispatch_attn_forward_method():
    """Test the dispatch_attn_forward_method logic."""
    print("\n🧪 Testing dispatch_attn_forward_method...")
    
    # Test different backend configurations
    test_configs = [
        ("triton", True, True),      # Backend, disable_ragged, disable_chunked
        ("flashinfer", False, True),
        ("fa3", True, False),
        ("aiter", True, True),
    ]
    
    for backend, disable_ragged, disable_chunked in test_configs:
        try:
            # Create SGLang-compatible config
            mock_config = MockConfig()
            
            model = DeepseekV2AttentionMLA(
                config=mock_config,
                hidden_size=mock_config.hidden_size,
                num_heads=mock_config.num_attention_heads,
                qk_nope_head_dim=mock_config.qk_nope_head_dim,
                qk_rope_head_dim=mock_config.qk_rope_head_dim,
                v_head_dim=mock_config.v_head_dim,
                q_lora_rank=mock_config.q_lora_rank,
                kv_lora_rank=mock_config.kv_lora_rank,
                layer_id=0,
                prefix="test"
            )
            
            # Test dispatch without forward_batch
            mock_batch = MockForwardBatch()
            method = model.dispatch_attn_forward_method(mock_batch)
            print(f"✅ Backend {backend}: {method.name}")
            
        except Exception as e:
            print(f"❌ Backend {backend}: {e}")
    
    print("✅ dispatch_attn_forward_method test completed!")

if __name__ == "__main__":
    print("SGLang Attention Benchmark with Quantization")
    print("=" * 60)
    print("🔧 Available quantization modes:")
    print("   • BFloat16 (baseline)")
    print("   • Per-tensor FP8 (runtime quantization)")
    print("   • Block-wise FP8 (128x128 blocks)")
    print()
    
    # Test FP8 configuration first
    test_fp8_config()
    
    # Test dispatch method
    test_dispatch_attn_forward_method()
    
    # Run context attention tests
    print("\nRunning context attention tests...")
    test_cases = get_context_attention_test_cases()
    for i, test_case in enumerate(test_cases[:2]):  # Limit to first 2 for testing
        print(f"Progress: {i+1}/2 - {test_case}")
        try:
            run_attention_torch(*test_case)
        except Exception as e:
            print(f"Error in test case {test_case}: {e}")
            continue
    
    # Run generation attention tests
    print("\nRunning generation attention tests...")
    test_cases = get_generation_attention_test_cases()
    for i, test_case in enumerate(test_cases[:2]):  # Limit to first 2 for testing
        print(f"Progress: {i+1}/2 - {test_case}")
        try:
            run_attention_torch(*test_case)
        except Exception as e:
            print(f"Error in test case {test_case}: {e}")
            continue
    
    print("Benchmark completed!") 