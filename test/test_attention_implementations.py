#!/usr/bin/env python3
"""
Qwen3-Next Attention Implementation Comparison

Compare three implementations:
1. aiter mha_batch_prefill (doesn't support head_dim=256, will fail)
2. Pure PyTorch implementation
3. Triton extend_attention_fwd (actual fallback used)
"""

import torch
import torch.nn.functional as F
import math
import time
import sys
sys.path.insert(0, '/sgl-workspace/sglang/python')

# Qwen3-Next Full Attention Configuration
BATCH_SIZE = 4
SEQ_LENS_Q = [2048, 2048]
SEQ_LENS_KV = [4096, 4096]
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2  # GQA
HEAD_DIM = 256
V_HEAD_DIM = 256

device = "cuda"
dtype = torch.bfloat16


def construct_qwen3_next_inputs():
    """Construct Qwen3-Next input data"""
    total_q_tokens = sum(SEQ_LENS_Q)
    total_kv_tokens = sum(SEQ_LENS_KV)
    
    # Query: [total_tokens, num_q_heads, head_dim]
    q = torch.randn(total_q_tokens, NUM_Q_HEADS, HEAD_DIM, dtype=dtype, device=device) * 0.1
    
    # KV cache: use [num_blocks, num_kv_heads, head_dim] format directly
    # Simplified: one block per token
    k_cache = torch.randn(total_kv_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device) * 0.1
    v_cache = torch.randn(total_kv_tokens, NUM_KV_HEADS, V_HEAD_DIM, dtype=dtype, device=device) * 0.1
    
    # indptr for variable-length sequences
    qo_indptr = torch.tensor([0] + [sum(SEQ_LENS_Q[:i+1]) for i in range(BATCH_SIZE)], 
                              dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0] + [sum(SEQ_LENS_KV[:i+1]) for i in range(BATCH_SIZE)],
                              dtype=torch.int32, device=device)
    
    # Block indices: simple sequential mapping
    kv_indices = torch.arange(total_kv_tokens, dtype=torch.int32, device=device)
    
    max_q_len = max(SEQ_LENS_Q)
    max_kv_len = max(SEQ_LENS_KV)
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    return {
        'q': q,
        'k_cache': k_cache,
        'v_cache': v_cache,
        'qo_indptr': qo_indptr,
        'kv_indptr': kv_indptr,
        'kv_indices': kv_indices,
        'max_q_len': max_q_len,
        'max_kv_len': max_kv_len,
        'softmax_scale': softmax_scale,
    }


def pytorch_attention(inputs):
    """Pure PyTorch implementation"""
    q = inputs['q']
    k_cache = inputs['k_cache']
    v_cache = inputs['v_cache']
    qo_indptr = inputs['qo_indptr']
    kv_indptr = inputs['kv_indptr']
    kv_indices = inputs['kv_indices']
    softmax_scale = inputs['softmax_scale']
    
    batch_size = len(qo_indptr) - 1
    num_groups = NUM_Q_HEADS // NUM_KV_HEADS
    output = torch.zeros_like(q)
    
    for batch_idx in range(batch_size):
        q_start, q_end = qo_indptr[batch_idx].item(), qo_indptr[batch_idx + 1].item()
        kv_start, kv_end = kv_indptr[batch_idx].item(), kv_indptr[batch_idx + 1].item()
        
        if q_end - q_start == 0 or kv_end - kv_start == 0:
            continue
        
        q_seq = q[q_start:q_end]
        block_indices = kv_indices[kv_start:kv_end]
        k_seq = k_cache[block_indices]
        v_seq = v_cache[block_indices]
        
        seq_len_q = q_end - q_start
        seq_len_kv = kv_end - kv_start
        
        for group_idx in range(num_groups):
            q_head_start = group_idx * NUM_KV_HEADS
            q_head_end = (group_idx + 1) * NUM_KV_HEADS
            q_group = q_seq[:, q_head_start:q_head_end, :]
            
            # Q @ K^T
            scores = torch.einsum('qhd,khd->qhk', q_group, k_seq) * softmax_scale
            
            # Causal mask
            mask = torch.triu(
                torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool, device=device),
                diagonal=seq_len_kv - seq_len_q + 1
            )
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
            
            # Softmax + @ V
            attn_weights = F.softmax(scores, dim=-1)
            o_group = torch.einsum('qhk,khd->qhd', attn_weights, v_seq)
            
            output[q_start:q_end, q_head_start:q_head_end, :] = o_group
    
    return output


def triton_attention(inputs):
    """Triton extend_attention_fwd (actual fallback used)"""
    from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd
    
    q = inputs['q']
    k_cache = inputs['k_cache']
    v_cache = inputs['v_cache']
    qo_indptr = inputs['qo_indptr']
    kv_indptr = inputs['kv_indptr']
    kv_indices = inputs['kv_indices']
    max_q_len = inputs['max_q_len']
    softmax_scale = inputs['softmax_scale']
    
    bs0 = len(qo_indptr)
    total_q_tokens = q.shape[0]
    
    o = q.new_empty((total_q_tokens, NUM_Q_HEADS * V_HEAD_DIM))
    
    extend_attention_fwd(
        q.view(-1, NUM_Q_HEADS, HEAD_DIM),
        k_cache.view(-1, NUM_KV_HEADS, HEAD_DIM),
        v_cache.view(-1, NUM_KV_HEADS, V_HEAD_DIM),
        o.view(-1, NUM_Q_HEADS, V_HEAD_DIM),
        k_cache.view(-1, NUM_KV_HEADS, HEAD_DIM),
        v_cache.view(-1, NUM_KV_HEADS, V_HEAD_DIM),
        qo_indptr[:bs0],
        kv_indptr[:bs0],
        kv_indices,
        None,
        True,  # causal
        None,
        max_q_len,
        softmax_scale,
        0.0,  # logits_soft_cap
        -1,  # sliding_window_size
    )
    
    return o.view(-1, NUM_Q_HEADS, V_HEAD_DIM)


def aiter_attention(inputs):
    """aiter mha_batch_prefill (only supports head_dim=128, will fail)"""
    try:
        from aiter.ops.mha import mha_batch_prefill_func
        
        q = inputs['q']
        k_cache = inputs['k_cache']
        v_cache = inputs['v_cache']
        qo_indptr = inputs['qo_indptr']
        kv_indptr = inputs['kv_indptr']
        kv_indices = inputs['kv_indices']
        max_q_len = inputs['max_q_len']
        max_kv_len = inputs['max_kv_len']
        softmax_scale = inputs['softmax_scale']
        
        bs0 = len(qo_indptr)
        
        o = mha_batch_prefill_func(
            q.contiguous().view(-1, NUM_Q_HEADS, HEAD_DIM),
            k_cache,
            v_cache,
            qo_indptr[:bs0],
            kv_indptr[:bs0],
            kv_indices,
            max_q_len,
            max_kv_len,
            causal=True,
            logits_soft_cap=0.0,
            alibi_slopes=None,
            return_lse=False,
            return_attn_probs=False,
        )
        
        return o
    except Exception as e:
        print(f"  ✗ aiter failed (expected): {str(e)[:80]}")
        return None


def create_new_attention_model():
    """Initialize the new attention model once"""
    try:
        sys.path.insert(0, '/app')
        from new_attention_generated import ModelNew
        model = ModelNew(NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, V_HEAD_DIM).to(device)
        return model
    except Exception as e:
        print(f"  ✗ Failed to create new attention model: {str(e)[:80]}")
        import traceback
        traceback.print_exc()
        return None


def new_attention(inputs, model=None):
    """New custom batch_prefill_attention implementation"""
    try:
        if model is None:
            return None
        
        q = inputs['q']
        k_cache = inputs['k_cache']
        v_cache = inputs['v_cache']
        qo_indptr = inputs['qo_indptr']
        kv_indptr = inputs['kv_indptr']
        kv_indices = inputs['kv_indices']
        softmax_scale = inputs['softmax_scale']
        
        o = model.forward(
            q,
            k_cache,
            v_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            softmax_scale,
        )
        
        return o
    except Exception as e:
        print(f"  ✗ New attention failed: {str(e)[:80]}")
        import traceback
        traceback.print_exc()
        return None


def test_implementation(name, func, inputs, warmup=3, runs=10):
    """Test single implementation and return output + timing"""
    print(f"\nTesting: {name}")
    
    try:
        # Warmup
        for _ in range(warmup):
            output = func(inputs)
            if output is None:
                return None, None
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            output = func(inputs)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / runs * 1000
        
        # Verify
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        
        if has_nan or has_inf:
            print(f"  ✗ Failed: Output contains NaN or Inf")
            return None, None
        
        print(f"  ✓ Success - {elapsed:.3f} ms")
        
        return output, elapsed
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def compare_outputs(out1, out2, name1, name2, tolerance=1e-2):
    """Compare two outputs and return if they match"""
    if out1 is None or out2 is None:
        return False, None, None
    
    diff = (out1.float() - out2.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    passed = max_diff < tolerance
    
    return passed, max_diff, mean_diff


def main():
    print("=" * 80)
    print("Qwen3-Next Attention Implementation Comparison")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  seq_lens_q: {SEQ_LENS_Q}")
    print(f"  seq_lens_kv: {SEQ_LENS_KV}")
    print(f"  num_q_heads: {NUM_Q_HEADS}")
    print(f"  num_kv_heads: {NUM_KV_HEADS} (GQA)")
    print(f"  head_dim: {HEAD_DIM}")
    print(f"  v_head_dim: {V_HEAD_DIM}")
    print(f"  device: {device}")
    print(f"  dtype: {dtype}")
    
    print(f"\nConstructing input data...")
    inputs = construct_qwen3_next_inputs()
    print(f"  ✓ Input data ready")
    
    # Run all implementations
    print(f"\n{'='*80}")
    print("Running Implementations")
    print("=" * 80)
    
    # Test 1: PyTorch Baseline
    out_pytorch, time_pytorch = test_implementation("PyTorch Baseline", pytorch_attention, inputs)
    
    # Test 2: Triton
    out_triton, time_triton = test_implementation("Triton extend_attention_fwd", triton_attention, inputs)
    
    # Test 3: aiter (expected to fail for head_dim=256)
    out_aiter, time_aiter = test_implementation("aiter mha_batch_prefill", aiter_attention, inputs)
    
    # Test 4: New custom implementation
    print(f"\nInitializing new attention model...")
    new_model = create_new_attention_model()
    if new_model is not None:
        print(f"  ✓ Model initialized")
        out_new, time_new = test_implementation("New Custom batch_prefill_attention", 
                                               lambda inp: new_attention(inp, new_model), inputs)
    else:
        out_new, time_new = None, None
    
    # Correctness Check
    print(f"\n{'='*80}")
    print("1. CORRECTNESS CHECK (vs PyTorch Baseline)")
    print("=" * 80)
    
    if out_pytorch is None:
        print("✗ PyTorch baseline failed - cannot validate correctness")
    else:
        # Check Triton
        triton_pass, triton_max_diff, triton_mean_diff = compare_outputs(
            out_pytorch, out_triton, "PyTorch", "Triton", tolerance=1e-2)
        
        if triton_pass:
            print(f"✓ Triton:        PASS (max_diff={triton_max_diff:.2e}, mean_diff={triton_mean_diff:.2e})")
        elif out_triton is not None:
            print(f"✗ Triton:        FAIL (max_diff={triton_max_diff:.2e}, mean_diff={triton_mean_diff:.2e})")
        else:
            print(f"✗ Triton:        FAIL (execution error)")
        
        # Check aiter
        aiter_pass, aiter_max_diff, aiter_mean_diff = compare_outputs(
            out_pytorch, out_aiter, "PyTorch", "aiter", tolerance=1e-2)
        
        if aiter_pass:
            print(f"✓ aiter:         PASS (max_diff={aiter_max_diff:.2e}, mean_diff={aiter_mean_diff:.2e})")
        elif out_aiter is not None:
            print(f"✗ aiter:         FAIL (max_diff={aiter_max_diff:.2e}, mean_diff={aiter_mean_diff:.2e})")
        else:
            print(f"✗ aiter:         FAIL (execution error, expected for head_dim=256)")
        
        # Check New Custom
        new_pass, new_max_diff, new_mean_diff = compare_outputs(
            out_pytorch, out_new, "PyTorch", "New Custom", tolerance=1e-2)
        
        if new_pass:
            print(f"✓ ModelAgent:    PASS (max_diff={new_max_diff:.2e}, mean_diff={new_mean_diff:.2e})")
        elif out_new is not None:
            print(f"✗ ModelAgent:    FAIL (max_diff={new_max_diff:.2e}, mean_diff={new_mean_diff:.2e})")
        else:
            print(f"✗ ModelAgent:    FAIL (execution error)")
    
    # Benchmark Results
    print(f"\n{'='*80}")
    print("2. BENCHMARK RESULTS")
    print("=" * 80)
    
    print(f"\nAbsolute Performance:")
    if time_pytorch is not None:
        print(f"  PyTorch Baseline:    {time_pytorch:8.3f} ms")
    else:
        print(f"  PyTorch Baseline:    FAILED")
    
    if time_triton is not None:
        print(f"  Triton:              {time_triton:8.3f} ms")
    else:
        print(f"  Triton:              FAILED")
    
    if time_aiter is not None:
        print(f"  aiter:               {time_aiter:8.3f} ms")
    else:
        print(f"  aiter:               FAILED")
    
    if time_new is not None:
        print(f"  ModelAgent:          {time_new:8.3f} ms")
    else:
        print(f"  ModelAgent:          FAILED")
    
    print(f"\nSpeedup vs PyTorch Baseline:")
    if time_pytorch is not None:
        if time_triton is not None:
            speedup_triton = time_pytorch / time_triton
            print(f"  Triton:              {speedup_triton:6.2f}x {'(faster)' if speedup_triton > 1 else '(slower)'}")
        else:
            print(f"  Triton:              N/A")
        
        if time_aiter is not None:
            speedup_aiter = time_pytorch / time_aiter
            print(f"  aiter:               {speedup_aiter:6.2f}x {'(faster)' if speedup_aiter > 1 else '(slower)'}")
        else:
            print(f"  aiter:               N/A")
        
        if time_new is not None:
            speedup_new = time_pytorch / time_new
            print(f"  ModelAgent:          {speedup_new:6.2f}x {'(faster)' if speedup_new > 1 else '(slower)'}")
        else:
            print(f"  ModelAgent:          N/A")
    
    # Final Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)
    
    if out_pytorch is not None and out_triton is not None and triton_pass:
        print("✓ Triton:      Correct and functional")
    elif out_triton is not None:
        print("✗ Triton:      Incorrect results")
    else:
        print("✗ Triton:      Failed to execute")
    
    if out_aiter is not None and aiter_pass:
        print("✓ aiter:       Correct and functional")
    elif out_aiter is not None:
        print("✗ aiter:       Incorrect results")
    else:
        print("✗ aiter:       Failed to execute (expected for head_dim=256)")
    
    if out_pytorch is not None and out_new is not None and new_pass:
        print("✓ ModelAgent:  Correct and functional")
    elif out_new is not None:
        print("✗ ModelAgent:  Incorrect results")
    else:
        print("✗ ModelAgent:  Failed to execute")
    
    print()


if __name__ == "__main__":
    main()
