
import os
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sglang.srt.layers.moe.token_dispatcher.pplx import PPLXDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput
import numpy as np

# Mocking envs for the test
from unittest.mock import patch

def _init_distributed(rank, world_size, master_addr='127.0.0.1', master_port='29500'):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}" # Isolate GPU for each process if needed, or rely on torch.cuda.set_device
    
    # Initialize process group with support for both CPU (Gloo) and CUDA (NCCL) operations
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def _cleanup_distributed():
    dist.destroy_process_group()

def _worker_fn(rank, world_size, num_experts, experts_per_token, hidden_dim):
    _init_distributed(rank, world_size)
    
    try:
        # PPLXDispatcher expects SGLANG_PPLX_NUM_MAX_DISPATCH_TOKENS_PER_RANK and SGLANG_PPLX_INTERNODE env vars
        # We can patch them or set them in env.
        # Since we are in a subprocess, patching might be tricky if not done carefully.
        # But we can set env vars before importing or using.
        
        # However, SGLang envs are often read at import time or lazily. 
        # PPLXDispatcher reads them in __init__.
        
        # We need to mock 'sglang.srt.environ.envs' lookups or set the underlying env vars.
        os.environ['SGLANG_PPLX_NUM_MAX_DISPATCH_TOKENS_PER_RANK'] = "1024"
        os.environ['SGLANG_PPLX_INTERNODE'] = "False"
        
        # Also, get_attention_dp_size is called. We should mock it to return 1 for simplicity or match world_size.
        with patch('sglang.srt.layers.moe.token_dispatcher.pplx.get_attention_dp_size', return_value=1), \
             patch('sglang.srt.layers.moe.token_dispatcher.pplx.envs.SGLANG_PPLX_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get', return_value=128), \
             patch('sglang.srt.layers.moe.token_dispatcher.pplx.envs.SGLANG_PPLX_INTERNODE.get', return_value=False):
            
            group = dist.group.WORLD
            
            # Instance
            dispatcher = PPLXDispatcher(
                group=group,
                num_experts=num_experts,
                experts_per_token=experts_per_token,
                hidden_dim=hidden_dim,
                hidden_dim_bytes=hidden_dim * 2, # float16
                hidden_dim_scale_bytes=0,
                block_size=128,
            )
            
            # --- Test Dispatch ---
            num_tokens = 10
            # Ensure we don't exceed max_num_tokens (128)
            
            # Input data
            hidden_states = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
            
            # Random routing
            # indices: (num_tokens, experts_per_token)
            # weights: (num_tokens, experts_per_token)
            
            topk_ids = torch.randint(0, num_experts, (num_tokens, experts_per_token), device="cuda", dtype=torch.int32)
            topk_weights = torch.softmax(torch.randn(num_tokens, experts_per_token, device="cuda", dtype=torch.float32), dim=-1)
            
            # Create StandardTopKOutput
            # StandardTopKOutput(topk_weights, topk_ids, router_logits)
            router_logits = torch.empty((num_tokens, num_experts), device="cuda", dtype=torch.float32)
            topk_output = StandardTopKOutput(topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits)
            
            dispatch_output = dispatcher.dispatch(hidden_states, topk_output)
            
            # Minimal check: sum of hidden states should be roughly preserved if we consider the routing? 
            # Not exactly, because we replicate for multiple experts.
            # But the 'combine' essentially does the reverse sum-reduction.
            
            # Let's perform a "fake" expert computation: identity * scalar
            # dispatch_output.hidden_states is (num_local_experts, max_tokens, hidden_dim)
            
            expert_out = dispatch_output.hidden_states.clone()
            
            # --- Test Combine ---
            from sglang.srt.layers.moe.token_dispatcher.pplx import PPLXCombineInput
            
            combine_input = PPLXCombineInput(
                hidden_states=expert_out,
                topk_ids=dispatch_output.topk_ids,
                topk_weights=dispatch_output.topk_weights
            )
            
            combined_output = dispatcher.combine(combine_input)
            
            # Verification:
            # If expert computation was Identity, combine(dispatch(x)) ~= x * weight_sum?
            # Actually, combine computes: out[i] = sum_k ( expert_out[k] * weight[k] )
            # Here expert_out[k] came from x[i].
            # So combined_output[i] should be approx x[i] * sum(weights[i])
            # Since weights sum to 1, combined_output should be approx x[i].
            
            # Check tolerances
            torch.testing.assert_close(combined_output, hidden_states, rtol=1e-3, atol=1e-3)
            
            print(f"Rank {rank}: Test Passed")
            
            dispatcher.destroy()

    except Exception as e:
        print(f"Rank {rank} failed with exception: {e}")
        raise e
    finally:
        _cleanup_distributed()

class TestPPLXDispatcher(unittest.TestCase):
    def test_pplx_dispatcher(self):
        # Configuration
        world_size = 4 # Use 4 GPUs
        if torch.cuda.device_count() < world_size:
            self.skipTest(f"Need at least {world_size} GPUs")
        
        num_experts = 8
        experts_per_token = 2
        hidden_dim = 64
        
        mp.spawn(
            _worker_fn,
            args=(world_size, num_experts, experts_per_token, hidden_dim),
            nprocs=world_size,
            join=True
        )

if __name__ == '__main__':
    print("running test...")
    unittest.main()
