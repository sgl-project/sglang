import unittest
import torch
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig

class TestFusedMoeActiveMap(unittest.TestCase):
    def setUp(self):
        self.device = "cuda"
        self.dtype = torch.bfloat16
        
    def test_ep_skipping(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Configuration
        num_tokens = 128
        top_k = 2
        hidden_size = 64
        intermediate_size = 128
        num_local_experts = 1
        num_experts = 2 # Global experts
        
        # Create inputs
        hidden_states = torch.randn(num_tokens, hidden_size, device=self.device, dtype=self.dtype)
        # w1: (E, 2*Inter, H)
        w1 = torch.randn(num_local_experts, intermediate_size * 2, hidden_size, device=self.device, dtype=self.dtype)
        # w2: (E, H, Inter)
        w2 = torch.randn(num_local_experts, hidden_size, intermediate_size, device=self.device, dtype=self.dtype)
        
        topk_weights = torch.ones(num_tokens, top_k, device=self.device, dtype=torch.float32) / top_k
        
        # Create topk_ids where some are local (0) and some are remote (1)
        # 0 is local. 1 is remote (since num_local_experts=1).
        topk_ids = torch.randint(0, 2, (num_tokens, top_k), device=self.device, dtype=torch.int32)
        
        # Force some patterns to ensure coverage
        topk_ids[0, :] = 0 # All local
        topk_ids[1, :] = 1 # All remote
        
        topk_output = StandardTopKOutput(topk_weights, topk_ids, topk_ids.shape[1])
        
        # Config to trigger filter_expert (EP mode)
        config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            top_k=top_k,
            activation="silu",
            inplace=False
        )
        
        output = fused_experts(
            hidden_states,
            w1,
            w2,
            topk_output,
            moe_runner_config=config,
            use_int8_w8a16=False
        )
        
        # Reference computation
        ref_output = torch.zeros_like(hidden_states)
        
        for i in range(num_tokens):
            for k in range(top_k):
                eid = topk_ids[i, k].item()
                weight = topk_weights[i, k].item()
                
                if eid < num_local_experts:
                    # Local expert
                    # Forward pass
                    x = hidden_states[i] # (H,)
                    
                    # GateUp
                    # w1[eid] is (2*Inter, H)
                    gate_up = torch.nn.functional.linear(x, w1[eid])
                    gate, up = gate_up.chunk(2, dim=-1)
                    act = torch.nn.functional.silu(gate) * up
                    
                    # Down
                    # w2[eid] is (H, Inter)
                    out = torch.nn.functional.linear(act, w2[eid])
                    
                    ref_output[i] += out * weight
        
        # Compare
        torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
        print("Test passed!")

if __name__ == "__main__":
    unittest.main()
