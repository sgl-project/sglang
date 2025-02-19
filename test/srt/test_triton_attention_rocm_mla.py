import random
import unittest

import torch

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd_grouped,
)

from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope1 import decode_attention_fwd_grouped_rope

device = 'cuda'
B = 1235

class TestTritonAttentionMLA(unittest.TestCase):

    def _set_all_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setUp(self):
        # Set seeds before each test method
        self._set_all_seeds(42)

    def test_rocm_fused_mla_kernel(self):
        q_input = torch.randn(B, 16, 576, device=device, dtype=torch.bfloat16)
        k_buffer = torch.randn(1045022, 1, 576, device=device, dtype=torch.bfloat16)
        v_buffer = torch.randn(1045022, 1, 512, device=device, dtype=torch.bfloat16)
        o = torch.randn(B, 16, 512, device=device, dtype=torch.bfloat16)
        kv_indptr = torch.arange(B + 1, device=device, dtype=torch.int32)
        kv_indices = torch.arange(41943040, device=device, dtype=torch.int32)
        k_pe_output = torch.empty(B, 1, 64, device=device, dtype=torch.bfloat16)
        kv_lora_rank = 512
        rotary_dim = 64
        cos_sin_cache = torch.randn(163840, 64, device=device, dtype=torch.bfloat16)
        positions = torch.tensor([B], device=device).unsqueeze(0).repeat(256, 1)
        attn_logits = torch.empty(B, 16, 16, 513, dtype=torch.float32, device=device)
        num_kv_splits = 16
        sm_scale = 0.1352337788608801
        logit_cap = 0.0

        decode_attention_fwd_grouped(q_input, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, num_kv_splits, sm_scale, logit_cap) 
        o_grouped = torch.randn(B, 16, 512, device=device, dtype=torch.bfloat16)
        attn_logits_grouped = torch.empty(B, 16, 16, 513, dtype=torch.float32, device=device)
        decode_attention_fwd_grouped_rope(q_input, k_buffer, v_buffer, o_grouped, kv_indptr, kv_indices, k_pe_output, kv_lora_rank,
                            rotary_dim, cos_sin_cache, positions, attn_logits_grouped, num_kv_splits, sm_scale, logit_cap, use_rope=True, is_neox_style=False)
        cos_sim = torch.nn.functional.cosine_similarity(
            o.flatten(), o_grouped.flatten(), dim=0
        )
        print(cos_sim.item())
        self.assertTrue(cos_sim.item() > 0.99)
        self.assertTrue(torch.allclose(o, o_grouped, atol=3e-2))

if __name__ == "__main__":
    unittest.main()
