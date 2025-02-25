import random
import unittest

import torch

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd_grouped,
)
from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
    decode_attention_fwd_grouped_rope,
)
from sglang.srt.layers.rotary_embedding import DeepseekScalingRotaryEmbedding


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

    def preprocess_kv_cache(self, kv_cache, kv_lora_rank):
        latent_cache = kv_cache
        v_input = latent_cache[..., :kv_lora_rank]
        v_input = v_input.contiguous().unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)
        k_input[..., :kv_lora_rank] = v_input

        return k_input, v_input

    def input_helper(
        self,
        B,
        H,
        S,
        kv_lora_rank,
        rotary_dim,
        qk_rope_head_dim,
        num_kv_splits,
        dtype,
        device,
        rope_base=10,
        rope_max_seq_len=16384,
        rope_scaling=1.0,
        is_neox_style=False,
    ):
        q = torch.randn(
            B, H, kv_lora_rank + qk_rope_head_dim, device=device, dtype=dtype
        )
        kv_cache = torch.randn(
            B * S, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
        )
        kv_indptr = torch.arange(B + 1, device=device) * S
        kv_indices = torch.arange(B * S, device=device)
        attn_logits = torch.empty(
            B, H, num_kv_splits, kv_lora_rank + 1, dtype=dtype, device=device
        )
        rotary_emb = DeepseekScalingRotaryEmbedding(
            qk_rope_head_dim,
            rotary_dim,
            rope_max_seq_len,
            rope_base,
            is_neox_style,
            rope_scaling,
            q.dtype,
            device="cpu",
        ).cuda()
        positions = torch.tensor([S], device=device).unsqueeze(0).repeat(B, 1)

        return kv_indptr, kv_indices, q, kv_cache, attn_logits, rotary_emb, positions

    def ref_compute_full_fwd(
        self,
        q,
        k_input,
        v_input,
        kv_lora_rank,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        sm_scale,
        logit_cap,
        rotary_emb,
        positions,
        use_rope,
        device="cuda",
    ):

        B, H = q.shape[0], q.shape[1]
        S = kv_indptr[1].item()
        qk_rope_head_dim = k_input.shape[-1] - kv_lora_rank

        q_input = torch.empty(B, H, kv_lora_rank + qk_rope_head_dim, dtype=q.dtype).to(
            device
        )
        q_nope_out, q_pe = q.split([kv_lora_rank, qk_rope_head_dim], dim=-1)
        k_pe_t = k_input.view(B, 1, S, -1)[:, :, -1:, kv_lora_rank:]

        if use_rope:
            q_pe, k_pe_t = rotary_emb(positions, q_pe.unsqueeze(2), k_pe_t)
            q_pe = q_pe.squeeze()

        k_input.view(B, 1, S, -1)[:, :, -1:, kv_lora_rank:] = k_pe_t

        q_input[..., :kv_lora_rank] = q_nope_out
        q_input[..., kv_lora_rank:] = q_pe

        B, H = q_input.shape[0], q_input.shape[1]
        kv_lora_rank = v_input.shape[-1]
        device = q_input.device

        attn_logits = torch.empty(
            B, H, num_kv_splits, kv_lora_rank + 1, dtype=q_input.dtype, device=device
        )
        o = torch.empty(B, H, kv_lora_rank, dtype=q_input.dtype, device=device)

        decode_attention_fwd_grouped(
            q_input,
            k_input,
            v_input,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            num_kv_splits,
            sm_scale,
            logit_cap,
        )

        return attn_logits, o, k_pe_t.squeeze()

    def _test_rocm_fused_mla_kernel(
        self,
        B,
        H,
        S,
        kv_lora_rank,
        qk_rope_head_dim,
        rotary_dim,
        dtype,
        use_rope,
        is_neox_style,
        num_kv_splits=2,
        sm_scale=1.0,
        logit_cap=0.0,
        device="cuda",
    ):
        kv_indptr, kv_indices, q, kv_cache, attn_logits, rotary_emb, positions = (
            self.input_helper(
                B,
                H,
                S,
                kv_lora_rank,
                rotary_dim,
                qk_rope_head_dim,
                num_kv_splits,
                dtype,
                device=device,
                is_neox_style=is_neox_style,
            )
        )

        k_input, v_input = self.preprocess_kv_cache(kv_cache, kv_lora_rank)
        k_pe_tokens = torch.empty(
            B, qk_rope_head_dim, dtype=kv_cache.dtype, device=device
        )
        tri_o = torch.empty(B, H, kv_lora_rank, dtype=kv_cache.dtype, device=device)

        decode_attention_fwd_grouped_rope(
            q,
            k_input,
            v_input,
            tri_o,
            kv_indptr,
            kv_indices,
            k_pe_tokens if use_rope else None,
            kv_lora_rank,
            rotary_dim if use_rope else None,
            rotary_emb.cos_sin_cache if use_rope else None,
            positions if use_rope else None,
            attn_logits,
            num_kv_splits,
            sm_scale,
            logit_cap,
            use_rope,
            is_neox_style,
        )

        tri_logits = attn_logits

        # reference
        ref_logits, ref_o, ref_k_pe_tokens = self.ref_compute_full_fwd(
            q,
            k_input,
            v_input,
            kv_lora_rank,
            kv_indptr,
            kv_indices,
            num_kv_splits,
            sm_scale,
            logit_cap,
            rotary_emb,
            positions,
            use_rope,
            device="cuda",
        )

        if use_rope:
            torch.testing.assert_close(
                ref_k_pe_tokens, k_pe_tokens.squeeze(), atol=1e-2, rtol=1e-2
            )
        torch.testing.assert_close(ref_logits, tri_logits, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(ref_o, tri_o, atol=1e-2, rtol=1e-2)

    def test_grouped_rocm_fused_mla(self):
        configs = [
            (1, 128, 2048, 512, 64, 64),
            (1, 128, 2048, 512, 128, 64),
            (1, 128, 2048, 512, 127, 64),
            (1, 128, 2050, 512, 127, 64),
            (1, 128, 2050, 512, 128, 64),
            (8, 128, 2048, 512, 64, 64),
            (8, 128, 2048, 512, 128, 64),
            (8, 128, 2048, 512, 127, 64),
            (8, 128, 2050, 512, 127, 64),
            (8, 128, 2050, 512, 128, 64),
        ]
        dtypes = [torch.bfloat16, torch.float32]
        use_rope_list = [True, False]
        is_neox_style_list = [True, False]

        for B, H, S, kv_lora_rank, qk_rope_head_dim, rotary_dim in configs:
            for dtype in dtypes:
                for use_rope in use_rope_list:
                    for is_neox_style in is_neox_style_list:
                        self._test_rocm_fused_mla_kernel(
                            B,
                            H,
                            S,
                            kv_lora_rank,
                            qk_rope_head_dim,
                            rotary_dim,
                            dtype,
                            use_rope,
                            is_neox_style,
                        )


if __name__ == "__main__":
    unittest.main()
