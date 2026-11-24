import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# backend-specific: the ROCm Triton compile of the extend kernel's block-mask branch.
register_amd_ci(est_time=25, stage="base-a-jit-kernel-unit", runner_config="amd")


class TestExtendAttentionBlockBidirectional(CustomTestCase):
    def test_extend_attention_block_bidirectional(self):
        dtype = torch.bfloat16
        device = get_device()
        torch.manual_seed(0)

        def indptr(lens):
            return torch.cat([lens.new_zeros(1), torch.cumsum(lens, 0)]).int()

        def check(H_Q, H_KV, D, prefix_lens, extend_lens, boundaries):
            prefix = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
            extend = torch.tensor(extend_lens, dtype=torch.int32, device=device)
            kv_indptr, qo_indptr = indptr(prefix), indptr(extend)
            n_kv, n_ext = int(prefix.sum()), int(extend.sum())
            k_buffer = torch.randn((n_kv, H_KV, D), dtype=dtype, device=device) * 0.2
            v_buffer = torch.randn_like(k_buffer)
            q_extend = torch.randn((n_ext, H_Q, D), dtype=dtype, device=device) * 0.2
            k_extend = torch.randn((n_ext, H_KV, D), dtype=dtype, device=device) * 0.2
            v_extend = torch.randn_like(k_extend)
            o = torch.empty_like(q_extend)
            extend_attention_fwd(
                q_extend=q_extend,
                k_extend=k_extend,
                v_extend=v_extend,
                o_extend=o,
                k_buffer=k_buffer,
                v_buffer=v_buffer,
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                kv_indices=torch.arange(n_kv, dtype=torch.int32, device=device),
                custom_mask=None,
                is_causal=True,
                mask_indptr=None,
                max_len_extend=int(extend.max()),
                k_scale=1.0,
                v_scale=1.0,
                sm_scale=D**-0.5,
                block_boundaries=torch.tensor(
                    boundaries, dtype=torch.int32, device=device
                ),
            )
            for i, (P, E) in enumerate(boundaries):
                P0, L = prefix_lens[i], prefix_lens[i] + extend_lens[i]
                qs, ks = (
                    slice(qo_indptr[i], qo_indptr[i + 1]),
                    slice(kv_indptr[i], kv_indptr[i + 1]),
                )
                k = torch.cat([k_buffer[ks], k_extend[qs]]).repeat_interleave(
                    H_Q // H_KV, dim=1
                )
                v = torch.cat([v_buffer[ks], v_extend[qs]]).repeat_interleave(
                    H_Q // H_KV, dim=1
                )
                q_abs = torch.arange(P0, L, device=device)[:, None]
                k_abs = torch.arange(L, device=device)[None, :]
                mask = (k_abs <= q_abs) | (
                    (P >= 0) & ((k_abs < P) | ((q_abs >= P) & (k_abs < E)))
                )
                s = torch.einsum("qhd,khd->qhk", q_extend[qs].float(), k.float())
                s = (s * D**-0.5).masked_fill(~mask[:, None, :], float("-inf"))
                ref = torch.einsum("qhk,khd->qhd", F.softmax(s, dim=-1), v.float())
                torch.testing.assert_close(o[qs].float(), ref, rtol=1e-2, atol=1e-2)

        # Causal, fully bidirectional, state end at or past the cached prefix end,
        # and a retracted request whose question block ends before its outputs (E < L).
        check(
            H_Q=8,
            H_KV=2,
            D=128,
            prefix_lens=[0, 0, 96, 100, 37],
            extend_lens=[150, 200, 70, 90, 131],
            boundaries=[(-1, -1), (0, 200), (96, 166), (130, 190), (90, 140)],
        )
        # Qwen3.5-shaped heads, tile-aligned (1088 = 17 * 64) and unaligned boundaries.
        check(
            H_Q=16,
            H_KV=4,
            D=256,
            prefix_lens=[0, 0, 0, 0],
            extend_lens=[1150, 1157, 3128, 1150],
            boundaries=[(-1, -1), (1088, 1157), (3059, 3128), (1088, 1120)],
        )


if __name__ == "__main__":
    unittest.main()
