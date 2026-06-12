"""Unit tests for RVV decode attention kernel (FP16/BF16)."""

import unittest

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.test.test_utils import CustomTestCase

from .rvv_utils import has_sgl_kernel_op, precision

torch.manual_seed(1234)


def _run_sdpa_forward_decode(
    query: torch.Tensor,
    output: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    scaling=None,
    enable_gqa=False,
    logit_cap=0.0,
) -> torch.Tensor:
    """Reference decode attention via PyTorch SDPA (supports logit_cap softcapping)."""
    # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
    query = query.movedim(0, query.dim() - 2)

    start_q = 0
    for seq_idx in range(seq_lens.shape[0]):
        seq_len_kv = seq_lens[seq_idx]
        end_q = start_q + 1

        per_req_query = query[:, start_q:end_q, :]
        req_pool_idx = req_pool_indices[seq_idx]
        per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
        per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
        per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

        if logit_cap > 0:
            # Manual attention with tanh softcap (PyTorch SDPA has no logit_cap param).
            # q: [H_Q, 1, D], k: [H_KV, Tk, D], v: [H_KV, Tk, D_V]
            q = per_req_query.unsqueeze(0).float()
            k = per_req_key.unsqueeze(0).float()
            v = per_req_value.unsqueeze(0).float()
            if enable_gqa:
                G = q.size(1) // k.size(1)
                k = k.repeat_interleave(G, dim=1)
                v = v.repeat_interleave(G, dim=1)
            scores = torch.matmul(q * scaling, k.transpose(-1, -2))
            scores = logit_cap * torch.tanh(scores / logit_cap)
            weights = torch.softmax(scores, dim=-1)
            per_req_out = (
                torch.matmul(weights, v)
                .to(per_req_query.dtype)
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
        else:
            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
        output[start_q:end_q, :, :] = per_req_out
        start_q = end_q

    return output


@unittest.skipUnless(
    has_sgl_kernel_op("decode_attention_cpu"),
    "decode_attention_cpu not available (non-RISC-V build)",
)
class TestRVVDecodeBase(CustomTestCase):
    """Base class for RVV FP decode attention tests."""

    def setUp(self):
        super().setUp()
        self.device = torch.device("cpu")
        self.dtypes = [torch.float16, torch.bfloat16]

    def _run_decode(
        self,
        B,
        H_Q,
        H_KV,
        D,
        D_V,
        seq_len=1024,
        dtype=torch.float16,
        logit_cap=0.0,
        num_kv_splits=2,
    ):
        device = self.device
        gen = torch.Generator(device=device)
        gen.manual_seed(1234)
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        enable_gqa = H_Q != H_KV

        q = torch.randn(B, H_Q, D, dtype=dtype, device=device, generator=gen)
        k_buffer = torch.randn(
            total_tokens, H_KV, D, dtype=dtype, device=device, generator=gen
        )
        v_buffer = torch.randn(
            total_tokens, H_KV, D_V, dtype=dtype, device=device, generator=gen
        )
        key = torch.randn(B, H_KV, D, dtype=dtype, device=device, generator=gen)
        value = torch.randn(B, H_KV, D_V, dtype=dtype, device=device, generator=gen)
        loc = torch.randint(
            0, total_tokens, (B,), dtype=torch.int64, device=device, generator=gen
        )

        k_buffer[loc] = key
        v_buffer[loc] = value

        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)
        o_ref = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)

        # Fragment the token map so cache access is not accidentally contiguous.
        random_indices = torch.randperm(total_tokens, device=device, generator=gen)
        req_to_token = random_indices.reshape(B, seq_len).to(torch.int32)

        b_req_idx = torch.arange(B, device=device, dtype=torch.int64)
        b_seq_len = torch.full((B,), seq_len, device=device, dtype=torch.int64)
        attn_logits = torch.empty(
            (B, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device=device
        )

        # Feed non-contiguous views to match the kernel contract.
        torch.ops.sgl_kernel.decode_attention_cpu(
            q.transpose(0, 1).contiguous().transpose(0, 1),
            k_buffer.transpose(0, 1).contiguous().transpose(0, 1),
            v_buffer.transpose(0, 1).contiguous().transpose(0, 1),
            o,
            key.transpose(0, 1).contiguous().transpose(0, 1),
            value.transpose(0, 1).contiguous().transpose(0, 1),
            loc,
            attn_logits,
            req_to_token,
            b_req_idx,
            b_seq_len,
            sm_scale,
            logit_cap,
        )

        _run_sdpa_forward_decode(
            q,
            o_ref,
            k_buffer,
            v_buffer,
            req_to_token,
            b_req_idx,
            b_seq_len,
            scaling=sm_scale,
            enable_gqa=enable_gqa,
            logit_cap=logit_cap,
        )

        prec = (
            precision["attention_decode_logit_cap"].get(
                q.dtype, precision["attention_decode"][q.dtype]
            )
            if logit_cap > 0.0
            else precision["attention_decode"][q.dtype]
        )
        cos_sim_threshold = (
            0.98 if (logit_cap > 0.0 and q.dtype == torch.bfloat16) else 0.99
        )
        cos_sim = torch.nn.functional.cosine_similarity(
            o.flatten(), o_ref.flatten(), dim=0
        )
        self.assertGreater(cos_sim.item(), cos_sim_threshold)
        torch.testing.assert_close(o, o_ref, atol=prec, rtol=prec)


class TestRVVDecodeMHA(TestRVVDecodeBase):
    """MHA path tests for RVV FP decode attention."""

    def test_mha(self):
        """Varied shapes cover head-count and seq-len boundary conditions."""
        configs = [
            (1, 1, 1, 64, 64, 128),
            (2, 8, 8, 128, 128, 63),
            (2, 8, 8, 128, 128, 65),
            (2, 8, 8, 128, 128, 129),
            (2, 8, 8, 128, 128, 256),
            (4, 16, 16, 64, 64, 512),
            (2, 17, 17, 127, 127, 128),
            (8, 8, 8, 128, 128, 128),
            # Asymmetric Q/V head dim.
            (2, 8, 8, 128, 64, 128),
            # Non-power-of-2 head dims exercise vector tails.
            (2, 8, 8, 33, 55, 64),
            (2, 8, 8, 80, 80, 64),
        ]
        for B, H_Q, H_KV, D, D_V, seq_len in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B, H_Q=H_Q, H_KV=H_KV, D=D, D_V=D_V, seq_len=seq_len, dtype=dtype
                ):
                    self._run_decode(B, H_Q, H_KV, D, D_V, seq_len=seq_len, dtype=dtype)

    def test_seq_len_one(self):
        """seq_len=1 hits the kv-split boundary: every split is empty or size-1."""
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self._run_decode(1, 1, 1, 64, 64, seq_len=1, dtype=dtype)


class TestRVVDecodeGQA(TestRVVDecodeBase):
    """GQA and MQA path tests for RVV FP decode attention."""

    def test_gqa(self):
        """GQA/MQA shapes verify the H_Q/H_KV ratio reduction is correct."""
        configs = [
            (2, 32, 8, 128, 128, 63),
            (2, 32, 8, 128, 128, 129),
            (2, 32, 8, 128, 128, 256),
            (4, 16, 1, 128, 128, 128),
            (2, 18, 3, 128, 128, 128),
            (2, 21, 7, 128, 128, 128),
            (1, 12, 3, 64, 64, 512),
            # Asymmetric Q/V head dim with GQA.
            (2, 32, 8, 128, 64, 128),
        ]
        for B, H_Q, H_KV, D, D_V, seq_len in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B, H_Q=H_Q, H_KV=H_KV, D=D, D_V=D_V, seq_len=seq_len, dtype=dtype
                ):
                    self._run_decode(B, H_Q, H_KV, D, D_V, seq_len=seq_len, dtype=dtype)


class TestRVVDecodeLogitCap(TestRVVDecodeBase):
    """logit_cap (tanh softcapping) path tests."""

    def test_logit_cap(self):
        """logit_cap > 0 activates the tanh-cap branch; must match manual SDPA."""
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self._run_decode(
                    2, 8, 8, 128, 128, seq_len=64, logit_cap=30.0, dtype=dtype
                )
                self._run_decode(
                    2, 8, 2, 128, 128, seq_len=64, logit_cap=30.0, dtype=dtype
                )


class TestRVVDecodeKvSplits2(TestRVVDecodeBase):
    """Production-default num_kv_splits=2 path for MHA and GQA."""

    def test_kv_splits_2(self):
        """num_kv_splits=2 is the production default; regression guard."""
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self._run_decode(
                    2, 8, 8, 128, 128, seq_len=64, dtype=dtype, num_kv_splits=2
                )
                self._run_decode(
                    2, 8, 2, 128, 128, seq_len=64, dtype=dtype, num_kv_splits=2
                )


@unittest.skipUnless(
    has_sgl_kernel_op("decode_attention_cpu"),
    "decode_attention_cpu not available (non-RISC-V build)",
)
class TestRVVDecodeValidation(CustomTestCase):
    """TORCH_CHECK guard tests for decode_attention_cpu."""

    def test_rejects_mixed_dtypes(self):
        """Mismatched key/query dtypes must be caught early rather than silently corrupting output."""
        q = torch.randn(1, 2, 32, dtype=torch.float16)
        k_buffer = torch.randn(8, 2, 32, dtype=torch.float16)
        v_buffer = torch.randn(8, 2, 32, dtype=torch.float16)
        output = torch.zeros(1, 2, 32, dtype=torch.float16)
        key = torch.randn(1, 2, 32, dtype=torch.bfloat16)
        value = torch.randn(1, 2, 32, dtype=torch.bfloat16)
        loc = torch.tensor([0], dtype=torch.int64)
        attn_logits = torch.empty((1, 2, 1, 33), dtype=torch.float32)
        req_to_token = torch.zeros((1, 4), dtype=torch.int32)
        req_pool_indices = torch.tensor([0], dtype=torch.int64)
        seq_lens = torch.tensor([1], dtype=torch.int64)

        with self.assertRaisesRegex(
            RuntimeError,
            "expect query, key, value, k_buffer, and v_buffer to have the same dtype",
        ):
            torch.ops.sgl_kernel.decode_attention_cpu(
                q,
                k_buffer,
                v_buffer,
                output,
                key,
                value,
                loc,
                attn_logits,
                req_to_token,
                req_pool_indices,
                seq_lens,
                1.0,
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
