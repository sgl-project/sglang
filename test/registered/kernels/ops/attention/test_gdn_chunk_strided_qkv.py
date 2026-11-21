"""chunk_gated_delta_rule on q / k / v split views of a packed projection.

The forward reads strided q / k / v without contiguous copies (q / k through
l2norm_fwd_strided, v in place in recompute_w_u_fwd) and skips the output zero
fill when no row is padding. Both must match the contiguous-copy path bit for
bit, on the output and on every slot of the state pool.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.fla import chunk as fla_chunk
from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.kernels.ops.attention.fla.l2norm import (
    l2norm_fwd,
    l2norm_fwd_strided,
    l2norm_fwd_strided_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def _split_views(num_tokens, num_heads, num_v_heads, head_dim, extra_cols, seed):
    """q / k / v as GDNAttnBackend.forward_extend passes them: views of a
    [T, q | k | v | extra] projection."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    qk_dim, v_dim = num_heads * head_dim, num_v_heads * head_dim
    mixed = torch.randn(
        num_tokens, 2 * qk_dim + v_dim + extra_cols, generator=g, device="cuda"
    ).to(torch.bfloat16)
    q, k, v = torch.split(mixed[:, : 2 * qk_dim + v_dim], [qk_dim, qk_dim, v_dim], -1)
    return (
        q.view(1, num_tokens, num_heads, head_dim),
        k.view(1, num_tokens, num_heads, head_dim),
        v.view(1, num_tokens, num_v_heads, head_dim),
    )


def _nan_empty_like(t, **kwargs):
    return torch.full_like(t, float("nan"), **kwargs)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestGDNChunkStridedQKV(CustomTestCase):
    def test_l2norm_fwd_strided_matches_contiguous_copy(self):
        for num_heads in (1, 2, 4, 8, 12, 16, 32):
            for head_dim in (64, 96, 128):
                for num_tokens in (1, 33, 1000):
                    with self.subTest(H=num_heads, D=head_dim, T=num_tokens):
                        _, k, _ = _split_views(
                            num_tokens, num_heads, 2 * num_heads, head_dim, 32, seed=1
                        )
                        self.assertTrue(l2norm_fwd_strided_supported(k))
                        got = l2norm_fwd_strided(k)
                        self.assertTrue(got.is_contiguous())
                        self.assertTrue(_same_bits(got, l2norm_fwd(k.contiguous())))

    def _inputs(self, lens, num_heads, num_v_heads, extra_cols, state_dtype, seed):
        num_tokens = sum(lens)
        q, k, v = _split_views(
            num_tokens, num_heads, num_v_heads, 128, extra_cols, seed=seed
        )
        g0 = torch.Generator(device="cuda").manual_seed(seed + 1)
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, num_tokens, num_v_heads, generator=g0, device="cuda")
        ).to(torch.bfloat16)
        beta = torch.sigmoid(
            torch.randn(1, num_tokens, num_v_heads, generator=g0, device="cuda")
        ).to(torch.bfloat16)
        num_slots = len(lens) + 5
        pool = (
            torch.randn(num_slots, num_v_heads, 128, 128, generator=g0, device="cuda")
            * 0.05
        ).to(state_dtype)
        indices = torch.randperm(num_slots, generator=g0, device="cuda")[: len(lens)]
        cu_seqlens = torch.tensor(
            [sum(lens[:i]) for i in range(len(lens) + 1)],
            dtype=torch.int32,
            device="cuda",
        )
        return q, k, v, g, beta, pool, indices.to(torch.int32), cu_seqlens

    def _run(self, q, k, v, g, beta, pool, indices, cu_seqlens, **kwargs):
        state = pool.clone()
        o, _, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=state,
            initial_state_indices=indices,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            **kwargs,
        )
        torch.cuda.synchronize()
        return o, state

    def test_split_views_match_contiguous_copies(self):
        cases = [
            # (lens, key heads, value heads, columns after v in the projection)
            ([700, 333, 1100, 1], 16, 32, 0),
            ([64, 129, 2000], 16, 48, 6144),
            ([5, 300, 77], 4, 8, 1024),
            ([1, 1, 1, 1], 2, 4, 16),
        ]
        for lens, num_heads, num_v_heads, extra_cols in cases:
            for state_dtype in (torch.float32, torch.bfloat16):
                for no_padding_rows in (False, True):
                    with self.subTest(
                        lens=lens,
                        H=num_heads,
                        HV=num_v_heads,
                        state_dtype=state_dtype,
                        no_padding_rows=no_padding_rows,
                    ):
                        q, k, v, *rest = self._inputs(
                            lens, num_heads, num_v_heads, extra_cols, state_dtype, 3
                        )
                        ref_o, ref_state = self._run(
                            q.contiguous(), k.contiguous(), v.contiguous(), *rest
                        )
                        with (
                            patch.object(
                                fla_chunk,
                                "_chunk_gated_delta_rule_strided_qkv",
                                wraps=fla_chunk._chunk_gated_delta_rule_strided_qkv,
                            ) as strided,
                            # Rows the kernel does not store would show as NaN.
                            patch.object(
                                torch, "empty_like", side_effect=_nan_empty_like
                            ),
                        ):
                            o, state = self._run(
                                q, k, v, *rest, no_padding_rows=no_padding_rows
                            )
                        strided.assert_called_once()
                        self.assertTrue(_same_bits(o, ref_o), "output differs")
                        self.assertTrue(_same_bits(state, ref_state), "state differs")

    def test_split_views_write_mapped_output(self):
        """Strided q / k / v with a mapped out / output_indices write the same
        bits into the caller's buffer as contiguous copies, and leave its
        unmapped rows alone."""
        lens, num_heads, num_v_heads = [300, 77], 4, 8
        q, k, v, *rest = self._inputs(
            lens, num_heads, num_v_heads, 1024, torch.float32, 5
        )
        num_tokens = sum(lens)
        gen = torch.Generator(device="cuda").manual_seed(9)
        output_indices = torch.randperm(num_tokens + 7, generator=gen, device="cuda")[
            :num_tokens
        ]
        sentinel = torch.full(
            (1, num_tokens + 7, num_v_heads, 128), 3.0, dtype=v.dtype, device="cuda"
        )
        ref_out = sentinel.clone()
        _, ref_state = self._run(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            *rest,
            out=ref_out,
            output_indices=ref_out.new_tensor(output_indices, dtype=torch.int64),
        )
        out = sentinel.clone()
        with patch.object(
            fla_chunk,
            "_chunk_gated_delta_rule_strided_qkv",
            wraps=fla_chunk._chunk_gated_delta_rule_strided_qkv,
        ) as strided:
            _, state = self._run(
                q,
                k,
                v,
                *rest,
                out=out,
                output_indices=out.new_tensor(output_indices, dtype=torch.int64),
            )
        strided.assert_called_once()
        self.assertTrue(_same_bits(out, ref_out), "mapped output differs")
        self.assertTrue(_same_bits(state, ref_state), "state differs")
        unmapped = torch.ones(num_tokens + 7, dtype=torch.bool, device="cuda")
        unmapped[output_indices] = False
        self.assertTrue((out[0, unmapped] == 3.0).all())

    def test_padding_rows_stay_zero(self):
        lens, num_padding = [100, 37], 40
        q, k, v, g, beta, pool, indices, cu_seqlens = self._inputs(
            lens + [num_padding], 16, 32, 0, torch.float32, 5
        )
        cu_seqlens, indices = cu_seqlens[:-1], indices[:-1]
        ref_o, ref_state = self._run(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g,
            beta,
            pool,
            indices,
            cu_seqlens,
        )
        with (
            patch.object(
                fla_chunk,
                "_chunk_gated_delta_rule_strided_qkv",
                wraps=fla_chunk._chunk_gated_delta_rule_strided_qkv,
            ) as strided,
            patch.object(torch, "empty_like", side_effect=_nan_empty_like),
        ):
            o, state = self._run(q, k, v, g, beta, pool, indices, cu_seqlens)
        strided.assert_called_once()
        self.assertTrue(_same_bits(o, ref_o))
        self.assertTrue(_same_bits(state, ref_state))
        self.assertTrue(bool((o[:, sum(lens) :] == 0).all()))

    def test_unaligned_views_take_the_copy_path(self):
        # A row stride that is not a multiple of 16 elements is not read in place.
        q, k, v, g, beta, pool, indices, cu_seqlens = self._inputs(
            [50, 70], 16, 32, 8, torch.float32, 7
        )
        self.assertFalse(l2norm_fwd_strided_supported(q))
        ref_o, ref_state = self._run(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g,
            beta,
            pool,
            indices,
            cu_seqlens,
        )
        with patch.object(fla_chunk, "_chunk_gated_delta_rule_strided_qkv") as strided:
            o, state = self._run(q, k, v, g, beta, pool, indices, cu_seqlens)
        strided.assert_not_called()
        self.assertTrue(_same_bits(o, ref_o))
        self.assertTrue(_same_bits(state, ref_state))


if __name__ == "__main__":
    unittest.main()
