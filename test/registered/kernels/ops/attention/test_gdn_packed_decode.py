import unittest

import torch

from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestGdnPackedDecode(CustomTestCase):
    def test_beam_replay_matches_materialized_bf16_state(self):
        torch.manual_seed(7)
        device = "cuda"
        rows, heads, value_heads, dim = 5, 2, 4, 64
        dtype = torch.bfloat16
        state = torch.randn(rows, value_heads, dim, dim, device=device, dtype=dtype)
        initial = state.clone()
        a_log = torch.randn(value_heads, device=device)
        dt_bias = torch.randn_like(a_log)
        stages = [
            (
                torch.randn(
                    rows, (2 * heads + value_heads) * dim, device=device, dtype=dtype
                ),
                torch.randn(rows, value_heads, device=device, dtype=dtype),
                torch.randn(rows, value_heads, device=device, dtype=dtype),
            )
            for _ in range(2)
        ]
        children = torch.arange(rows, device=device)
        parents = torch.tensor([1, 0, 1, 3, 4], device=device)
        groups = torch.tensor([0, 0, 0, 0, -1], device=device)
        slots = torch.tensor([0, 1], device=device)
        terminal = groups >= 0

        def decode(stage, indices, state, read_only=None, replay=False):
            qkv, a, b = stages[stage]
            out = qkv.new_empty((rows, 1, value_heads, dim))
            fused_recurrent_gated_delta_rule_packed_decode(
                qkv,
                a,
                b,
                a_log,
                dt_bias,
                dim**-0.5,
                state,
                out,
                indices,
                use_qk_l2norm_in_kernel=True,
                read_only_state_rows=read_only,
                beam_replay_group_ids=groups if replay else None,
                beam_replay_slots=slots if replay else None,
            )
            return out

        state[1:-1] = initial[0]
        expected_first = decode(0, children, state)
        expected_final = decode(1, parents, state, terminal)
        aliased = initial.clone()
        actual_first = decode(0, children, aliased, replay=True)
        actual_final = decode(1, parents, aliased, terminal, replay=True)
        torch.testing.assert_close(actual_first, expected_first, rtol=0, atol=0)
        torch.testing.assert_close(actual_final, expected_final, rtol=0, atol=0)
        torch.testing.assert_close(aliased[0], initial[0], rtol=0, atol=0)
        torch.testing.assert_close(aliased[-1], state[-1], rtol=0, atol=0)

    def test_read_only_siblings_match_private_slots(self):
        device = "cuda"
        torch.manual_seed(0)
        rows, heads, value_heads, key_dim, value_dim = 3, 2, 4, 16, 32
        qkv = torch.randn(
            rows,
            2 * heads * key_dim + value_heads * value_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        a = torch.randn(rows, value_heads, device=device, dtype=qkv.dtype)
        b = torch.randn_like(a)
        a_log = torch.randn(value_heads, device=device)
        dt_bias = torch.randn_like(a_log)
        initial = torch.randn(rows, value_heads, value_dim, key_dim, device=device)
        initial[1] = initial[0]

        def decode(indices, read_only=None):
            state = initial.clone()
            out = qkv.new_empty((rows, 1, value_heads, value_dim))
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=qkv,
                a=a,
                b=b,
                A_log=a_log,
                dt_bias=dt_bias,
                scale=key_dim**-0.5,
                initial_state=state,
                out=out,
                ssm_state_indices=torch.tensor(indices, device=device),
                use_qk_l2norm_in_kernel=True,
                read_only_state_rows=read_only,
            )
            return out, state

        expected, writable = decode([0, 1, 2])
        actual, aliased = decode(
            [0, 0, 2], torch.tensor([True, True, False], device=device)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(aliased[:2], initial[:2], rtol=0, atol=0)
        torch.testing.assert_close(aliased[2], writable[2], rtol=0, atol=0)
        self.assertFalse(torch.equal(aliased[2], initial[2]))

    def test_wide_batch_matches_split_launch(self):
        torch.manual_seed(0)
        device = "cuda"
        rows, heads, value_heads, key_dim, value_dim = 5120, 4, 16, 16, 48
        qkv = torch.randn(
            rows,
            2 * heads * key_dim + value_heads * value_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        a = torch.randn(rows, value_heads, device=device, dtype=qkv.dtype)
        b = torch.randn_like(a)
        a_log = torch.randn(value_heads, device=device)
        dt_bias = torch.randn_like(a_log)
        slots = torch.randperm(rows, device=device)
        initial_state = torch.randn(
            rows, value_heads, value_dim, key_dim, device=device
        )

        def decode(chunks):
            state = initial_state.clone()
            out = qkv.new_empty((rows, 1, value_heads, value_dim))
            for chunk in chunks:
                fused_recurrent_gated_delta_rule_packed_decode(
                    mixed_qkv=qkv[chunk],
                    a=a[chunk],
                    b=b[chunk],
                    A_log=a_log,
                    dt_bias=dt_bias,
                    scale=key_dim**-0.5,
                    initial_state=state,
                    out=out[chunk],
                    ssm_state_indices=slots[chunk],
                    use_qk_l2norm_in_kernel=True,
                )
            return out, state

        expected = decode((slice(2560), slice(2560, None)))
        actual = decode((slice(None),))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
