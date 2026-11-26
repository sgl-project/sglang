import unittest
from math import ceil

import torch

from sglang.kernels.ops.attention.fla.beam_replay_layout import (
    GDN_BEAM_REPLAY_WINDOW,
    records_per_state_slot,
)
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.kernels.ops.kvcache.copy_pages import copy_layered_pages
from sglang.srt.beam_search.beam_group import BeamGroup
from sglang.srt.beam_search.recurrent import GdnBeamReplayWindow
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestGdnPackedDecode(CustomTestCase):
    def test_window_replay_matches_dense_branches_across_checkpoints(self):
        torch.manual_seed(12)
        width, heads, dim = 5, 2, 64
        device, dtype = "cuda", torch.bfloat16
        per_slot = records_per_state_slot(
            dim, dim, torch.empty((), dtype=dtype).element_size()
        )
        arena_slots = ceil(GDN_BEAM_REPLAY_WINDOW * width / per_slot)
        slots = torch.arange(1, width + 1, device=device)
        arena = torch.arange(width + 1, width + 1 + arena_slots, device=device)
        shape = (heads, dim, dim)
        state = torch.randn(1 + width + arena_slots, *shape, device=device, dtype=dtype)
        reference = state[1:2].expand(width, *shape).clone()
        control = state[0:1].clone()
        a_log = torch.randn(heads, device=device)
        dt_bias = torch.randn_like(a_log)
        window = GdnBeamReplayWindow(slots, arena, per_slot)
        group = BeamGroup(
            beam_width=width, max_new_tokens=9, read_only_terminal_state=True
        )
        control_index = torch.zeros(1, device=device, dtype=slots.dtype)
        for generated in range(1, group.max_new_tokens):
            group.num_generated = generated
            qkv = torch.randn(width + 1, 3 * heads * dim, device=device, dtype=dtype)
            a = torch.randn(width + 1, heads, device=device, dtype=dtype)
            b = torch.randn_like(a)
            terminal = group.terminal_state_is_read_only()
            read_only = torch.tensor([terminal] * width + [False], device=device)
            headers = window.rows_for_decode(group)
            headers = torch.cat((headers, headers.new_full((1, headers.shape[1]), -1)))
            baseline = torch.cat((reference, control))
            expected = qkv.new_empty(width + 1, 1, heads, dim)
            actual = torch.empty_like(expected)
            for pool, indices, out, metadata in (
                (baseline, slots - 1, expected, None),
                (state, slots, actual, headers),
            ):
                indices = torch.cat(
                    (indices, control_index if metadata is not None else slots[-1:])
                )
                fused_recurrent_gated_delta_rule_packed_decode(
                    qkv,
                    a,
                    b,
                    a_log,
                    dt_bias,
                    dim**-0.5,
                    pool,
                    out,
                    indices,
                    use_qk_l2norm_in_kernel=True,
                    read_only_state_rows=read_only,
                    beam_replay_window_rows=metadata,
                )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(state[0], baseline[-1], rtol=0, atol=0)
            if terminal:
                break
            parents = torch.tensor([2, 0, 2, 4, 0], device=device)
            reference, control = baseline[parents], baseline[-1:]
            group.num_generated += 1
            copy = window.advance(group, parents)
            if copy is not None:
                src, dst = copy
                copy_layered_pages(state.unsqueeze(0), src=src, dst=dst)

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
