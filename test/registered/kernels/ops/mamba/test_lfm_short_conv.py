"""Correctness coverage for the gate-fused LFM2 short convolution."""

import unittest

import torch

from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.kernels.ops.mamba.lfm_short_conv import (
    can_dispatch_fused_lfm_short_conv,
    can_use_fused_lfm_short_conv,
    fused_lfm_short_conv_decode,
    fused_lfm_short_conv_prefill,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-large")

PAD_SLOT_ID = -1
requires_sm90 = unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "requires SM90 (Hopper)",
)


def _inputs(tokens: int, dim: int, slots: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def make(shape):
        return torch.randn(
            shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )

    return (
        make((tokens, dim)),
        make((tokens, dim)),
        make((tokens, dim)),
        make((dim, 3)),
        make((slots, dim, 2)),
    )


def _reference_prefill(
    b,
    c,
    x,
    weight,
    state,
    query_start_loc,
    cache_indices,
    has_initial_state,
    seq_lens,
):
    bx = b * x
    conv = causal_conv1d_fn(
        bx.transpose(0, 1).contiguous(),
        weight,
        None,
        conv_states=state,
        query_start_loc=query_start_loc,
        seq_lens_cpu=seq_lens,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation=None,
    ).transpose(0, 1)
    return c * conv


def _reference_decode(b, c, x, weight, state, cache_indices):
    conv = causal_conv1d_update(
        b * x,
        state,
        weight,
        None,
        activation=None,
        conv_state_indices=cache_indices,
    )
    return c * conv


@requires_sm90
class TestLFMShortConv(CustomTestCase):
    def test_prefill_matches_reference_for_variable_sequences(self):
        seq_lens = [1, 2, 33, 65]
        tokens = sum(seq_lens)
        b, c, x, weight, initial_state = _inputs(tokens, 2051, 3, seed=1)
        query_start_loc = torch.tensor(
            [0, 1, 3, 36, tokens], device="cuda", dtype=torch.int32
        )
        cache_indices = torch.tensor(
            [0, 1, PAD_SLOT_ID, 2], device="cuda", dtype=torch.int32
        )
        has_initial_state = torch.tensor(
            [False, True, True, False], device="cuda", dtype=torch.bool
        )

        expected_state = initial_state.clone()
        expected = _reference_prefill(
            b,
            c,
            x,
            weight,
            expected_state,
            query_start_loc,
            cache_indices,
            has_initial_state,
            seq_lens,
        )
        actual_state = initial_state.clone()
        actual = fused_lfm_short_conv_prefill(
            b,
            c,
            x,
            weight,
            actual_state,
            query_start_loc,
            cache_indices,
            has_initial_state,
            max(seq_lens),
        )

        active_tokens = torch.cat(
            [
                torch.full(
                    (seq_len,),
                    cache_index != PAD_SLOT_ID,
                    device="cuda",
                    dtype=torch.bool,
                )
                for seq_len, cache_index in zip(seq_lens, cache_indices.tolist())
            ]
        )
        torch.testing.assert_close(
            actual[active_tokens].float(),
            expected[active_tokens].float(),
            rtol=0.02,
            atol=0.05,
        )
        self.assertTrue(torch.equal(actual_state, expected_state))

    def test_decode_matches_reference_and_preserves_padded_slots(self):
        tokens, dim, slots = 5, 2051, 4
        b, c, x, weight, initial_state = _inputs(tokens, dim, slots, seed=2)
        cache_indices = torch.tensor(
            [2, PAD_SLOT_ID, 0, 3, PAD_SLOT_ID],
            device="cuda",
            dtype=torch.int32,
        )

        expected_state = initial_state.clone()
        expected = _reference_decode(b, c, x, weight, expected_state, cache_indices)
        actual_state = initial_state.clone()
        actual = fused_lfm_short_conv_decode(
            b, c, x, weight, actual_state, cache_indices
        )

        active_tokens = cache_indices != PAD_SLOT_ID
        torch.testing.assert_close(
            actual[active_tokens].float(),
            expected[active_tokens].float(),
            rtol=0.02,
            atol=0.05,
        )
        self.assertTrue(torch.equal(actual_state, expected_state))
        self.assertTrue(torch.equal(actual_state[1], initial_state[1]))

    def test_decode_cuda_graph_replay(self):
        tokens, dim, slots = 4, 2048, 4
        b, c, x, weight, initial_state = _inputs(tokens, dim, slots, seed=3)
        cache_indices = torch.arange(tokens, device="cuda", dtype=torch.int32)

        # Compile before capture, then prove the captured call reads live inputs and
        # writes the caller-owned state/output on replay.
        fused_lfm_short_conv_decode(
            b, c, x, weight, initial_state.clone(), cache_indices
        )
        graph_state = initial_state.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = fused_lfm_short_conv_decode(
                b, c, x, weight, graph_state, cache_indices
            )

        b.add_(torch.tensor(0.25, device="cuda", dtype=b.dtype))
        x.mul_(torch.tensor(0.75, device="cuda", dtype=x.dtype))
        expected_state = initial_state.clone()
        expected = _reference_decode(b, c, x, weight, expected_state, cache_indices)
        graph_state.copy_(initial_state)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=0.02, atol=0.05
        )
        self.assertTrue(torch.equal(graph_state, expected_state))

    def test_dispatch_is_narrow(self):
        b, c, x, weight, state = _inputs(4, 257, 4, seed=4)
        self.assertTrue(can_use_fused_lfm_short_conv(b, c, x, weight, None, state))
        self.assertFalse(
            can_dispatch_fused_lfm_short_conv(b, c, x, weight, None, state)
        )

        b_2048, c_2048, x_2048, weight_2048, state_2048 = _inputs(4, 2048, 4, seed=5)
        self.assertTrue(
            can_dispatch_fused_lfm_short_conv(
                b_2048, c_2048, x_2048, weight_2048, None, state_2048
            )
        )
        self.assertFalse(
            can_use_fused_lfm_short_conv(
                b, c, x, weight, torch.zeros(257, device="cuda"), state
            )
        )
        self.assertFalse(
            can_use_fused_lfm_short_conv(
                b, c, x, weight[:, :2].contiguous(), None, state
            )
        )
        self.assertFalse(
            can_use_fused_lfm_short_conv(b, c.float(), x, weight, None, state)
        )
        self.assertFalse(
            can_use_fused_lfm_short_conv(
                b.transpose(0, 1),
                c.transpose(0, 1),
                x.transpose(0, 1),
                weight,
                None,
                state,
            )
        )


if __name__ == "__main__":
    unittest.main()
