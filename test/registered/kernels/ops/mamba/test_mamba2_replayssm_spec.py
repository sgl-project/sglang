"""Mamba2 ReplaySSM verify commits the same SSM state as the per-draft snapshots."""

import unittest

import torch

from sglang.kernels.ops.mamba.mamba2_replayssm_spec import (
    fold_mamba2_replayssm_spec,
    record_mamba2_replayssm_spec,
)
from sglang.kernels.ops.mamba.triton_ops.mamba_ssm import selective_state_update
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

NUM_LAYERS, NUM_SLOTS = 2, 9
HV, V, G, K = 8, 64, 2, 128


class _Layer:
    def __init__(self, num_tokens: int):
        dev = "cuda"
        conv_dim = HV * V + 2 * G * K
        self.xbc = torch.randn(num_tokens, conv_dim, device=dev, dtype=torch.bfloat16)
        proj = torch.randn(num_tokens, HV + 40, device=dev, dtype=torch.bfloat16)
        self.dt = proj[:, 7 : 7 + HV]
        self.A = -torch.rand(HV, device=dev, dtype=torch.float32) * 4 - 0.1
        self.dt_bias = torch.randn(HV, device=dev, dtype=torch.float32)
        self.D = torch.randn(HV, device=dev, dtype=torch.float32)
        self.x, self.B, self.C = torch.split(self.xbc, [HV * V, G * K, G * K], dim=-1)


def _verify(layer, state, slots, num_reqs, T, intermediate):
    out = torch.empty(num_reqs, T, HV, V, device="cuda", dtype=torch.bfloat16)
    selective_state_update(
        state,
        layer.x.reshape(num_reqs, T, HV, V),
        layer.dt[:, :, None].expand(-1, -1, V).view(num_reqs, T, HV, V),
        layer.A[:, None, None].expand(-1, V, K),
        layer.B.reshape(num_reqs, T, G, K),
        layer.C.reshape(num_reqs, T, G, K),
        layer.D[:, None].expand(-1, V),
        dt_bias=layer.dt_bias[:, None].expand(-1, V),
        dt_softplus=True,
        state_batch_indices=slots,
        out=out,
        disable_state_update=True,
        intermediate_states_buffer=intermediate,
        cache_steps=T,
        intermediate_state_indices=torch.arange(
            num_reqs, device="cuda", dtype=torch.int32
        ),
    )
    return out


class TestMamba2ReplaySSMSpec(CustomTestCase):
    def test_fold_matches_snapshot_commit(self):
        for state_dtype in (torch.bfloat16, torch.float32):
            for with_track in (False, True):
                with self.subTest(state_dtype=state_dtype, with_track=with_track):
                    self._check(state_dtype, with_track)

    def _check(self, state_dtype, with_track):
        torch.manual_seed(0)
        T, num_reqs = 4, 5
        num_tokens = num_reqs * T
        slots = torch.tensor([3, 0, 7, -1, 5], device="cuda", dtype=torch.int32)
        last_step = torch.tensor([3, 0, 2, 1, -1], device="cuda", dtype=torch.int32)
        track_slots = torch.tensor([8, -1, 1, 2, 6], device="cuda", dtype=torch.int32)
        track_steps = torch.tensor([1, 0, 2, 0, -1], device="cuda", dtype=torch.int32)
        layers = [_Layer(num_tokens) for _ in range(NUM_LAYERS)]
        temporal = torch.randn(
            NUM_LAYERS, NUM_SLOTS, HV, V, K, device="cuda", dtype=state_dtype
        )

        expected = temporal.clone()
        intermediate = torch.zeros(
            NUM_LAYERS, num_reqs + 1, T, HV, V, K, device="cuda", dtype=state_dtype
        )
        ref_outs = [
            _verify(layers[l], expected[l], slots, num_reqs, T, intermediate[l])
            for l in range(NUM_LAYERS)
        ]
        for i in range(num_reqs):
            if slots[i] < 0 or last_step[i] < 0:
                continue
            if with_track and track_slots[i] >= 0 and track_steps[i] >= 0:
                expected[:, track_slots[i]] = intermediate[:, i, track_steps[i]]
            expected[:, slots[i]] = intermediate[:, i, last_step[i]]

        rows, ring_t = num_reqs + 1, T + 2
        x_ring = torch.zeros(NUM_LAYERS, rows, ring_t, HV, V, device="cuda").bfloat16()
        b_ring = torch.zeros(NUM_LAYERS, rows, ring_t, G, K, device="cuda").bfloat16()
        dt_ring = torch.zeros(NUM_LAYERS, rows, ring_t, HV, device="cuda")
        decay_ring = torch.zeros_like(dt_ring)
        actual = temporal.clone()
        for l, layer in enumerate(layers):
            out = _verify(layer, actual[l], slots, num_reqs, T, None)
            torch.testing.assert_close(out, ref_outs[l], rtol=0, atol=0)
            record_mamba2_replayssm_spec(
                x=layer.x,
                B=layer.B,
                dt=layer.dt,
                dt_bias=layer.dt_bias,
                A=layer.A,
                x_ring=x_ring[l],
                b_ring=b_ring[l],
                dt_ring=dt_ring[l],
                decay_ring=decay_ring[l],
                num_steps=T,
            )
        torch.testing.assert_close(actual, temporal, rtol=0, atol=0)
        fold_mamba2_replayssm_spec(
            temporal=actual,
            x_ring=x_ring,
            b_ring=b_ring,
            dt_ring=dt_ring,
            decay_ring=decay_ring,
            state_indices=slots,
            last_correct_step_indices=last_step,
            mamba_track_indices=track_slots if with_track else None,
            mamba_steps_to_track=track_steps if with_track else None,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_src_rows_select_ring_rows(self):
        torch.manual_seed(1)
        T, num_reqs = 3, 2
        layer = _Layer(num_reqs * T)
        x_ring = torch.zeros(1, num_reqs, T, HV, V, device="cuda").bfloat16()
        b_ring = torch.zeros(1, num_reqs, T, G, K, device="cuda").bfloat16()
        dt_ring = torch.zeros(1, num_reqs, T, HV, device="cuda")
        decay_ring = torch.zeros_like(dt_ring)
        record_mamba2_replayssm_spec(
            x=layer.x,
            B=layer.B,
            dt=layer.dt,
            dt_bias=layer.dt_bias,
            A=layer.A,
            x_ring=x_ring[0],
            b_ring=b_ring[0],
            dt_ring=dt_ring[0],
            decay_ring=decay_ring[0],
            num_steps=T,
        )
        temporal = torch.randn(1, 4, HV, V, K, device="cuda")
        slots = torch.tensor([2, 1], device="cuda", dtype=torch.int32)
        last_step = torch.tensor([2, 1], device="cuda", dtype=torch.int32)
        swapped = temporal.clone()
        fold_mamba2_replayssm_spec(
            temporal=swapped,
            x_ring=x_ring.flip(1).contiguous(),
            b_ring=b_ring.flip(1).contiguous(),
            dt_ring=dt_ring.flip(1).contiguous(),
            decay_ring=decay_ring.flip(1).contiguous(),
            state_indices=slots,
            last_correct_step_indices=last_step,
            src_rows=torch.tensor([1, 0], device="cuda", dtype=torch.int32),
        )
        direct = temporal.clone()
        fold_mamba2_replayssm_spec(
            temporal=direct,
            x_ring=x_ring,
            b_ring=b_ring,
            dt_ring=dt_ring,
            decay_ring=decay_ring,
            state_indices=slots,
            last_correct_step_indices=last_step,
        )
        torch.testing.assert_close(swapped, direct, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
