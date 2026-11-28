"""Mamba2 ReplaySSM verify and commit match FlashInfer's per-draft snapshot path."""

import types
import unittest

import torch

from sglang.kernels.ops.mamba.mamba2_replayssm_spec import (
    commit_mamba2_replayssm_after_verify,
    mamba2_replayssm_ring_len,
    verify_mamba2_replayssm_spec,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

NUM_LAYERS, NUM_SLOTS = 2, 9
HV, V, G, K = 8, 64, 2, 128
MAX_T = 4


class _Layer:
    def __init__(self, num_reqs: int, T: int):
        dev = "cuda"
        n = num_reqs * T
        xbc = torch.randn(n, HV * V + 2 * G * K, device=dev, dtype=torch.bfloat16)
        x, B, C = torch.split(xbc, [HV * V, G * K, G * K], dim=-1)
        proj = torch.randn(n, HV + 40, device=dev, dtype=torch.bfloat16)
        self.A = -torch.rand(HV, device=dev, dtype=torch.float32) * 4 - 0.1
        dt_bias = torch.randn(HV, device=dev, dtype=torch.bfloat16)
        D = torch.randn(HV, device=dev, dtype=torch.bfloat16)
        shape = (num_reqs, T, HV, V)
        self.args = dict(
            x=x.view(shape),
            dt=proj[:, 7 : 7 + HV][:, :, None].expand(-1, -1, V).view(shape),
            A=self.A[:, None, None].expand(-1, V, K),
            B=B.view(num_reqs, T, G, K),
            C=C.view(num_reqs, T, G, K),
            D=D[:, None].expand(-1, V),
            dt_bias=dt_bias[:, None].expand(-1, V),
        )
        self.shape = shape


def _snapshot_verify(layer, state, slots):
    from flashinfer.mamba import selective_state_update

    num_reqs, T = layer.shape[:2]
    a = layer.args
    out = torch.empty(layer.shape, device="cuda", dtype=torch.bfloat16)
    intermediate = torch.zeros(num_reqs, T, HV, V, K, device="cuda", dtype=state.dtype)
    selective_state_update(
        state,
        a["x"],
        a["dt"],
        a["A"],
        a["B"],
        a["C"],
        a["D"],
        dt_bias=a["dt_bias"],
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
    return out, intermediate


def _assert_rel_close(actual, expected, tol):
    err = (actual.float() - expected.float()).abs().max()
    scale = expected.float().abs().max().clamp_min(1e-6)
    assert err / scale < tol, f"relative error {(err / scale).item():.3e} >= {tol}"


class TestMamba2ReplaySSMSpec(CustomTestCase):
    def test_commit_matches_snapshot_path(self):
        for state_dtype in (torch.float32, torch.bfloat16):
            with self.subTest(state_dtype=state_dtype):
                self._check(state_dtype)

    def _check(self, state_dtype):
        """Two verify/commit rounds, the second with a window shorter than the ring.

        A row whose last step is -1 keeps its state, a padded row (slot -1) writes
        nothing, and a track slot gets the state after its step only when that step
        was accepted.
        """
        torch.manual_seed(0)
        ring_len = mamba2_replayssm_ring_len(MAX_T)
        dev = "cuda"
        slots = torch.tensor([3, 0, 7, 5, -1], device=dev, dtype=torch.int32)
        rounds = [
            (
                MAX_T,
                torch.tensor([3, 0, 2, -1, 1], device=dev, dtype=torch.int32),
                torch.tensor([8, -1, 1, 2, 6], device=dev, dtype=torch.int32),
                torch.tensor([1, 0, 3, 0, 0], device=dev, dtype=torch.int32),
            ),
            (
                2,
                torch.tensor([1, 0, 1, 1, 1], device=dev, dtype=torch.int32),
                torch.tensor([6, 4, -1, 2, 6], device=dev, dtype=torch.int32),
                torch.tensor([0, 1, 0, 1, 0], device=dev, dtype=torch.int32),
            ),
        ]
        temporal = torch.randn(NUM_LAYERS, NUM_SLOTS, HV, V, K, device=dev)
        actual = temporal.to(state_dtype)
        expected = actual.clone()
        spec_state = types.SimpleNamespace(
            temporal=actual,
            replayssm_rawv=torch.zeros(
                NUM_LAYERS, NUM_SLOTS, HV, ring_len, V, device=dev
            ).bfloat16(),
            replayssm_rawk=torch.zeros(
                NUM_LAYERS, NUM_SLOTS, G, ring_len, K, device=dev
            ).bfloat16(),
            replayssm_g=torch.zeros(NUM_LAYERS, NUM_SLOTS, HV, ring_len, device=dev),
            conv=[],
            intermediate_conv_window=[],
        )
        cursor = torch.zeros(NUM_SLOTS, device=dev, dtype=torch.int32)
        tol = 1e-2 if state_dtype == torch.float32 else 2e-2

        for T, last_step, track_slots, track_steps in rounds:
            layers = [_Layer(len(slots), T) for _ in range(NUM_LAYERS)]
            for l, layer in enumerate(layers):
                ref_out, intermediate = _snapshot_verify(layer, expected[l], slots)
                out = torch.empty(layer.shape, device=dev, dtype=torch.bfloat16)
                verify_mamba2_replayssm_spec(
                    state=actual[l],
                    x_ring=spec_state.replayssm_rawv[l],
                    b_ring=spec_state.replayssm_rawk[l],
                    dt_ring=spec_state.replayssm_g[l],
                    cursor=cursor,
                    out=out,
                    state_batch_indices=slots,
                    **layer.args,
                )
                _assert_rel_close(out[:-1], ref_out[:-1], tol)
                for i, slot in enumerate(slots.tolist()):
                    if slot < 0:
                        continue
                    step = last_step[i].item()
                    track = track_steps[i].item()
                    if 0 <= track <= step and track_slots[i] >= 0:
                        expected[l, track_slots[i]] = intermediate[i, track]
                    if step >= 0:
                        expected[l, slot] = intermediate[i, step]
            commit_mamba2_replayssm_after_verify(
                spec_state=spec_state,
                A_by_layer=[layer.A for layer in layers],
                state_batch_indices=slots,
                last_correct_step_indices=last_step,
                mamba_track_indices=track_slots,
                mamba_steps_to_track=track_steps,
            )
            _assert_rel_close(actual, expected, tol)
            live = slots >= 0
            written = set(slots[live & (last_step >= 0)].tolist())
            tracked = live & (track_steps >= 0) & (track_steps <= last_step)
            written |= set(track_slots[tracked & (track_slots >= 0)].tolist())
            unchanged = [s for s in range(NUM_SLOTS) if s not in written]
            torch.testing.assert_close(
                actual[:, unchanged], expected[:, unchanged], rtol=0, atol=0
            )
            expected.copy_(actual)
        self.assertEqual(cursor.count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
