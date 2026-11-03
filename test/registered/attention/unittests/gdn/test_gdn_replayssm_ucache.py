"""Deferred-fold GDN ReplaySSM verify (--enable-gdn-replayssm-ucache) against an
fp32 token-by-token recurrence.

Drives SGLang's glue (cursor gather, FlashInfer u-cache verify, cursor commit) for
many verify steps with random accept counts, padded rows and folds. A wrong cursor
commit, a gather of the wrong slot or a fold that drops records shows up as drift
from the fp32 truth, which advances each slot's state by exactly its accepted tokens.
"""

import unittest

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
    (10, 0),
    (10, 7),
):
    pytest.skip(
        "FlashInfer u-cache verify requires SM100/SM107", allow_module_level=True
    )
pytest.importorskip("flashinfer.gdn_kernels.gdn_replay_mtp_umma")

from sglang.srt.layers.attention.linear.gdn_replayssm_ucache import (  # noqa: E402
    GdnUcacheVerifier,
    allocate_gdn_ucache,
    commit_gdn_ucache_cursors,
)

H, HV, K, V = 4, 12, 128, 128
NUM_DRAFT_TOKENS = 4
NUM_SLOTS = 8
# Real rows on slots 3, 5, 1, 6; slot 0 (the reserved padding slot) and -1 are padding.
CACHE_INDICES = [3, 0, 5, -1, 1, 6]
NUM_STEPS = 32
# FlashInfer's bounds against a ring-free fp32 recurrence: one verify from the
# initial fp16 state, and over the whole horizon (2 * 8e-3).
FIRST_STEP_TOL = 8e-3
DRIFT_TOL = 1.6e-2
# Relative to the largest reference output (about 0.15 here), so a few-percent
# gain or gate error fails; measured 0.27% on B200, bound about 4x that.
REL_TOL = 1e-2
# An in-place weight update at this step must reach the verify's gate parameters.
DT_BIAS_UPDATE_STEP = NUM_STEPS // 2


def _fp32_window(state, q, k, v, a, b, A_log, dt_bias):
    """Runs one row's T tokens from ``state`` [HV, V, K]; returns the per-token
    outputs [T, HV, V] and the states after each token [T, HV, V, K]."""
    group = HV // H
    q_hat = F.normalize(q.float(), dim=-1).repeat_interleave(group, dim=1) * K**-0.5
    k_hat = F.normalize(k.float(), dim=-1).repeat_interleave(group, dim=1)
    log_decay = -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float())
    beta = torch.sigmoid(b.float())
    outputs, states = [], []
    for t in range(q.shape[0]):
        state = state * torch.exp(log_decay[t])[:, None, None]
        pred = torch.einsum("hvk,hk->hv", state, k_hat[t])
        u = (v[t].float() - pred) * beta[t][:, None]
        state = state + u[:, :, None] * k_hat[t][:, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", state, q_hat[t]))
        states.append(state)
    return torch.stack(outputs), torch.stack(states)


class TestGdnReplaySsmUcache(CustomTestCase):
    def test_multistep_random_accept_matches_fp32_recurrence(self):
        device = torch.device("cuda")
        gen = torch.Generator(device=device).manual_seed(0)

        def randn(*shape, scale=1.0):
            return torch.randn(*shape, generator=gen, device=device) * scale

        T = NUM_DRAFT_TOKENS
        buffers = allocate_gdn_ucache(
            num_layers=1,
            num_slots=NUM_SLOTS,
            num_k_heads=H,
            num_v_heads=HV,
            head_k_dim=K,
            head_v_dim=V,
            max_num_draft_tokens=T,
            device="cuda",
        )
        # Poisoned rings: a record read outside the live window turns into NaN.
        for ring in (buffers.k_ring, buffers.u_ring, buffers.g_ring):
            ring.fill_(float("nan"))
        state = (randn(NUM_SLOTS, HV, V, K, scale=0.5)).to(torch.float16)
        truth = state.float().clone()
        A_log = torch.full((HV,), -3.0, device=device) + randn(HV, scale=0.3).abs()
        dt_bias = randn(HV, scale=0.5).to(torch.bfloat16)
        cache_indices = torch.tensor(CACHE_INDICES, dtype=torch.int32, device=device)
        rows = [r for r, slot in enumerate(CACHE_INDICES) if slot > 0]
        batch = len(CACHE_INDICES)
        verifier = GdnUcacheVerifier(device=device, first_layer_id=0)

        worst = worst_first = ref_max = 0.0
        folds = {r: 0 for r in rows}
        for step in range(NUM_STEPS):
            if step == DT_BIAS_UPDATE_STEP:
                dt_bias.copy_(randn(HV, scale=0.5))
            q, k = randn(batch * T, H, K), randn(batch * T, H, K)
            v = randn(batch * T, HV, V, scale=0.5)
            a, b = randn(batch * T, HV, scale=0.5), randn(batch * T, HV)
            q, k, v, a, b = (x.to(torch.bfloat16) for x in (q, k, v, a, b))
            hist_before = buffers.hist_len.clone()
            output = verifier.target_verify(
                layer_id=0,
                query=q,
                key=k,
                value=v,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                state=state,
                k_ring=buffers.k_ring[0],
                u_ring=buffers.u_ring[0],
                g_ring=buffers.g_ring[0],
                buffers=buffers,
                cache_indices=cache_indices,
                num_draft_tokens=T,
            ).view(batch, T, HV, V)
            num_accept = torch.randint(
                1, T + 1, (batch,), generator=gen, device=device, dtype=torch.int32
            )
            commit_gdn_ucache_cursors(
                buffers=buffers, slots=cache_indices, num_accept_tokens=num_accept
            )
            for r in range(batch):
                if r not in rows:
                    self.assertEqual(output[r].abs().max().item(), 0.0)
                    continue
                slot = CACHE_INDICES[r]
                tokens = slice(r * T, (r + 1) * T)
                y_ref, states = _fp32_window(
                    truth[slot],
                    q[tokens],
                    k[tokens],
                    v[tokens],
                    a[tokens],
                    b[tokens],
                    A_log,
                    dt_bias,
                )
                err = (output[r].float() - y_ref).abs().max().item()
                worst = max(worst, err)
                if step == 0:
                    worst_first = max(worst_first, err)
                ref_max = max(ref_max, y_ref.abs().max().item())
                truth[slot] = states[int(num_accept[r]) - 1]
                folds[r] += int(hist_before[slot] >= buffers.flush_min)

        torch.cuda.synchronize()
        report = (
            f"|y - fp32 truth|: first step {worst_first:.2e}, horizon {worst:.2e}, "
            f"relative {worst / ref_max:.2%} of max |y| {ref_max:.3f}"
        )
        print(report)
        self.assertLess(worst_first, FIRST_STEP_TOL, report)
        self.assertLess(worst, DRIFT_TOL, report)
        self.assertLess(worst / ref_max, REL_TOL, report)
        self.assertTrue(all(count > 0 for count in folds.values()), folds)
        # Padding slot 0 is never verified or committed.
        self.assertEqual(int(buffers.hist_len[0]), 0)


if __name__ == "__main__":
    unittest.main()
