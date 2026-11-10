"""Deferred-fold GDN ReplaySSM verify (--enable-gdn-replayssm-ucache) against an
fp32 token-by-token recurrence.

Drives SGLang's glue (cursor gather, FlashInfer u-cache verify, cursor commit) for
many verify steps with random accept counts, padded rows and folds. A wrong cursor
commit, a gather of the wrong slot or a fold that drops records shows up as drift
from the fp32 truth, which advances each slot's state by exactly its accepted tokens.
"""

import os

# As in serving (the server-args hook): FlashInfer fixes the materialize kernel's
# state dtype at the first flashinfer.gdn_kernels import; the verify must not care.
os.environ.setdefault("GDN_UCACHE_STATE_DTYPE", "fp16")

import unittest  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
    (10, 0),
    (10, 3),
    (10, 7),
):
    pytest.skip(
        "FlashInfer u-cache verify requires SM100/SM103/SM107", allow_module_level=True
    )
pytest.importorskip("flashinfer.gdn_kernels.gdn_replay_mtp_umma")

from sglang.srt.layers.attention.linear.gdn_replayssm_ucache import (  # noqa: E402
    GdnUcacheVerifier,
    allocate_gdn_ucache,
    commit_gdn_ucache_cursors,
    reset_gdn_ucache_cursors,
)
from sglang.srt.runtime_context import get_context  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402


def setUpModule():
    # GdnUcacheVerifier reads the exec config when it starts; publish a minimal one.
    get_context().set_server_args(ServerArgs(model_path="dummy"))


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
        self._check_random_accept(max_num_draft_tokens=NUM_DRAFT_TOKENS, widths=[4])

    def test_adaptive_widths_share_one_pool(self):
        """Adaptive tiers verify 1..8 tokens per step against one pool whose fold
        threshold comes from the widest tier; every width, including steps that
        switch width mid-window, must match the recurrence."""
        self._check_random_accept(max_num_draft_tokens=8, widths=[1, 2, 4, 6, 8])

    def _check_random_accept(self, *, max_num_draft_tokens, widths):
        device = torch.device("cuda")
        gen = torch.Generator(device=device).manual_seed(0)

        def randn(*shape, scale=1.0):
            return torch.randn(*shape, generator=gen, device=device) * scale

        buffers = allocate_gdn_ucache(
            num_layers=1,
            num_slots=NUM_SLOTS,
            num_k_heads=H,
            num_v_heads=HV,
            head_k_dim=K,
            head_v_dim=V,
            max_num_draft_tokens=max_num_draft_tokens,
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
        # One verifier per width, as each adaptive tier builds its own backend.
        verifiers = {
            T: GdnUcacheVerifier(device=device, first_layer_id=0) for T in widths
        }

        worst = worst_first = ref_max = 0.0
        folds = {r: 0 for r in rows}
        for step in range(NUM_STEPS):
            T = widths[0]
            if len(widths) > 1:
                pick = torch.randint(len(widths), (1,), generator=gen, device=device)
                T = widths[int(pick)]
            if step == DT_BIAS_UPDATE_STEP:
                dt_bias.copy_(randn(HV, scale=0.5))
            # q/k/v are views into one packed [q | k | v] row per token, as in
            # serving: the verifier copies them out with a single split launch.
            qkv = torch.cat(
                (
                    randn(batch * T, H * K),
                    randn(batch * T, H * K),
                    randn(batch * T, HV * V, scale=0.5),
                ),
                dim=1,
            ).to(torch.bfloat16)
            q, k, v = torch.split(qkv, [H * K, H * K, HV * V], dim=1)
            q, k, v = (
                q.view(batch * T, H, K),
                k.view(batch * T, H, K),
                v.view(batch * T, HV, V),
            )
            a, b = randn(batch * T, HV, scale=0.5), randn(batch * T, HV)
            a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
            hist_before = buffers.hist_len.clone()
            output = (
                verifiers[T]
                .target_verify(
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
                )
                .view(batch, T, HV, V)
            )
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

    def test_extend_reset_empties_recycled_slot_windows(self):
        """A target extend into a slot whose previous owner left a non-empty window
        (a recycled, copy-on-write or loaded-back slot) must verify from the new
        checkpoint alone; a stale window would replay the old owner's records."""
        device = torch.device("cuda")
        gen = torch.Generator(device=device).manual_seed(1)

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
        # Finite garbage records: a stale window shifts the output instead of NaN.
        for ring in (buffers.k_ring, buffers.u_ring):
            ring.copy_(randn(*ring.shape).to(ring.dtype))
        buffers.g_ring.copy_(-randn(*buffers.g_ring.shape).abs() * 0.1)
        # Previous owners' cursors, including on the padding slot 0.
        buffers.hist_len.copy_(torch.arange(NUM_SLOTS, device=device) % 13 + 1)
        buffers.cache_base.copy_(torch.arange(NUM_SLOTS, device=device) * 5 % 32)
        hist_before = buffers.hist_len.clone()
        base_before = buffers.cache_base.clone()

        # The extend batch: slots 3 and 5 plus padding rows (0 and -1).
        extend_slots = torch.tensor([3, 0, 5, -1], dtype=torch.int32, device=device)
        reset_gdn_ucache_cursors(buffers=buffers, slots=extend_slots)
        reset = [3, 5]
        for slot in range(NUM_SLOTS):
            if slot in reset:
                self.assertEqual(int(buffers.hist_len[slot]), 0, slot)
                self.assertEqual(int(buffers.cache_base[slot]), 0, slot)
            else:
                self.assertEqual(int(buffers.hist_len[slot]), int(hist_before[slot]))
                self.assertEqual(int(buffers.cache_base[slot]), int(base_before[slot]))

        # The extend wrote a fresh checkpoint; the first verify must read only it.
        state = randn(NUM_SLOTS, HV, V, K, scale=0.5).to(torch.float16)
        rows = torch.tensor(reset, dtype=torch.int32, device=device)
        batch = len(reset)
        q = randn(batch * T, H, K).to(torch.bfloat16)
        k = randn(batch * T, H, K).to(torch.bfloat16)
        v = randn(batch * T, HV, V, scale=0.5).to(torch.bfloat16)
        a = randn(batch * T, HV, scale=0.5).to(torch.bfloat16)
        b = randn(batch * T, HV).to(torch.bfloat16)
        A_log = torch.full((HV,), -3.0, device=device)
        dt_bias = randn(HV, scale=0.5).to(torch.bfloat16)
        output = (
            GdnUcacheVerifier(device=device, first_layer_id=0)
            .target_verify(
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
                cache_indices=rows,
                num_draft_tokens=T,
            )
            .view(batch, T, HV, V)
        )
        for r, slot in enumerate(reset):
            tokens = slice(r * T, (r + 1) * T)
            y_ref, _ = _fp32_window(
                state[slot].float(),
                q[tokens],
                k[tokens],
                v[tokens],
                a[tokens],
                b[tokens],
                A_log,
                dt_bias,
            )
            err = (output[r].float() - y_ref).abs().max().item()
            self.assertLess(err, FIRST_STEP_TOL, f"slot {slot}: {err:.2e}")


if __name__ == "__main__":
    unittest.main()
