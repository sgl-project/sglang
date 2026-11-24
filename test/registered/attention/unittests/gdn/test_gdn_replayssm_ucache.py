"""Deferred-fold GDN ReplaySSM verify (--enable-gdn-replayssm-ucache) against an
fp32 token-by-token recurrence.

Drives SGLang's glue (cursor gather, FlashInfer u-cache verify, cursor commit) for
many verify steps with random accept counts, padded rows and folds. A wrong cursor
commit, a gather of the wrong slot or a fold that drops records shows up as drift
from the fp32 truth, which advances each slot's state by exactly its accepted tokens.
On the archs of the vendored flush (SM103, SM107) the launch options must not change a
bit, and the fused gated norm must equal the stock one.
"""

import os

# As in serving (the server-args hook): FlashInfer fixes the materialize kernel's
# state dtype at the first flashinfer.gdn_kernels import; the verify must not care.
os.environ.setdefault("GDN_UCACHE_STATE_DTYPE", "fp16")

import unittest  # noqa: E402
from contextlib import ExitStack  # noqa: E402

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
# SM103 / SM107 run the vendored flush build; SM100 FlashInfer's.
VENDORED = torch.cuda.get_device_capability() in ((10, 3), (10, 7))
if not VENDORED:
    pytest.importorskip("flashinfer.gdn_kernels.gdn_replay_mtp_umma")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.attention.linear.gdn_replayssm_ucache import (  # noqa: E402
    GdnUcacheVerifier,
    allocate_gdn_ucache,
    commit_gdn_ucache_cursors,
    gdn_ucache_fused_norm_applies,
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

    @unittest.skipUnless(VENDORED, "the fp16 key ring is a vendored-kernel build")
    def test_fp16_key_ring_matches_fp32_recurrence(self):
        with envs.SGLANG_ENABLE_GDN_UCACHE_FP16_KEY_RING.override(True):
            self._check_random_accept(
                max_num_draft_tokens=NUM_DRAFT_TOKENS,
                widths=[4],
                key_ring_dtype=torch.float16,
            )

    def test_adaptive_widths_share_one_pool(self):
        """Adaptive tiers verify 1..8 tokens per step against one pool whose fold
        threshold comes from the widest tier; every width, including steps that
        switch width mid-window, must match the recurrence."""
        self._check_random_accept(max_num_draft_tokens=8, widths=[1, 2, 4, 6, 8])

    def _check_random_accept(
        self, *, max_num_draft_tokens, widths, key_ring_dtype=torch.bfloat16
    ):
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
        self.assertEqual(buffers.k_ring.dtype, key_ring_dtype)
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


# Launch options of the vendored flush (and request-row rings): none may change a bit.
_OPTION_CASES = {
    "strided_qkv": dict(strided=True),
    "pdl": dict(env=("PDL",)),
    "flush_first": dict(env=("FLUSH_FIRST",)),
    "request_rings": dict(request_rows=True),
    "served": dict(strided=True, env=("PDL", "FLUSH_FIRST"), request_rows=True),
}
OPT_BATCH = 24
OPT_SLOTS = 40
OPT_STEPS = 20


@unittest.skipUnless(VENDORED, "launch options of the vendored SM103 / SM107 flush")
class TestGdnUcacheVendoredOptions(CustomTestCase):
    def _run(self, *, strided=False, env=(), request_rows=False, fused_norm=None):
        """OPT_STEPS verifies of OPT_BATCH rows (two of them padded) with random accepts;
        returns the outputs, the state pool and each slot's rings."""
        device = torch.device("cuda")
        gen = torch.Generator(device=device).manual_seed(7)

        def randn(*shape, scale=1.0):
            return torch.randn(*shape, generator=gen, device=device) * scale

        T = NUM_DRAFT_TOKENS
        slots = torch.randperm(OPT_SLOTS - 1, generator=gen, device=device)[:OPT_BATCH] + 1
        slots[3] = 0
        slots[11] = -1
        slots = slots.int()
        # Request rows: another permutation, so a ring indexed by the wrong one shows.
        req_rows = (torch.randperm(OPT_BATCH, generator=gen, device=device) + 1).int()
        state = randn(OPT_SLOTS, HV, V, K, scale=0.5).to(torch.float16)
        A_log = torch.full((HV,), -3.0, device=device) + randn(HV, scale=0.3).abs()
        dt_bias = randn(HV, scale=0.5).to(torch.bfloat16)
        accepts = torch.randint(1, T + 1, (OPT_STEPS, OPT_BATCH), generator=gen, device=device)
        inputs = []
        for _ in range(OPT_STEPS):
            qkv = torch.cat(
                (
                    randn(OPT_BATCH * T, H * K),
                    randn(OPT_BATCH * T, H * K),
                    randn(OPT_BATCH * T, HV * V, scale=0.5),
                ),
                dim=1,
            ).to(torch.bfloat16)
            gates = torch.cat(
                (randn(OPT_BATCH * T, HV), randn(OPT_BATCH * T, HV, scale=0.5)), dim=1
            ).to(torch.bfloat16)
            z = randn(OPT_BATCH * T, HV, V).to(torch.bfloat16)
            inputs.append((qkv, gates, z))
        with ExitStack() as stack:
            for name in env:
                stack.enter_context(
                    getattr(envs, f"SGLANG_ENABLE_GDN_UCACHE_{name}").override(True)
                )
            if strided:
                stack.enter_context(envs.SGLANG_ENABLE_GDN_UCACHE_STRIDED_QKV.override(True))
            buffers = allocate_gdn_ucache(
                num_layers=1,
                num_slots=OPT_BATCH + 1 if request_rows else OPT_SLOTS,
                num_k_heads=H,
                num_v_heads=HV,
                head_k_dim=K,
                head_v_dim=V,
                max_num_draft_tokens=T,
                device="cuda",
                by_request_row=request_rows,
            )
            verifier = GdnUcacheVerifier(
                device=device, first_layer_id=0, by_request_row=request_rows
            )
            outputs = []
            for step, (qkv, gates, z) in enumerate(inputs):
                q, k, v = torch.split(qkv, [H * K, H * K, HV * V], dim=1)
                q, k, v = q.view(-1, H, K), k.view(-1, H, K), v.view(-1, HV, V)
                b, a = gates[:, :HV], gates[:, HV:]
                out = verifier.target_verify(
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
                    cache_indices=slots,
                    num_draft_tokens=T,
                    request_rows=req_rows if request_rows else None,
                    fused_norm=fused_norm(z) if fused_norm is not None else None,
                )
                outputs.append(out.clone())
                commit_gdn_ucache_cursors(
                    buffers=buffers,
                    slots=slots,
                    num_accept_tokens=accepts[step].int(),
                    request_rows=req_rows if request_rows else None,
                )
        torch.cuda.synchronize()
        # Each live slot's rings, wherever they live.
        live = [r for r in range(OPT_BATCH) if int(slots[r]) > 0]
        ring_row = [int(req_rows[r]) if request_rows else int(slots[r]) for r in live]
        rings = [
            torch.stack([x[0, i] for i in ring_row]).view(torch.int16)
            if x.dtype != torch.float32
            else torch.stack([x[0, i] for i in ring_row])
            for x in (buffers.k_ring, buffers.u_ring, buffers.g_ring)
        ]
        cursors = torch.stack(
            [buffers.hist_len[ring_row], buffers.cache_base[ring_row]]
        )
        return outputs, state, rings, cursors, inputs

    def test_launch_options_are_bitwise(self):
        ref_out, ref_state, ref_rings, ref_cursors, _ = self._run()
        for name, case in _OPTION_CASES.items():
            out, state, rings, cursors, _ = self._run(**case)
            for step, (x, y) in enumerate(zip(out, ref_out)):
                self.assertTrue(
                    torch.equal(x.view(torch.int16), y.view(torch.int16)),
                    f"{name}: output differs at step {step}",
                )
            self.assertTrue(
                torch.equal(state.view(torch.int16), ref_state.view(torch.int16)),
                f"{name}: state differs",
            )
            for which, x, y in zip("kug", rings, ref_rings):
                self.assertTrue(torch.equal(x, y), f"{name}: {which} ring differs")
            self.assertTrue(torch.equal(cursors, ref_cursors), f"{name}: cursors differ")

    def test_fused_norm_equals_stock_gated_norm(self):
        """SGLANG_ENABLE_GDN_UCACHE_FUSED_NORM (strided, dependent launches): the flush's
        output is the stock gated RMSNorm of the unfused output, bit for bit."""
        from sglang.kernels.ops.attention.fla.layernorm_gated import (
            RMSNorm as RMSNormGated,
        )
        from sglang.kernels.ops.attention.linear.gdn_ucache_rubin.gdn_replay_mtp_umma_rubin import (
            GdnFusedNorm,
        )

        self.assertTrue(
            gdn_ucache_fused_norm_applies(
                OPT_BATCH * NUM_DRAFT_TOKENS * HV, torch.device("cuda")
            )
        )
        norm = RMSNormGated(
            V,
            eps=1e-6,
            group_size=None,
            norm_before_gate=True,
            dtype=torch.bfloat16,
            device="cuda",
            activation="sigmoid",
        )
        with torch.no_grad():
            norm.weight.copy_(
                torch.randn(V, generator=torch.Generator().manual_seed(3)) * 0.2 + 1.0
            )
        case = dict(strided=True, env=("PDL",))
        plain, _, _, _, inputs = self._run(**case)
        fused, _, _, _, _ = self._run(
            **case,
            fused_norm=lambda z: GdnFusedNorm(z=z, weight=norm.weight, eps=norm.eps),
        )
        for step, (x, y, (_, _, z)) in enumerate(zip(plain, fused, inputs)):
            ref = norm(x.reshape(-1, V), z.reshape(-1, V)).view(y.shape)
            self.assertTrue(
                torch.equal(ref.view(torch.int16), y.view(torch.int16)),
                f"fused norm differs at step {step}: max "
                f"{(ref.float() - y.float()).abs().max().item():.3e}",
            )


if __name__ == "__main__":
    unittest.main()
