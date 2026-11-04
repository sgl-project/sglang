"""Single-GPU correctness and contract tests for the Kimi-K3 K1 launch
(``kda_pre``, ROCm / FlyDSL): one KDA layer from the attention-residual seam to
the gated core output.

K1 is rank-local -- its ABI carries no peer buffer and no rank -- so unlike the
MoE and whole-layer launches this needs one GPU, not eight.

What is checked, and why each:

* the seam, against a torch reference: aggregation then ``input_layernorm``.
  Same formula as the layer launch's seam, so a divergence localizes to K1.
* the in-projection, against the seam output the kernel itself produced. Same
  reasoning as the layer test: comparing the projection against the kernel's
  own x keeps a rounding difference in the seam from being reported as a GEMM
  bug.
* the **conv-state footprint**, which is the contract K1 would have to share
  with SGLang's speculative conv-window pool. Nothing about wiring K1 into the
  model is safe to design until the indices are measured, because FlyDSL's
  ``buffer_load`` is OOB-clamped: reading past the buffer returns a clamped
  address rather than faulting, so a window that is one entry short loses
  precision silently.

Run directly with ``python test/registered/amd/test_kimi_k3_mono_kda_pre.py``.
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# One GPU: K1 is rank-local. gfx950 only, so the MI35x suite. The run itself
# is about 10 s; the budget covers the first FlyDSL build on a cold JIT cache.
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

BF = torch.bfloat16
COS_MIN = 0.999
# One request of L tokens: the shape a DSPARK-7 verify step has at concurrency 1.
REQS = 1
QLEN = 8
NVB = 3
MAX_BLOCKS = 8
EPS = 1e-5


def _randn(gen, *shape, scale=1.0, dtype=BF):
    return (
        torch.randn(*shape, generator=gen, device=gen.device, dtype=torch.float32)
        * scale
    ).to(dtype)


def _cos(a, b):
    return float(
        F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0, eps=1e-12)
    )


def _rms(x, weight, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def _seam_reference(prefix, blocks, nvb, ares_nw, ares_qk, in_nw, eps, out_eps):
    """Aggregate the banked sources and the prefix, then input_layernorm."""
    rows = torch.cat([blocks[:, :nvb, :], prefix.unsqueeze(1)], dim=1)
    scores = (_rms(rows, ares_nw, eps) * ares_qk.float()).sum(-1)
    mixed = (torch.softmax(scores, dim=-1).unsqueeze(-1) * rows.float()).sum(dim=1)
    return _rms(mixed, in_nw, out_eps).to(BF)


def _kda_reference(case):
    """core_out in torch, from the kernel's own in-projection.

    Derived from the kernel source rather than from SGLang's KDA path: the two
    agree on the math but not on argument conventions, so reproducing
    ``kda_conv`` / ``kda_gates`` / ``kda_recur`` / ``kda_norm`` directly is
    what makes a mismatch mean "the kernel is wrong" instead of "the harness
    guessed the convention wrong".

    Feeds on ``proj_dbg`` for the same reason the layer test feeds on the
    kernel's own x: a rounding difference upstream must not be reported here.
    """
    kda = case.kda
    L, hd, nh, proj_w = case.L, kda.HD, kda.NH, kda.PROJ
    win = kda.CONV_W - 1
    acc_idx = case.num_acc - 1
    slot_c = int(case.st_idx[0, 0])
    slot0 = int(case.st_idx[0, acc_idx])
    proj = case.proj_dbg.float()

    # --- conv: the carried window then this step's tokens, then silu
    xv = proj[:, : 3 * proj_w].t()  # [3 * PROJ, L]
    window = case.conv_state_in[slot_c][:, acc_idx : acc_idx + win].float()
    seq = torch.cat([window, xv], dim=1)
    cw = case.conv_w.float()
    conv = torch.stack(
        [sum(cw[:, q] * seq[:, j + q] for q in range(kda.CONV_W)) for j in range(L)],
        dim=1,
    )
    conv = (conv * torch.sigmoid(conv)).t()  # [L, 3 * PROJ]
    q_all, k_all, v_all = (
        conv[:, i * proj_w : (i + 1) * proj_w].reshape(L, nh, hd) for i in range(3)
    )

    def l2(x, scale):
        return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + 1e-6) * scale

    q_all = l2(q_all, hd**-0.5)
    k_all = l2(k_all, 1.0)

    fa = proj[:, kda.FA_0 : kda.FA_0 + hd]  # [L, HD]
    out = torch.empty(case.S, proj_w, dtype=torch.float32, device=proj.device)
    for h in range(nh):
        beta = torch.sigmoid(proj[:, kda.BETA_0 + h])  # [L]
        g1 = fa.to(BF).float() @ case.w_fb[h * hd : (h + 1) * hd].float().t()
        g = g1 + case.dt_bias[h * hd : (h + 1) * hd].float()
        gate = -5.0 * torch.sigmoid(torch.exp(case.a_log[h].float()) * g)
        decay = torch.exp(gate)  # [L, HD], per-K

        state = case.rstate_in[slot0, h].float().clone()  # [V, K]
        o = torch.empty(L, hd, dtype=torch.float32, device=proj.device)
        for j in range(L):
            state = state * decay[j].unsqueeze(0)
            delta = (v_all[j, h] - state @ k_all[j, h]) * beta[j]
            state = state + delta.unsqueeze(1) * k_all[j, h].unsqueeze(0)
            o[j] = state @ q_all[j, h]

        r = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)
        g2 = proj[:, kda.G2_0 + h * hd : kda.G2_0 + (h + 1) * hd]
        out[:, h * hd : (h + 1) * hd] = o * r * case.on_w.float() * torch.sigmoid(g2)
    return out.to(BF)


class K1Case:
    """One synthetic K1 launch: its inputs, and the tensors it wrote."""

    def __init__(self, device, *, num_acc, state_len=None, write_idx=-1):
        from sglang.kernels.ops.moe.k3_mono_flydsl.attention import kda

        self.kda = kda
        self.device = device
        self.L = QLEN
        self.S = REQS * QLEN
        # K1 rebases the conv window on every launch, reading up to
        # acc_idx + CONV_W - 1 past the start. See test_conv_state_footprint.
        self.state_len = state_len or (self.L + kda.CONV_W - 1)
        self.num_acc = num_acc
        self.write_idx = write_idx

        g = torch.Generator(device=device).manual_seed(5)
        s, h, dim = self.S, kda.HIDDEN, 3 * kda.PROJ
        self.prefix = _randn(g, s, h, scale=0.3)
        self.blocks = _randn(g, s, MAX_BLOCKS, h, scale=0.3)
        self.ares_nw = (1 + _randn(g, h, scale=0.1)).to(BF)
        self.ares_qk = _randn(g, h, scale=0.02)
        self.in_nw = (1 + _randn(g, h, scale=0.1)).to(BF)
        self.w_in = _randn(g, kda.NPROJ, h, scale=0.02)
        self.w_fb = _randn(g, kda.NH * kda.HD, kda.HD, scale=0.02)
        self.conv_w = _randn(g, dim, kda.CONV_W, scale=0.3, dtype=torch.float32)
        self.a_log = _randn(g, kda.NH, scale=0.1, dtype=torch.float32)
        self.dt_bias = _randn(g, kda.NH * kda.HD, scale=0.1, dtype=torch.float32)
        self.on_w = (1 + _randn(g, kda.HD, scale=0.1)).to(BF)

        # Slot 0 is an ordinary slot: the kernel's "no state" marker is a
        # negative index, matching SGLang's null_block_id = -1. Starting at 0
        # is what covers the request that owns scratch row 0, which the pool
        # hands out like any other.
        slots = torch.arange(self.L, dtype=torch.int32, device=device)
        self.st_idx = slots.view(REQS, self.L).contiguous()
        self.num_acc_t = torch.full((REQS,), num_acc, dtype=torch.int32, device=device)
        nslots = int(slots.max()) + 1
        # Always drawn at full width and truncated, so a short-buffer case is a
        # prefix of the full one and every later draw is unchanged.
        full = _randn(g, nslots, dim, self.L + kda.CONV_W - 1, scale=0.5)
        self.conv_state = full[:, :, : self.state_len].contiguous()
        self.conv_state_in = self.conv_state.clone()
        self.rstate = _randn(
            g, nslots, kda.NH, kda.HD, kda.HD, scale=0.05, dtype=torch.float32
        )
        self.rstate_in = self.rstate.clone()

        self.core_out = torch.zeros(self.S, kda.PROJ, dtype=BF, device=device)
        self.x_dbg = torch.zeros(self.S, h, dtype=BF, device=device)
        self.proj_dbg = torch.zeros(self.S, kda.NPROJ, dtype=BF, device=device)

    def run(self):
        kda = self.kda
        key = kda.KdaPreBuild(
            tokens=self.S,
            qlen=self.L,
            nblocks=NVB,
            delta=False,
            state_len=self.state_len,
            eps=EPS,
            out_eps=EPS,
            onorm_eps=EPS,
            debug=True,
            write_idx=self.write_idx,
        )
        scratch = torch.zeros(
            kda.scratch_bytes(key), dtype=torch.uint8, device=self.device
        )
        epoch = torch.ones(1, dtype=torch.int32, device=self.device)
        kda.kda_pre(
            key,
            prefix=self.prefix,
            delta=None,
            blocks=self.blocks,
            ares_nw=self.ares_nw,
            ares_qk=self.ares_qk,
            in_nw=self.in_nw,
            w_in=self.w_in,
            w_fb=self.w_fb,
            conv_w=self.conv_w,
            conv_state=self.conv_state,
            a_log=self.a_log,
            dt_bias=self.dt_bias,
            on_w=self.on_w,
            rstate=self.rstate,
            st_idx=self.st_idx,
            num_acc=self.num_acc_t,
            core_out=self.core_out,
            scratch=scratch,
            layer=1,
            epoch=epoch,
            x_dbg=self.x_dbg,
            proj_dbg=self.proj_dbg,
        )
        torch.cuda.synchronize()
        return self


def _skip_reasons():
    if not torch.cuda.is_available():
        return "needs a GPU"
    if not torch.version.hip:
        return "ROCm only"
    return None


class TestKimiK3MonoKdaPre(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        reason = _skip_reasons()
        if reason:
            raise unittest.SkipTest(reason)
        cls.device = torch.device("cuda", 0)
        torch.cuda.set_device(cls.device)
        # acc_idx = num_acc - 1 reaches its maximum when every drafted token of
        # the previous step was accepted. That is the case the conv window has
        # to be widest for, so make it the default.
        cls.case = K1Case(cls.device, num_acc=QLEN).run()

    def test_seam_matches_torch(self):
        case = self.case
        want = _seam_reference(
            case.prefix,
            case.blocks,
            NVB,
            case.ares_nw,
            case.ares_qk,
            case.in_nw,
            EPS,
            EPS,
        )
        cos = _cos(case.x_dbg, want)
        print(f"seam: cos {cos:.6f}")
        self.assertGreater(cos, COS_MIN, "the attention-residual seam differs")

    def test_in_proj_matches_its_own_x(self):
        case = self.case
        # Rows past 6284 are padding the kernel never defines.
        real = 6284
        want = F.linear(case.x_dbg, case.w_in[:real])
        cos = _cos(case.proj_dbg[:, :real], want)
        print(f"in_proj: cos {cos:.6f}")
        self.assertGreater(cos, COS_MIN, "the in-projection differs")

    def test_core_out_matches_torch(self):
        case = self.case
        out = case.core_out
        self.assertTrue(bool(torch.isfinite(out).all()), "core_out is not finite")
        self.assertTrue(bool(torch.count_nonzero(out) > 0), "core_out is all zero")
        cos = _cos(out, _kda_reference(case))
        print(f"core_out: cos {cos:.6f}")
        self.assertGreater(cos, COS_MIN, "the KDA recurrence differs")

    def test_every_acceptance_count(self):
        """num_acc picks both the conv window offset and the initial state slot.

        The default case uses the widest offset; these cover the rest, since a
        caller has no control over how many drafted tokens get accepted.
        """
        for num_acc in range(1, QLEN + 1):
            case = K1Case(self.device, num_acc=num_acc).run()
            cos = _cos(case.core_out, _kda_reference(case))
            with self.subTest(num_acc=num_acc):
                self.assertGreater(cos, COS_MIN, f"num_acc={num_acc}: cos {cos:.6f}")
        print(f"acceptance counts 1..{QLEN}: all match")

    def test_slot_zero_is_a_real_slot(self):
        """Row 0 of the speculative scratch must compute like any other row.

        SGLang's ``build_verify_intermediate_state_indices`` gives request slot
        i scratch row i, so row 0 goes to whichever request holds req_pool
        index 0 -- there is no way to avoid it. The default case already starts
        its slots at 0; this pins the behaviour so the kernel's sentinel cannot
        drift back to treating 0 as "no state".
        """
        case = self.case
        self.assertEqual(int(case.st_idx[0, 0]), 0, "the case must exercise slot 0")
        cos = _cos(case.core_out, _kda_reference(case))
        self.assertGreater(cos, COS_MIN, f"slot 0 computed wrong (cos {cos:.6f})")
        touched = [
            j
            for j in range(case.rstate.shape[0])
            if not torch.equal(case.rstate[j], case.rstate_in[j])
        ]
        self.assertIn(0, touched, "slot 0 was skipped by the state write")

    def test_negative_slot_is_the_sentinel(self):
        """A negative slot means "no state": zero output, nothing written."""
        case = K1Case(self.device, num_acc=QLEN)
        case.st_idx = torch.full_like(case.st_idx, -1)
        case.run()
        self.assertEqual(
            int(torch.count_nonzero(case.core_out)),
            0,
            "a negative slot should produce a zero core_out",
        )
        self.assertTrue(
            torch.equal(case.rstate, case.rstate_in),
            "a negative slot should write no recurrent state",
        )
        self.assertTrue(
            torch.equal(case.conv_state, case.conv_state_in),
            "a negative slot should write no conv state",
        )
        print("negative slot: no output, no state written")

    def test_seeded_window_equals_the_rolling_convention(self):
        """A caller may own the conv buffer and reseed it every step.

        K1's own convention is a rolling buffer it rebases each launch, with
        ``num_acc`` saying where the live window starts. A caller that owns the
        buffer instead can hand it the window directly at offset 0 and pass
        ``num_acc = 1``. This has to compute the same thing, because that is
        what lets K1 take its three carried values from SGLang's committed conv
        state rather than from a buffer SGLang does not maintain.

        It also drops the highest read index from ``acc_idx + CONV_W - 1`` to
        ``CONV_W - 1``, which is why a privately owned buffer sidesteps the
        one-short problem entirely.
        """
        kda = self.case.kda
        win = kda.CONV_W - 1
        acc = 5  # an interior acceptance count, so the offsets really differ
        rolling = K1Case(self.device, num_acc=acc).run()

        slot_c = int(rolling.st_idx[0, 0])
        acc_idx = acc - 1
        seeded = K1Case(self.device, num_acc=1)
        # The window the rolling case used, placed at offset 0.
        seeded.conv_state[slot_c][:, :win] = rolling.conv_state_in[slot_c][
            :, acc_idx : acc_idx + win
        ]
        # ... and the initial state the rolling case selected.
        seeded.rstate[int(seeded.st_idx[0, 0])] = rolling.rstate_in[
            int(rolling.st_idx[0, acc_idx])
        ]
        seeded.conv_state_in = seeded.conv_state.clone()
        seeded.rstate_in = seeded.rstate.clone()
        seeded.run()

        cos = _cos(seeded.core_out, rolling.core_out)
        equal = bool(torch.equal(seeded.core_out, rolling.core_out))
        print(f"seeded vs rolling: cos {cos:.6f} (equal={equal})")
        self.assertGreater(
            cos, COS_MIN, "seeding the window at offset 0 is not equivalent"
        )

    def test_core_out_depends_on_the_initial_state(self):
        """Control: the recurrence must actually read the slot it selects.

        ``num_acc`` picks which of the previous step's states to start from. If
        core_out were the same with that slot perturbed, the reference above
        would be matching a kernel that ignores its initial state, and the
        test would prove nothing about the state wiring 5b depends on.
        """
        case = self.case
        acc_idx = case.num_acc - 1
        other = K1Case(self.device, num_acc=case.num_acc)
        slot0 = int(other.st_idx[0, acc_idx])
        other.rstate[slot0] += 1.0
        other.rstate_in = other.rstate.clone()
        other.run()
        cos = _cos(other.core_out, case.core_out)
        print(f"perturbed initial state: cos vs unperturbed {cos:.6f}")
        self.assertLess(
            cos, COS_MIN, "core_out ignored the initial state slot num_acc selects"
        )
        self.assertGreater(
            _cos(other.core_out, _kda_reference(other)),
            COS_MIN,
            "the reference does not track the perturbed initial state",
        )

    def test_conv_state_footprint(self):
        """The contract a caller has to satisfy, measured rather than assumed.

        K1 rebases the conv window every launch: the new buffer is the three
        entries starting one past the accepted token, then this step's L token
        inputs. So it reads up to index ``acc_idx + CONV_W - 1`` of the old
        buffer, and ``acc_idx`` reaches ``L - 1`` whenever every drafted token
        was accepted.
        """
        case = self.case
        kda = case.kda
        win = kda.CONV_W - 1
        acc_idx = case.num_acc - 1
        slot_c = int(case.st_idx[0, 0])
        before = case.conv_state_in[slot_c]
        after = case.conv_state[slot_c]

        # The carried window: new[0 : win] == old[acc_idx + 1 : acc_idx + 1 + win]
        self.assertTrue(
            torch.equal(after[:, :win], before[:, acc_idx + 1 : acc_idx + 1 + win]),
            "the carried conv window is not the entries past the accepted token",
        )
        # The appended tokens: new[win:] == this step's conv channel inputs,
        # which are the first 3 * PROJ rows of the in-projection.
        want = case.proj_dbg[:, : 3 * kda.PROJ].t().contiguous()
        self.assertTrue(
            torch.equal(after[:, win:], want),
            "the appended conv entries are not this step's token inputs",
        )
        highest_read = acc_idx + win
        print(
            f"conv footprint: reads up to index {highest_read} of "
            f"state_len {case.state_len} (acc_idx {acc_idx})"
        )
        # The requirement this test exists to pin down.
        self.assertEqual(
            highest_read,
            case.state_len - 1,
            "state_len must be qlen + CONV_W - 1; a shorter buffer is read "
            "out of bounds, which buffer_load clamps instead of faulting",
        )

    def test_recurrent_state_footprint(self):
        """Which rstate slots K1 reads and writes.

        It starts from the slot of the last accepted token and writes one slot
        per token of this step -- so a caller owes it ``st_idx[req, 0 : L]``
        with the previous step's states still in place.
        """
        case = self.case
        touched = [
            j
            for j in range(case.rstate.shape[0])
            if not torch.equal(case.rstate[j], case.rstate_in[j])
        ]
        want = sorted(int(s) for s in case.st_idx[0].tolist())
        print(f"rstate: wrote slots {touched}, st_idx holds {want}")
        self.assertEqual(touched, want, "K1 did not write exactly st_idx[req, :L]")

    def test_one_short_state_len_is_wrong_and_silent(self):
        """A buffer one entry short is read out of bounds, and nothing says so.

        This is the case a caller would hit by handing K1 SGLang's speculative
        conv window, which is ``num_draft_tokens + CONV_W - 2`` wide -- one
        less than K1 reads. The two cases below share the first entries, so the
        in-bounds part of the carry must agree and only the entry that came
        from past the end may differ. It does, and the launch still succeeds:
        the failure is silent.
        """
        kda = self.case.kda
        win = kda.CONV_W - 1
        acc_idx = QLEN - 1
        slot_c = int(self.case.st_idx[0, 0])
        long_after = self.case.conv_state[slot_c]

        short = K1Case(self.device, num_acc=QLEN, state_len=QLEN + kda.CONV_W - 2).run()
        short_after = short.conv_state[slot_c]

        self.assertTrue(
            torch.equal(short_after[:, : win - 1], long_after[:, : win - 1]),
            "the in-bounds part of the carried window should be unaffected",
        )
        self.assertFalse(
            torch.equal(short_after[:, win - 1], long_after[:, win - 1]),
            "a one-short buffer happened to read the right value; this test "
            "no longer demonstrates the out-of-bounds read",
        )
        print(
            f"short state_len {short.state_len}: carry entry {win - 1} differs "
            f"(read index {acc_idx + win} does not exist), launch succeeded"
        )

    def test_block_write_layer_snapshots_the_prefix(self):
        """write_idx >= 0 stores the updated prefix into that bank row.

        This is the seam the whole-layer launch's ``reset`` mode assumes has
        already happened, so the two have to agree on the row.
        """
        row = NVB
        case = K1Case(self.device, num_acc=QLEN, write_idx=row).run()
        self.assertTrue(
            torch.equal(case.blocks[:, row, :], case.prefix),
            "the block-write row does not hold the updated prefix",
        )


if __name__ == "__main__":
    unittest.main()
