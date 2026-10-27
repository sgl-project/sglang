"""GDN ReplaySSM spec verify committed through the DSPARK/DFLASH backend path.

DSPARK and DFLASH verify through ``GDNAttnBackend.forward_extend`` and commit
through ``HybridLinearAttnBackend.update_mamba_state_after_mtp_verify``. Under
``--enable-linear-replayssm-spec`` the verify reads each checkpoint once and
appends the draft window to the compact request ring; the commit folds the
accepted prefix into the checkpoint and rolls back conv.

Checked against a float64 sequential GDN reference (conv included) over a long
chain of verify/commit steps, calibrated against the snapshot path (recurrent
verify + per-draft snapshot scatter) on the same inputs. Track slots must hold
the state of the tracked step, non-finite rejected drafts must not reach
accepted outputs or the committed state, the request cursors must move only
for rows with a state slot, and CUDA graph replay must match eager.
"""

import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_after_verify,
    commit_gdn_replayssm_spec,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    GDNKernelDispatcher,
)
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")

# Qwen3.5/3.8 GDN head shape (HV / H = 3); DFlash block 8 and the default
# --linear-replayssm-cache-len.
HEADS, V_HEADS, HEAD_DIM, CONV_WIDTH, NUM_LAYERS = 2, 6, 128, 4, 2
DIM = (2 * HEADS + V_HEADS) * HEAD_DIM
WIDTH, RING = 8, 16
NUM_REAL = 6
NUM_SLOTS = NUM_REAL + 1  # one graph-padding slot
NUM_ROWS = 12  # request rows: spec_state_size + 1
NUM_POOL_SLOTS = 16  # persistent mamba slots, track slots included

RING_FIELDS = (
    "replayssm_d",
    "replayssm_k",
    "replayssm_g",
    "replayssm_rawv",
    "replayssm_rawk",
)


def _randn(*shape, dtype=torch.bfloat16, scale=0.2):
    return (torch.randn(*shape, device="cuda") * scale).to(dtype)


def _make_layers():
    return [
        SimpleNamespace(
            layer_id=i,
            num_q_heads=HEADS,
            num_k_heads=HEADS,
            num_v_heads=V_HEADS,
            head_q_dim=HEAD_DIM,
            head_k_dim=HEAD_DIM,
            head_v_dim=HEAD_DIM,
            q_dim=HEADS * HEAD_DIM,
            k_dim=HEADS * HEAD_DIM,
            v_dim=V_HEADS * HEAD_DIM,
            conv_weights=_randn(DIM, CONV_WIDTH, scale=0.5),
            bias=_randn(DIM),
            activation="silu",
            # Decay rates from ~0.98 to ~0.05 per token across heads.
            A_log=torch.empty(V_HEADS, device="cuda").uniform_(-4.0, 1.0),
            dt_bias=_randn(V_HEADS),
        )
        for i in range(NUM_LAYERS)
    ]


def _conv_window(dtype):
    # Deduplicated sliding-window conv scratch as MambaPool allocates it for
    # CUDA linear chains: step t's window is phys[..., t : t + CONV_WIDTH - 1].
    phys = torch.zeros(
        NUM_LAYERS, NUM_ROWS, DIM, WIDTH + CONV_WIDTH - 2, device="cuda", dtype=dtype
    )
    return phys.as_strided(
        (NUM_LAYERS, NUM_ROWS, WIDTH, DIM, CONV_WIDTH - 1),
        (phys.stride(0), phys.stride(1), 1, phys.stride(2), 1),
    )


class _Arm:
    """One verify/commit path over its own copy of the pool contents."""

    def __init__(self, template, *, replayssm, dispatcher, ring_len=RING, lazy=False):
        conv = template["conv"].clone()
        temporal = template["temporal"].clone()
        if replayssm:

            def ring(heads, last, dtype):
                return torch.zeros(
                    NUM_LAYERS,
                    NUM_ROWS,
                    heads,
                    ring_len,
                    *last,
                    device="cuda",
                    dtype=dtype,
                )

            self.state = MambaPool.SpeculativeState(
                conv=[conv],
                temporal=temporal,
                intermediate_ssm=None,
                intermediate_conv_window=[_conv_window(conv.dtype)],
                replayssm_d=ring(V_HEADS, (HEAD_DIM,), torch.bfloat16),
                replayssm_k=ring(HEADS, (HEAD_DIM,), torch.bfloat16),
                replayssm_g=ring(V_HEADS, (), torch.float32),
                # Low parts of the bf16 compact d and normalized k.
                replayssm_rawv=ring(V_HEADS, (HEAD_DIM,), torch.bfloat16),
                replayssm_rawk=ring(HEADS, (HEAD_DIM,), torch.bfloat16),
            )

            def cursor(dtype):
                return torch.zeros(NUM_ROWS, device="cuda", dtype=dtype)

            self.pool = SimpleNamespace(
                replayssm_is_kda=False,
                replayssm_spec_fold=False,
                replayssm_spec_write_pos=cursor(torch.int32),
                replayssm_cache_base=cursor(torch.int32),
                replayssm_is_flush=cursor(torch.int8),
            )
        else:
            self.state = MambaPool.SpeculativeState(
                conv=[conv],
                temporal=temporal,
                intermediate_ssm=temporal.new_zeros(
                    NUM_LAYERS, NUM_ROWS, WIDTH, *temporal.shape[2:]
                ),
                intermediate_conv_window=[_conv_window(conv.dtype)],
            )
            self.pool = SimpleNamespace()
        self.dispatcher = dispatcher
        self.lazy = lazy

    def backend(self, query_start_loc, cache_indices):
        backend = GDNAttnBackend.__new__(GDNAttnBackend)
        backend.req_to_token_pool = SimpleNamespace(
            size=NUM_ROWS - 1,
            mamba2_layer_cache=self.state.at_layer_idx,
            get_speculative_mamba2_params_all_layers=lambda: self.state,
            translate_mamba_indices=lambda ids: ids,
            mamba_pool=self.pool,
            short_conv_pool=SimpleNamespace(
                conv_state=None, intermediate_conv_state=None
            ),
            ngram_pool=SimpleNamespace(context=None, intermediate_context=None),
        )
        backend.forward_metadata = ForwardMetadata(
            query_start_loc=query_start_loc, mamba_cache_indices=cache_indices
        )
        backend.verify_intermediate_state_indices = torch.arange(
            NUM_ROWS, device="cuda", dtype=torch.int32
        )
        backend.kernel_dispatcher = self.dispatcher
        backend.mis_metadata = None
        backend._use_strided_target_verify_qkv = False
        hybrid = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
        hybrid.linear_attn_backend = backend
        self.hybrid = hybrid
        return backend

    def commit(self, accept_steps, req_rows, track_slots=None, track_steps=None):
        if self.lazy:
            # The generic spec commit of an fp32 state: fold only when the
            # next window would not fit, so later verifies replay history.
            metadata = self.hybrid.linear_attn_backend.forward_metadata
            commit_gdn_replayssm_after_verify(
                spec_state=self.state,
                write_pos=self.pool.replayssm_spec_write_pos,
                cache_base=self.pool.replayssm_cache_base,
                is_flush=self.pool.replayssm_is_flush,
                state_batch_indices=metadata.mamba_cache_indices[: len(req_rows)],
                replay_indices=req_rows,
                accept_lens=accept_steps + 1,
                last_correct_step_indices=accept_steps,
                max_spec_len=WIDTH,
                fold_every_commit=False,
            )
            return
        self.hybrid.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=accept_steps,
            mamba_track_indices=track_slots,
            mamba_steps_to_track=track_steps,
            model=None,
            req_pool_indices=req_rows,
        )


def _template(ssm_dtype):
    return {
        "temporal": _randn(
            NUM_LAYERS, NUM_POOL_SLOTS, V_HEADS, HEAD_DIM, HEAD_DIM, dtype=ssm_dtype
        ),
        "conv": _randn(NUM_LAYERS, NUM_POOL_SLOTS, DIM, CONV_WIDTH - 1),
    }


def _batch(num_tokens, req_rows):
    return SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        req_pool_indices=req_rows,
        input_ids=torch.zeros(num_tokens, device="cuda", dtype=torch.int64),
        spec_info=SimpleNamespace(draft_token_num=WIDTH),
    )


def _verify(backend, layers, batch, inputs):
    return [
        backend.forward_extend(layer, batch, mixed_qkv, a, b)
        for layer, (mixed_qkv, a, b) in zip(layers, inputs)
    ]


def _rows_and_slots(num_slots=NUM_SLOTS, num_real=NUM_REAL):
    """Scrambled request rows (row 0 is reserved) and persistent slots for the
    real requests; graph-padding slots carry request row 0 and slot -1."""
    req_rows = torch.zeros(num_slots, device="cuda", dtype=torch.int64)
    req_rows[:num_real] = torch.randperm(NUM_ROWS - 1, device="cuda")[:num_real] + 1
    cache_indices = torch.full((num_slots,), -1, device="cuda", dtype=torch.int32)
    cache_indices[:num_real] = torch.randperm(NUM_POOL_SLOTS, device="cuda")[
        :num_real
    ].int()
    return req_rows, cache_indices


def _dense_inputs(num_slots=NUM_SLOTS):
    return [
        (
            _randn(num_slots * WIDTH, DIM),
            _randn(num_slots * WIDTH, V_HEADS),
            _randn(num_slots * WIDTH, V_HEADS),
        )
        for _ in range(NUM_LAYERS)
    ]


def _gdn_reference(layer, conv_state, ssm_state, mixed_qkv, a, b):
    """float64 sequential GDN over one window: conv1d with its state, l2-normed
    q/k, scalar decay, delta rule. Returns per-step outputs, states and conv
    windows."""
    steps = mixed_qkv.shape[0]
    x = torch.cat([conv_state.double(), mixed_qkv.double().T], dim=1)
    taps = x.unfold(1, CONV_WIDTH, 1)  # [DIM, steps, CONV_WIDTH]
    conv = (taps * layer.conv_weights.double()[:, None, :]).sum(-1)
    conv = F.silu(conv + layer.bias.double()[:, None]).T
    q, k, v = conv.split([HEADS * HEAD_DIM, HEADS * HEAD_DIM, V_HEADS * HEAD_DIM], -1)
    group = V_HEADS // HEADS

    def l2norm(t):
        t = t.view(steps, HEADS, HEAD_DIM)
        t = t / torch.sqrt((t * t).sum(-1, keepdim=True) + 1e-6)
        return t.repeat_interleave(group, dim=1)

    q, k = l2norm(q) * HEAD_DIM**-0.5, l2norm(k)
    v = v.view(steps, V_HEADS, HEAD_DIM)
    g = -torch.exp(layer.A_log.double()) * F.softplus(
        a.double() + layer.dt_bias.double()
    )
    beta = torch.sigmoid(b.double())
    state = ssm_state.double()
    outputs, states = [], []
    for t in range(steps):
        state = state * torch.exp(g[t])[:, None, None]
        delta = beta[t][:, None] * (v[t] - torch.einsum("hvk,hk->hv", state, k[t]))
        state = state + delta[:, :, None] * k[t][:, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", state, q[t]))
        states.append(state)
    windows = [x[:, t + 1 : t + CONV_WIDTH] for t in range(steps)]
    return torch.stack(outputs), states, windows


def _relative_error(actual, expected):
    expected = expected.double()
    error = (actual.double() - expected).abs().max()
    return (error / expected.abs().max().clamp(min=1e-30)).item()


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGDNReplaySSMDirectCommit(CustomTestCase):
    def setUp(self):
        triton = LinearAttnKernelBackend.TRITON
        self.dispatcher = GDNKernelDispatcher(triton, triton, triton)

    def _replayssm(self, template):
        return _Arm(template, replayssm=True, dispatcher=self.dispatcher)

    def test_chain_matches_reference(self):
        """Chained verify/commit steps with random accepted lengths: outputs
        and committed states track the float64 reference no worse than the
        snapshot path, without error growth, and touch nothing outside the
        batch."""
        for ssm_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(ssm_dtype=ssm_dtype):
                self._check_chain(ssm_dtype, num_steps=256)

    def test_lazy_fold_chain_replays_history(self):
        """The generic spec commit of an fp32 state folds only when the next
        window would not fit the ring, so verifies replay committed history;
        the outputs stay as close to the reference as the snapshot path's."""
        self._check_chain(torch.float32, num_steps=128, lazy=True)

    def _check_chain(self, ssm_dtype, num_steps, lazy=False):
        torch.manual_seed(7)
        layers = _make_layers()
        template = _template(ssm_dtype)
        req_rows, cache_indices = _rows_and_slots()
        real_rows, real_slots = req_rows[:NUM_REAL], cache_indices[:NUM_REAL].long()
        qsl = torch.arange(
            0, NUM_SLOTS * WIDTH + 1, WIDTH, device="cuda", dtype=torch.int32
        )
        arms = {
            "replayssm": _Arm(
                template,
                replayssm=True,
                dispatcher=self.dispatcher,
                ring_len=4 * WIDTH if lazy else RING,
                lazy=lazy,
            ),
            # The Triton recurrent verify writes a snapshot per draft on every
            # GPU.
            "snapshot": _Arm(template, replayssm=False, dispatcher=self.dispatcher),
        }
        backends = {name: arm.backend(qsl, cache_indices) for name, arm in arms.items()}
        replayssm = arms["replayssm"]
        untouched_rows = [
            r for r in range(NUM_ROWS) if r not in set(real_rows.tolist())
        ]
        untouched_slots = [
            s for s in range(NUM_POOL_SLOTS) if s not in set(real_slots.tolist())
        ]
        ring_before = replayssm.state.replayssm_d[:, untouched_rows].clone()
        reference = {
            "temporal": template["temporal"].double(),
            "conv": template["conv"].double(),
        }
        # Under the lazy fold the checkpoint lags the committed state by the
        # history kept in the ring, so only the outputs are compared; they
        # depend on the checkpoint and the replayed history together.
        keys = ("out",) if lazy else ("out", "state")
        errors = {name: {key: [] for key in keys} for name in arms}
        batch = _batch(NUM_SLOTS * WIDTH, req_rows)
        generator = torch.Generator().manual_seed(11)
        expected_base = torch.zeros(NUM_REAL, dtype=torch.int64)
        max_history = 0
        for _ in range(num_steps):
            inputs = _dense_inputs()
            accept = torch.randint(1, WIDTH + 1, (NUM_REAL,), generator=generator)
            outputs = {
                name: _verify(backends[name], layers, batch, inputs) for name in arms
            }
            ref_out = []
            for layer, (mixed_qkv, a, b) in zip(layers, inputs):
                layer_out = []
                for i in range(NUM_REAL):
                    slot, n = int(real_slots[i]), int(accept[i])
                    rows = slice(i * WIDTH, (i + 1) * WIDTH)
                    out, states, windows = _gdn_reference(
                        layer,
                        reference["conv"][layer.layer_id, slot],
                        reference["temporal"][layer.layer_id, slot],
                        mixed_qkv[rows],
                        a[rows],
                        b[rows],
                    )
                    layer_out.append(out)
                    reference["temporal"][layer.layer_id, slot] = states[n - 1]
                    reference["conv"][layer.layer_id, slot] = windows[n - 1]
                ref_out.append(torch.cat(layer_out))
            for name, arm in arms.items():
                arm.commit((accept - 1).cuda(), req_rows[:NUM_REAL])
                out = torch.stack(
                    [
                        o.reshape(-1, V_HEADS, HEAD_DIM)[: NUM_REAL * WIDTH]
                        for o in outputs[name]
                    ]
                )
                errors[name]["out"].append(_relative_error(out, torch.stack(ref_out)))
                if not lazy:
                    errors[name]["state"].append(
                        _relative_error(
                            arm.state.temporal[:, real_slots],
                            reference["temporal"][:, real_slots],
                        )
                    )
                # Conv rollback is an exact copy of the accepted inputs.
                self.assertTrue(
                    torch.equal(
                        arm.state.conv[0][:, real_slots].double(),
                        reference["conv"][:, real_slots],
                    ),
                    name,
                )
            # The graph-padding slot (state -1) outputs zeros.
            for out in outputs["replayssm"]:
                padding = out.reshape(-1, V_HEADS, HEAD_DIM)[NUM_REAL * WIDTH :]
                self.assertEqual(padding.abs().max().item(), 0.0)
            pool = replayssm.pool
            if lazy:
                history = pool.replayssm_spec_write_pos[real_rows]
                max_history = max(max_history, int(history.max()))
                continue
            # Cursors: folded and reset every commit, ring origin advanced.
            expected_base = (expected_base + accept) % RING
            self.assertEqual(
                pool.replayssm_spec_write_pos[real_rows].abs().sum().item(), 0
            )
            self.assertEqual(pool.replayssm_is_flush[real_rows].abs().sum().item(), 0)
            self.assertEqual(
                pool.replayssm_cache_base[real_rows].cpu().tolist(),
                expected_base.tolist(),
            )

        if lazy:
            # Later verifies saw committed history.
            self.assertGreater(max_history, WIDTH)
        pool = replayssm.pool
        for cursor in (
            pool.replayssm_spec_write_pos,
            pool.replayssm_cache_base,
            pool.replayssm_is_flush,
        ):
            self.assertEqual(cursor[untouched_rows].abs().sum().item(), 0)
        self.assertTrue(
            torch.equal(replayssm.state.replayssm_d[:, untouched_rows], ring_before)
        )
        self.assertTrue(
            torch.equal(
                replayssm.state.temporal[:, untouched_slots],
                template["temporal"][:, untouched_slots],
            )
        )

        # Relative errors: mean over the chain, early window (after a burn-in
        # that lets the state reach its working magnitude) and late window.
        def mean(values):
            return sum(values) / len(values)

        summary = {
            name: {
                key: (mean(values), mean(values[32:96]), mean(values[-64:]))
                for key, values in errs.items()
            }
            for name, errs in errors.items()
        }
        print(f"\n{ssm_dtype} relative error (mean, early, late): {summary}")
        for key in keys:
            values = errors["replayssm"][key]
            chain_mean, early, late = summary["replayssm"][key]
            self.assertTrue(all(map(math.isfinite, values)), key)
            # No error growth over the chain.
            self.assertLessEqual(late, 2 * early + 1e-6, f"{ssm_dtype} {key}")
            # Within rounding of the snapshot path on the same inputs: the
            # verify dots round their operands to bf16 (bf16 state) or TF32
            # (fp32 state), as the snapshot kernels do.
            baseline = summary["snapshot"][key][0]
            self.assertLessEqual(chain_mean, 1.25 * baseline, f"{ssm_dtype} {key}")

    def test_track_slot_holds_tracked_step(self):
        """A track slot receives the state and conv window of its tracked step
        and leaves the active commit unchanged; step -1 tracks nothing."""
        for ssm_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(ssm_dtype=ssm_dtype):
                self._check_track(ssm_dtype)

    def _check_track(self, ssm_dtype):
        torch.manual_seed(99)
        layers = _make_layers()
        template = _template(ssm_dtype)
        req_rows, cache_indices = _rows_and_slots()
        real_rows = req_rows[:NUM_REAL]
        active = cache_indices[:NUM_REAL].long()
        unused = [s for s in range(NUM_POOL_SLOTS) if s not in set(active.tolist())]
        track_slots = torch.tensor(unused[:NUM_REAL], device="cuda")
        accept = torch.tensor([8, 5, 3, 1, 6, 8], device="cuda")
        track_steps = torch.tensor([7, 2, -1, 0, 5, 3], device="cuda")
        qsl = torch.arange(
            0, NUM_SLOTS * WIDTH + 1, WIDTH, device="cuda", dtype=torch.int32
        )
        batch = _batch(NUM_SLOTS * WIDTH, req_rows)
        tracked, plain, at_step = (self._replayssm(template) for _ in range(3))
        backends = [
            arm.backend(qsl, cache_indices) for arm in (tracked, plain, at_step)
        ]

        # Warm-up steps move the ring origin off zero.
        for _ in range(3):
            inputs = _dense_inputs()
            warm_accept = torch.randint(1, WIDTH + 1, (NUM_REAL,), device="cuda")
            for arm, backend in zip((tracked, plain, at_step), backends):
                _verify(backend, layers, batch, inputs)
                arm.commit(warm_accept - 1, real_rows)
        before = tracked.state.temporal.clone()

        inputs = _dense_inputs()
        for backend in backends:
            _verify(backend, layers, batch, inputs)
        tracked.commit(accept - 1, real_rows, track_slots, track_steps)
        plain.commit(accept - 1, real_rows)
        at_step.commit(
            torch.where(track_steps >= 0, track_steps, accept - 1), real_rows
        )

        # The active commit does not see the track slots. Compiling the track
        # path in may change the rounding of the active fold by one ulp.
        eps = torch.finfo(ssm_dtype).eps
        torch.testing.assert_close(
            tracked.state.temporal[:, active].float(),
            plain.state.temporal[:, active].float(),
            rtol=eps,
            atol=1e-6,
        )
        self.assertTrue(
            torch.equal(
                tracked.state.conv[0][:, active], plain.state.conv[0][:, active]
            )
        )
        for i, step in enumerate(track_steps.tolist()):
            track = int(track_slots[i])
            if step < 0:
                self.assertTrue(
                    torch.equal(tracked.state.temporal[:, track], before[:, track])
                )
                continue
            # The tracked state is the active state of a commit that accepted
            # exactly the tracked step (same fold, one rounding).
            torch.testing.assert_close(
                tracked.state.temporal[:, track].float(),
                at_step.state.temporal[:, int(active[i])].float(),
                rtol=eps,
                atol=1e-6,
            )
            self.assertTrue(
                torch.equal(
                    tracked.state.conv[0][:, track],
                    at_step.state.conv[0][:, int(active[i])],
                )
            )

    def test_rejected_non_finite_drafts_stay_contained(self):
        """Non-finite activations at rejected draft positions leave the
        accepted outputs, the committed state, conv, the cursors and the later
        steps bit-identical to a run with finite rejected drafts, also under
        the lazy fold of the generic spec commit, whose verifies replay the
        committed history."""
        for ssm_dtype, lazy in (
            (torch.bfloat16, False),
            (torch.float32, False),
            (torch.float32, True),
        ):
            with self.subTest(ssm_dtype=ssm_dtype, lazy=lazy):
                self._check_non_finite_drafts(ssm_dtype, lazy)

    def _check_non_finite_drafts(self, ssm_dtype, lazy):
        torch.manual_seed(2024)
        layers = _make_layers()
        template = _template(ssm_dtype)
        req_rows, cache_indices = _rows_and_slots()
        qsl = torch.arange(
            0, NUM_SLOTS * WIDTH + 1, WIDTH, device="cuda", dtype=torch.int32
        )
        batch = _batch(NUM_SLOTS * WIDTH, req_rows)
        clean, poisoned = (
            _Arm(
                template,
                replayssm=True,
                dispatcher=self.dispatcher,
                ring_len=4 * WIDTH if lazy else RING,
                lazy=lazy,
            )
            for _ in range(2)
        )
        backends = [arm.backend(qsl, cache_indices) for arm in (clean, poisoned)]
        accept = [1, 3, WIDTH, 5, 2, 7]
        poisoned_steps = (0, 2)
        for step in range(4):
            if lazy and step == 2:
                # The second poisoned window follows committed history.
                history = clean.pool.replayssm_spec_write_pos[req_rows[:NUM_REAL]]
                self.assertGreater(int(history.min()), 0)
            inputs = _dense_inputs()
            bad_inputs = [tuple(x.clone() for x in layer) for layer in inputs]
            if step in poisoned_steps:
                for mixed_qkv, a, b in bad_inputs:
                    for i, n in enumerate(accept):
                        rejected = slice(i * WIDTH + n, (i + 1) * WIDTH)
                        mixed_qkv[rejected] = float("nan")
                        a[rejected] = float("inf")
                        b[rejected] = float("nan")
            expected = _verify(backends[0], layers, batch, inputs)
            actual = _verify(backends[1], layers, batch, bad_inputs)
            for out, ref in zip(actual, expected):
                out = out.reshape(-1, V_HEADS, HEAD_DIM)
                ref = ref.reshape(-1, V_HEADS, HEAD_DIM)
                for i, n in enumerate(accept):
                    # The steps after a poisoned one compare whole windows.
                    if step not in poisoned_steps:
                        n = WIDTH
                    rows = slice(i * WIDTH, i * WIDTH + n)
                    self.assertTrue(torch.equal(out[rows], ref[rows]), (step, i))
            accept_steps = torch.tensor(accept, device="cuda") - 1
            for arm in (clean, poisoned):
                arm.commit(accept_steps, req_rows[:NUM_REAL])
            self.assertTrue(torch.equal(poisoned.state.temporal, clean.state.temporal))
            self.assertTrue(torch.equal(poisoned.state.conv[0], clean.state.conv[0]))
            for name in (
                "replayssm_spec_write_pos",
                "replayssm_cache_base",
                "replayssm_is_flush",
            ):
                self.assertTrue(
                    torch.equal(
                        getattr(poisoned.pool, name), getattr(clean.pool, name)
                    ),
                    name,
                )

    def test_row_without_state_slot_keeps_cursors(self):
        """A committed row whose state slot is -1 keeps its cursors, so once it
        has a slot again its verify matches a fresh pool's."""
        torch.manual_seed(77)
        layers = _make_layers()
        template = _template(torch.bfloat16)
        req_rows, cache_indices = _rows_and_slots()
        qsl = torch.arange(
            0, NUM_SLOTS * WIDTH + 1, WIDTH, device="cuda", dtype=torch.int32
        )
        batch = _batch(NUM_SLOTS * WIDTH, req_rows)
        arm = self._replayssm(template)
        slotless = cache_indices.clone()
        slotless[1] = -1
        _verify(arm.backend(qsl, slotless), layers, batch, _dense_inputs())
        arm.commit(
            torch.full((NUM_REAL,), WIDTH - 1, device="cuda"), req_rows[:NUM_REAL]
        )
        pool, row = arm.pool, int(req_rows[1])
        self.assertEqual(int(pool.replayssm_spec_write_pos[row]), 0)
        self.assertEqual(int(pool.replayssm_is_flush[row]), 0)
        self.assertEqual(int(pool.replayssm_cache_base[row]), 0)
        others = req_rows[[0, 2, 3, 4, 5]]
        self.assertEqual(pool.replayssm_cache_base[others].tolist(), [WIDTH % RING] * 5)

        fresh = self._replayssm(template)
        fresh.state.temporal.copy_(arm.state.temporal)
        fresh.state.conv[0].copy_(arm.state.conv[0])
        fresh.pool.replayssm_cache_base.copy_(pool.replayssm_cache_base)
        inputs = _dense_inputs()
        out = _verify(arm.backend(qsl, cache_indices), layers, batch, inputs)
        expected = _verify(fresh.backend(qsl, cache_indices), layers, batch, inputs)
        for actual, ref in zip(out, expected):
            real = slice(0, NUM_REAL * WIDTH)
            self.assertTrue(
                torch.equal(
                    actual.reshape(-1, V_HEADS, HEAD_DIM)[real],
                    ref.reshape(-1, V_HEADS, HEAD_DIM)[real],
                )
            )

    def test_cuda_graph_replay_matches_eager(self):
        """Verify captured once replays with new inputs, request rows, slots
        and ring origins exactly like eager verify."""
        for ssm_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(ssm_dtype=ssm_dtype):
                self._check_graph(ssm_dtype)

    def _check_graph(self, ssm_dtype, num_replays=3):
        torch.manual_seed(4321)
        layers = _make_layers()
        template = _template(ssm_dtype)
        num_tokens = NUM_SLOTS * WIDTH
        qsl = torch.arange(0, num_tokens + 1, WIDTH, device="cuda", dtype=torch.int32)
        static_rows, static_cache = _rows_and_slots()
        static_inputs = [
            (
                _randn(num_tokens, DIM),
                _randn(num_tokens, V_HEADS),
                _randn(num_tokens, V_HEADS),
            )
            for _ in layers
        ]
        graph_arm = self._replayssm(template)
        graph_backend = graph_arm.backend(qsl, static_cache)
        capture_batch = _batch(num_tokens, static_rows)

        def run():
            return [
                graph_backend.forward_extend(layer, capture_batch, mixed_qkv, a, b)
                for layer, (mixed_qkv, a, b) in zip(layers, static_inputs)
            ]

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = run()

        real = slice(0, NUM_REAL * WIDTH)
        for _ in range(num_replays):
            req_rows, cache_indices = _rows_and_slots()
            for static in static_inputs:
                for tensor in static:
                    tensor.copy_(_randn(*tensor.shape))
            static_rows.copy_(req_rows)
            static_cache.copy_(cache_indices)

            eager_arm = self._replayssm(template)
            base = torch.randint(0, RING, (NUM_ROWS,), device="cuda", dtype=torch.int32)
            for arm in (graph_arm, eager_arm):
                arm.state.temporal.copy_(template["temporal"])
                arm.state.conv[0].copy_(template["conv"])
                for name in RING_FIELDS:
                    getattr(arm.state, name).zero_()
                arm.pool.replayssm_cache_base.copy_(base)
            graph.replay()

            expected = _verify(
                eager_arm.backend(qsl, cache_indices),
                layers,
                _batch(num_tokens, req_rows),
                static_inputs,
            )
            for actual, ref in zip(static_out, expected):
                self.assertTrue(
                    torch.equal(
                        actual.reshape(-1, V_HEADS, HEAD_DIM)[real],
                        ref.reshape(-1, V_HEADS, HEAD_DIM)[real],
                    )
                )
            for name in RING_FIELDS:
                self.assertTrue(
                    torch.equal(
                        getattr(graph_arm.state, name), getattr(eager_arm.state, name)
                    ),
                    name,
                )
            accept_steps = torch.randint(0, WIDTH, (NUM_REAL,), device="cuda")
            graph_arm.commit(accept_steps, req_rows[:NUM_REAL])
            eager_arm.commit(accept_steps, req_rows[:NUM_REAL])
            self.assertTrue(
                torch.equal(graph_arm.state.temporal, eager_arm.state.temporal)
            )
            self.assertTrue(
                torch.equal(graph_arm.state.conv[0], eager_arm.state.conv[0])
            )
            self.assertTrue(
                torch.equal(
                    graph_arm.pool.replayssm_cache_base,
                    eager_arm.pool.replayssm_cache_base,
                )
            )


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGDNReplaySSMSpecCursors(CustomTestCase):
    def test_rows_without_state_slot_keep_their_cursors(self):
        def cursor(values, dtype):
            out = torch.zeros(NUM_ROWS, device="cuda", dtype=dtype)
            for row, value in values.items():
                out[row] = value
            return out

        # Rows 3 and 7 have slots (7 folds this commit); 5 and 9 have none.
        write_pos = cursor({3: 2, 5: 1, 7: 4, 11: 6}, torch.int32)
        cache_base = cursor({3: 5, 5: 3, 7: 14, 9: 9, 11: 1}, torch.int32)
        is_flush = cursor({5: 1, 7: 1, 11: 1}, torch.int8)
        before = [t.clone() for t in (write_pos, cache_base, is_flush)]
        commit_gdn_replayssm_spec(
            write_pos=write_pos,
            cache_base=cache_base,
            is_flush=is_flush,
            num_accepted=torch.tensor([3, 2, 8, 1], device="cuda"),
            replay_indices=torch.tensor([3, 5, 7, 9], device="cuda"),
            state_batch_indices=torch.tensor(
                [2, -1, 4, -1], device="cuda", dtype=torch.int32
            ),
            max_cache_len=RING,
            max_spec_len=WIDTH,
            fold_every_commit=True,
            null_block_id=-1,
        )

        expected = [t.clone() for t in before]
        # Row 3 appends its 3 accepted records; row 7 first retires its folded
        # 4 records (origin 14 + 4 wraps to 2), then holds the 8 new ones.
        for cursor_t, row3, row7 in zip(expected, (5, 5, 1), (8, 2, 1)):
            cursor_t[3], cursor_t[7] = row3, row7
        for name, actual, want in zip(
            ("write_pos", "cache_base", "is_flush"),
            (write_pos, cache_base, is_flush),
            expected,
        ):
            self.assertEqual(actual.tolist(), want.tolist(), name)


if __name__ == "__main__":
    unittest.main()
