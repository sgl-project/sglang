"""GDN ReplaySSM fold-every-commit: fused ring-write + commit fold.

The production kernel targets bitwise parity with the recurrent verify and
per-draft snapshot baseline. These tests allow an absolute error up to FP32_ATOL
for committed/tracked state and downstream outputs, and verify that bound
through 256 chained commits. Ring-write output and untouched/null slots remain
exact.
"""

import unittest

import torch

from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
    commit_gdn_replayssm_fold_all_layers,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.gdn_attention import (
    GDN_ATOL,
    GDN_RTOL,
    GDNAttentionCase,
    build_gdn_attention_fixture,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

B, T = 3, 4
H, HV = 4, 8
K = V = 64
NUM_SLOTS = 8
DEVICE = "cuda"
# Absolute allowance for this deterministic regression case, not a general
# numerical-error guarantee for ReplaySSM.
FP32_ATOL = 2 * torch.finfo(torch.float32).eps


def _make_window(step_seed: int):
    gen = torch.Generator(device=DEVICE).manual_seed(step_seed)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=gen)

    return {
        "q": rand(1, B * T, H, K),
        "k": rand(1, B * T, H, K),
        "v": rand(1, B * T, HV, V),
        "a": rand(B * T, HV),
        "b": rand(B * T, HV),
    }


def _run_verify(inputs, gating, state, slots, *, snapshots=None, rings=None):
    kwargs = {}
    if snapshots is not None:
        kwargs.update(
            intermediate_states_buffer=snapshots,
            intermediate_state_indices=slots,
            cache_steps=T,
        )
    if rings is not None:
        # Per-layer views, matching the backend's mamba2_layer_cache slices.
        kwargs.update(
            cache_ring=True,
            replayssm_rawv=rings["rawv"][0],
            replayssm_rawk=rings["rawk"][0],
            replayssm_g=rings["g"][0],
            replayssm_beta=rings["beta"][0],
        )
    cu_seqlens = torch.arange(0, B * T + 1, step=T, dtype=torch.int32, device=DEVICE)
    return fused_sigmoid_gating_delta_rule_update(
        A_log=gating["A_log"],
        dt_bias=gating["dt_bias"],
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        b=inputs["b"],
        a=inputs["a"],
        initial_state_source=state,
        initial_state_indices=slots,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
        disable_state_update=True,
        **kwargs,
    )


def _make_rings(dtype=torch.bfloat16):
    return {
        "rawv": torch.zeros(1, NUM_SLOTS, HV, T, V, device=DEVICE, dtype=dtype),
        "rawk": torch.zeros(1, NUM_SLOTS, H, T, K, device=DEVICE, dtype=dtype),
        "g": torch.zeros(1, NUM_SLOTS, HV, T, device=DEVICE, dtype=torch.float32),
        "beta": torch.zeros(1, NUM_SLOTS, HV, T, device=DEVICE, dtype=torch.float32),
    }


def _fold(state, rings, slots, accept_lens, track_slots=None, track_steps=None):
    commit_gdn_replayssm_fold_all_layers(
        checkpoint_state=state,
        rawv_cache=rings["rawv"],
        rawk_cache=rings["rawk"],
        g_cache=rings["g"],
        beta_cache=rings["beta"],
        ssm_state_indices=slots,
        accept_lens=accept_lens,
        max_cache_len=T,
        num_k_heads=H,
        mamba_track_indices=track_slots,
        mamba_steps_to_track=track_steps,
        null_block_id=-1,
    )


class TestGdnReplayssmSpecFold(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.gating = {
            "A_log": torch.randn(HV, device=DEVICE) * 0.1,
            "dt_bias": torch.randn(HV, device=DEVICE) * 0.1,
        }
        self.slots = torch.tensor([5, 2, 7], dtype=torch.int32, device=DEVICE)
        self.accept_lens = torch.tensor([3, 1, 4], dtype=torch.int32, device=DEVICE)

    def _state(self, dtype):
        gen = torch.Generator(device=DEVICE).manual_seed(1)
        return torch.randn(
            NUM_SLOTS, HV, K, V, device=DEVICE, dtype=dtype, generator=gen
        )

    def test_ring_write_does_not_change_verify_output(self):
        for dtype in (torch.float32, torch.bfloat16):
            state = self._state(dtype)
            inputs = _make_window(11)
            out_plain = _run_verify(inputs, self.gating, state.clone(), self.slots)
            out_ring = _run_verify(
                inputs, self.gating, state.clone(), self.slots, rings=_make_rings()
            )
            self.assertTrue(torch.equal(out_plain, out_ring), f"{dtype=}")

    def test_fold_matches_snapshot_baseline(self):
        for dtype in (torch.float32, torch.bfloat16):
            state = self._state(dtype)
            inputs = _make_window(22)

            snapshots = torch.zeros(NUM_SLOTS, T, HV, K, V, device=DEVICE, dtype=dtype)
            _run_verify(
                inputs, self.gating, state.clone(), self.slots, snapshots=snapshots
            )

            fold_state = state.clone().unsqueeze(0).contiguous()
            rings = _make_rings()
            _run_verify(inputs, self.gating, fold_state[0], self.slots, rings=rings)
            _fold(fold_state, rings, self.slots, self.accept_lens)

            for s, n in zip(self.slots.tolist(), self.accept_lens.tolist()):
                torch.testing.assert_close(
                    snapshots[s, n - 1],
                    fold_state[0, s],
                    rtol=0,
                    atol=FP32_ATOL,
                    msg=f"{dtype=} slot={s} accept_len={n}",
                )
            untouched = set(range(NUM_SLOTS)) - set(self.slots.tolist())
            for s in untouched:
                self.assertTrue(torch.equal(fold_state[0, s], state[s]))

    def test_track_store_and_null_slots(self):
        dtype = torch.float32
        state = self._state(dtype)
        inputs = _make_window(33)

        snapshots = torch.zeros(NUM_SLOTS, T, HV, K, V, device=DEVICE, dtype=dtype)
        _run_verify(inputs, self.gating, state.clone(), self.slots, snapshots=snapshots)

        fold_state = state.clone().unsqueeze(0).contiguous()
        rings = _make_rings()
        _run_verify(inputs, self.gating, fold_state[0], self.slots, rings=rings)

        track_slots = torch.tensor([1, 0, 3], dtype=torch.int64, device=DEVICE)
        track_steps = torch.tensor([1, -1, 2], dtype=torch.int64, device=DEVICE)
        slots_with_null = self.slots.clone()
        slots_with_null[1] = -1
        _fold(
            fold_state,
            rings,
            slots_with_null,
            self.accept_lens,
            track_slots=track_slots,
            track_steps=track_steps,
        )

        torch.testing.assert_close(
            fold_state[0, 1], snapshots[5, 1], rtol=0, atol=FP32_ATOL
        )
        torch.testing.assert_close(
            fold_state[0, 3], snapshots[7, 2], rtol=0, atol=FP32_ATOL
        )
        # Row 1's state slot is replaced with -1, and its track step is -1, so
        # neither its original state slot 2 nor tracking slot 0 is written.
        self.assertTrue(torch.equal(fold_state[0, 2], state[2]))
        self.assertTrue(torch.equal(fold_state[0, 0], state[0]))

    def test_long_chain_error_stays_bounded(self):
        """This regression case remains within FP32_ATOL through 256 commits."""
        num_iters = 256
        for dtype in (torch.float32, torch.bfloat16):
            base_state = self._state(dtype)
            fold_state = base_state.clone().unsqueeze(0).contiguous()
            snapshots = torch.zeros(NUM_SLOTS, T, HV, K, V, device=DEVICE, dtype=dtype)
            gen = torch.Generator().manual_seed(7)
            for it in range(num_iters):
                inputs = _make_window(1000 + it)
                accept_lens = torch.randint(1, T + 1, (B,), generator=gen).to(
                    device=DEVICE, dtype=torch.int32
                )

                out_base = _run_verify(
                    inputs, self.gating, base_state, self.slots, snapshots=snapshots
                )
                for s, n in zip(self.slots.tolist(), accept_lens.tolist()):
                    base_state[s] = snapshots[s, n - 1]

                rings = _make_rings()
                out_fold = _run_verify(
                    inputs, self.gating, fold_state[0], self.slots, rings=rings
                )
                _fold(fold_state, rings, self.slots, accept_lens)

                torch.testing.assert_close(
                    out_base,
                    out_fold,
                    rtol=0,
                    atol=FP32_ATOL,
                    msg=f"{dtype=} {it=}",
                )
                torch.testing.assert_close(
                    base_state,
                    fold_state[0],
                    rtol=0,
                    atol=FP32_ATOL,
                    msg=f"{dtype=} {it=}",
                )

    def test_compact_backend_matches_dense_verify_and_commit(self):
        draft_token_num = 4
        case = GDNAttentionCase(
            name="gdn_replayssm_compact_ragged",
            backend="triton",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_k_heads=2,
            num_v_heads=2,
            page_size=16,
            prefix_lens=(4, 7, 5),
            extend_lens=(draft_token_num,) * 3,
            enable_linear_replayssm_spec=True,
        )
        fixture = build_gdn_attention_fixture(self, case)
        linear_backend = fixture.backend.linear_attn_backend
        spec_state = fixture.runner.req_to_token_pool.mamba_pool.mamba_cache
        layer_cache = fixture.runner.req_to_token_pool.mamba2_layer_cache(0)
        cache_indices = fixture.runner.req_to_token_pool.get_mamba_indices(
            fixture.forward_batch.req_pool_indices
        )

        # Exercise history-dependent conv values; the standard tiny fixture
        # uses an identity conv, which would not catch cross-request packing.
        with torch.no_grad():
            fixture.actual_module.attn.conv_weights[:, 0] = 0.25
            conv = layer_cache.conv[0]
            gen = torch.Generator(device=DEVICE).manual_seed(2025)
            conv_values = torch.randn(
                (cache_indices.shape[0], *conv.shape[1:]),
                dtype=conv.dtype,
                device=DEVICE,
                generator=gen,
            )
            conv[cache_indices] = conv_values * 0.1

        initial_conv = spec_state.conv[0].clone()
        initial_temporal = spec_state.temporal.clone()
        base_input_ids = fixture.forward_batch.input_ids.clone()
        base_positions = fixture.forward_batch.positions.clone()
        base_out_cache_loc = fixture.forward_batch.out_cache_loc.clone()

        def reset_cache():
            spec_state.conv[0].copy_(initial_conv)
            spec_state.temporal.copy_(initial_temporal)
            for tensor in spec_state.intermediate_conv_window:
                tensor.zero_()
            for name in (
                "replayssm_rawv",
                "replayssm_rawk",
                "replayssm_g",
                "replayssm_beta",
            ):
                getattr(spec_state, name).zero_()

        def run_verify(token_indices, layout):
            batch = fixture.forward_batch
            batch.input_ids = base_input_ids[token_indices]
            batch.positions = base_positions[token_indices]
            batch.out_cache_loc = base_out_cache_loc[token_indices]
            batch.seq_lens = torch.tensor(
                case.prefix_lens, dtype=torch.int32, device=DEVICE
            )
            batch.seq_lens_cpu = torch.tensor(
                case.prefix_lens, dtype=torch.int32, device="cpu"
            )
            batch.seq_lens_sum = sum(case.prefix_lens)
            batch.spec_info = DFlashVerifyInput(
                draft_token=batch.input_ids,
                positions=batch.positions,
                draft_token_num=draft_token_num,
                ragged_verify_layout=layout,
            )
            with torch.no_grad(), forward_context(
                ForwardContext(attn_backend=fixture.backend)
            ):
                linear_backend.init_forward_metadata(batch)
                return fixture.actual_module(
                    batch,
                    fixture.mixed_qkv[token_indices],
                    fixture.a[token_indices],
                    fixture.b[token_indices],
                )

        dense_indices = torch.arange(case.num_input_tokens, device=DEVICE)
        dense_out = run_verify(dense_indices, layout=None)
        accept_lens = torch.tensor([3, 2, 1], dtype=torch.int64, device=DEVICE)
        fixture.backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=accept_lens - 1,
            mamba_track_indices=None,
            mamba_steps_to_track=None,
            model=None,
        )
        dense_conv = spec_state.conv[0].clone()
        dense_temporal = spec_state.temporal.clone()

        reset_cache()
        verify_lens = (4, 2, 1)
        live_indices = torch.tensor(
            [0, 1, 2, 3, 4, 5, 8], dtype=torch.int64, device=DEVICE
        )
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens,
            device=torch.device(DEVICE),
            grid=(8,),
        )
        # One graph-tier tail token sits outside qsl[-1] and must use the
        # ghost row without affecting the seven live packed tokens.
        graph_indices = torch.cat((live_indices, torch.tensor([6], device=DEVICE)))
        compact_out = run_verify(graph_indices, layout=layout)
        fixture.backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=accept_lens - 1,
            mamba_track_indices=None,
            mamba_steps_to_track=None,
            model=None,
        )

        torch.testing.assert_close(
            compact_out[:, : live_indices.shape[0]],
            dense_out[:, live_indices],
            atol=GDN_ATOL,
            rtol=GDN_RTOL,
        )
        self.assertTrue(
            torch.equal(compact_out[:, -1], torch.zeros_like(compact_out[:, -1]))
        )
        torch.testing.assert_close(spec_state.conv[0], dense_conv, rtol=0, atol=0)
        torch.testing.assert_close(
            spec_state.temporal,
            dense_temporal,
            rtol=0,
            atol=FP32_ATOL,
        )


if __name__ == "__main__":
    unittest.main()
