"""GDN ReplaySSM fold-every-commit: fused ring-write + commit fold.

The production kernel targets bitwise parity with the recurrent verify and
per-draft snapshot baseline. These tests allow an absolute error up to FP32_ATOL
for committed/tracked state and downstream outputs, and verify that bound
through 256 chained commits. Ring-write output and untouched/null slots remain
exact.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
    commit_gdn_replayssm_fold_all_layers,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

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

    def test_ring_write_accepts_strided_qkv_views(self):
        inputs = _make_window(12)
        packed_qkv = torch.cat(
            [inputs[name].reshape(B * T, -1) for name in ("q", "k", "v")],
            dim=-1,
        )
        q, k, v = packed_qkv.split([H * K, H * K, HV * V], dim=-1)
        strided_qkv = {
            "q": q.view(1, B * T, H, K),
            "k": k.view(1, B * T, H, K),
            "v": v.view(1, B * T, HV, V),
        }
        self.assertTrue(
            all(not tensor.is_contiguous() for tensor in strided_qkv.values())
        )

        def run(qkv):
            state = self._state(torch.float32).unsqueeze(0).contiguous()
            rings = _make_rings()
            output = _run_verify(
                {**inputs, **qkv},
                self.gating,
                state[0],
                self.slots,
                rings=rings,
            )
            _fold(state, rings, self.slots, self.accept_lens)
            return {"output": output, "state": state, **rings}

        contiguous = run(
            {name: tensor.contiguous() for name, tensor in strided_qkv.items()}
        )
        strided = run(strided_qkv)
        for name in contiguous:
            self.assertTrue(torch.equal(contiguous[name], strided[name]), name)

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


# DSPARK/DFLASH verify a whole draft block; model-shaped GDN heads.
POOL_T, POOL_LAYERS, POOL_SIZE, POOL_SPEC = 16, 2, 24, 8
POOL_H, POOL_HV, POOL_K, POOL_V, POOL_WIDTH = 4, 8, 128, 128, 4
POOL_CONV_DIM = 2 * POOL_H * POOL_K + POOL_HV * POOL_V


def _pool(*, spec: bool, fold_gdn: bool) -> MambaPool:
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=POOL_HV * POOL_V,
        n_groups=POOL_H,
        num_heads=POOL_HV,
        head_dim=POOL_V,
        state_size=POOL_K,
        conv_kernel=POOL_WIDTH,
    )
    cache_params = Mamba2CacheParams(
        shape=shape,
        layers=list(range(POOL_LAYERS)),
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
    )
    return MambaPool(
        size=POOL_SIZE,
        spec_state_size=POOL_SPEC,
        cache_params=cache_params,
        mamba_layer_ids=list(range(POOL_LAYERS)),
        device=DEVICE,
        speculative_num_draft_tokens=POOL_T,
        enable_linear_replayssm_spec=spec,
        replayssm_spec_fold_gdn=fold_gdn,
    )


def _commit_through_backend(pool, slots, last_correct, track_slots, track_steps):
    """Run HybridLinearAttnBackend.update_mamba_state_after_mtp_verify, the
    DSPARK/DFLASH commit, against ``pool`` with this step's slot metadata."""
    linear_backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(mamba_cache_indices=slots),
        req_to_token_pool=SimpleNamespace(
            mamba_pool=pool,
            get_speculative_mamba2_params_all_layers=(
                pool.get_speculative_mamba2_params_all_layers
            ),
        ),
        accept_lens_pool=None,
        _translate_mamba_indices=lambda indices: indices,
    )
    backend = MagicMock(linear_attn_backend=linear_backend)
    HybridLinearAttnBackend.update_mamba_state_after_mtp_verify(
        backend,
        last_correct_step_indices=last_correct,
        mamba_track_indices=track_slots,
        mamba_steps_to_track=track_steps,
        model=None,
    )


class TestGdnReplayssmSpecFoldDflashCommit(CustomTestCase):
    """--enable-linear-replayssm-spec for DSPARK/DFLASH on GDN: the pool keeps
    one raw-input window per mamba slot, and the backend commit folds it into
    ``temporal`` instead of scattering per-draft snapshots."""

    def test_pool_layout(self):
        pool = _pool(spec=True, fold_gdn=True)
        cache = pool.mamba_cache
        self.assertTrue(pool.replayssm_spec_fold)
        self.assertFalse(pool.replayssm_is_kda)
        # Indexed by mamba slot, one verify window long.
        slots = POOL_SIZE + 1
        self.assertEqual(
            tuple(cache.replayssm_rawv.shape),
            (POOL_LAYERS, slots, POOL_HV, POOL_T, POOL_V),
        )
        self.assertEqual(
            tuple(cache.replayssm_rawk.shape),
            (POOL_LAYERS, slots, POOL_H, POOL_T, POOL_K),
        )
        for ring in (cache.replayssm_g, cache.replayssm_beta):
            self.assertEqual(tuple(ring.shape), (POOL_LAYERS, slots, POOL_HV, POOL_T))
        self.assertIsNone(cache.replayssm_d)
        self.assertIsNone(cache.replayssm_k)
        self.assertIsNone(cache.intermediate_ssm)
        self.assertIsNone(pool.replayssm_spec_write_pos)
        self.assertIsNone(pool.replayssm_cache_base)

        # Without the DSPARK/DFLASH fold, GDN keeps compact replay.
        compact = _pool(spec=True, fold_gdn=False)
        self.assertFalse(compact.replayssm_spec_fold)
        self.assertIsNotNone(compact.mamba_cache.replayssm_d)
        self.assertIsNotNone(compact.replayssm_cache_base)

    def test_commit_matches_snapshot_scatter(self):
        fold = _pool(spec=True, fold_gdn=True)
        snap = _pool(spec=False, fold_gdn=False)
        fc, sc = fold.mamba_cache, snap.mamba_cache
        gen = torch.Generator(device=DEVICE).manual_seed(0)

        def rand(*shape, dtype=torch.bfloat16):
            return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=gen)

        temporal0 = rand(*fc.temporal.shape, dtype=torch.float32) * 0.1
        conv0 = rand(*fc.conv[0].shape)
        for cache in (fc, sc):
            cache.temporal.copy_(temporal0)
            cache.conv[0].copy_(conv0)

        # Requests on the last pool slots (the rings follow the pool, not the
        # request count) plus a padded row; accept lengths include the bonus.
        slots = torch.tensor(
            [POOL_SIZE, POOL_SIZE - 1, 17, 9, 3, -1], dtype=torch.int32, device=DEVICE
        )
        accept_lens = torch.tensor(
            [1, POOL_T, 7, 12, 4, 1], dtype=torch.int32, device=DEVICE
        )
        last_correct = accept_lens - 1
        # Extra-buffer tracking: two requests cross a track boundary.
        track_slots = torch.tensor(
            [POOL_SIZE - 2, 20, 0, 5, 0, 0], dtype=torch.int64, device=DEVICE
        )
        track_steps = torch.tensor(
            [0, 9, -1, 11, -1, -1], dtype=torch.int64, device=DEVICE
        )
        n = slots.numel()
        rows = torch.arange(n, dtype=torch.int32, device=DEVICE)
        cu_seqlens = torch.arange(
            0, n * POOL_T + 1, POOL_T, dtype=torch.int32, device=DEVICE
        )
        cu_seqlens[-1] = cu_seqlens[-2]  # padded row: empty token range
        valid_tokens = (slots >= 0).repeat_interleave(POOL_T)
        gating = {
            "A_log": rand(POOL_HV, dtype=torch.float32) * 0.5,
            "dt_bias": rand(POOL_HV, dtype=torch.float32) * 0.5,
        }
        conv_weight, conv_bias = rand(POOL_CONV_DIM, POOL_WIDTH), rand(POOL_CONV_DIM)

        for layer in range(POOL_LAYERS):
            mixed = rand(n * POOL_T, POOL_CONV_DIM)
            a, b = rand(n * POOL_T, POOL_HV), rand(n * POOL_T, POOL_HV)
            outputs = {}
            for name, pool in (("fold", fold), ("snap", snap)):
                layer_cache = pool.mamba2_layer_cache(layer)
                x = mixed.view(n, POOL_T, POOL_CONV_DIM).transpose(1, 2)
                y = causal_conv1d_update(
                    x,
                    layer_cache.conv[0],
                    conv_weight,
                    conv_bias,
                    "silu",
                    conv_state_indices=slots,
                    intermediate_conv_window=layer_cache.intermediate_conv_window[0],
                    intermediate_state_indices=rows,
                )
                y = y.transpose(1, 2).reshape(n * POOL_T, POOL_CONV_DIM)
                q, k, v = torch.split(
                    y,
                    [POOL_H * POOL_K, POOL_H * POOL_K, POOL_HV * POOL_V],
                    dim=-1,
                )
                if name == "fold":
                    kwargs = dict(
                        cache_ring=True,
                        replayssm_rawv=layer_cache.replayssm_rawv,
                        replayssm_rawk=layer_cache.replayssm_rawk,
                        replayssm_g=layer_cache.replayssm_g,
                        replayssm_beta=layer_cache.replayssm_beta,
                    )
                else:
                    kwargs = dict(
                        intermediate_states_buffer=layer_cache.intermediate_ssm,
                        intermediate_state_indices=rows,
                        cache_steps=POOL_T,
                    )
                outputs[name] = fused_sigmoid_gating_delta_rule_update(
                    A_log=gating["A_log"],
                    dt_bias=gating["dt_bias"],
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    q=q.reshape(1, n * POOL_T, POOL_H, POOL_K),
                    k=k.reshape(1, n * POOL_T, POOL_H, POOL_K),
                    v=v.reshape(1, n * POOL_T, POOL_HV, POOL_V),
                    a=a,
                    b=b,
                    initial_state_source=layer_cache.temporal,
                    initial_state_indices=slots,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True,
                    is_kda=False,
                    disable_state_update=True,
                    **kwargs,
                )
            self.assertTrue(
                torch.equal(
                    outputs["fold"].reshape(n * POOL_T, -1)[valid_tokens],
                    outputs["snap"].reshape(n * POOL_T, -1)[valid_tokens],
                ),
                f"{layer=}",
            )

        for pool in (fold, snap):
            _commit_through_backend(pool, slots, last_correct, track_slots, track_steps)

        torch.testing.assert_close(fc.temporal, sc.temporal, rtol=0, atol=FP32_ATOL)
        self.assertTrue(torch.equal(fc.conv[0], sc.conv[0]))
        written = set(slots.tolist()) | {
            s for s, t in zip(track_slots.tolist(), track_steps.tolist()) if t >= 0
        }
        untouched = [s for s in range(POOL_SIZE + 1) if s not in written]
        self.assertTrue(torch.equal(fc.temporal[:, untouched], temporal0[:, untouched]))
        for s in written - {-1}:
            self.assertFalse(torch.equal(fc.temporal[:, s], temporal0[:, s]), f"{s=}")


class TestGdnReplayssmSpecFoldGate(CustomTestCase):
    """Which models and workers get the GDN fold."""

    def _decide(self, *, algo, gdn, kda, enabled=True):
        configurator = SimpleNamespace(
            hybrid_gdn_config=object() if gdn else None,
            hybrid_kda_config=object() if kda else None,
        )
        exec_cfg = SimpleNamespace(
            mamba=SimpleNamespace(enable_linear_replayssm_spec=enabled)
        )
        module = "sglang.srt.mem_cache.kv_cache_configurator"
        with (
            patch(f"{module}.get_exec", return_value=exec_cfg),
            patch(
                f"{module}.get_spec",
                return_value=SimpleNamespace(speculative_algorithm=algo),
            ),
        ):
            return KVCacheConfigurator._gdn_replayssm_spec_fold(configurator)

    def test_gate(self):
        for algo in ("DFLASH", "DSPARK", "dflash"):
            self.assertTrue(self._decide(algo=algo, gdn=True, kda=False))
        self.assertFalse(
            self._decide(algo="DFLASH", gdn=True, kda=False, enabled=False)
        )
        # MTP / EAGLE commit through spec_utils (GDN compact replay).
        for algo in ("EAGLE", "NEXTN", None):
            self.assertFalse(self._decide(algo=algo, gdn=True, kda=False))
        # KDA already folds on the DSPARK/DFLASH path.
        self.assertFalse(self._decide(algo="DFLASH", gdn=False, kda=True))
        with self.assertRaises(ValueError):
            self._decide(algo="DFLASH", gdn=False, kda=False)
        # The GDN ring-write verify has no ragged layout.
        for mode in ("cap-accept", "compact"):
            with envs.SGLANG_RAGGED_VERIFY_MODE.override(mode):
                with self.assertRaises(ValueError):
                    self._decide(algo="DFLASH", gdn=True, kda=False)


if __name__ == "__main__":
    unittest.main()
