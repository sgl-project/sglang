"""Lossless GDN tuple verify (--enable-linear-lossless-verify).

Target-verify keeps each draft step's (u, k, g) tuple instead of a full
[HV, V, K] state snapshot, and the commit replays the accepted prefix. Every
check here is exact (``torch.equal``) against the snapshot verify, driven
through the real pools, the real verify kernel and the real commit entry point
``HybridLinearAttnBackend.update_mamba_state_after_mtp_verify``.
"""

import types
import unittest

import torch

from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention import hybrid_linear_attn_backend as hlab
from sglang.srt.mem_cache.memory_pool import (
    MambaPool,
    get_tensor_size_bytes,
    spec_intermediate_bytes_per_req,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-large-amd")

DEVICE = "cuda"
NUM_LAYERS = 2
K = V = 128
CONV_KERNEL = 4
MAMBA_SLOTS = 12
# Five verify rows: per-row mamba slots and mamba-track destination slots.
SLOTS = [3, 0, 7, 5, 9]
TRACK_SLOTS = [10, 11, 1, 2, 4]
N = len(SLOTS)
# (k-heads, v-heads) per TP rank: Qwen3.6-35B-A3B at TP=1, and the shape that
# selects the gfx950-specific verify launch config on ROCm.
HEADS = [(16, 32), (4, 16)]


def _params(h, hv, ssm_dtype):
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=hv * V,
        n_groups=h,
        num_heads=hv,
        head_dim=V,
        state_size=K,
        conv_kernel=CONV_KERNEL,
    )
    return Mamba2CacheParams(
        shape=shape,
        layers=list(range(NUM_LAYERS)),
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=ssm_dtype),
    )


def _pool(lossless, d, h, hv, ssm_dtype, spec_slots=N, topk=1):
    return MambaPool(
        size=MAMBA_SLOTS,
        spec_state_size=spec_slots,
        cache_params=_params(h, hv, ssm_dtype),
        mamba_layer_ids=list(range(NUM_LAYERS)),
        device=DEVICE,
        speculative_num_draft_tokens=d,
        speculative_eagle_topk=topk,
        enable_linear_lossless_verify=lossless,
    )


def _fill(pools, seed):
    """Give every pool the same persistent state and conv windows."""
    for pool in pools:
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        cache = pool.mamba_cache

        def rand_like(t, scale=1.0):
            return (torch.randn(t.shape, generator=gen, device=DEVICE) * scale).to(
                t.dtype
            )

        cache.temporal.copy_(rand_like(cache.temporal, 0.1))
        for conv in cache.conv:
            conv.copy_(rand_like(conv))
        for phys in pool._intermediate_conv_window_phys:
            phys.copy_(rand_like(phys))


def _inputs(d, h, hv, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    t = N * d

    def rand(*shape):
        return torch.randn(*shape, generator=gen, device=DEVICE)

    return [
        dict(
            q=rand(1, t, h, K).bfloat16(),
            k=rand(1, t, h, K).bfloat16(),
            v=rand(1, t, hv, V).bfloat16(),
            a=rand(t, hv).bfloat16(),
            b=rand(t, hv).bfloat16(),
            A_log=rand(hv).abs().log(),
            dt_bias=rand(hv),
        )
        for _ in range(NUM_LAYERS)
    ]


def _verify(pool, inputs, d, rows):
    """GDNAttnBackend.forward_extend's target-verify call, for every layer."""
    slots = torch.tensor(SLOTS, dtype=torch.int32, device=DEVICE)
    cu = torch.arange(0, N * d + 1, d, dtype=torch.int32, device=DEVICE)
    outs = []
    for layer, x in enumerate(inputs):
        lc = pool.mamba_cache.at_layer_idx(layer)
        lossless = lc.intermediate_ssm_u is not None
        outs.append(
            fused_sigmoid_gating_delta_rule_update(
                **x,
                initial_state_source=lc.temporal,
                initial_state_indices=slots,
                cu_seqlens=cu,
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                disable_state_update=True,
                intermediate_states_buffer=None if lossless else lc.intermediate_ssm,
                intermediate_state_indices=rows,
                cache_steps=d,
                u_states_buffer=lc.intermediate_ssm_u,
                k_states_buffer=lc.intermediate_ssm_k,
                g_states_buffer=lc.intermediate_ssm_g,
            )
        )
    return outs


def _commit(pool, last, track_idx, track_steps, req_pool_indices=None):
    """Drive the real commit hook with a minimal stand-in for the backend."""
    slots = torch.tensor(SLOTS, dtype=torch.int64, device=DEVICE)
    slot_of_req = torch.full((64,), -1, dtype=torch.int64, device=DEVICE)
    if req_pool_indices is not None:
        slot_of_req[req_pool_indices] = slots
    req_pool = types.SimpleNamespace(
        get_speculative_mamba2_params_all_layers=lambda: pool.mamba_cache,
        get_mamba_indices=lambda r: slot_of_req[r],
        mamba_pool=types.SimpleNamespace(replayssm_is_kda=False),
        short_conv_pool=types.SimpleNamespace(
            conv_state=None, intermediate_conv_state=None
        ),
        ngram_pool=types.SimpleNamespace(context=None, intermediate_context=None),
    )
    linear = types.SimpleNamespace(
        _translate_mamba_indices=lambda x: x,
        forward_metadata=types.SimpleNamespace(mamba_cache_indices=slots),
        req_to_token_pool=req_pool,
        accept_lens_pool=None,
        verify_intermediate_state_indices=torch.arange(
            N + 1, dtype=torch.int32, device=DEVICE
        ),
    )
    backend = types.SimpleNamespace(linear_attn_backend=linear)
    backend._update_ple_state_after_mtp_verify = lambda *a: (
        hlab.HybridLinearAttnBackend._update_ple_state_after_mtp_verify(backend, *a)
    )
    hlab.HybridLinearAttnBackend.update_mamba_state_after_mtp_verify(
        backend,
        last_correct_step_indices=last,
        mamba_track_indices=track_idx,
        mamba_steps_to_track=track_steps,
        model=None,
        req_pool_indices=req_pool_indices,
    )


def _accept_pattern(d):
    """Every accepted length (none, one, half, all) plus track crossings."""
    last = torch.tensor(
        [0, d - 1, min(1, d - 1), d // 2, -1], dtype=torch.int64, device=DEVICE
    )
    track = torch.tensor(
        [0, max(d - 2, 0), min(1, d - 1), -1, -1], dtype=torch.int64, device=DEVICE
    )
    track = torch.minimum(track, last)
    track_idx = torch.tensor(TRACK_SLOTS, dtype=torch.int64, device=DEVICE)
    return last, track, track_idx


class TestGdnLosslessVerify(CustomTestCase):
    def _run_pair(self, d, h, hv, ssm_dtype, seed, pp_rows=False):
        spec_slots = 8 if pp_rows else N
        stock = _pool(False, d, h, hv, ssm_dtype, spec_slots)
        lossless = _pool(True, d, h, hv, ssm_dtype, spec_slots)
        _fill([stock, lossless], seed)
        before = lossless.mamba_cache.temporal.clone()
        inputs = _inputs(d, h, hv, seed + 1)
        if pp_rows:
            req_rows = torch.tensor([4, 1, 6, 0, 2], device=DEVICE)
            rows = req_rows.to(torch.int32)
        else:
            req_rows = None
            rows = torch.arange(N + 1, dtype=torch.int32, device=DEVICE)
        outs_stock = _verify(stock, inputs, d, rows)
        outs_lossless = _verify(lossless, inputs, d, rows)
        last, track, track_idx = _accept_pattern(d)
        _commit(stock, last, track_idx, track, req_rows)
        _commit(lossless, last, track_idx, track, req_rows)
        torch.cuda.synchronize()
        return stock, lossless, outs_stock, outs_lossless, before, last, track_idx

    def _assert_identical(self, stock, lossless, outs_stock, outs_lossless):
        for a, b in zip(outs_stock, outs_lossless):
            self.assertTrue(torch.equal(a, b), "verify output differs")
        s, t = stock.mamba_cache, lossless.mamba_cache
        diff = (s.temporal.float() - t.temporal.float()).abs().max().item()
        self.assertTrue(torch.equal(s.temporal, t.temporal), f"max|d|={diff}")
        for a, b in zip(s.conv, t.conv):
            self.assertTrue(torch.equal(a, b), "conv state differs")

    def test_scratch_bytes_match_allocation(self):
        for lossless in (False, True):
            for d in (2, 4, 8):
                with self.subTest(lossless=lossless, d=d):
                    pool = _pool(lossless, d, 16, 32, torch.float32)
                    c = pool.mamba_cache
                    ssm = (
                        [
                            c.intermediate_ssm_u,
                            c.intermediate_ssm_k,
                            c.intermediate_ssm_g,
                        ]
                        if lossless
                        else [c.intermediate_ssm]
                    )
                    real = get_tensor_size_bytes(ssm) + get_tensor_size_bytes(
                        pool._intermediate_conv_window_phys
                    )
                    per_req = spec_intermediate_bytes_per_req(
                        _params(16, 32, torch.float32),
                        d,
                        speculative_eagle_topk=1,
                        enable_linear_lossless_verify=lossless,
                    )
                    self.assertEqual(per_req * (N + 1), real)
                    self.assertEqual(c.intermediate_ssm is None, lossless)

    def test_bit_exact_against_snapshot_verify(self):
        for h, hv in HEADS:
            for d in (2, 3, 4, 8, 16):
                with self.subTest(h=h, hv=hv, d=d):
                    stock, lossless, outs_s, outs_l, before, last, track_idx = (
                        self._run_pair(d, h, hv, torch.float32, seed=10 * d + h)
                    )
                    self._assert_identical(stock, lossless, outs_s, outs_l)
                    # Rows with nothing accepted or tracked keep their state.
                    touched = torch.zeros(
                        MAMBA_SLOTS + 1, dtype=torch.bool, device=DEVICE
                    )
                    slots = torch.tensor(SLOTS, device=DEVICE)
                    touched[slots[last >= 0]] = True
                    touched[track_idx[:3]] = True
                    t = lossless.mamba_cache.temporal
                    self.assertTrue(torch.equal(t[:, ~touched], before[:, ~touched]))

    def test_bf16_state_pool(self):
        # The tuples stay fp32, so a bf16 pool commits bf16(h_t) like the
        # snapshot verify does.
        for d in (4, 8):
            with self.subTest(d=d):
                stock, lossless, outs_s, outs_l, *_ = self._run_pair(
                    d, 16, 32, torch.bfloat16, seed=d
                )
                self.assertEqual(
                    lossless.mamba_cache.intermediate_ssm_u.dtype, torch.float32
                )
                self._assert_identical(stock, lossless, outs_s, outs_l)

    def test_pp_stable_request_rows(self):
        # SGLANG_ENABLE_PP_SPEC keys verify rows by req_pool_indices; the
        # commit then replays the tuples from those rows.
        with envs.SGLANG_ENABLE_PP_SPEC.override(True):
            stock, lossless, outs_s, outs_l, *_ = self._run_pair(
                4, 16, 32, torch.float32, seed=7, pp_rows=True
            )
        self._assert_identical(stock, lossless, outs_s, outs_l)

    def test_track_replay_reads_pre_verify_state(self):
        # Replaying the accepted step in place first and the track step second
        # (the order a naive port uses) stores a wrong prefix-cache state.
        d = 4
        stock, _, _, _, before, last, track_idx = self._run_pair(
            d, 16, 32, torch.float32, seed=3
        )
        pool = _pool(True, d, 16, 32, torch.float32)
        _fill([pool], 3)
        _verify(
            pool,
            _inputs(d, 16, 32, 4),
            d,
            torch.arange(N + 1, dtype=torch.int32, device=DEVICE),
        )
        c = pool.mamba_cache
        slots = torch.tensor(SLOTS, device=DEVICE)
        rows = torch.arange(N, device=DEVICE)
        _, track, _ = _accept_pattern(d)
        hlab._reconstruct_ssm_from_tuples(
            c.temporal,
            c.intermediate_ssm_u,
            c.intermediate_ssm_k,
            c.intermediate_ssm_g,
            slots,
            rows,
            last,
        )
        hlab._reconstruct_ssm_from_tuples(
            c.temporal,
            c.intermediate_ssm_u,
            c.intermediate_ssm_k,
            c.intermediate_ssm_g,
            slots,
            rows,
            track,
            dest_indices=track_idx,
        )
        tracked = track_idx[track >= 0]
        self.assertFalse(
            torch.equal(c.temporal[:, tracked], stock.mamba_cache.temporal[:, tracked])
        )

    def test_out_of_range_rows_are_skipped(self):
        pool = _pool(True, 4, 16, 32, torch.float32)
        _fill([pool], 5)
        c = pool.mamba_cache
        snap = c.temporal.clone()
        cache_rows = c.intermediate_ssm_u.shape[1]
        hlab._reconstruct_ssm_from_tuples(
            c.temporal,
            c.intermediate_ssm_u,
            c.intermediate_ssm_k,
            c.intermediate_ssm_g,
            torch.tensor([-1, 0, 3, MAMBA_SLOTS + 1], device=DEVICE),
            torch.tensor([0, 1, cache_rows, 3], device=DEVICE),
            torch.zeros(4, dtype=torch.int64, device=DEVICE),
            dest_indices=torch.tensor([2, -1, MAMBA_SLOTS + 1, 2], device=DEVICE),
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(c.temporal, snap))

    @unittest.skipIf(torch.version.hip is not None, "inline PTX is CUDA-only")
    def test_ptx_and_portable_replay_agree(self):
        d = 8
        stock, _, _, _, before, last, _ = self._run_pair(
            d, 16, 32, torch.float32, seed=11
        )
        for use_ptx in (True, False):
            with self.subTest(use_ptx=use_ptx):
                pool = _pool(True, d, 16, 32, torch.float32)
                _fill([pool], 11)
                _verify(
                    pool,
                    _inputs(d, 16, 32, 12),
                    d,
                    torch.arange(N + 1, dtype=torch.int32, device=DEVICE),
                )
                c = pool.mamba_cache
                slots = torch.tensor(SLOTS, device=DEVICE)
                hlab._reconstruct_ssm_from_tuples(
                    c.temporal,
                    c.intermediate_ssm_u,
                    c.intermediate_ssm_k,
                    c.intermediate_ssm_g,
                    slots,
                    torch.arange(N, device=DEVICE),
                    last,
                    use_ptx=use_ptx,
                )
                accepted = slots[last >= 0]
                self.assertTrue(
                    torch.equal(
                        c.temporal[:, accepted], stock.mamba_cache.temporal[:, accepted]
                    )
                )

    def test_rejects_tree_verify(self):
        with self.assertRaises(ValueError):
            _pool(True, 4, 16, 32, torch.float32, topk=2)


if __name__ == "__main__":
    unittest.main()
