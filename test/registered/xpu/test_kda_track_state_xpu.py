import unittest

import torch

from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=300, suite="stage-b-test-1-gpu-xpu")

CHUNK_SIZE = 64
# Gap between the pool's per-slot pitch and H*V*K. A kernel that assumes a
# contiguous pool addresses slot i at i*H*V*K, which lands in this gap.
ENVELOPE_PAD = 256
PAD_SENTINEL = -12345.0


def _make_varlen_inputs(seed, lens, num_heads=2, head_dim=128, device="xpu"):
    """Packed varlen KDA inputs: [1, sum(lens), H, D] plus a zero fp32 state pool."""
    generator = torch.Generator().manual_seed(seed)
    total = sum(lens)

    def randn(*shape, dtype=torch.bfloat16):
        # CPU generator + transfer: torch.Generator(device="xpu") is not a
        # portable construction across the XPU torch builds this suite runs on.
        return torch.randn(*shape, generator=generator, dtype=torch.float32).to(
            device=device, dtype=dtype
        )

    q = randn(1, total, num_heads, head_dim)
    k = randn(1, total, num_heads, head_dim)
    v = (0.1 * randn(1, total, num_heads, head_dim, dtype=torch.float32)).to(
        torch.bfloat16
    )
    gate = randn(1, total, num_heads, head_dim)
    beta = torch.sigmoid(randn(1, total, num_heads, dtype=torch.float32)).to(
        torch.bfloat16
    )
    a_log = randn(num_heads, dtype=torch.float32)
    dt_bias = randn(num_heads * head_dim, dtype=torch.float32)
    state = torch.zeros(
        len(lens), num_heads, head_dim, head_dim, device=device, dtype=torch.float32
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    return q, k, v, gate, beta, a_log, dt_bias, state, cu_seqlens


def _run_chunk_kda(q, k, v, gate, beta, a_log, dt_bias, state, cu_seqlens, **kwargs):
    return chunk_kda(
        # chunk_kda writes in place (the attention output lands in v, the gate
        # cumsum in g); hand every run fresh copies so runs stay independent.
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=gate.clone(),
        beta=beta.clone(),
        scale=q.shape[-1] ** -0.5,
        initial_state=state,
        initial_state_indices=torch.arange(
            state.shape[0], device=state.device, dtype=torch.int32
        ),
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        **kwargs,
    )


@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestKdaTrackStateXpu(CustomTestCase):
    @torch.inference_mode()
    def test_track_state_snapshots_fp32_accumulator(self):
        """Bug regression: the XPU chunk kernel had no track path at all, so the
        mamba radix prefix cache could not snapshot the SSM state at the last
        chunk boundary of an unaligned sequence. `track_state` must carry the
        in-kernel fp32 accumulator: identical to the fp32 final state of a run
        truncated at the boundary, and strictly more precise than the bf16 `h`
        row for the same boundary.
        """
        # seq0: 100 tokens, unaligned -> snapshot at the 64-token boundary
        # (start of chunk 1). seq1: 64 tokens, aligned -> not tracked.
        lens = [100, 64]
        q, k, v, gate, beta, a_log, dt_bias, state, cu_seqlens = _make_varlen_inputs(
            0, lens
        )
        num_heads, head_dim = q.shape[2], q.shape[3]

        track_state = torch.full(
            (len(lens), num_heads, head_dim, head_dim),
            float("nan"),
            device=state.device,
            dtype=torch.float32,
        )
        track_chunk_idx = torch.tensor([1, -1], dtype=torch.int32, device=state.device)
        _, h = _run_chunk_kda(
            q,
            k,
            v,
            gate,
            beta,
            a_log,
            dt_bias,
            state,
            cu_seqlens,
            output_intermediate_states=True,
            track_state=track_state,
            track_chunk_idx=track_chunk_idx,
        )

        # The untracked row must stay untouched; the tracked row must be finite.
        self.assertTrue(torch.all(torch.isnan(track_state[1])))
        self.assertFalse(torch.any(torch.isnan(track_state[0])))

        # Reference: truncate seq0 at the boundary; the pool's fp32 row then
        # receives the in-place final state for the same prefix -- the
        # established fp32 path the snapshot must agree with.
        ref_state = torch.zeros(
            1, num_heads, head_dim, head_dim, device=state.device, dtype=torch.float32
        )
        ref_cu_seqlens = torch.tensor(
            [0, CHUNK_SIZE], dtype=torch.int32, device=state.device
        )
        _run_chunk_kda(
            q[:, :CHUNK_SIZE],
            k[:, :CHUNK_SIZE],
            v[:, :CHUNK_SIZE],
            gate[:, :CHUNK_SIZE],
            beta[:, :CHUNK_SIZE],
            a_log,
            dt_bias,
            ref_state,
            ref_cu_seqlens,
        )
        torch.testing.assert_close(track_state[0], ref_state[0], rtol=1e-5, atol=1e-5)

        # The guard: h packs one row per (seq, chunk); row 1 is seq0's state at
        # the boundary, rounded to bf16. If the snapshot were re-routed through
        # h, it could not match the fp32 reference above.
        self.assertTrue(
            torch.equal(h[0, 1].float(), track_state[0].to(torch.bfloat16).float()),
            "h row should be exactly the bf16 rounding of the fp32 snapshot",
        )
        self.assertFalse(
            torch.equal(track_state[0], track_state[0].to(torch.bfloat16).float()),
            "test inputs must make bf16 rounding lossy",
        )

    @torch.inference_mode()
    def test_envelope_strided_initial_state_pool(self):
        """Bug regression: the XPU chunk kernel addressed a state slot as
        index*H*V*K, so a pool whose stride(0) spans all layers (page-major /
        unified memory) had every slot but the first read and written at the
        wrong offset. Slot pitch must come from initial_state.stride(0): results
        match a contiguous pool, and no write leaves a slot.
        """
        lens = [100, 64]
        q, k, v, gate, beta, a_log, dt_bias, state, cu_seqlens = _make_varlen_inputs(
            0, lens
        )
        num_slots, num_heads, v_dim, k_dim = state.shape
        slot_elems = num_heads * v_dim * k_dim
        pitch = slot_elems + ENVELOPE_PAD

        # Distinct non-zero content per slot: against an all-zero pool, reading
        # the wrong slot returns zeros either way and the bug stays invisible.
        generator = torch.Generator().manual_seed(7)
        base = (
            0.1
            * torch.randn(
                num_slots,
                num_heads,
                v_dim,
                k_dim,
                generator=generator,
                dtype=torch.float32,
            )
        ).to(state.device)
        track_chunk_idx = torch.tensor([1, -1], dtype=torch.int32, device=state.device)

        def run(pool):
            track = torch.full(
                (num_slots, num_heads, v_dim, k_dim),
                float("nan"),
                device=state.device,
                dtype=torch.float32,
            )
            out, _ = _run_chunk_kda(
                q,
                k,
                v,
                gate,
                beta,
                a_log,
                dt_bias,
                pool,
                cu_seqlens,
                output_intermediate_states=True,
                track_state=track,
                track_chunk_idx=track_chunk_idx,
            )
            return out, track

        contiguous_pool = base.clone()
        self.assertEqual(contiguous_pool.stride(0), slot_elems)
        expected_out, expected_track = run(contiguous_pool)

        envelope = torch.full(
            (num_slots, pitch), PAD_SENTINEL, device=state.device, dtype=torch.float32
        )
        strided_pool = torch.as_strided(
            envelope,
            (num_slots, num_heads, v_dim, k_dim),
            (pitch, v_dim * k_dim, k_dim, 1),
        )
        strided_pool.copy_(base)
        self.assertEqual(strided_pool.stride(0), pitch)
        actual_out, actual_track = run(strided_pool)

        torch.testing.assert_close(
            actual_out.float(), expected_out.float(), rtol=1e-2, atol=1e-2
        )
        # INPLACE_UPDATE: the pool holds the final state, so this covers the
        # write path as well as the read.
        torch.testing.assert_close(strided_pool, contiguous_pool, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            actual_track, expected_track, rtol=1e-5, atol=1e-5, equal_nan=True
        )
        self.assertTrue(
            torch.all(envelope[:, slot_elems:] == PAD_SENTINEL),
            "kernel wrote past a pool slot",
        )


if __name__ == "__main__":
    unittest.main()
