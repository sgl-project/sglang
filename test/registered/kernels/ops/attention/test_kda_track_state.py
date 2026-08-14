import unittest

import torch

from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=180, stage="base-c", runner_config="4-gpu-gb300")

CHUNK_SIZE = 64

_BACKENDS = {"triton": chunk_kda}
try:
    from sglang.kernels.ops.attention.helion.kda_prefill import (
        chunk_kda as helion_chunk_kda,
    )

    _BACKENDS["helion"] = helion_chunk_kda
except ImportError:
    pass


def _make_varlen_inputs(seed, lens, num_heads=2, head_dim=128):
    """Packed varlen KDA inputs: [1, sum(lens), H, D] plus a zero fp32 state pool."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(lens)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=generator, device="cuda", dtype=dtype)

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
        len(lens), num_heads, head_dim, head_dim, device="cuda", dtype=torch.float32
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    return q, k, v, gate, beta, a_log, dt_bias, state, cu_seqlens


def _run_chunk_kda(
    chunk_kda_fn, q, k, v, gate, beta, a_log, dt_bias, state, cu_seqlens, **kwargs
):
    return chunk_kda_fn(
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
            state.shape[0], device="cuda", dtype=torch.int32
        ),
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        **kwargs,
    )


class TestKdaTrackState(CustomTestCase):
    @torch.inference_mode()
    def test_track_state_snapshots_fp32_accumulator(self):
        """Bug regression: the mamba radix track path snapshots the SSM state at
        the last chunk boundary of unaligned sequences into the fp32 state pool.
        It used to read the per-chunk states `h` (activation dtype, bf16), so a
        prefix-cache hit restored a bf16-rounded state while a cache miss kept
        fp32. `track_state` must carry the in-kernel fp32 accumulator: identical
        to the fp32 final state of a run truncated at the boundary, and strictly
        more precise than the bf16 `h` row for the same boundary.
        """
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        for backend, chunk_kda_fn in _BACKENDS.items():
            with self.subTest(backend=backend):
                self._check_track_state(chunk_kda_fn)

    def _check_track_state(self, chunk_kda_fn):
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
            device="cuda",
            dtype=torch.float32,
        )
        track_chunk_idx = torch.tensor([1, -1], dtype=torch.int32, device="cuda")
        _, h = _run_chunk_kda(
            chunk_kda_fn,
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
        # receives the in-place final state for the same prefix — the
        # established fp32 path the snapshot must agree with.
        ref_state = torch.zeros(
            1, num_heads, head_dim, head_dim, device="cuda", dtype=torch.float32
        )
        ref_cu_seqlens = torch.tensor([0, CHUNK_SIZE], dtype=torch.int32, device="cuda")
        _run_chunk_kda(
            chunk_kda_fn,
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


if __name__ == "__main__":
    unittest.main()
