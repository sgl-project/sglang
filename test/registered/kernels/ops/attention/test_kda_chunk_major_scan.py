"""The gfx950 chunk-major KDA prefill scans against the token-major scan."""

import unittest

import torch

from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: the chunk-major gated key and fused-output scan run on gfx950 only.
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_gfx95_supported(), "gfx950 chunk-major KDA scan")
class TestKDAChunkMajorScan(CustomTestCase):
    """The chunk-major gated key must not change the checkpointing scan's math
    (bit-exact with token-major), and the fused-output scan must match it within
    rounding, across tail chunks, both intra paths, padded rows and track snapshots."""

    HEADS, DIM = 12, 128

    def _run(self, lengths, *, pad_last, track, chunk_major, fused_output):
        from sglang.srt.environ import envs

        g = torch.Generator(device="cuda").manual_seed(sum(lengths))
        shape = (1, sum(lengths), self.HEADS, self.DIM)
        rnd = lambda *s: torch.randn(*s, generator=g, device="cuda")  # noqa: E731
        q, k, v, gate = (rnd(*shape).bfloat16() for _ in range(4))
        beta = rnd(1, shape[1], self.HEADS).bfloat16()
        state = rnd(len(lengths) + 2, self.HEADS, self.DIM, self.DIM) * 0.05
        indices = torch.arange(1, len(lengths) + 1, dtype=torch.int32, device="cuda")
        if pad_last:
            indices[-1] = -1
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        kwargs = {}
        if track is not None:
            kwargs = dict(
                track_state=torch.full(
                    (len(track), self.HEADS, self.DIM, self.DIM),
                    float("nan"),
                    device="cuda",
                ),
                track_chunk_idx=torch.tensor(track, dtype=torch.int32, device="cuda"),
                output_intermediate_states=True,
            )
        with (
            envs.SGLANG_ROCM_KDA_CHUNK_MAJOR_SCAN.override(chunk_major),
            envs.SGLANG_ROCM_KDA_FUSED_OUTPUT_MAX_TOKENS.override(
                1 << 30 if fused_output else 0
            ),
        ):
            out = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=gate,
                beta=beta,
                initial_state=state,
                initial_state_indices=indices,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu,
                A_log=rnd(self.HEADS) * 0.5,
                dt_bias=rnd(self.HEADS * self.DIM) * 0.1,
                lower_bound=-5.0,
                beta_is_raw=True,
                **kwargs,
            )
        out = out[0] if isinstance(out, tuple) else out
        valid = sum(lengths[:-1]) if pad_last else sum(lengths)
        return out[:, :valid].float(), state, kwargs.get("track_state")

    @torch.inference_mode()
    def test_chunk_major_and_fused_output_match_token_major(self):
        cases = (
            # 8K in one sequence: the standalone recompute kernel writes kg.
            ([8192], False, None),
            # Ragged with tail chunks; few chunks: the fused inter-solve writes kg.
            ([5, 64, 130, 17], False, None),
            ([1, 63, 64, 65, 200, 1000, 77], True, None),
            ([700, 300, 1100], False, [3, 0, 16]),
        )
        for lengths, pad_last, track in cases:
            ref = self._run(
                lengths,
                pad_last=pad_last,
                track=track,
                chunk_major=False,
                fused_output=False,
            )
            for fused_output in (False, True):
                with self.subTest(lengths=lengths[:4], fused_output=fused_output):
                    got = self._run(
                        lengths,
                        pad_last=pad_last,
                        track=track,
                        chunk_major=True,
                        fused_output=fused_output,
                    )
                    for name, a, b in zip(("o", "state", "track"), got, ref):
                        if b is None:
                            continue
                        if fused_output:
                            torch.testing.assert_close(
                                a, b, atol=2e-3, rtol=1e-2, msg=name
                            )
                        else:
                            self.assertTrue(torch.equal(a, b), name)


if __name__ == "__main__":
    unittest.main()
