import unittest

import torch

from sglang.srt.batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode
from sglang.srt.layers.attention.linear.gdn_backend import (
    torch_chunk_gated_delta_rule,
    torch_gdn_gating,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="stage-b", runner_config="1-gpu-large")

C = 64


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGDNMultiSeqPrefill(unittest.TestCase):
    """Multi-sequence (packed) prefill of torch_chunk_gated_delta_rule.

    forward_extend packs N prompts into one batch=1 row delimited by query_start_loc. The
    delta-rule recurrence is per-sequence, so a packed forward must equal running each sequence
    standalone (batch=1) with its own ssm_states slot. Regression test for the crash on
    rl-auz-icky-day-cs5y where the packed row was treated as one sequence.
    """

    def _run(self, seq_lens, HV=12, HK=4, K=128, V=128, seed=0, with_init=False):
        torch.manual_seed(seed)
        dev = "cuda"
        A_log = torch.randn(HV, device=dev)
        dt_bias = torch.randn(HV, device=dev)
        n = len(seq_lens)
        total = sum(seq_lens)
        starts = [0]
        for sl in seq_lens:
            starts.append(starts[-1] + sl)
        qsl = torch.tensor(starts, dtype=torch.int32, device=dev)
        cache_indices = torch.arange(n, dtype=torch.long, device=dev)

        q = torch.randn(1, total, HK, K, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, total, HK, K, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, total, HV, V, device=dev, dtype=torch.bfloat16)
        a = torch.randn(total, HV, device=dev, dtype=torch.bfloat16)
        b = torch.randn(total, HV, device=dev, dtype=torch.bfloat16)
        g, beta = torch_gdn_gating(A_log, a, b, dt_bias)

        ssm = None
        if with_init:
            ssm = torch.randn(n, HV, V, K, device=dev, dtype=torch.float32)

        with set_batch_invariant_mode(True):
            packed_out, packed_state, _ = torch_chunk_gated_delta_rule(
                q, k, v, g=g, beta=beta,
                ssm_states=ssm, cache_indices=cache_indices, query_start_loc=qsl,
            )
            # Per-sequence reference: each sequence standalone with its own slot state.
            for i in range(n):
                s, e = starts[i], starts[i + 1]
                qi_sl = torch.tensor([0, e - s], dtype=torch.int32, device=dev)
                ssm_i = ssm[i : i + 1] if ssm is not None else None
                gi, bi = torch_gdn_gating(A_log, a[s:e], b[s:e], dt_bias)
                ref_out, ref_state, _ = torch_chunk_gated_delta_rule(
                    q[:, s:e], k[:, s:e], v[:, s:e], g=gi, beta=bi,
                    ssm_states=ssm_i,
                    cache_indices=torch.zeros(1, dtype=torch.long, device=dev),
                    query_start_loc=qi_sl,
                )
                self.assertTrue(
                    torch.equal(packed_out[:, s:e], ref_out),
                    msg=f"seq {i} (len {e - s}) output not bit-identical to standalone",
                )
                self.assertTrue(
                    torch.equal(packed_state[i : i + 1], ref_state),
                    msg=f"seq {i} (len {e - s}) final state not bit-identical to standalone",
                )

    def test_multiseq_unequal_lengths(self):
        # chunk-aligned, misaligned, single-token, multi-chunk; the exact crash case had N=19.
        for lens in [[130, 65, 3], [64, 64], [1, 70, 200], [63, 127, 64], [7] * 19]:
            self._run(lens, seed=hash(tuple(lens)) & 0xFFFF)

    def test_multiseq_with_initial_state(self):
        # Non-zero prefix states (exercises the read-side transpose path).
        self._run([130, 65, 3], with_init=True, seed=7)

    def test_single_seq_unchanged(self):
        # N==1 must degenerate to the previous single-sequence path.
        for pl in [64, 60, 128, 1, 127]:
            self._run([pl], seed=pl)


if __name__ == "__main__":
    unittest.main()
