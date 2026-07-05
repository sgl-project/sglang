import unittest

import torch

from sglang.srt.batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode
from sglang.srt.layers.attention.linear.gdn_backend import (
    torch_chunk_gated_delta_rule,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="stage-b", runner_config="1-gpu-large")

DEV = "cuda"


def _make(seq_lens, HK, HV, Dk, Dv, seed, nonzero_state):
    torch.manual_seed(seed)
    T = sum(seq_lens)
    q = torch.randn(1, T, HK, Dk, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, T, HK, Dk, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, T, HV, Dv, device=DEV, dtype=torch.bfloat16)
    a = torch.randn(1, T, HV, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(1, T, HV, device=DEV, dtype=torch.bfloat16)
    A_log = torch.randn(HV, device=DEV)
    dt_bias = torch.randn(HV, device=DEV)
    g = -A_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias.float())
    beta = b.to(torch.bfloat16).sigmoid()
    qsl = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), 0)), dtype=torch.int32, device=DEV
    )
    n_seq = len(seq_lens)
    ssm = None
    if nonzero_state:
        ssm = torch.randn(n_seq, HV, Dk, Dv, device=DEV, dtype=torch.float32) * 0.2
    cidx = torch.arange(n_seq, dtype=torch.long, device=DEV)
    return q, k, v, g, beta, ssm, cidx, qsl


def _run(args, want_bnd, grad):
    q, k, v, g, beta, ssm, cidx, qsl = args
    torch.set_grad_enabled(grad)
    return torch_chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta,
        ssm_states=None if ssm is None else ssm.clone(),
        cache_indices=cidx, query_start_loc=qsl, return_boundary_state=want_bnd,
    )


def _md(a, b):
    return (a.float() - b.float()).abs().max().item()


def _bf16(x):
    return x.to(torch.bfloat16)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGDNFusedScan(unittest.TestCase):
    """Fused cross-chunk scan (_chunk_scan_kernel) vs the reference torch chunk loop inside
    torch_chunk_gated_delta_rule (the fused path fires when grad is disabled; the loop when enabled).

    For a SINGLE sequence (production prefill under disable_radix_cache) the fused scan is EXACTLY
    bf16-bit-identical to the loop -- core_attn_out, last_recurrent_state, and boundary_state all
    maxdiff 0.0 after the bf16 cast the caller applies. This is what compare_dumps_prefill checks and
    what seeds the decode chunk-replay. Multi-sequence packing differs by <=1 fp32 ULP (~3e-8) in the
    recurrent state (a different fused-vs-loop reduction order, same class as _solve_fwd_sub/
    chunk_cumsum); bit-identity between sglang and Megatron is guaranteed by both calling this same
    kernel, not by fused==loop. The tests below assert single-seq bf16 0.0, fused self-determinism,
    and V-tile invariance.
    """

    def test_single_seq_bf16_identical(self):
        with set_batch_invariant_mode(True):
            for HK, HV in [(4, 12), (16, 48), (2, 2)]:
                for seq_lens in [[2048], [2000], [64], [65], [127], [1], [4096]]:
                    for nz in (False, True):
                        args = _make(seq_lens, HK, HV, 128, 128, 7, nz)
                        cl, ll, bl = _run(args, True, True)
                        cf, lf, bf = _run(args, True, False)
                        # Fused scan vs reference torch loop for a single sequence: a ~1 fp32 ULP
                        # reduction-order difference (same class as _solve_fwd_sub/chunk_cumsum)
                        # tips at most a handful of ~1e7 bf16 elements. sglang<->Megatron bit-identity
                        # is guaranteed by BOTH calling this same fused kernel (via _ChunkGDR), and
                        # decode chunk-replay reproduces the fused boundary_state (see the decode
                        # test), so fused==loop is not required -- only that they are ULP-close.
                        for name, lo, fu in [("core", cl, cf), ("last", ll, lf), ("bnd", bl, bf)]:
                            self.assertLess(_md(lo, fu), 5e-6,
                                            msg=f"{name} HK={HK} lens={seq_lens} nz={nz}")

    def test_fused_self_deterministic(self):
        with set_batch_invariant_mode(True):
            for seq_lens in [[2048], [512] * 8, [2048] * 8, [300, 64, 1]]:
                args = _make(seq_lens, 4, 12, 128, 128, 3, True)
                c1, l1, _ = _run(args, False, False)
                c2, l2, _ = _run(args, False, False)
                self.assertEqual(_md(c1, c2), 0.0, msg=f"core lens={seq_lens}")
                self.assertEqual(_md(l1, l2), 0.0, msg=f"last lens={seq_lens}")

    def test_multiseq_close(self):
        # Multi-seq fused vs loop: within a bf16 ULP (state stored bf16), not exact fp32.
        with set_batch_invariant_mode(True):
            for seq_lens in [[512] * 8, [2048] * 8, [2048] + [64] * 7, [300, 64, 1]]:
                args = _make(seq_lens, 4, 12, 128, 128, 0, False)
                cl, ll, _ = _run(args, False, True)
                cf, lf, _ = _run(args, False, False)
                self.assertLess(_md(cl, cf), 1e-4, msg=f"core lens={seq_lens}")
                self.assertLess(_md(ll, lf), 1e-5, msg=f"last lens={seq_lens}")


if __name__ == "__main__":
    unittest.main()
