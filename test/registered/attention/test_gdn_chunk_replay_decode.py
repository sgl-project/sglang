import unittest

import torch

from sglang.srt.batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    torch_chunk_gated_delta_rule,
    torch_gdn_gating,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="stage-b", runner_config="1-gpu-large")

C = 64


class _FakeLayer:
    """Minimal stand-in for RadixLinearAttention holding the dims/params the replay uses."""

    def __init__(self, HV, HK, K, V, A_log, dt_bias):
        self.layer_id = 0
        self.num_k_heads = HK
        self.num_v_heads = HV
        self.head_k_dim = K
        self.head_v_dim = V
        self.head_q_dim = K
        self.num_q_heads = HK
        self.q_dim = HK * K
        self.k_dim = HK * K
        self.v_dim = HV * V
        self.A_log = A_log
        self.dt_bias = dt_bias


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGDNChunkReplayDecode(unittest.TestCase):
    """Differentiable incremental chunk-replay decode (batch-invariant GDN path).

    Each decode token must reproduce, bit-for-bit, row `position` of a single full-sequence
    `torch_chunk_gated_delta_rule` forward (what Megatron computes when teacher-forcing the
    whole prompt+response). This depends on the fp32 bmm routing to the M-invariant Triton
    kernel (no cuBLAS short-circuit). Also verifies gradients flow (autograd through the same
    production graph) with no NaNs.
    """

    def _make_backend(self):
        be = GDNAttnBackend.__new__(GDNAttnBackend)
        be.gdn_replay_cache = {}
        return be

    def _full_ref(self, mixed_all, a_all, b_all, layer):
        T = mixed_all.shape[0]
        qf, kf, vf = torch.split(
            mixed_all, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
        )
        q = qf.view(1, T, layer.num_k_heads, layer.head_k_dim)
        k = kf.view(1, T, layer.num_k_heads, layer.head_k_dim)
        v = vf.view(1, T, layer.num_v_heads, layer.head_v_dim)
        g, beta = torch_gdn_gating(layer.A_log, a_all, b_all, layer.dt_bias)
        core, _, _ = torch_chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            ssm_states=None,
            cache_indices=torch.zeros(1, dtype=torch.long, device="cuda"),
            query_start_loc=torch.tensor([0, T], dtype=torch.int32, device="cuda"),
        )
        return core[0]  # [T, HV, V]

    def test_bit_identical_to_full_sequence(self):
        with set_batch_invariant_mode(True):
            HV, HK, K, V = 12, 4, 128, 128  # GQA repeat 3, per-rank TP=4 shapes
            A_log = torch.randn(HV, device="cuda")
            dt_bias = torch.randn(HV, device="cuda")
            layer = _FakeLayer(HV, HK, K, V, A_log, dt_bias)
            dim = layer.q_dim + layer.k_dim + layer.v_dim
            # prompt_len % 64 both == 0 and != 0; single crossing and multi-crossing.
            for pl, nd in [(64, 8), (60, 8), (30, 20), (128, 10), (63, 4), (1, 70), (127, 4)]:
                for seed in range(2):
                    torch.manual_seed(seed * 977 + pl)
                    T = pl + nd
                    mixed = torch.randn(T, dim, device="cuda", dtype=torch.bfloat16)
                    a = torch.randn(T, HV, device="cuda", dtype=torch.bfloat16)
                    b = torch.randn(T, HV, device="cuda", dtype=torch.bfloat16)
                    ref = self._full_ref(mixed, a, b, layer)

                    be = self._make_backend()
                    cidx = torch.tensor([7], dtype=torch.long, device="cuda")
                    be._seed_gdn_replay_cache(
                        layer,
                        mixed[:pl],
                        a[:pl],
                        b[:pl],
                        torch.tensor([0, pl], dtype=torch.int32, device="cuda"),
                        cidx,
                        None,
                    )
                    for p in range(pl, T):
                        out = be._gdn_chunk_replay_decode(
                            layer, mixed[p : p + 1], a[p : p + 1], b[p : p + 1], cidx
                        )
                        self.assertTrue(
                            torch.equal(out[0, 0], ref[p]),
                            msg=f"pl={pl} nd={nd} seed={seed} token={p} not bit-identical",
                        )

    def test_pad_slot_returns_zeros(self):
        with set_batch_invariant_mode(True):
            HV, HK, K, V = 12, 4, 128, 128
            layer = _FakeLayer(
                HV, HK, K, V, torch.randn(HV, device="cuda"), torch.randn(HV, device="cuda")
            )
            dim = layer.q_dim + layer.k_dim + layer.v_dim
            be = self._make_backend()
            out = be._gdn_chunk_replay_decode(
                layer,
                torch.randn(1, dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(1, HV, device="cuda", dtype=torch.bfloat16),
                torch.randn(1, HV, device="cuda", dtype=torch.bfloat16),
                torch.tensor([PAD_SLOT_ID], dtype=torch.long, device="cuda"),
            )
            self.assertTrue(bool((out == 0).all()))
            self.assertEqual(len(be.gdn_replay_cache), 0)

    def test_differentiable(self):
        with set_batch_invariant_mode(True):
            HV, HK, K, V = 4, 2, 128, 128
            A_log = torch.randn(HV, device="cuda")
            dt_bias = torch.randn(HV, device="cuda")
            layer = _FakeLayer(HV, HK, K, V, A_log, dt_bias)
            dim = layer.q_dim + layer.k_dim + layer.v_dim
            pl, nd = 60, 6
            T = pl + nd
            torch.manual_seed(0)
            base = torch.randn(T, dim, device="cuda", dtype=torch.float32)
            a = torch.randn(T, HV, device="cuda", dtype=torch.float32)
            b = torch.randn(T, HV, device="cuda", dtype=torch.float32)

            mixed = base.clone().requires_grad_(True)
            be = self._make_backend()
            cidx = torch.tensor([3], dtype=torch.long, device="cuda")
            be._seed_gdn_replay_cache(
                layer,
                mixed[:pl].to(torch.bfloat16),
                a[:pl],
                b[:pl],
                torch.tensor([0, pl], dtype=torch.int32, device="cuda"),
                cidx,
                None,
            )
            p_target = pl + 3
            out = None
            for p in range(pl, p_target + 1):
                o = be._gdn_chunk_replay_decode(
                    layer, mixed[p : p + 1].to(torch.bfloat16), a[p : p + 1], b[p : p + 1], cidx
                )
                if p == p_target:
                    out = o
            out.float().pow(2).sum().backward()
            self.assertIsNotNone(mixed.grad)
            self.assertFalse(torch.isnan(mixed.grad).any())
            # gradient reaches the partial-chunk rows since the last 64-boundary
            comp = (pl // C) * C
            self.assertGreater((mixed.grad[comp : p_target + 1].abs().sum(-1) > 0).sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
