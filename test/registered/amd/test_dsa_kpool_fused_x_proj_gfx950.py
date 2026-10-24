"""gfx950 k-pool indexer: the stacked x projection used by target verify."""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    INDEX_HEAD_DIM,
    kpool_write_tail_and_maybe_compress,
)
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

HIDDEN, N_HEADS, Q_LORA, TOKENS = 4096, 32, 1024, 6


class _Linear:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x):
        return F.linear(x, self.weight), None


class _LoRALinear(_Linear):
    """A LoRA-wrapped linear as the indexer sees it: base weight plus a delta in forward."""

    set_lora = True

    def __init__(self, weight, delta):
        super().__init__(weight)
        self.delta = delta

    def __call__(self, x):
        return F.linear(x, self.weight + self.delta.to(self.weight.dtype)), None


def _eager(method):
    return getattr(method, "_torchdynamo_orig_callable", method)


def _bf16_weight(*shape, dtype=torch.bfloat16):
    return (torch.randn(*shape, device="cuda") * 0.02).bfloat16().to(dtype)


def _indexer() -> IndexerKPool:
    """The attributes the target-verify projections read, as a GLM-5.3-Flash NoPE indexer."""
    indexer = IndexerKPool.__new__(IndexerKPool)
    torch.nn.Module.__init__(indexer)
    indexer.layer_id = 0
    indexer.head_dim = INDEX_HEAD_DIM
    indexer.n_heads = N_HEADS
    indexer.rope_head_dim = 0
    indexer.skip_rope = True
    indexer.wq_b = _Linear(_bf16_weight(N_HEADS * INDEX_HEAD_DIM, Q_LORA))
    indexer.wk = _Linear(_bf16_weight(INDEX_HEAD_DIM, HIDDEN))
    # held in fp32, loaded from a bf16 checkpoint
    indexer.weights_proj = _Linear(_bf16_weight(N_HEADS, HIDDEN, dtype=torch.float32))
    indexer.index_kpool_compress_gate = torch.nn.Parameter(
        _bf16_weight(INDEX_HEAD_DIM, HIDDEN), requires_grad=False
    )
    indexer.k_norm = LayerNorm(INDEX_HEAD_DIM, dtype=torch.float32).cuda()
    with torch.no_grad():
        indexer.k_norm.weight.uniform_(0.5, 1.5)
        indexer.k_norm.bias.uniform_(-0.1, 0.1)
    indexer.x_proj_weight = None
    return indexer


@unittest.skipUnless(
    torch.cuda.is_available() and is_gfx95_supported(), "requires gfx950"
)
class TestKPoolFusedXProj(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    @torch.no_grad()
    def test_fused_matches_unfused(self):
        indexer = _indexer()
        indexer.fuse_x_projections()
        self.assertIsNotNone(indexer.x_proj_weight)
        x = torch.randn(TOKENS, HIDDEN, device="cuda").bfloat16()
        q_lora = torch.randn(TOKENS, Q_LORA, device="cuda").bfloat16()

        query, key, gate, head = indexer._get_q_k_fused_x_proj(q_lora, x)
        ref_query, ref_key, _, _ = indexer._get_q_k_bf16(
            q_lora, x, positions=None, enable_dual_stream=False, forward_batch=None
        )
        ref_gate = indexer._compute_gate_score_if_missing(x, None)
        ref_head = _eager(IndexerKPool._project_head_gates)(indexer, x)

        self.assertTrue(torch.equal(query, ref_query))
        # the fused path feeds LayerNorm the fp32 projection; the reference rounds it to bf16 first
        torch.testing.assert_close(key.float(), ref_key.float(), atol=2e-2, rtol=2e-2)
        # the k-pool tail stores the score in bf16, which is where the reference rounds it
        torch.testing.assert_close(
            gate.bfloat16().float(), ref_gate.float(), atol=1e-3, rtol=1.6e-2
        )
        torch.testing.assert_close(head, ref_head, atol=1e-5, rtol=1e-4)

    @torch.no_grad()
    def test_refresh_keeps_the_address(self):
        indexer = _indexer()
        indexer.fuse_x_projections()
        fused = indexer.x_proj_weight
        address = fused.data_ptr()
        d = INDEX_HEAD_DIM
        # a partial reload writes the parameters in place, as the weight loaders do
        indexer.wk.weight.copy_(_bf16_weight(d, HIDDEN))
        indexer.weights_proj.weight.copy_(
            _bf16_weight(N_HEADS, HIDDEN, dtype=torch.float32)
        )
        indexer.fuse_x_projections()
        self.assertIs(indexer.x_proj_weight, fused)
        self.assertEqual(indexer.x_proj_weight.data_ptr(), address)
        self.assertTrue(torch.equal(fused[:d], indexer.wk.weight))
        self.assertTrue(
            torch.equal(fused[2 * d :], indexer.weights_proj.weight.bfloat16())
        )

    @torch.no_grad()
    def test_not_fused_when_weights_proj_is_not_bf16_exact(self):
        indexer = _indexer()
        indexer.weights_proj.weight.add_(1e-6)
        indexer.fuse_x_projections()
        self.assertIsNone(indexer.x_proj_weight)

    @torch.no_grad()
    def test_reload_that_is_not_bf16_exact_is_rejected(self):
        indexer = _indexer()
        indexer.fuse_x_projections()
        stacked = indexer.x_proj_weight.clone()
        indexer.weights_proj.weight.add_(1e-6)
        with self.assertRaisesRegex(ValueError, "not bf16-exact"):
            indexer.fuse_x_projections()
        self.assertTrue(torch.equal(indexer.x_proj_weight, stacked))

    @torch.no_grad()
    def test_lora_adapter_takes_the_unfused_path(self):
        x = torch.randn(TOKENS, HIDDEN, device="cuda").bfloat16()
        q_lora = torch.randn(TOKENS, Q_LORA, device="cuda").bfloat16()
        for name in ("wk", "weights_proj"):
            with self.subTest(adapter=name):
                indexer = _indexer()
                indexer.fuse_x_projections()
                self.assertTrue(indexer._use_fused_x_proj())
                base = getattr(indexer, name).weight
                setattr(indexer, name, _LoRALinear(base, torch.randn_like(base) * 0.02))
                self.assertFalse(indexer._use_fused_x_proj())
                # the stack still holds the base weights, so it would drop the delta
                fused = indexer._get_q_k_fused_x_proj(q_lora, x)
                if name == "wk":
                    got, want = (
                        fused[1],
                        indexer._get_q_k_bf16(
                            q_lora,
                            x,
                            positions=None,
                            enable_dual_stream=False,
                            forward_batch=None,
                        )[1],
                    )
                else:
                    got, want = (
                        fused[3],
                        _eager(IndexerKPool._project_head_gates)(indexer, x),
                    )
                self.assertFalse(
                    torch.allclose(got.float(), want.float(), atol=2e-2, rtol=2e-2)
                )

    @torch.no_grad()
    def test_tail_write_accepts_a_column_slice_score(self):
        pool_size, n = 4, TOKENS
        pool = type(
            "Pool",
            (),
            dict(
                page_size=256,
                index_head_dim=INDEX_HEAD_DIM,
                index_page_size=64,
                index_kpool=pool_size,
                tail_extra_slots=n,
                quant_block_size=128,
            ),
        )()
        tail_size = pool_size + n
        key = torch.randn(n, INDEX_HEAD_DIM, device="cuda").bfloat16()
        # the score as the fused path hands it over: columns d:2d of the fp32 projection
        proj = torch.randn(n, 2 * INDEX_HEAD_DIM + N_HEADS, device="cuda")
        score_slice = proj[:, INDEX_HEAD_DIM : 2 * INDEX_HEAD_DIM]
        self.assertNotEqual(score_slice.stride(0), INDEX_HEAD_DIM)
        ape = torch.randn(pool_size, INDEX_HEAD_DIM, device="cuda")
        tail_k0 = torch.randn(1, tail_size, INDEX_HEAD_DIM, device="cuda").bfloat16()
        tail_s0 = torch.randn_like(tail_k0)
        page_nbytes = 64 * INDEX_HEAD_DIM + 64 * 4

        def run(score):
            buf = torch.zeros((2, page_nbytes), dtype=torch.uint8, device="cuda")
            tail_k, tail_s = tail_k0.clone(), tail_s0.clone()
            kpool_write_tail_and_maybe_compress(
                pool=pool,
                buf=buf,
                key=key,
                score=score,
                tail_k=tail_k,
                tail_score=tail_s,
                ape=ape,
                req_pool_indices=torch.zeros(1, dtype=torch.int64, device="cuda"),
                write_start=torch.tensor([3], dtype=torch.int32, device="cuda"),
                tail_logical_start=torch.zeros(1, dtype=torch.int32, device="cuda"),
                write_loc=torch.tensor([[0, 1]], dtype=torch.int64, device="cuda"),
                out_cache_loc=torch.arange(1, n + 1, dtype=torch.int64, device="cuda"),
                num_draft_tokens=n,
                round_scale=False,
            )
            return buf, tail_k, tail_s

        got = run(score_slice)
        want = run(score_slice.contiguous())
        for a, b in zip(got, want):
            self.assertTrue(torch.equal(a, b))
        # the tail rounds the fp32 score to bf16, as the unfused bf16 GEMM output would be
        slots = [(3 + i) % tail_size for i in range(n)]
        self.assertTrue(torch.equal(got[2][0, slots], score_slice.bfloat16()))


if __name__ == "__main__":
    unittest.main()
