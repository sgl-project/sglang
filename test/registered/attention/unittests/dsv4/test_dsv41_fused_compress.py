"""Check decode numerics and packed writes through the real cache pool."""

import types
import unittest

import torch
import torch.nn.functional as F

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

HIDDEN = 2048
HEAD_DIM = 512
# `[0] * 2 + [2] * 18 + [1] * 20` as the served model lays the ratios out; the
# ratio-2 and ratio-1 source layers are read back off the pool rather than
# assumed.
RATIOS = [0] * 2 + [2] * 18 + [1] * 20
KV_SOURCE_LAYERS = (2, 8, 14, 20)
POOL_PAGE_SIZE = 256
POOL_FULL_SIZE = 4 * POOL_PAGE_SIZE
MAX_REQS = 256
INDEX_HEAD_DIM = 128
ROPE_DIM = 64
EPS = 1e-6


def _make_pool():
    """The real `DeepSeekV4TokenToKVPool`, so the buffers, their dtypes, the page
    sizes and the pair-state ring all come from the code under test."""
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    return DeepSeekV4TokenToKVPool(
        max_num_reqs=MAX_REQS,
        swa_size=POOL_FULL_SIZE,
        c4_size=0,
        c128_size=0,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=POOL_PAGE_SIZE,
        swa_page_size=128,
        dtype=torch.float8_e4m3fn,
        c4_state_dtype=torch.float32,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=HEAD_DIM - ROPE_DIM,
        qk_rope_head_dim=ROPE_DIM,
        indexer_head_dim=INDEX_HEAD_DIM,
        layer_num=len(RATIOS),
        device="cuda",
        enable_memory_saver=False,
        compression_ratios=RATIOS,
        kv_source_layers=KV_SOURCE_LAYERS,
        full_size=POOL_FULL_SIZE,
    )


def _build(n: int, ratio: int, seed: int):
    """One decode step against the real pool."""
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        _low_ratio_compression_metadata,
    )
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
        DeepseekV41Compressor,
        DeepseekV41Indexer,
        RMSNorm,
    )
    from sglang.srt.model_loader.utils import set_default_torch_dtype

    g = torch.Generator(device="cuda").manual_seed(seed)
    with set_default_torch_dtype(torch.bfloat16):
        compressor = DeepseekV41Compressor(
            HIDDEN, HEAD_DIM, ratio, EPS, fused_compress=True
        ).cuda()
        k_norm = RMSNorm(INDEX_HEAD_DIM, EPS).cuda()
        wk = torch.nn.Linear(
            HEAD_DIM, INDEX_HEAD_DIM, bias=False, dtype=torch.bfloat16
        ).cuda()
    # Not `ones` anywhere: a constant weight cannot catch a wrong element index.
    with torch.no_grad():
        for module in (compressor, k_norm, wk):
            for param in module.parameters():
                param.copy_(torch.randn(param.shape, generator=g, device="cuda") * 0.1)
    x = torch.randn(n, HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)

    # Every token is the last of its group, so every row completes and writes.
    pos = torch.arange(n, device="cuda", dtype=torch.int64) * ratio + (ratio - 1)
    raw_out_loc = (torch.arange(n, device="cuda", dtype=torch.int32) + 1) * ratio + (
        ratio - 1
    )
    out_loc, _ = _low_ratio_compression_metadata(
        ratio, raw_out_loc.to(torch.int64) + 1, raw_out_loc
    )

    ang = torch.randn(int(pos.max()) + 2, ROPE_DIM // 2, generator=g, device="cuda")
    freqs = torch.polar(torch.ones_like(ang), ang)

    pool = _make_pool()
    layer_id = pool.sources_by_ratio[ratio][0]
    # Use the real pool: get_extra_key_buffer exposes a uint8 store as an fp8 view.
    kv_cache = pool.get_extra_key_buffer(layer_id)
    compressed_page = pool.get_extra_key_page_size(layer_id)
    index_cache = pool.get_index_k_with_scale_buffer(layer_id)
    # Ratio 1 pools nothing, so it has no pair state -- the pool asserts on the
    # accessor rather than returning an empty one.
    state = pool.get_attention_compress_states(layer_id) if ratio == 2 else None
    assert int(out_loc.max()) < kv_cache.shape[0] * compressed_page, (
        "the test's slots must fit the pool"
    )

    indexer = types.SimpleNamespace(
        owns_k=True,
        wk=wk,
        k_norm=k_norm,
        rope_head_dim=ROPE_DIM,
        index_head_dim=INDEX_HEAD_DIM,
    )
    # The real method, bound to the stand-in: it reads only `wk` and
    # `index_head_dim`, and it is what decides whether the in-tree GEMM serves `wk`.
    captured = {}

    def forward_wk(latent):
        projected = DeepseekV41Indexer.forward_wk(indexer, latent)
        captured.update(latent=latent, projected=projected)
        return projected

    indexer.forward_wk = forward_wk

    if state is not None:
        # Seeded rather than zeroed, so a completing row that fails to read its
        # partner shows up instead of pairing against zeros.
        state.kv_score_buffer.kv_score.copy_(
            torch.randn(
                state.kv_score_buffer.kv_score.shape,
                generator=g,
                device="cuda",
                dtype=torch.float32,
            )
        )
    layer = types.SimpleNamespace(
        compressor=compressor,
        indexer=indexer,
        layer_id=layer_id,
        compress_ratio=ratio,
        rope_head_dim=ROPE_DIM,
        freqs_cis=freqs,
    )
    backend = types.SimpleNamespace(
        token_to_kv_pool=pool,
        forward_metadata=types.SimpleNamespace(
            core_metadata=types.SimpleNamespace(
                raw_out_loc=raw_out_loc, c1_out_loc=out_loc, c2_out_loc=out_loc
            )
        ),
    )
    return types.SimpleNamespace(
        backend=backend,
        layer=layer,
        x=x,
        pos=pos,
        freqs=freqs,
        out_loc=out_loc,
        kv_cache=kv_cache,
        index_cache=index_cache,
        compressor=compressor,
        k_norm=k_norm,
        captured=captured,
        pair_state=(
            state.kv_score_buffer.kv_score.clone() if state is not None else None
        ),
        ring_size=state.ring_size if state is not None else 0,
        compressed_page=compressed_page,
        req=torch.arange(n, device="cuda", dtype=torch.int64),
    )


def _reference(t, ratio: int):
    """Check numerics independently, then check byte-exact packing of actual latents."""
    from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
    from sglang.kernels.ops.attention.dsv4.rope_pack_indexer import (
        rope_fake_quant_pack_indexer,
    )
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail
    from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_compressed_kv

    if ratio == 1:
        pooled = t.compressor.wkv(t.x)
    else:
        fused = t.compressor.project_fused(t.x)
        kv, score = fused[..., :HEAD_DIM], fused[..., HEAD_DIM:]
        # `translate_from_req_position_to_state_loc` for the slot `pos - 1`
        # left: the ring puts the read and the write on different rows.
        read = t.req * t.ring_size + (t.pos - 1) % t.ring_size
        partner_kv = t.pair_state[read, :HEAD_DIM]
        partner_score = t.pair_state[read, HEAD_DIM:]
        pooled = (
            torch.stack([partner_kv, kv], dim=1)
            * torch.stack([partner_score, score], dim=1).softmax(dim=1)
        ).sum(dim=1)
    live = t.out_loc > 0
    latent = t.captured["latent"]
    projected = t.captured["projected"]
    if live.any():
        normalized_input = pooled[live].bfloat16().double()
        normalized = normalized_input * torch.rsqrt(
            normalized_input.square().mean(dim=-1, keepdim=True) + EPS
        )
        expected_latent = normalized * t.compressor.norm.weight.double()
        # Pooling and normalization may differ by up to two bf16 ULPs.
        torch.testing.assert_close(
            latent[live].float(), expected_latent.float(), rtol=2**-6, atol=2**-20
        )
        expected_projection = F.linear(
            latent[live].float(), t.layer.indexer.wk.weight.float()
        )
        torch.testing.assert_close(
            projected[live].float(), expected_projection, rtol=1e-2, atol=1e-2
        )

    group_pos = t.pos & ~(ratio - 1)
    # fused_store_cache takes uint8, while the pool exposes an fp8 view;
    # compare and call any() through uint8 views of both buffers.
    kv_cache = torch.zeros(
        t.kv_cache.shape, dtype=torch.uint8, device=t.kv_cache.device
    )
    if not live.any():
        return kv_cache, torch.zeros_like(t.index_cache)
    fused_store_cache(
        input=fake_quant_compressed_kv(
            rope_tail(latent[live], t.freqs[group_pos[live]], ROPE_DIM)
        ),
        cache=kv_cache,
        indices=t.out_loc[live],
        page_size=t.compressed_page,
        type="flashmla",
    )
    index_cache = torch.zeros_like(t.index_cache)
    rope_fake_quant_pack_indexer(
        t.k_norm(projected[live]),
        t.freqs[group_pos[live]],
        ROPE_DIM,
        cache=index_cache,
        loc=t.out_loc[live],
    )
    return kv_cache, index_cache


class TestFusedLowRatioCompress(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # The pool reads the scheduler's config bags (`spec`, for the pair-ring
        # size), so they have to be published before one can be built.
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=POOL_PAGE_SIZE)
        )

    def _check_step(self, t, ratio: int):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        DeepseekV4AttnBackend._low_ratio_compress_decode_fused(
            t.backend, t.layer, t.x, t.req, t.pos
        )
        ref_kv, ref_index = _reference(t, ratio)
        torch.cuda.synchronize()

        got_kv = t.kv_cache.view(torch.uint8)
        if (t.out_loc > 0).any():
            self.assertTrue(got_kv.any(), "the main-KV write published nothing")
            self.assertTrue(t.index_cache.any(), "the index-K write published nothing")
        self.assertTrue(
            torch.equal(got_kv, ref_kv),
            f"{ratio=}: {int((got_kv != ref_kv).sum())} main-KV cache "
            f"bytes differ from the unfused chain",
        )
        self.assertTrue(
            torch.equal(t.index_cache, ref_index),
            f"{ratio=}: {int((t.index_cache != ref_index).sum())} index-K "
            f"cache bytes differ from the unfused chain",
        )

    def test_ratio_2_matches_exact_reference(self):
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
            DeepseekV41Compressor,
        )

        torch.manual_seed(0)
        hidden, head_dim, n = 5120, 512, 37
        comp = DeepseekV41Compressor(hidden, head_dim, compress_ratio=2, eps=1e-6)
        comp = comp.cuda()
        # Read wkv/wgate from either the fused [2D, K] layout or split projections;
        # both must satisfy the same numerical contract.
        if comp.use_fused_gate:
            w = comp.wkv_gate.weight
            w_kv, w_gate = w[:head_dim], w[head_dim:]
        else:
            w_kv, w_gate = comp.wkv.weight, comp.wgate.weight
        self.assertEqual(w_kv.dtype, torch.bfloat16)
        self.assertEqual(w_gate.dtype, torch.bfloat16)

        x = torch.randn(n, hidden, device="cuda", dtype=torch.bfloat16)
        kv, score = comp.project(x)

        # bf16 x bf16 products are exact in fp32 and fp64, so an fp64 GEMM over
        # the same inputs is the reference up to the fp32 accumulation order.
        x64 = x.double().cpu()
        ref_kv = x64 @ w_kv.double().cpu().t()
        ref_score = x64 @ w_gate.double().cpu().t()

        for out, ref in ((kv, ref_kv), (score, ref_score)):
            self.assertEqual(out.dtype, torch.float32)
            torch.testing.assert_close(out.double().cpu(), ref, rtol=1e-4, atol=1e-4)
            # Well below what a bf16-rounded output could reach.
            bf16_err = (out.bfloat16().double().cpu() - ref).abs().max()
            self.assertLess((out.double().cpu() - ref).abs().max(), bf16_err)

    def test_matches_the_unfused_chain(self):
        for ratio in (1, 2):
            for n in (1, 64):
                with self.subTest(ratio=ratio, n=n):
                    self._check_step(_build(n, ratio, seed=100 + n + ratio), ratio)


if __name__ == "__main__":
    unittest.main()
