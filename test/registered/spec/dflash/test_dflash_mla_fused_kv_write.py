# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_DFLASH_MLA_FUSED_KV_WRITE: the one-kernel RoPE, q, k and FP8 KV rows of forward_absorb_core's Triton
branch are bitwise equal to the stock rotary_emb + torch.cat + MLATokenToKVPool.set_kv_buffer path, and the kernel
engages only where stock runs exactly those ops."""

import types
import unittest

import torch

from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

# kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, fused_qkv_a output width (q_lora 1536 + 512 + 64)
NOPE, ROPE, QNOPE, LATENT = 512, 64, 128, 2112
POOL_SLOTS, PAGE, LAYERS, START_LAYER = 8192, 64, 5, 3
MAX_POS = 8192
DEV = "cuda"
# FP8 E4M3 edge values: the 448 / 464 rounding and saturation boundary, overflow, infinities, subnormals, signed zero.
EDGES = [448.0, 456.0, 464.0, 466.0, 480.0, 512.0, 1e4, float("inf"), -466.0, -1e4, -float("inf"),
         2.0**-10, 1.5 * 2.0**-9, 2.0**-7, 3 * 2.0**-8, -0.0, 0.0, -(2.0**-9)]


def _rope(is_neox_style=False):
    """The draft's rotary (DeepseekV2AttentionMLA rewrites YaRN to deepseek_yarn): interleaved pairs, fp32 cos | sin."""
    from sglang.srt.layers.rotary_embedding.factory import get_rope_wrapper

    return get_rope_wrapper(
        ROPE,
        rotary_dim=ROPE,
        max_position=MAX_POS,
        base=50000,
        rope_scaling={
            "rope_type": "deepseek_yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
        is_neox_style=is_neox_style,
        device=DEV,
    ).to(DEV)


def _pool(dtype=torch.float8_e4m3fn, cls=None, gen=None):
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    pool = (cls or MLATokenToKVPool)(
        size=POOL_SLOTS,
        page_size=PAGE,
        dtype=dtype,
        kv_lora_rank=NOPE,
        qk_rope_head_dim=ROPE,
        layer_num=LAYERS,
        device=DEV,
        enable_memory_saver=False,
        start_layer=START_LAYER,
    )
    if gen is not None:
        for buf in pool.kv_buffer:
            buf.copy_(torch.randint(0, 256, buf.shape, dtype=torch.uint8, device=DEV, generator=gen))
    return pool


def _inputs(seed, tokens, heads, plant, max_pos):
    """One layer's attention inputs in the strided layouts forward_absorb_prepare hands over; fresh tensors per call."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    q_full = (torch.randn(tokens, heads, QNOPE + ROPE, generator=g, device=DEV) * 2).bfloat16()
    # w_kc bmm output [H, T, 512] viewed as [T, H, 512]
    q_nope = (torch.randn(heads, tokens, NOPE, generator=g, device=DEV) * 3).bfloat16().transpose(0, 1)
    latent = (torch.randn(tokens, LATENT, generator=g, device=DEV) * 4).bfloat16()
    k_nope = (torch.randn(tokens, NOPE, generator=g, device=DEV) * 2).bfloat16().unsqueeze(1)
    q_pe = q_full[..., QNOPE:]
    k_pe = latent[:, LATENT - ROPE :].unsqueeze(1)
    if plant:
        edges = torch.tensor(EDGES, device=DEV)
        k_nope.view(-1)[7 : 7 + edges.numel()] = edges.bfloat16()
        k_pe[tokens - 1, 0, 3 : 3 + edges.numel() // 2] = edges[: edges.numel() // 2].bfloat16()
    # Distinct slots; slot 0 is the padding slot.
    loc = torch.randperm(POOL_SLOTS - 1, generator=g, device=DEV)[:tokens] + 1
    positions = torch.randint(0, max_pos, (tokens,), generator=g, device=DEV)
    return dict(q_nope=q_nope, q_pe=q_pe, k_nope=k_nope, k_pe=k_pe, loc=loc, positions=positions)


def _stock(pool, layer_id, x, rope):
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc

    q_pe, k_pe = x["q_pe"], x["k_pe"]
    if rope is not None:
        q_pe, k_pe = rope(x["positions"], q_pe, k_pe)
    q = torch.cat([x["q_nope"], q_pe], dim=-1)
    k = torch.cat([x["k_nope"], k_pe], dim=-1)
    pool.set_kv_buffer(types.SimpleNamespace(layer_id=layer_id), KVWriteLoc(x["loc"]), k, x["k_nope"])
    return q, k


def _fused(pool, layer_id, x, rope):
    from sglang.kernels.ops.kvcache.mla_fused_kv_write import mla_concat_qk_store_fp8_kv
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
        _mla_fused_kv_write_folds_rope,
        _mla_fused_kv_write_rows,
    )

    rows = _mla_fused_kv_write_rows(
        pool=pool, layer_id=layer_id, attention_backend="triton", dcp_size=1, k_scale=None
    )
    assert rows is not None and (rope is None or _mla_fused_kv_write_folds_rope(rope))
    return mla_concat_qk_store_fp8_kv(
        q_nope=x["q_nope"],
        q_pe=x["q_pe"],
        k_nope=x["k_nope"],
        k_pe=x["k_pe"],
        kv_rows=rows,
        loc=x["loc"],
        positions=x["positions"],
        cos_sin_cache=None if rope is None else rope.cos_sin_cache,
    )


def _bytes(x):
    return x.contiguous().view(torch.uint8)


class TestDFlashMlaFusedKvWrite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        cls.rope = _rope()

    @classmethod
    def tearDownClass(cls):
        reset_context()

    def _differences(self, seed, make_inputs, rope):
        """Byte differences between the stock and fused paths: q, k and every layer of the whole pool."""
        gen = torch.Generator(device=DEV).manual_seed(10_000 + seed)
        ref_pool = _pool(gen=gen)
        got_pool = _pool()
        for r, g in zip(ref_pool.kv_buffer, got_pool.kv_buffer):
            g.copy_(r)
        layer_id = START_LAYER + 1
        q_ref, k_ref = _stock(ref_pool, layer_id, make_inputs(), rope)
        q_got, k_got = _fused(got_pool, layer_id, make_inputs(), rope)
        torch.cuda.synchronize()
        diffs = {
            "q": int((_bytes(q_ref) != _bytes(q_got)).sum()),
            "k": int((_bytes(k_ref) != _bytes(k_got)).sum()),
        }
        for i, (r, g) in enumerate(zip(ref_pool.kv_buffer, got_pool.kv_buffer)):
            diffs[f"kv layer {START_LAYER + i}"] = int((r != g).sum())
        return {name: n for name, n in diffs.items() if n}

    def _matrix(self, rope):
        """76 cases, 0 differing bytes: 64 seeds at the draft shape (8 rows, 6 local heads; every 4th with FP8 edge
        values planted) + 1 / 16 rows and 12 heads with edge values."""
        cases = [(s, 8, 6, s % 4 == 0) for s in range(64)]
        cases += [(100 + s, t, h, True) for s in range(4) for t, h in ((1, 6), (16, 6), (8, 12))]
        self.assertEqual(len(cases), 76)
        max_pos = self.rope.cos_sin_cache.shape[0]
        bad = []
        for seed, tokens, heads, plant in cases:
            diffs = self._differences(
                seed, lambda: _inputs(seed, tokens, heads, plant, max_pos), rope
            )
            if diffs:
                bad.append(f"seed {seed} T{tokens} H{heads} plant={plant}: {diffs}")
        self.assertEqual(bad, [], f"{len(bad)}/76 cases differ")

    def test_matrix_bitwise(self):
        self._matrix(rope=None)

    def test_matrix_bitwise_rope(self):
        """As test_matrix_bitwise with the RoPE folded in, against the draft's YaRN rotary_emb in place."""
        self._matrix(rope=self.rope)

    def test_rope_bitwise_at_scale(self):
        """3.7M rotated pairs (4 x 4096 rows x 7 vectors x 32 pairs) at positions over the whole cache: a contraction
        form that differs from fused_rope_kernel's in a few ulps of rare pairs cannot pass."""
        max_pos = self.rope.cos_sin_cache.shape[0]
        for seed in range(4):
            with self.subTest(seed=seed):
                diffs = self._differences(
                    200 + seed, lambda: _inputs(200 + seed, 4096, 6, False, max_pos), self.rope
                )
                self.assertEqual(diffs, {})

    def test_fp8_store_exhaustive(self):
        """Every BF16 bit pattern (NaN, infinities, subnormals, signed zero) in both the nope and the rope part of
        the KV row: the fused store == torch's BF16 -> float8_e4m3fn cast in set_kv_buffer."""
        pats = torch.arange(65536, dtype=torch.int32, device=DEV).to(torch.int16).view(torch.bfloat16)
        tokens = 65536 // ROPE

        def make_inputs():
            return dict(
                q_nope=pats.repeat(NOPE // ROPE).flip(0).view(tokens, 1, NOPE),
                q_pe=pats.flip(0).view(tokens, 1, ROPE),
                k_nope=pats.repeat(NOPE // ROPE).view(tokens, 1, NOPE),
                k_pe=pats.view(tokens, 1, ROPE),
                loc=torch.arange(1, tokens + 1, dtype=torch.int64, device=DEV),
                positions=torch.zeros(tokens, dtype=torch.int64, device=DEV),
            )

        self.assertEqual(self._differences(0, make_inputs, None), {})

    def test_rows_only_for_the_plain_fp8_pool(self):
        """The fused write engages only where the Triton backend stores through plain MLATokenToKVPool.set_kv_buffer's
        FP8 cast + scatter, and returns the addressed layer's buffer (pool layers start at START_LAYER)."""
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
            _mla_fused_kv_write_rows,
        )

        class _DerivedPool(MLATokenToKVPool):
            pass

        fp8 = _pool()
        physical = _pool()
        physical.requires_physical_write_loc = True
        ok = dict(pool=fp8, layer_id=START_LAYER + 2, attention_backend="triton", dcp_size=1, k_scale=None)
        self.assertEqual(_mla_fused_kv_write_rows(**ok).data_ptr(), fp8.kv_buffer[2].data_ptr())
        refused = {
            "flashinfer backend": dict(attention_backend="flashinfer"),
            "dcp 2": dict(dcp_size=2),
            "k_scale": dict(k_scale=torch.ones(1, device=DEV)),
            "bf16 pool": dict(pool=_pool(dtype=torch.bfloat16)),
            "derived pool": dict(pool=_pool(cls=_DerivedPool)),
            "physical-loc pool": dict(pool=physical),
        }
        for name, change in refused.items():
            with self.subTest(name):
                self.assertIsNone(_mla_fused_kv_write_rows(**{**ok, **change}))

    def test_rope_folds_only_for_the_interleaved_fp32_kernel(self):
        """The RoPE is left to the fused kernel only for the arithmetic it reproduces."""
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
            _mla_fused_kv_write_folds_rope,
        )

        self.assertTrue(_mla_fused_kv_write_folds_rope(self.rope))
        bf16_cache = _rope()
        bf16_cache.cos_sin_cache = bf16_cache.cos_sin_cache.bfloat16()
        for name, rope in {"no rotary": None, "neox pairs": _rope(is_neox_style=True), "bf16 cache": bf16_cache}.items():
            with self.subTest(name):
                self.assertFalse(_mla_fused_kv_write_folds_rope(rope))


if __name__ == "__main__":
    unittest.main()
