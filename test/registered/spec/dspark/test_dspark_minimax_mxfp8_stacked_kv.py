import unittest
from types import MethodType, SimpleNamespace

import torch

from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.srt.models.dspark_minimax import (
    DSparkMiniMaxAttention,
    DSparkMiniMaxDraftModel,
)
from sglang.srt.models.minimax_common.amd.dspark_minimax_hip import (
    BlockFp8CtxKvWriter,
)
from sglang.srt.runtime_context import get_platform
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="4-gpu-b200")
# backend-specific: the gfx950 block-fp8 ctx-KV writer and fused markov step bias.
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


class Projection:
    def __init__(self, method, weight, scales, bias):
        self.quant_method = method
        self.weight = weight
        self.weight_scale_inv_swizzled = scales
        self.bias = bias

    def __call__(self, x):
        return self.quant_method.apply(self, x, self.bias), None


@unittest.skipUnless(
    torch.cuda.is_available() and get_platform().is_blackwell,
    "MXFP8 FlashInfer requires Blackwell",
)
class TestMiniMaxStackedKv(CustomTestCase):
    def test_cutedsl(self):
        self._check_backend("flashinfer_cutedsl")

    def test_cutlass(self):
        self._check_backend("flashinfer_cutlass")

    @torch.inference_mode()
    def _check_backend(self, backend):
        """Aligned swizzled KV tiles preserve layer order and partial-RoPE K/V."""
        from flashinfer import block_scale_interleave

        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", fp8_gemm_runner_backend=backend)
        )
        method = Fp8LinearMethod(
            Fp8Config(
                is_checkpoint_fp8_serialized=True,
                use_mxfp8=True,
                weight_block_size=[1, 32],
            )
        )
        g = torch.Generator(device="cuda").manual_seed(42)
        for q_size, hidden in ((1024, 6144), (128, 160)):
            layers = []
            for _ in range(5):
                n = q_size + 256
                weight = torch.randn(n, hidden, device="cuda", generator=g).to(
                    torch.float8_e4m3fn
                )
                scales = torch.randint(
                    121,
                    129,
                    (n, hidden // 32),
                    device="cuda",
                    generator=g,
                    dtype=torch.uint8,
                )
                norm = GemmaRMSNorm(128, eps=1e-6).to(
                    device="cuda", dtype=torch.bfloat16
                )
                norm.weight.copy_(torch.randn(128, device="cuda", generator=g))
                attn = SimpleNamespace(
                    qkv_proj=Projection(
                        method, weight, block_scale_interleave(scales), None
                    ),
                    q_size=q_size,
                    kv_size=128,
                    num_kv_heads=1,
                    head_dim=128,
                    k_norm=norm,
                    rotary_emb=get_rope(
                        128,
                        rotary_dim=64,
                        max_position=4096,
                        base=10000.0,
                        is_neox_style=True,
                    ).cuda(),
                )
                for name in ("kv_proj_only", "apply_k_norm", "apply_k_rope"):
                    setattr(
                        attn,
                        name,
                        MethodType(getattr(DSparkMiniMaxAttention, name), attn),
                    )
                layers.append(SimpleNamespace(self_attn=attn))
            model = SimpleNamespace(layers=layers)
            stacked = DSparkMiniMaxDraftModel._stacked_ctx_kv_params(model)
            self.assertIsNotNone(stacked)
            for rows in (1, 8, 64, 512):
                with self.subTest(hidden=hidden, rows=rows):
                    x = torch.randn(
                        rows, hidden, device="cuda", dtype=torch.bfloat16, generator=g
                    )
                    positions = torch.arange(rows, device="cuda") + 17
                    ks, vs = DSparkMiniMaxDraftModel._project_ctx_kv_stacked(
                        model, ctx_hidden=x, positions=positions, stacked=stacked
                    )
                    for layer, k, v in zip(layers, ks, vs):
                        attn = layer.self_attn
                        rk, rv = attn.kv_proj_only(x)
                        rk = attn.apply_k_rope(positions, attn.apply_k_norm(rk))
                        torch.testing.assert_close(k.flatten(1), rk, rtol=0, atol=0)
                        torch.testing.assert_close(v.flatten(1), rv, rtol=0, atol=0)
            for attr, value in (("q_size", 64), ("kv_size", 32)):
                original = getattr(layers[0].self_attn, attr)
                setattr(layers[0].self_attn, attr, value)
                model._mxfp8_ctx_kv_cache = False
                self.assertIsNone(DSparkMiniMaxDraftModel._stacked_ctx_kv_params(model))
                setattr(layers[0].self_attn, attr, original)
            layers[0].self_attn.qkv_proj.bias = torch.zeros(n, device="cuda")
            model._mxfp8_ctx_kv_cache = False
            self.assertIsNone(DSparkMiniMaxDraftModel._stacked_ctx_kv_params(model))


class _BlockFp8Projection:
    """A qkv_proj converted from MXFP8 on gfx950: a bpreshuffled block-fp8 weight
    carrying its rowwise-fp8 copy, run by aiter_w8a8_block_fp8_linear."""

    def __init__(self, n, k, g):
        from aiter.ops.shuffle import shuffle_weight

        def fp8(rows):
            w = torch.randn(rows, k, device="cuda", generator=g) * 0.5
            return shuffle_weight(w.to(torch.float8_e4m3fn), (16, 16))

        self.quant_method = SimpleNamespace(weight_block_size=[128, 128])
        self.bias = None
        self.weight = fp8(n)
        self.weight_scale_inv = (
            torch.rand(n // 128, k // 128, device="cuda", generator=g) * 0.01 + 0.001
        )
        self.weight._ptpc_weight = fp8(n)
        self.weight._ptpc_scale = (
            torch.rand(n, 1, device="cuda", generator=g) * 0.01 + 0.001
        )

    def __call__(self, x):
        out = fp8_utils.aiter_w8a8_block_fp8_linear(
            x, self.weight, [128, 128], self.weight_scale_inv
        )
        return out, None


class _Fc:
    def __init__(self, n, k, g):
        self.input_size = k
        self.weight = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(
            torch.bfloat16
        )

    def __call__(self, x):
        return x @ self.weight.t(), None


@unittest.skipUnless(
    torch.cuda.is_available()
    and is_gfx95_supported()
    and fp8_utils._use_aiter_bpreshuffle_gfx95,
    "the block-fp8 ctx-KV path is gfx95 + aiter bpreshuffle",
)
class TestMiniMaxBlockFp8CtxKv(CustomTestCase):
    HIDDEN, Q_SIZE, LAYERS = 512, 256, 5

    def _draft(self, g):
        hidden = self.HIDDEN
        rope = get_rope(
            128, rotary_dim=64, max_position=8192, base=1e7, is_neox_style=True
        ).cuda()
        layers = []
        for i in range(self.LAYERS):
            norm = GemmaRMSNorm(128, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
            norm.weight.copy_(torch.randn(128, device="cuda", generator=g) * 0.3)
            attn = SimpleNamespace(
                qkv_proj=_BlockFp8Projection(self.Q_SIZE + 256, hidden, g),
                attn=SimpleNamespace(layer_id=i, k_scale=None, v_scale=None),
                q_size=self.Q_SIZE,
                kv_size=128,
                num_kv_heads=1,
                head_dim=128,
                k_norm=norm,
                rotary_emb=rope,
            )
            for name in ("kv_proj_only", "apply_k_norm", "apply_k_rope"):
                setattr(
                    attn, name, MethodType(getattr(DSparkMiniMaxAttention, name), attn)
                )
            layers.append(SimpleNamespace(self_attn=attn))
        hidden_norm = GemmaRMSNorm(hidden, eps=1e-6).to(
            device="cuda", dtype=torch.bfloat16
        )
        hidden_norm.weight.copy_(torch.randn(hidden, device="cuda", generator=g) * 0.3)
        model = SimpleNamespace(
            layers=layers,
            fc=_Fc(hidden, 5 * hidden, g),
            _fc_gatherer=lambda x: x,
            # block-fp8 qkv_proj rules out the shared bf16 / MXFP8 fused write, as on gfx950
            _build_fused_kv_write_bundle=lambda pool: None,
            hidden_norm=hidden_norm,
            _fused_kv_write_cache=None,
            _mxfp8_ctx_kv_cache=False,
        )
        for cls, names in (
            (DSparkDraftMixin, ("_fused_kv_write_bundle",)),
            (
                DSparkMiniMaxDraftModel,
                ("_stacked_ctx_kv_params", "project_target_hidden"),
            ),
        ):
            for name in names:
                setattr(model, name, MethodType(getattr(cls, name), model))
        model._ctx_kv_writer = BlockFp8CtxKvWriter(model)
        return model, self._pool()

    def _pool(self):
        return MHATokenToKVPool(
            size=8192,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=128,
            layer_num=self.LAYERS,
            device="cuda",
            enable_memory_saver=False,
            enable_alt_stream=False,
        )

    @staticmethod
    def _cache(pool, layers):
        return torch.stack(
            [
                torch.cat([pool.get_key_buffer(i), pool.get_value_buffer(i)], dim=-1)
                for i in range(layers)
            ]
        )

    @torch.inference_mode()
    def test_stacked_injection_writes_the_per_layer_cache(self):
        """The stacked injection writes the K/V cache of the per-layer one: its norm,
        rope and store bit for bit, its one K/V GEMM within a bf16 ulp of the qkv
        GEMMs (a different N may accumulate in another order)."""
        g = torch.Generator(device="cuda").manual_seed(0)
        model, pool = self._draft(g)
        self.assertIsNotNone(model._ctx_kv_writer._ctx_kv(pool))
        # rowwise GEMM with committed rows, block GEMM with every row written
        for bs, verify in ((2, 8), (40, 8)):
            with self.subTest(rows=bs * verify):
                rows = bs * verify
                loc_2d = (
                    torch.randperm(4096, device="cuda", generator=g)[:rows] + 1
                ).view(bs, verify)
                kwargs = dict(
                    target_hidden=torch.randn(
                        rows, 5 * self.HIDDEN, device="cuda", generator=g
                    ).to(torch.bfloat16),
                    pool=pool,
                    positions=torch.randint(
                        0, 8000, (rows,), device="cuda", generator=g
                    ),
                    cache_loc=loc_2d.reshape(-1),
                )
                if bs == 2:
                    kwargs["cache_loc_2d"] = loc_2d
                    kwargs["commit_lens"] = torch.tensor(
                        [3, 8], device="cuda", dtype=torch.int32
                    )
                caches = []
                for write in (
                    DSparkDraftMixin.write_target_hidden_kv,
                    DSparkMiniMaxDraftModel.write_target_hidden_kv,
                ):
                    for i in range(self.LAYERS):
                        pool.get_key_buffer(i).zero_()
                        pool.get_value_buffer(i).zero_()
                    write(model, **kwargs)
                    caches.append(self._cache(pool, self.LAYERS).float())
                torch.testing.assert_close(caches[1], caches[0], rtol=2**-6, atol=2**-5)
        layer = model.layers[0].self_attn
        del layer.qkv_proj.weight._ptpc_weight
        self.assertIsNone(model._ctx_kv_writer._build(pool))

    @torch.inference_mode()
    def test_writer_rebuilds_for_a_new_pool(self):
        """The writer caches every layer's K/V buffer address per pool; a new pool, even
        one at a freed pool's id, must not get the old addresses."""
        g = torch.Generator(device="cuda").manual_seed(2)
        model, pool = self._draft(g)
        writer = model._ctx_kv_writer
        meta = writer._ctx_kv(pool).meta
        self.assertIs(writer._ctx_kv(pool).meta, meta)
        stale_id = id(pool)
        # keep the old buffers alive, so the new pool's sit elsewhere
        old_buffers = [*pool.k_buffer, *pool.v_buffer]
        del pool
        pool = self._pool()
        # CPython hands a freed object's address to the next one of its size
        self.assertEqual(id(pool), stale_id)
        self.assertNotEqual(
            pool.get_key_buffer(0).data_ptr(), old_buffers[0].data_ptr()
        )
        self.assertEqual(
            writer._ctx_kv(pool).meta[0, 0].item(), pool.get_key_buffer(0).data_ptr()
        )

    @torch.inference_mode()
    def test_fused_k_norm_rope_store_writes_the_separate_ops_bytes(self):
        """The fused write stores what GemmaRMSNorm, sgl_kernel's bf16-cache rope
        and the pool store write, for one and two KV heads, committed rows or a
        flat loc list."""
        from sglang.kernels.ops.speculative.dspark.dspark_gfx950 import (
            fused_kv_gemma_norm_rope_write,
        )
        from sglang.srt.mem_cache.memory_pool import KVWriteLoc

        g = torch.Generator(device="cuda").manual_seed(1)
        rope = get_rope(
            128, rotary_dim=64, max_position=8192, base=1e7, is_neox_style=True
        ).cuda()
        for heads in (1, 2):
            with self.subTest(kv_heads=heads):
                layers, rows, size = 3, 24, 128 * heads
                pool = MHATokenToKVPool(
                    size=4096,
                    page_size=1,
                    dtype=torch.bfloat16,
                    head_num=heads,
                    head_dim=128,
                    layer_num=layers,
                    device="cuda",
                    enable_memory_saver=False,
                    enable_alt_stream=False,
                )
                norms = []
                for _ in range(layers):
                    norm = GemmaRMSNorm(128, eps=1e-6).to(
                        device="cuda", dtype=torch.bfloat16
                    )
                    norm.weight.copy_(torch.randn(128, device="cuda", generator=g))
                    norms.append(norm)
                kv = (
                    torch.randn(rows, layers * 2 * size, device="cuda", generator=g).to(
                        torch.bfloat16
                    )
                    * 4
                )
                positions = torch.randint(0, 8000, (rows,), device="cuda", generator=g)
                locs = torch.randperm(4000, device="cuda", generator=g)[:rows] + 1
                commit_lens = torch.tensor([5, 8, 1], device="cuda", dtype=torch.int32)
                meta = torch.tensor(
                    [
                        [
                            pool.get_key_buffer(i).data_ptr(),
                            pool.get_value_buffer(i).data_ptr(),
                            size,
                            size,
                        ]
                        for i in range(layers)
                    ],
                    dtype=torch.int64,
                    device="cuda",
                )
                for committed in (True, False):
                    expected = []
                    for i, norm in enumerate(norms):
                        k = kv[:, 2 * i * size : (2 * i + 1) * size]
                        v = kv[:, (2 * i + 1) * size : (2 * i + 2) * size]
                        k = norm(k.reshape(-1, 128)).view(rows, size)
                        _, k = rope(positions, torch.empty_like(k), k)
                        k_buf = pool.get_key_buffer(i)
                        v_buf = pool.get_value_buffer(i)
                        k_buf.zero_()
                        v_buf.zero_()
                        k = k.view(rows, heads, 128)
                        v = v.reshape(rows, heads, 128)
                        if committed:
                            pool.set_kv_buffer_prefix_valid(
                                SimpleNamespace(layer_id=i),
                                locs.view(3, 8),
                                commit_lens,
                                k,
                                v,
                            )
                        else:
                            pool.set_kv_buffer(
                                SimpleNamespace(layer_id=i),
                                KVWriteLoc(locs, physical=True),
                                k,
                                v,
                            )
                        expected.append(torch.cat([k_buf, v_buf], dim=-1).clone())
                        k_buf.zero_()
                        v_buf.zero_()
                    fused_kv_gemma_norm_rope_write(
                        kv,
                        meta,
                        torch.stack([n.weight.data for n in norms]),
                        rope.cos_sin_cache.to(torch.bfloat16),
                        positions,
                        locs,
                        heads,
                        128,
                        1e-6,
                        commit_lens=commit_lens if committed else None,
                        locs_row_width=8 if committed else None,
                    )
                    got = self._cache(pool, layers)
                    self.assertTrue(
                        torch.equal(
                            got.view(torch.int16),
                            torch.stack(expected).view(torch.int16),
                        )
                    )


@unittest.skipUnless(is_gfx95_supported(), "the fused markov step bias is gfx950-only")
class TestMiniMaxFusedStepBiasMarkov(CustomTestCase):
    def test_minimax_head_keeps_the_greedy_kernel_and_the_bias(self):
        """The MiniMax draft's gfx950 head computes the bias VanillaMarkov does, and still
        hands upstream's one-row greedy kernel its w2: overriding step_bias alone would
        turn that kernel off."""
        from sglang.srt.models.dspark import VanillaMarkov
        from sglang.srt.models.minimax_common.amd import dspark_minimax_hip
        from sglang.srt.speculative.dspark_components.dspark_hip import (
            FusedStepBiasMarkov,
        )

        torch.manual_seed(0)
        plain = VanillaMarkov(vocab_size=5003, markov_rank=256).to(
            device="cuda", dtype=torch.bfloat16
        )
        head = dspark_minimax_hip.markov_head(plain)
        self.assertIs(type(head), FusedStepBiasMarkov)
        head.load_state_dict(plain.state_dict())
        self.assertIs(head.greedy_w2_weight(), head.markov_w2.weight)
        tokens = torch.randint(0, 5003, (3, 7), device="cuda")
        got, state = head.step_bias(tokens, hidden_states=None, state=None)
        ref, _ = plain.step_bias(tokens, hidden_states=None, state=None)
        self.assertIsNone(state)
        torch.testing.assert_close(got.float(), ref.float(), rtol=2**-7, atol=1e-6)
        # ranks the kernel cannot tile keep VanillaMarkov
        odd = VanillaMarkov(vocab_size=5003, markov_rank=96).to(
            device="cuda", dtype=torch.bfloat16
        )
        self.assertIs(dspark_minimax_hip.markov_head(odd), odd)


if __name__ == "__main__":
    unittest.main()
