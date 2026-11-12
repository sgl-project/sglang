"""MLU latent attention and one-shot/chunked MHA numerical coverage."""

import unittest
from types import SimpleNamespace as NS

import torch
import torch_mlu  # noqa: F401

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlu.attention.mla_backend import MLUMLABackend
from sglang.srt.hardware_backend.mlu.memory_pool import MLUMLATokenToKVPool
from sglang.srt.layers.attention.merge_state import merge_state
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.deepseek_common.attention_backend_handler import (
    handle_attention_mlu,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=60, suite="pr-test-1-mlu")


class TestMLAMLU(CustomTestCase):
    def setUp(self):
        super().setUp()
        parallel = get_parallel().override(
            attn_dcp_size=1, attn_dcp_rank=0, attn_cp_size=1, dcp_enabled=False
        )
        parallel.__enter__()
        self.addCleanup(parallel.__exit__, None, None, None)

    def _fixture(self, qlens, lens, mode=ForwardMode.EXTEND, mixed=None):
        torch.manual_seed(11)
        pool = MLUMLATokenToKVPool(
            size=256,
            page_size=16,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=1,
            device="mlu",
            enable_memory_saver=False,
        )
        req = torch.zeros(2, 64, dtype=torch.int32, device="mlu")
        for i, pages in enumerate(((5, 1, 8, 2), (7, 3, 9, 6))):
            req[i] = torch.cat(
                [torch.arange(p * 16, (p + 1) * 16, device="mlu") for p in pages]
            )
        with get_parallel().override(attn_dcp_size=1, attn_cp_size=1):
            backend = MLUMLABackend(
                NS(
                    device="mlu",
                    page_size=16,
                    model_config=NS(
                        context_len=64,
                        kv_lora_rank=512,
                        hf_config=NS(architectures=["DeepseekV2ForCausalLM"]),
                    ),
                    req_to_token_pool=NS(req_to_token=req),
                    token_to_kv_pool=pool,
                    kv_cache_dtype=torch.bfloat16,
                )
            )
        layer = NS(
            layer_id=0,
            tp_q_head_num=8,
            tp_k_head_num=1,
            tp_v_head_num=1,
            qk_head_dim=576,
            v_head_dim=512,
            scaling=192**-0.5,
            logit_cap=0,
            is_cross_attention=False,
            attn_type=AttentionType.DECODER,
        )
        cache = torch.randn(sum(lens), 1, 576, device="mlu", dtype=torch.bfloat16) * 0.2
        slots = torch.cat([req[i, :n] for i, n in enumerate(lens)])
        pool.set_kv_buffer(layer, slots, cache, None)
        loc = torch.cat(
            [req[i, n - m : n] for i, (m, n) in enumerate(zip(qlens, lens))]
        )
        fb = NS(
            batch_size=2,
            req_pool_indices=torch.arange(2, device="mlu"),
            seq_lens=torch.tensor(lens, device="mlu"),
            seq_lens_cpu=torch.tensor(lens),
            extend_seq_lens=torch.tensor(qlens, device="mlu"),
            extend_seq_lens_cpu=qlens,
            extend_prefix_lens_cpu=[n - m for n, m in zip(lens, qlens)],
            mixed_prefill_reqs=mixed,
            forward_mode=mode,
            out_cache_loc=loc,
            attn_attend_prefix_cache=None,
            mha_one_shot=None,
            mha_return_lse=None,
        )
        backend.init_forward_metadata(fb)
        q = torch.randn(sum(qlens), 8, 576, device="mlu", dtype=torch.bfloat16) * 0.2
        chunks = cache.split(lens)
        current = torch.cat([x[-m:] for x, m in zip(chunks, qlens)])
        pool.set_kv_buffer(layer, loc, torch.zeros_like(current), None)
        return NS(
            pool=pool,
            backend=backend,
            layer=layer,
            fb=fb,
            q=q,
            cache=cache,
            current=current,
            lens=lens,
            qlens=qlens,
        )

    @staticmethod
    def _reference(q, k, v, qlens, lens, scale, return_lse=False):
        outputs = []
        lses = []
        for qi, ki, vi in zip(
            q.cpu().float().split(qlens),
            k.cpu().float().split(lens),
            v.cpu().float().split(lens),
        ):
            logits = (
                torch.einsum("qhd,khd->hqk", qi, ki.expand(-1, qi.shape[1], -1)) * scale
            )
            mask = torch.arange(ki.shape[0])[None, :] > (
                torch.arange(qi.shape[0])[:, None] + ki.shape[0] - qi.shape[0]
            )
            logits.masked_fill_(mask.unsqueeze(0), -torch.inf)
            lses.append(logits.logsumexp(-1).T)
            outputs.append(
                torch.einsum(
                    "hqk,khd->qhd", logits.softmax(-1), vi.expand(-1, qi.shape[1], -1)
                )
            )
        if return_lse:
            return torch.cat(outputs), torch.cat(lses)
        return torch.cat(outputs)

    def test_latent_prefill_decode_mixed(self):
        for mode, qlens, lens, mixed in (
            (ForwardMode.EXTEND, [21, 35], [21, 35], None),
            (ForwardMode.EXTEND, [3, 5], [21, 35], None),
            (ForwardMode.DECODE, [1, 1], [21, 35], None),
            (ForwardMode.MIXED, [3, 1], [21, 35], 1),
            (ForwardMode.MIXED, [3, 1], [3, 35], 1),
            (ForwardMode.MIXED, [1, 1], [1, 35], 1),
            (ForwardMode.MIXED, [1, 1], [21, 35], 1),
            (ForwardMode.MIXED, [1, 1], [21, 35], 0),
            (ForwardMode.MIXED, [3, 5], [21, 35], 2),
            (ForwardMode.MIXED, [3, 5], [3, 5], 2),
            (ForwardMode.EXTEND, [3, 5], [3, 35], None),
        ):
            with self.subTest(mode=mode, qlens=qlens):
                f = self._fixture(qlens, lens, mode, mixed)
                actual = f.backend.forward(
                    f.q,
                    f.current,
                    None,
                    f.layer,
                    f.fb,
                )
                expected = self._reference(
                    f.q, f.cache, f.cache[..., :512], qlens, lens, f.layer.scaling
                )
                torch.testing.assert_close(
                    actual.cpu().view_as(expected).float(),
                    expected,
                    atol=0.004,
                    rtol=0.025,
                )
                with_lse, lse = f.backend.forward(
                    f.q,
                    f.current,
                    None,
                    f.layer,
                    f.fb,
                    save_kv_cache=False,
                    return_lse=True,
                )
                self.assertEqual(with_lse.shape, actual.shape)
                self.assertEqual(lse.shape, (sum(qlens), 8))
                self.assertEqual(lse.dtype, torch.float32)
                torch.testing.assert_close(with_lse, actual, atol=0.004, rtol=0.025)
                _, expected_lse = self._reference(
                    f.q,
                    f.cache,
                    f.cache[..., :512],
                    qlens,
                    lens,
                    f.layer.scaling,
                    return_lse=True,
                )
                torch.testing.assert_close(
                    lse.cpu(), expected_lse, atol=0.003, rtol=0.003
                )
                gathered = f.pool.get_key_buffer(0)[f.fb.out_cache_loc.long()]
                torch.testing.assert_close(gathered, f.current, rtol=0, atol=0)

    def test_absorb_core_query_scaling(self):
        """Absorbed MLA must scale both Q components before attention."""
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
            DeepseekMLAForwardMixin,
        )

        class Attention(DeepseekMLAForwardMixin):
            current_attention_backend = "mlu"
            dcp_size = 1
            num_local_heads = 8
            kv_lora_rank = 512
            v_head_dim = 128
            use_deep_gemm_bmm = False
            rotary_emb = None
            next_skip_topk = None

        for scale in (None, 2.0):
            with self.subTest(scale=scale):
                f = self._fixture([1, 1], [21, 35], ForwardMode.DECODE)
                attn = Attention()
                attn.w_vc = (
                    torch.eye(512, 128, device="mlu", dtype=torch.bfloat16)
                    .unsqueeze(0)
                    .repeat(8, 1, 1)
                )
                attn.o_proj = lambda x: (x, None)
                attn.attn_mqa = lambda q, k, v, fb, **kwargs: f.backend.forward(
                    q, k, v, f.layer, fb, **kwargs
                )
                scaling = None if scale is None else f.q.new_tensor(scale)
                actual = attn.forward_absorb_core(
                    q_pe=f.q[..., 512:],
                    k_pe=f.current[..., 512:],
                    q_nope_out=f.q[..., :512],
                    k_nope=f.current[..., :512],
                    forward_batch=f.fb,
                    zero_allocator=None,
                    positions=None,
                    topk_indices=None,
                    llama_4_scaling=scaling,
                )
                expected = self._reference(
                    f.q if scale is None else f.q * scaling,
                    f.cache,
                    f.cache[..., :512],
                    f.qlens,
                    f.lens,
                    f.layer.scaling,
                )[..., :128].flatten(1)
                torch.testing.assert_close(
                    actual.cpu().float(), expected, atol=0.004, rtol=0.025
                )
                torch.testing.assert_close(
                    f.pool.get_key_buffer(0)[f.fb.out_cache_loc.long()],
                    f.current,
                    rtol=0,
                    atol=0,
                )

    def test_one_shot_and_chunked_mha(self):
        f = self._fixture([3, 5], [21, 35])
        layer = NS(**vars(f.layer))
        layer.tp_k_head_num = layer.tp_v_head_num = 8
        layer.qk_head_dim = 192
        layer.v_head_dim = 128
        q = f.q[..., :192].contiguous()
        k = torch.randn(56, 8, 192, device="mlu", dtype=torch.bfloat16) * 0.2
        v = torch.randn(56, 8, 128, device="mlu", dtype=torch.bfloat16) * 0.2
        expected, expected_lse = self._reference(
            q, k, v, f.qlens, f.lens, layer.scaling, return_lse=True
        )
        f.fb.mha_one_shot = True
        one = f.backend.forward(q, k, v, layer, f.fb, save_kv_cache=False)
        torch.testing.assert_close(
            one.cpu().view_as(expected).float(), expected, atol=0.004, rtol=0.025
        )
        one_with_lse, one_lse = f.backend.forward(
            q, k, v, layer, f.fb, save_kv_cache=False, return_lse=True
        )
        self.assertEqual(one_with_lse.shape, expected.shape)
        self.assertEqual(one_lse.dtype, torch.float32)
        torch.testing.assert_close(one_with_lse.flatten(1), one, atol=0.004, rtol=0.025)
        torch.testing.assert_close(one_lse.cpu(), expected_lse, atol=0.003, rtol=0.003)
        f.fb.mha_one_shot = False
        f.fb.mha_return_lse = True
        f.fb.attn_attend_prefix_cache = False
        suffix_k = torch.cat([k[18:21], k[51:56]])
        suffix_v = torch.cat([v[18:21], v[51:56]])
        out, lse = f.backend.forward(
            q, suffix_k, suffix_v, layer, f.fb, save_kv_cache=False
        )
        f.fb.attn_attend_prefix_cache = True
        # The second chunk has no KV for request 0. Merge its raw vendor
        # output/LSE directly, including the empty-row +inf LSE sentinel.
        for idx, (starts, ends) in enumerate(
            (((0, 21), (18, 39)), ((18, 39), (18, 51)))
        ):
            ck = torch.cat([k[a:b] for a, b in zip(starts, ends)])
            cv = torch.cat([v[a:b] for a, b in zip(starts, ends)])
            lengths = [b - a for a, b in zip(starts, ends)]
            f.fb.prefix_chunk_idx = 0
            f.fb.prefix_chunk_cu_seq_lens = [
                torch.tensor(
                    [0, lengths[0], sum(lengths)], dtype=torch.int32, device="mlu"
                )
            ]
            f.fb.prefix_chunk_max_seq_lens = [max(lengths)]
            part, plse = f.backend.forward(q, ck, cv, layer, f.fb, save_kv_cache=False)
            out, lse = merge_state(part, plse, out, lse)
        torch.testing.assert_close(out.cpu().float(), expected, atol=0.004, rtol=0.025)
        torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.003, rtol=0.003)

    def test_merge_output_buffers(self):
        torch.manual_seed(17)
        prefix = torch.randn(7, 4, 128, device="mlu", dtype=torch.bfloat16)
        suffix = torch.randn_like(prefix)
        p_lse = torch.randn(7, 4, device="mlu", dtype=torch.float32)
        s_lse = torch.randn_like(p_lse)
        # Build the reference with mathematical empty-state LSE (-inf).
        prefix[0] = 0
        p_lse[0] = -torch.inf
        expected_lse = torch.logaddexp(p_lse.cpu(), s_lse.cpu())
        expected = prefix.cpu().float() * (p_lse.cpu() - expected_lse).exp().unsqueeze(
            -1
        ) + suffix.cpu().float() * (s_lse.cpu() - expected_lse).exp().unsqueeze(-1)
        # TMO flash_attention returns zero output and +inf LSE for empty KV.
        # update_out_and_lse must consume that sentinel without normalization.
        p_lse[0] = torch.inf
        for layout in ("contiguous", "transposed", "inplace"):
            with self.subTest(layout=layout):
                p = prefix.clone()
                pl = p_lse.clone()
                out = p if layout == "inplace" else torch.empty_like(p)
                out_lse = (
                    pl
                    if layout == "inplace"
                    else torch.empty(4, 7, device="mlu").T
                    if layout == "transposed"
                    else torch.empty_like(pl)
                )
                actual, actual_lse = merge_state(p, pl, suffix, s_lse, out, out_lse)
                self.assertIs(actual, out)
                self.assertIs(actual_lse, out_lse)
                torch.testing.assert_close(
                    actual.cpu().float(), expected, atol=0.015, rtol=0.01
                )
                torch.testing.assert_close(
                    actual_lse.cpu(), expected_lse, atol=1e-5, rtol=1e-5
                )

    def test_deepseek_rope(self):
        """MLU RoPE must construct without the CUDA/vLLM fallback import."""
        from sglang.srt.layers.rotary_embedding.rope_variant import (
            DeepseekScalingRotaryEmbedding,
        )

        rope = DeepseekScalingRotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=128,
            base=10000,
            is_neox_style=False,
            scaling_factor=4,
            dtype=torch.bfloat16,
            device="mlu",
        ).to("mlu")
        q = torch.randn(5, 8, 64, device="mlu", dtype=torch.bfloat16)
        k = torch.randn(5, 1, 64, device="mlu", dtype=torch.bfloat16)
        positions = torch.tensor([0, 1, 16, 31, 63], device="mlu")
        actual_q, actual_k = rope(positions, q, k)
        cos, sin = rope.cos_sin_cache[positions].cpu().float().chunk(2, dim=-1)
        for actual, original in ((actual_q, q), (actual_k, k)):
            pairs = original.cpu().float().reshape(5, -1, 32, 2)
            real, imag = pairs[..., 0], pairs[..., 1]
            expected = torch.stack(
                (
                    real * cos[:, None] - imag * sin[:, None],
                    imag * cos[:, None] + real * sin[:, None],
                ),
                dim=-1,
            ).flatten(-2)
            torch.testing.assert_close(
                actual.cpu().float(), expected, atol=0.04, rtol=0.02
            )

    def test_routing(self):
        with envs.SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD.override(10):
            attn = NS(
                disable_chunked_prefix_cache=False, chunked_prefix_cache_threshold=10
            )
            fb = NS(
                forward_mode=ForwardMode.EXTEND,
                extend_prefix_lens_cpu=[0, 0],
                seq_lens_sum=32,
                get_max_chunk_capacity=lambda: 32,
            )
            self.assertEqual(handle_attention_mlu(attn, fb), AttnForwardMethod.MHA)
            fb.extend_prefix_lens_cpu = [5, 5]
            self.assertEqual(
                handle_attention_mlu(attn, fb), AttnForwardMethod.MHA_ONE_SHOT
            )
            fb.seq_lens_sum = 33
            self.assertEqual(
                handle_attention_mlu(attn, fb), AttnForwardMethod.MHA_CHUNKED_KV
            )
            attn.disable_chunked_prefix_cache = True
            self.assertEqual(handle_attention_mlu(attn, fb), AttnForwardMethod.MLA)
            attn.disable_chunked_prefix_cache = False
            fb.extend_prefix_lens_cpu = [0, 9]
            self.assertEqual(handle_attention_mlu(attn, fb), AttnForwardMethod.MLA)
            for mode in (ForwardMode.DECODE, ForwardMode.MIXED):
                fb.forward_mode = mode
                self.assertEqual(handle_attention_mlu(attn, fb), AttnForwardMethod.MLA)

    def test_default_prefix_threshold_is_local(self):
        threshold = envs.SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD
        with threshold.override(8192):
            threshold.clear()
            attn = NS(
                disable_chunked_prefix_cache=False,
                chunked_prefix_cache_threshold=threshold.get(),
            )
            fb = NS(
                forward_mode=ForwardMode.EXTEND,
                extend_prefix_lens_cpu=[1],
                seq_lens_sum=32,
                get_max_chunk_capacity=lambda: 32,
            )
            self.assertEqual(
                handle_attention_mlu(attn, fb), AttnForwardMethod.MHA_ONE_SHOT
            )
            fb.seq_lens_sum = 33
            self.assertEqual(
                handle_attention_mlu(attn, fb), AttnForwardMethod.MHA_CHUNKED_KV
            )
            attn.disable_chunked_prefix_cache = True
            self.assertEqual(handle_attention_mlu(attn, fb), AttnForwardMethod.MLA)
            self.assertFalse(threshold.is_set())
            self.assertEqual(threshold.get(), 8192)
            self.assertEqual(attn.chunked_prefix_cache_threshold, 8192)

    def test_pool_gather_grid_boundary(self):
        from triton.runtime import driver

        max_grid_x, _, _ = driver.active.utils.get_max_grid_size(
            torch.mlu.current_device()
        )
        for rope_dim in (64, 0):
            pool = MLUMLATokenToKVPool(
                size=16,
                page_size=16,
                dtype=torch.bfloat16,
                kv_lora_rank=512,
                qk_rope_head_dim=rope_dim,
                layer_num=2,
                device="mlu",
                enable_memory_saver=False,
            )
            layer = NS(layer_id=0)
            values = torch.randn(8, 1, 512 + rope_dim, device="mlu").bfloat16()
            pool.set_kv_buffer(
                layer,
                torch.arange(8, device="mlu"),
                values.float(),
                None,
                layer_id_override=1,
            )
            for count in (0, 9, max_grid_x + 3):
                with self.subTest(rope_dim=rope_dim, count=count):
                    # Match the parent pool contract: device-local, flat indices.
                    # Repeated slots cover large gathers and partial row tiles.
                    loc = torch.arange(count, device="mlu") % 8
                    nope, rope = pool.get_mla_kv_buffer(
                        layer, loc, torch.float32, layer_id_override=1
                    )
                    expected = values[loc].float()
                    torch.testing.assert_close(
                        nope, expected[..., :512], rtol=0, atol=0
                    )
                    self.assertTrue(nope.is_contiguous())
                    if rope_dim:
                        torch.testing.assert_close(
                            rope, expected[..., 512:], rtol=0, atol=0
                        )
                        self.assertTrue(rope.is_contiguous())
                    else:
                        self.assertIsNone(rope)

    def test_pool_split_write_gather_and_move(self):
        f = self._fixture([3, 5], [21, 35])
        with get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0):
            f.pool.set_mla_kv_buffer(
                f.layer, f.fb.out_cache_loc, f.current[..., :512], f.current[..., 512:]
            )
        nope, rope = f.pool.get_mla_kv_buffer(f.layer, f.fb.out_cache_loc)
        torch.testing.assert_close(
            torch.cat((nope, rope), -1), f.current, rtol=0, atol=0
        )
        target = torch.arange(200, 208, device="mlu")
        f.pool.move_kv_cache(target, f.fb.out_cache_loc)
        torch.testing.assert_close(
            f.pool.get_key_buffer(0)[target], f.current, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
