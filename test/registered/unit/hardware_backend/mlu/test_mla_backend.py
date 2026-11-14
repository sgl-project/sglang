"""MLU latent attention and one-shot/chunked MHA numerical coverage."""

import unittest
from contextlib import ExitStack
from functools import partial
from types import SimpleNamespace as NS

import torch

from sglang.srt.hardware_backend.mlu.attention.mla_backend import MLUMLABackend
from sglang.srt.hardware_backend.mlu.memory_pool import MLUMLATokenToKVPool
from sglang.srt.layers.attention.merge_state import merge_state
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=45, suite="pr-test-1-mlu")


class TestMLUMLABackend(CustomTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(11)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            get_parallel().override(attn_dcp_size=1, attn_cp_size=1, dcp_enabled=False)
        )

    def _create_fixture(
        self, query_lens, kv_lens, mode=ForwardMode.EXTEND, mixed_prefill_reqs=None
    ):
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
        pages = torch.tensor(
            [[5, 1, 8, 2], [7, 3, 9, 6]], device="mlu", dtype=torch.int32
        )
        offsets = torch.arange(16, device="mlu", dtype=torch.int32)
        req_to_token = (pages[..., None] * 16 + offsets).reshape(2, 64)
        backend = MLUMLABackend(
            NS(
                device="mlu",
                page_size=16,
                model_config=NS(
                    context_len=64,
                    kv_lora_rank=512,
                    hf_config=NS(architectures=["DeepseekV2ForCausalLM"]),
                ),
                req_to_token_pool=NS(req_to_token=req_to_token),
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
        loc = torch.cat(
            [
                req_to_token[i, kv_len - query_len : kv_len]
                for i, (query_len, kv_len) in enumerate(zip(query_lens, kv_lens))
            ]
        )
        forward_batch = NS(
            batch_size=2,
            req_pool_indices=torch.arange(2, device="mlu"),
            seq_lens=torch.tensor(kv_lens, device="mlu"),
            seq_lens_cpu=torch.tensor(kv_lens),
            extend_seq_lens=torch.tensor(query_lens, device="mlu"),
            extend_seq_lens_cpu=query_lens,
            extend_prefix_lens_cpu=[
                kv_len - query_len for kv_len, query_len in zip(kv_lens, query_lens)
            ],
            mixed_prefill_reqs=mixed_prefill_reqs,
            forward_mode=mode,
            out_cache_loc=loc,
            attn_attend_prefix_cache=None,
            mha_one_shot=None,
            mha_return_lse=None,
        )
        backend.init_forward_metadata(forward_batch)
        return NS(
            pool=pool,
            backend=backend,
            layer=layer,
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            kv_lens=kv_lens,
            query_lens=query_lens,
        )

    def _create_latent_fixture(
        self, query_lens, kv_lens, mode=ForwardMode.EXTEND, mixed_prefill_reqs=None
    ):
        fixture = self._create_fixture(query_lens, kv_lens, mode, mixed_prefill_reqs)
        fixture.cache = (
            torch.randn(sum(kv_lens), 1, 576, device="mlu", dtype=torch.bfloat16) * 0.2
        )
        slots = torch.cat(
            [fixture.req_to_token[i, :length] for i, length in enumerate(kv_lens)]
        )
        fixture.pool.set_kv_buffer(fixture.layer, slots, fixture.cache, None)
        fixture.q = (
            torch.randn(sum(query_lens), 8, 576, device="mlu", dtype=torch.bfloat16)
            * 0.2
        )
        fixture.current = torch.cat(
            [
                chunk[-length:]
                for chunk, length in zip(fixture.cache.split(kv_lens), query_lens)
            ]
        )
        # Clear current-token slots so the test also verifies the forward KV write.
        fixture.pool.set_kv_buffer(
            fixture.layer,
            fixture.forward_batch.out_cache_loc,
            torch.zeros_like(fixture.current),
            None,
        )
        return fixture

    @staticmethod
    def _reference(q, k, v, query_lens, kv_lens, scale):
        outputs = []
        lses = []
        for qi, ki, vi in zip(
            q.cpu().float().split(query_lens),
            k.cpu().float().split(kv_lens),
            v.cpu().float().split(kv_lens),
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
        return torch.cat(outputs), torch.cat(lses)

    def test_latent_prefill_decode_mixed(self):
        for mode, query_lens, kv_lens, mixed_prefill_reqs in (
            # No cached prefixes.
            (ForwardMode.EXTEND, [21, 35], [21, 35], None),
            # Both requests have cached prefixes.
            (ForwardMode.EXTEND, [3, 5], [21, 35], None),
            # One decode token per request.
            (ForwardMode.DECODE, [1, 1], [21, 35], None),
            # Cached-prefix prefill plus decode.
            (ForwardMode.MIXED, [3, 1], [21, 35], 1),
            # Uncached prefill plus decode.
            (ForwardMode.MIXED, [3, 1], [3, 35], 1),
            # Uncached one-token prefill plus decode.
            (ForwardMode.MIXED, [1, 1], [1, 35], 1),
            # Cached one-token prefill plus decode.
            (ForwardMode.MIXED, [1, 1], [21, 35], 1),
            # All requests are decode.
            (ForwardMode.MIXED, [1, 1], [21, 35], 0),
            # All requests are cached-prefix prefill.
            (ForwardMode.MIXED, [3, 5], [21, 35], 2),
            # All requests are uncached prefill.
            (ForwardMode.MIXED, [3, 5], [3, 5], 2),
            # Only one request has a cached prefix.
            (ForwardMode.EXTEND, [3, 5], [3, 35], None),
        ):
            with self.subTest(
                mode=mode,
                query_lens=query_lens,
                kv_lens=kv_lens,
                mixed_prefill_reqs=mixed_prefill_reqs,
            ):
                # Create the backend, batch, and input tensors.
                fixture = self._create_latent_fixture(
                    query_lens, kv_lens, mode, mixed_prefill_reqs
                )
                forward = partial(
                    fixture.backend.forward,
                    q=fixture.q,
                    k=fixture.current,
                    v=None,
                    layer=fixture.layer,
                    forward_batch=fixture.forward_batch,
                )

                # Compute latent attention outputs and LSE on CPU.
                expected, expected_lse = self._reference(
                    fixture.q,
                    fixture.cache,
                    fixture.cache[..., :512],
                    query_lens,
                    kv_lens,
                    fixture.layer.scaling,
                )

                # Write current KV, then reuse it while requesting LSE.
                actual = forward()
                with_lse, lse = forward(save_kv_cache=False, return_lse=True)

                # Compare outputs, LSE, and stored KV against the references.
                torch.testing.assert_close(
                    actual.cpu().float(),
                    expected.flatten(1),
                    atol=3e-2,
                    rtol=3e-2,
                )
                torch.testing.assert_close(with_lse, actual, atol=3e-2, rtol=3e-2)
                torch.testing.assert_close(
                    lse.cpu(), expected_lse, atol=1e-3, rtol=1e-3
                )
                gathered = fixture.pool.get_key_buffer(0)[
                    fixture.forward_batch.out_cache_loc.long()
                ]
                torch.testing.assert_close(gathered, fixture.current, rtol=0, atol=0)

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
                # Create model projections and the MLA backend inputs.
                fixture = self._create_latent_fixture(
                    [1, 1], [21, 35], ForwardMode.DECODE
                )
                attn = Attention()
                attn.w_vc = (
                    torch.eye(512, 128, device="mlu", dtype=torch.bfloat16)
                    .unsqueeze(0)
                    .repeat(8, 1, 1)
                )
                attn.o_proj = lambda x: (x, None)
                attn.attn_mqa = lambda q, k, v, forward_batch, **kwargs: (
                    fixture.backend.forward(
                        q, k, v, fixture.layer, forward_batch, **kwargs
                    )
                )
                scaling = None if scale is None else fixture.q.new_tensor(scale)

                # Compute the scaled attention reference on CPU.
                expected, _ = self._reference(
                    fixture.q if scale is None else fixture.q * scaling,
                    fixture.cache,
                    fixture.cache[..., :512],
                    fixture.query_lens,
                    fixture.kv_lens,
                    fixture.layer.scaling,
                )
                expected = expected[..., :128].flatten(1)

                # Run the model attention path on MLU.
                actual = attn.forward_absorb_core(
                    q_pe=fixture.q[..., 512:],
                    k_pe=fixture.current[..., 512:],
                    q_nope_out=fixture.q[..., :512],
                    k_nope=fixture.current[..., :512],
                    forward_batch=fixture.forward_batch,
                    zero_allocator=None,
                    positions=None,
                    topk_indices=None,
                    llama_4_scaling=scaling,
                )

                # Check the projected output and stored latent KV.
                torch.testing.assert_close(
                    actual.cpu().float(),
                    expected,
                    atol=3e-2,
                    rtol=3e-2,
                )
                torch.testing.assert_close(
                    fixture.pool.get_key_buffer(0)[
                        fixture.forward_batch.out_cache_loc.long()
                    ],
                    fixture.current,
                    rtol=0,
                    atol=0,
                )

    def test_one_shot_and_chunked_mha(self):
        # Create dense Q/K/V and a shared forward entry for both prefill paths.
        fixture = self._create_fixture([3, 5], [21, 35])
        layer = fixture.layer
        forward_batch = fixture.forward_batch
        layer.tp_k_head_num = layer.tp_v_head_num = 8
        layer.qk_head_dim = 192
        layer.v_head_dim = 128
        q = (
            torch.randn(
                sum(fixture.query_lens), 8, 192, device="mlu", dtype=torch.bfloat16
            )
            * 0.2
        )
        k = (
            torch.randn(
                sum(fixture.kv_lens), 8, 192, device="mlu", dtype=torch.bfloat16
            )
            * 0.2
        )
        v = (
            torch.randn(
                sum(fixture.kv_lens), 8, 128, device="mlu", dtype=torch.bfloat16
            )
            * 0.2
        )
        forward = partial(
            fixture.backend.forward,
            q=q,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=False,
        )
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        prefix_k, prefix_v = [], []
        suffix_k, suffix_v = [], []
        for req_k, req_v, prefix_len in zip(
            k.split(fixture.kv_lens), v.split(fixture.kv_lens), prefix_lens
        ):
            prefix_k.append(req_k[:prefix_len])
            prefix_v.append(req_v[:prefix_len])
            suffix_k.append(req_k[prefix_len:])
            suffix_v.append(req_v[prefix_len:])

        # Prepare metadata for every prefix chunk before running attention.
        forward_batch.prefix_chunk_cu_seq_lens = []
        forward_batch.prefix_chunk_max_seq_lens = []
        prefix_chunks = []
        chunk_size = prefix_lens[0]
        for start in range(0, max(prefix_lens), chunk_size):
            chunk_k = [req_k[start : start + chunk_size] for req_k in prefix_k]
            chunk_v = [req_v[start : start + chunk_size] for req_v in prefix_v]
            lengths = [chunk.shape[0] for chunk in chunk_k]
            forward_batch.prefix_chunk_cu_seq_lens.append(
                torch.tensor(
                    [0, lengths[0], sum(lengths)], dtype=torch.int32, device="mlu"
                )
            )
            forward_batch.prefix_chunk_max_seq_lens.append(max(lengths))
            prefix_chunks.append((torch.cat(chunk_k), torch.cat(chunk_v)))

        # Compute full causal attention outputs and LSE on CPU.
        expected, expected_lse = self._reference(
            q, k, v, fixture.query_lens, fixture.kv_lens, layer.scaling
        )

        # Run one-shot prefill with the full K/V.
        forward_batch.mha_one_shot = True
        one_shot_output = forward(k=k, v=v)
        one_shot_output_with_lse, one_shot_lse = forward(k=k, v=v, return_lse=True)

        # Compute causal attention over the suffix.
        forward_batch.mha_one_shot = False
        forward_batch.mha_return_lse = True
        forward_batch.attn_attend_prefix_cache = False
        chunked_output, chunked_lse = forward(
            k=torch.cat(suffix_k), v=torch.cat(suffix_v)
        )

        # Merge each prefix chunk into the suffix result. Request 0 has no KV
        # in the last chunk, whose zero output and +inf LSE are merged as-is.
        forward_batch.attn_attend_prefix_cache = True
        for chunk_idx, (chunk_k, chunk_v) in enumerate(prefix_chunks):
            forward_batch.prefix_chunk_idx = chunk_idx
            prefix_output, prefix_lse = forward(k=chunk_k, v=chunk_v)
            chunked_output, chunked_lse = merge_state(
                prefix_output, prefix_lse, chunked_output, chunked_lse
            )

        # Compare both paths against the same CPU reference.
        torch.testing.assert_close(
            one_shot_output.cpu().float(),
            expected.flatten(1),
            atol=3e-2,
            rtol=3e-2,
        )
        for path, output in (
            ("one_shot", one_shot_output_with_lse),
            ("chunked", chunked_output),
        ):
            with self.subTest(path=path, result="output"):
                torch.testing.assert_close(
                    output.cpu().float(), expected, atol=3e-2, rtol=3e-2
                )
        for path, lse in (("one_shot", one_shot_lse), ("chunked", chunked_lse)):
            with self.subTest(path=path, result="lse"):
                torch.testing.assert_close(
                    lse.cpu(), expected_lse, atol=1e-3, rtol=1e-3
                )


if __name__ == "__main__":
    unittest.main()
