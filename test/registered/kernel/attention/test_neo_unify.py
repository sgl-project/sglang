import unittest

import torch
from transformers.cache_utils import DynamicCache

from sglang.kernels.ops.attention.neo_unify import (
    _neo_fa3,
    build_image_token_end,
    neo_unify_attention,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestNeoUnifyGPU(CustomTestCase):
    def compare_backend(self, backend, *, image_aware=True):
        torch.manual_seed(42)
        for dtype in (torch.float16, torch.bfloat16):
            for batch, qlen, prefix, heads, kvheads, dim in (
                (1, 1, 7, 4, 1, 32),
                (2, 65, 11, 4, 2, 64),
                (1, 257, 0, 8, 2, 128),
                (2, 129, 17, 4, 1, 256),
            ):
                q = torch.randn(
                    batch, heads, qlen, dim, device="cuda", dtype=dtype
                ).transpose(1, 2)
                # Noncontiguous token and head-dimension strides exercise the adapter.
                k = torch.randn(
                    batch, qlen + prefix, kvheads, dim * 2, device="cuda", dtype=dtype
                )[..., ::2]
                v = torch.randn_like(k)
                ids = torch.arange(qlen, device="cuda").expand(batch, -1).clone()
                ids[:, 1 : min(70, qlen)] = 1
                if qlen > 90:
                    ids[:, 80:100] = 80
                if batch > 1:
                    ids[1] = torch.arange(qlen, device="cuda")
                    ids[1, 2 : min(33, qlen)] = 2
                ends = build_image_token_end(ids, prefix)
                cases = [(False, None), (True, None)]
                if image_aware:
                    cases.append((True, ends))
                for causal, boundaries in cases:
                    with self.subTest(
                        backend=backend,
                        dtype=dtype,
                        shape=q.shape,
                        causal=causal,
                        image=boundaries is not None,
                    ):
                        expected = neo_unify_attention(
                            q,
                            k,
                            v,
                            image_token_end=boundaries,
                            causal=causal,
                            softmax_scale=0.17,
                            backend="torch",
                        )
                        actual = neo_unify_attention(
                            q,
                            k,
                            v,
                            image_token_end=boundaries,
                            causal=causal,
                            softmax_scale=0.17,
                            backend=backend,
                        )
                        tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
                        torch.testing.assert_close(
                            actual, expected, atol=tolerance, rtol=tolerance
                        )

    def test_triton(self):
        self.compare_backend("triton")

    def test_fa3(self):
        if torch.cuda.get_device_capability()[0] != 9 or _neo_fa3() is None:
            self.skipTest("Hopper and the optional image_token_end FA3 build required")
        self.compare_backend("fa3")

    def test_fa3_standard(self):
        if torch.cuda.get_device_capability()[0] != 9:
            self.skipTest("Hopper required")
        self.compare_backend("fa3", image_aware=False)

    def test_image_does_not_see_future_text(self):
        torch.manual_seed(7)
        q = torch.randn(1, 129, 4, 64, dtype=torch.bfloat16, device="cuda")
        k, v = torch.randn(2, 1, 129, 2, 64, dtype=q.dtype, device=q.device)
        ids = torch.arange(129, device=q.device)
        ids[1:66] = 1
        ends = build_image_token_end(ids)
        original = neo_unify_attention(
            q, k, v, image_token_end=ends, causal=True, backend="triton"
        )
        k[:, 66:] += 10
        v[:, 66:] -= 10
        changed = neo_unify_attention(
            q, k, v, image_token_end=ends, causal=True, backend="triton"
        )
        torch.testing.assert_close(original[:, :66], changed[:, :66], atol=0, rtol=0)

    @torch.no_grad()
    def test_model_prefill_and_denoise(self):
        from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_chat import (
            NEOLLMConfig,
        )
        from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_neo_chat import (
            prepare_flash_kv_cache,
        )
        from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
            Qwen3Attention,
            create_block_causal_mask,
            create_neo_attention_mask,
        )

        torch.manual_seed(42)
        config = NEOLLMConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
        )
        config._attn_implementation = "eager"
        layer = Qwen3Attention(config, 0).cuda().to(torch.bfloat16).eval()
        backends = ["triton"]
        if torch.cuda.get_device_capability()[0] == 9 and _neo_fa3() is not None:
            backends.append("fa3")
        ids = torch.arange(129, device="cuda")
        ids[1:66] = 1
        indexes = torch.stack((ids, torch.zeros_like(ids), torch.zeros_like(ids)))
        hidden = torch.randn(1, 129, 128, dtype=torch.bfloat16, device="cuda")
        legacy_cache = DynamicCache(config=config)
        expected, _ = layer.forward_und(
            hidden, indexes, create_block_causal_mask(ids), legacy_cache
        )
        for backend in backends:
            with self.subTest(backend=backend):
                cache = DynamicCache(config=config)
                actual, _ = layer.forward_und(
                    hidden, indexes, create_neo_attention_mask(ids, backend), cache
                )
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
                torch.testing.assert_close(
                    cache.layers[0].keys, legacy_cache.layers[0].keys, rtol=0, atol=0
                )
                prepare_flash_kv_cache(cache, current_len=65, batch_size=1)
                prefix_k = cache.layers[0].flash_k_cache[:, :129].clone()
                prefix_v = cache.layers[0].flash_v_cache[:, :129].clone()
                pointer = cache.layers[0].flash_k_cache.data_ptr()
                gen_indexes = torch.stack(
                    (
                        torch.full((65,), 129, device="cuda"),
                        torch.arange(65, device="cuda"),
                        torch.zeros(65, dtype=torch.long, device="cuda"),
                    )
                )
                for _ in range(3):
                    gen_hidden = torch.randn(
                        1, 65, 128, dtype=hidden.dtype, device=hidden.device
                    )
                    config.neo_denoise_backend = "legacy"
                    legacy, _ = layer.forward_gen(
                        gen_hidden, gen_indexes, None, cache, update_cache=False
                    )
                    # Poison the current-image segment to detect skipped writes.
                    cache.layers[0].flash_k_cache[:, 129:].fill_(float("nan"))
                    cache.layers[0].flash_v_cache[:, 129:].fill_(float("nan"))
                    config.neo_denoise_backend = backend
                    output, _ = layer.forward_gen(
                        gen_hidden, gen_indexes, None, cache, update_cache=False
                    )
                    torch.testing.assert_close(output, legacy, rtol=2e-2, atol=2e-2)
                    self.assertEqual(cache.get_seq_length(), 129)
                    self.assertEqual(cache.layers[0].flash_k_cache.data_ptr(), pointer)
                    torch.testing.assert_close(
                        cache.layers[0].flash_k_cache[:, :129], prefix_k, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        cache.layers[0].flash_v_cache[:, :129], prefix_v, rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
