import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.neo_unify import (
    build_image_token_end,
    neo_unify_attention,
    resolve_neo_backend,
)
from sglang.multimodal_gen.configs.sensenova_u1 import get_neo_attention_backends
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestNeoUnify(CustomTestCase):
    def test_sm89_support_neo_is_explicit_only(self):
        q = SimpleNamespace(
            is_cuda=True,
            dtype=torch.bfloat16,
            shape=(1, 128, 16, 128),
            device=torch.device("cuda"),
        )
        module = "sglang.kernels.ops.attention.neo_unify"
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)),
            patch(f"{module}._neo_fa3", return_value=object()),
        ):
            self.assertEqual(
                resolve_neo_backend(q, image_aware=True, backend="fa3"), "fa3"
            )
            self.assertEqual(
                resolve_neo_backend(q, image_aware=True, backend="auto"), "triton"
            )

    def test_fa3_packing_preserves_batch_boundaries(self):
        # Emulate only the optional extension boundary. Its varlen inputs are
        # consumed independently, so a wrong packing/offset changes the output.
        def extension(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            cache_seqlens,
            image_token_end,
            max_seqlen_q,
            causal,
            softmax_scale,
        ):
            outputs = []
            for b in range(k_cache.shape[0]):
                start, stop = cu_seqlens_q[b : b + 2].tolist()
                length = int(cache_seqlens[b])
                self.assertLessEqual(stop - start, max_seqlen_q)
                self.assertTrue(causal)
                queries = q[start:stop].transpose(0, 1)
                keys = k_cache[b, :length].transpose(0, 1).repeat_interleave(2, 0)
                values = v_cache[b, :length].transpose(0, 1).repeat_interleave(2, 0)
                kp = torch.arange(length)
                qp = torch.arange(stop - start) + length - (stop - start)
                allow = (kp <= qp[:, None]) | (kp < image_token_end[start:stop, None])
                outputs.append(
                    torch.nn.functional.scaled_dot_product_attention(
                        queries, keys, values, attn_mask=allow, scale=softmax_scale
                    ).transpose(0, 1)
                )
            return torch.cat(outputs)

        torch.manual_seed(3)
        q = torch.randn(2, 5, 4, 32)
        k, v = torch.randn(2, 2, 8, 2, 32)
        ends = build_image_token_end(
            torch.tensor([[0, 1, 1, 2, 3], [0, 0, 0, 1, 2]]), 3
        )
        expected = neo_unify_attention(
            q, k, v, image_token_end=ends, causal=True, backend="torch"
        )
        module = "sglang.kernels.ops.attention.neo_unify"
        with (
            patch(f"{module}.resolve_neo_backend", return_value="fa3"),
            patch(f"{module}._neo_fa3", return_value=extension),
        ):
            actual = neo_unify_attention(
                q, k, v, image_token_end=ends, causal=True, backend="fa3"
            )
        torch.testing.assert_close(actual, expected)

    def test_backend_config(self):
        self.assertEqual(
            get_neo_attention_backends({"neo_prefill_backend": "triton"}),
            {"neo_prefill_backend": "triton", "neo_denoise_backend": "auto"},
        )
        for config in ({"foo": "fa3"}, {"neo_denoise_backend": "fa4"}, "triton"):
            with self.assertRaises(ValueError):
                get_neo_attention_backends(config)

    def test_block_boundaries_match_model_mask(self):
        ids = torch.tensor([[0, 1, 1, 1, 2, 3, 3, 4], [0, 0, 1, 2, 3, 3, 4, 5]])
        ends = build_image_token_end(ids, prefix_len=5)
        qpos = torch.arange(8) + 5
        kpos = torch.arange(13)
        actual = (kpos <= qpos[:, None]) | (kpos < ends[:, :, None])
        expected = (ids[:, :, None] == ids[:, None, :]) | (
            torch.arange(8) <= torch.arange(8)[:, None]
        )
        expected = torch.cat((torch.ones(2, 8, 5, dtype=torch.bool), expected), -1)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(ends.dtype, torch.int32)

    def test_invalid_blocks(self):
        with self.assertRaises(ValueError):
            build_image_token_end(torch.tensor([0, 1, 0]))
        with self.assertRaises(ValueError):
            build_image_token_end(torch.tensor([0.0, 1.0]))
        with self.assertRaises(ValueError):
            build_image_token_end(torch.tensor([0, 1]), prefix_len=-1)

    def test_future_tokens_cannot_change_earlier_image(self):
        torch.manual_seed(1)
        q = torch.randn(1, 8, 4, 16)
        k, v = torch.randn(2, 1, 8, 2, 16)
        ends = build_image_token_end(torch.tensor([0, 1, 1, 1, 2, 3, 3, 4]))
        original = neo_unify_attention(
            q, k, v, image_token_end=ends, causal=True, backend="torch"
        )
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[:, 4:] += 100
        changed_v[:, 4:] -= 100
        changed = neo_unify_attention(
            q, changed_k, changed_v, image_token_end=ends, causal=True, backend="torch"
        )
        torch.testing.assert_close(original[:, :4], changed[:, :4], rtol=0, atol=0)
        self.assertFalse(torch.equal(original[:, 4:], changed[:, 4:]))

    def test_denoising_reads_current_kv_without_mutation(self):
        torch.manual_seed(2)
        q = torch.randn(2, 3, 4, 16)
        k, v = torch.randn(2, 2, 8, 2, 16)
        prefix_k, prefix_v = k[:, :5].clone(), v[:, :5].clone()
        first = neo_unify_attention(q, k, v, backend="torch")
        v[:, 5:] += 10
        second = neo_unify_attention(q, k, v, backend="torch")
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(torch.equal(prefix_k, k[:, :5]))
        self.assertTrue(torch.equal(prefix_v, v[:, :5]))

    def test_validation(self):
        q = torch.zeros(1, 3, 4, 16)
        kv = torch.zeros(1, 5, 2, 16)
        with self.assertRaises(ValueError):
            neo_unify_attention(q, kv, kv, backend="unknown")
        with self.assertRaises(ValueError):
            neo_unify_attention(
                q, kv, kv, image_token_end=torch.zeros(3, dtype=torch.int32)
            )
        with self.assertRaises(ValueError):
            neo_unify_attention(
                q,
                kv,
                kv,
                causal=True,
                image_token_end=torch.zeros(2, dtype=torch.int32),
            )
        with self.assertRaises(RuntimeError):
            neo_unify_attention(q, kv, kv, backend="fa3")


if __name__ == "__main__":
    unittest.main()
