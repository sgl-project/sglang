"""Reference-example configuration and logical KV gather regressions."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from examples.runtime.sid_gr.attention import SIDReferenceAttentionBackend
from examples.runtime.sid_gr.launch_server import validate_configuration
from examples.runtime.sid_gr.model import (
    SIDFullReductionLinear,
    SIDLogitsProcessor,
    SIDQwen3ForCausalLM,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestSIDReferenceConfiguration(CustomTestCase):
    def test_qualified_configuration_and_rejections(self):
        config = dict(
            enable_gr=True,
            dtype="bfloat16",
            page_size=1,
            tp_size=1,
            pp_size=1,
            ep_size=1,
            dp_size=1,
            attn_dp_size=1,
            attn_cp_size=1,
            attention_backend="sid_gr_reference",
            chunked_prefill_size=-1,
            disable_radix_cache=True,
            disable_overlap_schedule=True,
            cuda_graph_backend_decode="disabled",
            cuda_graph_backend_prefill="disabled",
            kv_cache_dtype="auto",
        )
        for tp_size in (1, 2, 4):
            validate_configuration({**config, "tp_size": tp_size})
        for tp_size in (0, 3, 8):
            with self.subTest(tp_size=tp_size), self.assertRaises(ValueError):
                validate_configuration({**config, "tp_size": tp_size})
        for key, invalid in dict(
            page_size=16,
            dcp_size=2,
            pp_size=2,
            ep_size=2,
            dp_size=2,
            attn_dp_size=2,
            attn_cp_size=2,
            dtype="float16",
            chunked_prefill_size=256,
            disable_radix_cache=False,
            disable_overlap_schedule=False,
            enable_gr=False,
            cuda_graph_backend_decode="full",
            cuda_graph_backend_prefill="full",
            quantization="fp8",
            kv_cache_dtype="fp8_e4m3",
            enable_lora=True,
            enable_layernorm_sp=True,
            enable_fp32_lm_head=True,
            enable_torch_compile=True,
            speculative_algorithm="EAGLE",
            prefill_attention_backend="flashinfer",
            decode_attention_backend="flashinfer",
        ).items():
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate_configuration({**config, key: invalid})

    def test_reject_unqualified_tp_gemm_geometry(self):
        parallel = SimpleNamespace(tp_size=2, pp_size=1, attn_tp_size=2)
        config = SimpleNamespace(
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=64,
            hidden_size=2048,
            intermediate_size=1024,
            vocab_size=256,
        )
        with patch("examples.runtime.sid_gr.model.get_parallel", return_value=parallel):
            with self.assertRaisesRegex(ValueError, "GEMM geometry"):
                SIDQwen3ForCausalLM(config)


class TestSIDFullReductionProjection(CustomTestCase):
    def test_cancellation_keeps_full_dot_product_and_disjoint_outputs(self):
        # Splitting this dot product in two loses 1/256 at the BF16 partial
        # rounding boundary: (1 + 1/256) + (-1 + 0) becomes 0.
        x = torch.ones(1, 4, dtype=torch.bfloat16)
        weight = torch.tensor([[1, 1 / 256, -1, 0]], dtype=x.dtype).repeat(8, 1)
        expected = torch.nn.functional.linear(x, weight)
        partials = [
            torch.nn.functional.linear(a, b)
            for a, b in zip(x.chunk(2, -1), weight.chunk(2, -1))
        ]
        self.assertFalse(torch.equal(partials[0] + partials[1], expected))
        for size in (2, 4):
            outputs = []
            for rank in range(size):
                group = SimpleNamespace(
                    rank_in_group=rank, all_gather=Mock(return_value=x)
                )
                projection = SimpleNamespace(
                    tp_group=group,
                    weight=weight.chunk(size, 0)[rank].contiguous(),
                    output_size=weight.shape[0],
                )
                local_input = x.chunk(size, -1)[rank].contiguous()
                output, bias = SIDFullReductionLinear.forward(projection, local_input)
                self.assertIsNone(bias)
                group.all_gather.assert_called_once()
                torch.testing.assert_close(
                    group.all_gather.call_args.args[0], local_input
                )
                outputs.append(output)
            assembled = outputs[0]
            for output in outputs[1:]:
                assembled = assembled + output
            torch.testing.assert_close(assembled, expected, atol=0, rtol=0)
            self.assertTrue(torch.all(torch.stack(outputs).count_nonzero(0) == 1))

    def test_lm_head_padding_is_removed_before_vocab_gather(self):
        hidden = torch.ones(2, 4, dtype=torch.bfloat16)
        head = SimpleNamespace(weight=torch.full((64, 4), -1, dtype=hidden.dtype))
        processor = SimpleNamespace(vocab_size=256)
        for out in (
            None,
            torch.empty(2, 64, dtype=hidden.dtype),
            torch.empty(2, 64, dtype=torch.float32),
            torch.empty(2, 256, dtype=hidden.dtype),
        ):
            with self.subTest(buffer=None if out is None else (out.shape, out.dtype)):
                with patch(
                    "examples.runtime.sid_gr.model.get_parallel",
                    return_value=SimpleNamespace(tp_size=4),
                ):
                    logits = SIDLogitsProcessor._compute_lm_head(
                        processor, hidden, head, out=out
                    )
                self.assertEqual(logits.shape, (2, 64))
                self.assertEqual(logits.dtype, hidden.dtype)
                self.assertTrue(logits.is_contiguous())
                self.assertTrue(torch.all(logits == -4))
                if (
                    out is not None
                    and out.shape == logits.shape
                    and out.dtype == logits.dtype
                ):
                    self.assertIs(logits, out)


class TestSIDReferenceKV(CustomTestCase):
    def make_backend(self):
        backend = SIDReferenceAttentionBackend.__new__(SIDReferenceAttentionBackend)
        # Noncontiguous slots; both children share generated ancestor slot 9.
        mapping = torch.tensor([[11, 4, 7, 9, 5], [11, 4, 7, 9, 12]])
        backend.req_to_token_pool = SimpleNamespace(req_to_token=mapping)
        key = torch.arange(32, dtype=torch.float32).reshape(16, 1, 2)
        value = -key
        backend.token_to_kv_pool = SimpleNamespace(get_kv_buffer=lambda _: (key, value))
        backend._contexts = {11: 3}
        backend.tp_size = 1
        backend.beam_decode_attn = Mock(side_effect=lambda q, *args, **kwargs: q)
        return backend

    def batch(self, lengths=(5, 5)):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend=lambda: False, is_decode=lambda: True
            ),
            seq_lens_cpu=torch.tensor(lengths),
            req_pool_indices=torch.tensor([0, 1]),
        )

    def test_shared_ancestor_gather_and_reordered_rows(self):
        for order in ([0, 1], [1, 0]):
            with self.subTest(order=order):
                backend = self.make_backend()
                batch = self.batch()
                batch.req_pool_indices = torch.tensor(order)
                backend.init_forward_metadata(batch)
                layer = SimpleNamespace(
                    head_dim=2,
                    tp_q_head_num=2,
                    tp_k_head_num=1,
                    layer_id=0,
                    is_cross_attention=False,
                    logit_cap=0,
                )
                q = torch.zeros(2, 4)
                backend.forward_decode(q, q, q, layer, batch, save_kv_cache=False)
                args = backend.beam_decode_attn.call_args.args
                key, value = backend.token_to_kv_pool.get_kv_buffer(0)
                torch.testing.assert_close(args[1], key[[11, 4, 7]][None])
                torch.testing.assert_close(args[2], value[[11, 4, 7]][None])
                generated = [9, 9] + ([5, 12] if order == [0, 1] else [12, 5])
                torch.testing.assert_close(args[3], key[generated][None])
                torch.testing.assert_close(args[4], value[generated][None])
                self.assertEqual(args[5][0, 0, 0].tolist(), [[0, 1], [2, 3]])
                self.assertEqual(args[6], 2)

    def test_tp_dispatch_uses_full_heads_and_local_kv(self):
        backend = self.make_backend()
        backend.tp_size = 4
        batch = self.batch()
        backend.init_forward_metadata(batch)
        layer = SimpleNamespace(
            head_dim=2,
            tp_q_head_num=2,
            tp_k_head_num=1,
            layer_id=0,
            is_cross_attention=False,
            logit_cap=0,
        )
        q = torch.zeros(2, 4)
        with patch(
            "examples.runtime.sid_gr.attention.beam_decode_with_reference_splits",
            side_effect=lambda q, *args, **kwargs: q,
        ) as kernel:
            backend.forward_decode(q, q, q, layer, batch, save_kv_cache=False)
        self.assertEqual(kernel.call_args.kwargs["reference_heads"], 8)
        self.assertEqual(kernel.call_args.args[1].shape[-2], 1)
        backend.beam_decode_attn.assert_not_called()

    def test_reject_mixed_groups_and_unequal_lengths(self):
        backend = self.make_backend()
        with self.assertRaisesRegex(ValueError, "equal-length"):
            backend.init_forward_metadata(self.batch((5, 4)))
        backend.req_to_token_pool.req_to_token[1, 0] = 10
        with self.assertRaisesRegex(ValueError, "one beam group"):
            backend.init_forward_metadata(self.batch())

    def test_prompt_slot_reuse_replaces_context_length(self):
        backend = self.make_backend()
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[4],
            out_cache_loc=torch.tensor([11, 4, 7, 9]),
        )
        backend.init_forward_metadata(batch)
        self.assertEqual(backend._contexts[11], 4)
        backend.init_forward_metadata(self.batch())
        self.assertEqual(backend.forward_metadata.steps, 1)
        batch.extend_prefix_lens_cpu = [1]
        with self.assertRaisesRegex(ValueError, "uncached prompt"):
            backend.init_forward_metadata(batch)


if __name__ == "__main__":
    unittest.main()
