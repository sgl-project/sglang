"""Loading ModelOpt NVFP4 Llama 4 checkpoints: q/k rows and fused expert scales."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.mllama4 import Llama4ForConditionalGeneration as Llama4
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HEAD_DIM, HEADS, KV_HEADS, HIDDEN = 4, 2, 1, 32


def _model():
    config = SimpleNamespace(
        head_dim=HEAD_DIM,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        original_num_attention_heads=HEADS,
        original_total_num_kv_heads=KV_HEADS,
    )
    model = SimpleNamespace(language_model=SimpleNamespace(config=config))
    model._transform_expert_name = lambda name, is_weight=False: (
        Llama4._transform_expert_name(model, name, is_weight)
    )
    return model


def _rows(n, cols, dtype):
    """Each row holds its own index, so a permutation shows in column 0."""
    return torch.arange(n).repeat_interleave(cols).view(n, cols).to(dtype)


class TestLlama4Nvfp4Loader(CustomTestCase):
    def test_packed_weights_and_block_scales_get_the_rows_of_a_bf16_weight(self):
        """An NVFP4 q/k weight packs two values per byte and has per-row FP8
        block scales. Loading failed on the packed width, and the scales kept
        their rows while the weight's moved."""
        model = _model()
        for proj, heads in (("q_proj", HEADS), ("k_proj", KV_HEADS)):
            n = heads * HEAD_DIM
            prefix = f"language_model.model.layers.0.self_attn.{proj}"
            _, bf16 = Llama4.permute_qk_weight_for_rotary(
                model, prefix + ".weight", _rows(n, HIDDEN, torch.bfloat16)
            )
            expected = bf16[:, 0].long()
            self.assertNotEqual(expected.tolist(), list(range(n)))
            for suffix, cols, dtype in (
                (".weight", HIDDEN // 2, torch.uint8),  # two FP4 values per byte
                (".weight_scale", HIDDEN // 16, torch.float8_e4m3fn),
            ):
                with self.subTest(proj=proj, suffix=suffix):
                    _, out = Llama4.permute_qk_weight_for_rotary(
                        model, prefix + suffix, _rows(n, cols, dtype)
                    )
                    self.assertEqual(tuple(out.shape), (n, cols))
                    self.assertEqual(
                        out[:, 0].float().long().tolist(), expected.tolist()
                    )
            for suffix in (".weight_scale_2", ".input_scale", ".weight_scale"):
                scalar = torch.tensor(2.0)
                _, out = Llama4.permute_qk_weight_for_rotary(
                    model, prefix + suffix, scalar
                )
                self.assertIs(out, scalar)

    def test_int4_packed_weights_and_their_scales_keep_their_rows(self):
        """Compressed-tensors INT4 names its weight weight_packed, which is not
        permuted, so its (non-FP8) group scales must keep their rows too."""
        model = _model()
        n = HEADS * HEAD_DIM
        prefix = "language_model.model.layers.0.self_attn.q_proj"
        for suffix, tensor in (
            (".weight_packed", _rows(n, HIDDEN // 8, torch.int32)),
            (".weight_scale", _rows(n, HIDDEN // 32, torch.bfloat16)),
        ):
            with self.subTest(suffix=suffix):
                _, out = Llama4.permute_qk_weight_for_rotary(
                    model, prefix + suffix, tensor
                )
                self.assertTrue(torch.equal(out, tensor))

    def test_fused_expert_scales_load_per_expert_like_the_weights(self):
        """A fused checkpoint stores the block scales of all experts in one
        [experts, in, out] tensor. It was assigned whole to every expert and
        failed; it must load per expert, transposed, gate_up split in two."""
        model = _model()
        experts, inter, scale_cols = 3, 8, HIDDEN // 16
        calls = []
        params = {}
        for name in ("w13_weight_scale", "w2_weight_scale"):
            param = torch.nn.Parameter(torch.zeros(experts, 1), requires_grad=False)
            param.weight_loader = lambda p, w, n, shard_id, expert_id: calls.append(
                (n, shard_id, expert_id, w.clone())
            )
            params["language_model.model.layers.0.feed_forward.experts." + name] = param
        prefix = "language_model.model.layers.0.feed_forward.experts."
        gate_up = torch.randn(experts, scale_cols, 2 * inter)  # [E, in/16, 2*out]
        down = torch.randn(experts, inter // 16 or 1, HIDDEN)  # [E, in/16, out]
        loaded = set()
        for name, tensor in (
            ("gate_up_proj_weight_scale", gate_up),
            ("down_proj_weight_scale", down),
        ):
            self.assertTrue(
                Llama4._handle_expert_scale_params(
                    model, prefix + name, tensor, params, experts, loaded
                )
            )
        w1, w3 = gate_up.chunk(2, dim=-1)
        expected = [("w1", e, w1[e].T) for e in range(experts)]
        expected += [("w3", e, w3[e].T) for e in range(experts)]
        expected += [("w2", e, down[e].T) for e in range(experts)]
        self.assertEqual(len(calls), len(expected))
        for (_, shard, expert, tensor), (want_shard, want_expert, want) in zip(
            calls, expected
        ):
            self.assertEqual((shard, expert), (want_shard, want_expert))
            self.assertTrue(torch.equal(tensor, want))
        self.assertEqual(loaded, set(params))

    def test_a_scalar_expert_scale_still_fills_every_expert(self):
        """A scale without an expert index or an expert dimension still goes
        to every expert."""
        model = _model()
        name = "language_model.model.layers.0.feed_forward.experts.w2_weight_scale_2"
        param = torch.nn.Parameter(torch.zeros(3), requires_grad=False)
        Llama4._handle_expert_scale_params(
            model,
            "language_model.model.layers.0.feed_forward.experts.down_proj_weight_scale_2",
            torch.tensor(0.5),
            {name: param},
            3,
            set(),
        )
        self.assertEqual(param.data.tolist(), [0.5, 0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
