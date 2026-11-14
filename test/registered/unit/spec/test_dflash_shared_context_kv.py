"""DFlash shared context K/V drafts: one projection writes the context K/V that
every draft layer reads, kept in a one-layer draft pool."""

import unittest
from unittest.mock import Mock, patch

import torch
from transformers import Qwen3Config

from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, LAYERS, EPS = 32, 4, 2, 8, 3, 1e-6
TARGET_LAYER_IDS = [0, 1]
FEATURES = len(TARGET_LAYER_IDS) * HIDDEN


def _config(**overrides):
    fields = dict(
        architectures=["DFlashDraftModel"],
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=64,
        rms_norm_eps=EPS,
        rope_theta=1e4,
        max_position_embeddings=4096,
        dflash_config={
            "block_size": 4,
            "target_layer_ids": TARGET_LAYER_IDS,
            "shared_context_kv": True,
        },
    )
    fields.update(overrides)
    return Qwen3Config(**fields)


def _rmsnorm(x, weight):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight


class TestDFlashSharedContextKV(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        # CPU CI does not enable the CPU engine; keep RoPE on the native path.
        self.enterContext(
            patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        )
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")
        torch.manual_seed(0)

    def _model(self, **overrides):
        model = DFlashDraftModel(_config(**overrides))
        for param in model.shared_context_kv.parameters():
            param.data.normal_()
        return model

    def test_parse(self):
        config = {"num_hidden_layers": 2, "dflash_config": {"shared_context_kv": True}}
        self.assertTrue(
            parse_dflash_draft_config(draft_hf_config=config).shared_context_kv
        )
        config = {"num_hidden_layers": 2, "dflash_config": {}}
        self.assertFalse(
            parse_dflash_draft_config(draft_hf_config=config).shared_context_kv
        )

    def test_one_pool_layer_and_no_fc(self):
        model = self._model()
        self.assertEqual(model.end_layer, 1)
        self.assertEqual({l.self_attn.attn.layer_id for l in model.layers}, {0})
        self.assertFalse(hasattr(model, "fc") or hasattr(model, "hidden_norm"))
        self.assertFalse(model.supports_fused_context_kv)
        self.assertEqual(
            {
                n
                for n, _ in model.named_parameters()
                if n.startswith("shared_context_kv.")
            },
            {
                "shared_context_kv.input_norm.weight",
                "shared_context_kv.kv_proj.weight",
                "shared_context_kv.k_norm.weight",
            },
        )

    def test_load_weights(self):
        model = self._model()
        checkpoint = {
            "shared_context_kv.input_norm.weight": torch.randn(FEATURES),
            "shared_context_kv.kv_proj.weight": torch.randn(
                2 * KV_HEADS * HEAD_DIM, FEATURES
            ),
            "shared_context_kv.k_norm.weight": torch.randn(HEAD_DIM),
        }
        model.load_weights(checkpoint.items())
        params = dict(model.named_parameters())
        for name, weight in checkpoint.items():
            torch.testing.assert_close(params[name], weight, rtol=0, atol=0)

    def test_write_matches_reference(self):
        model = self._model()
        shared, attn = model.shared_context_kv, model.layers[0].self_attn
        n = 5
        target_hidden = torch.randn(n, FEATURES)
        positions = torch.arange(n) + 7

        x = _rmsnorm(target_hidden, shared.input_norm.weight)
        kv = (x @ shared.kv_proj.weight.T).view(n, 2, KV_HEADS, HEAD_DIM)
        k = _rmsnorm(kv[:, 0], shared.k_norm.weight).reshape(n, -1)
        k = attn.apply_k_rope(positions, k).view(n, KV_HEADS, HEAD_DIM)
        v = kv[:, 1]

        with torch.no_grad():
            pool = Mock()
            shared.write_context_kv(
                attn=attn,
                pool=pool,
                positions=positions,
                target_hidden=target_hidden,
                cache_loc=torch.arange(n),
            )
            pool.layer_router.layer.assert_called_once_with(0)
            args = pool.layer_router.layer.return_value.store.call_args.args
            torch.testing.assert_close(args[2], k)
            torch.testing.assert_close(args[3], v)

            pool = Mock()
            commit_lens = torch.tensor([3, 2], dtype=torch.int32)
            shared.write_context_kv(
                attn=attn,
                pool=pool,
                positions=positions,
                target_hidden=target_hidden,
                cache_loc=torch.arange(n),
                cache_loc_2d=torch.arange(6).view(2, 3),
                commit_lens=commit_lens,
            )
            args = pool.set_kv_buffer_prefix_valid.call_args.args
            self.assertIs(args[0], attn.attn)
            torch.testing.assert_close(args[3], k)
            torch.testing.assert_close(args[4], v)

    def test_layers_need_one_attention_window(self):
        with self.assertRaisesRegex(ValueError, "one attention window"):
            DFlashDraftModel(
                _config(
                    layer_types=[
                        "sliding_attention",
                        "full_attention",
                        "sliding_attention",
                    ],
                    sliding_window=16,
                    use_sliding_window=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
