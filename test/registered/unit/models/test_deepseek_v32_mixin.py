"""Hermetic unit tests for the DeepSeek V3.2/DSA model mixin."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models.deepseek_common import v32_mixin as v32  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepseekV32Mixin(CustomTestCase):
    @staticmethod
    def attention_kwargs(config, **overrides):
        kwargs = dict(
            config=config,
            hidden_size=7168,
            qk_rope_head_dim=64,
            q_lora_rank=1536,
            max_position_embeddings=163840,
            rope_theta=10000,
            rope_scaling={"rope_type": "deepseek_yarn"},
            quant_config=None,
            layer_id=4,
            alt_stream=None,
            prefix="model.layers.4.self_attn.indexer",
            is_nextn=False,
            skip_rope=False,
        )
        kwargs.update(overrides)
        return kwargs

    def test_init_attention_preserves_dsa_skip_policy_and_indexer_arguments(self):
        mixin = v32.DeepseekV32Mixin()
        config = SimpleNamespace(indexer_rope_interleave=True)
        indexer = object()

        with (
            mock.patch.object(v32, "is_deepseek_dsa", return_value=True),
            mock.patch.object(
                v32, "dsa_layer_skips_topk", side_effect=[False, True]
            ) as skips_topk,
            mock.patch.object(v32, "get_dsa_index_kpool", return_value=1),
            mock.patch.object(v32, "get_dsa_index_n_heads", return_value=64),
            mock.patch.object(v32, "get_dsa_index_head_dim", return_value=128),
            mock.patch.object(v32, "get_dsa_index_topk", return_value=2048),
            mock.patch.object(v32, "Indexer", return_value=indexer) as indexer_cls,
        ):
            mixin.init_v32_attention(**self.attention_kwargs(config))

        self.assertTrue(mixin.use_dsa)
        self.assertFalse(mixin.skip_topk)
        self.assertTrue(mixin.next_skip_topk)
        self.assertIs(mixin.indexer, indexer)
        skips_topk.assert_has_calls([mock.call(config, 4), mock.call(config, 5)])
        indexer_kwargs = indexer_cls.call_args.kwargs
        self.assertFalse(indexer_kwargs["is_neox_style"])
        self.assertEqual(indexer_kwargs["prefix"], "model.layers.4.self_attn.indexer")
        self.assertIs(indexer_kwargs["config"], config)

    def test_init_attention_uses_kpool_indexer_and_forwards_skip_rope(self):
        mixin = v32.DeepseekV32Mixin()
        config = SimpleNamespace(indexer_rope_interleave=False)
        indexer = object()

        with (
            mock.patch.object(v32, "is_deepseek_dsa", return_value=True),
            mock.patch.object(v32, "dsa_layer_skips_topk", return_value=False),
            mock.patch.object(v32, "get_dsa_index_kpool", return_value=2),
            mock.patch.object(v32, "get_dsa_index_n_heads", return_value=64),
            mock.patch.object(v32, "get_dsa_index_head_dim", return_value=128),
            mock.patch.object(v32, "get_dsa_index_topk", return_value=2048),
            mock.patch.object(v32, "IndexerKPool", return_value=indexer) as indexer_cls,
        ):
            mixin.init_v32_attention(**self.attention_kwargs(config, skip_rope=True))

        self.assertIs(mixin.indexer, indexer)
        self.assertTrue(indexer_cls.call_args.kwargs["skip_rope"])
        self.assertTrue(indexer_cls.call_args.kwargs["is_neox_style"])

    def test_nextn_always_constructs_an_indexer_with_reused_topk(self):
        mixin = v32.DeepseekV32Mixin()
        config = SimpleNamespace()

        with (
            mock.patch.object(v32, "is_deepseek_dsa", return_value=True),
            mock.patch.object(v32, "dsa_layer_skips_topk") as skips_topk,
            mock.patch.object(v32, "get_dsa_index_kpool", return_value=1),
            mock.patch.object(v32, "get_dsa_index_n_heads", return_value=64),
            mock.patch.object(v32, "get_dsa_index_head_dim", return_value=128),
            mock.patch.object(v32, "get_dsa_index_topk", return_value=2048),
            mock.patch.object(v32, "Indexer", return_value=object()) as indexer_cls,
        ):
            mixin.init_v32_attention(**self.attention_kwargs(config, is_nextn=True))

        self.assertTrue(mixin.skip_topk)
        self.assertTrue(mixin.next_skip_topk)
        skips_topk.assert_not_called()
        indexer_cls.assert_called_once()

    def test_non_dsa_attention_has_no_indexer(self):
        mixin = v32.DeepseekV32Mixin()

        with (
            mock.patch.object(v32, "is_deepseek_dsa", return_value=False),
            mock.patch.object(v32, "Indexer") as indexer_cls,
            mock.patch.object(v32, "IndexerKPool") as kpool_cls,
        ):
            mixin.init_v32_attention(**self.attention_kwargs(SimpleNamespace()))

        self.assertFalse(mixin.use_dsa)
        self.assertIsNone(mixin.skip_topk)
        self.assertIsNone(mixin.next_skip_topk)
        self.assertIsNone(mixin.indexer)
        indexer_cls.assert_not_called()
        kpool_cls.assert_not_called()

    def test_communicator_selection_matches_prefill_cp_setting(self):
        mixin = v32.DeepseekV32Mixin()
        generic_communicator = object()
        dsa_communicator = object()
        kwargs = dict(
            layer_scatter_modes=object(),
            input_layernorm=object(),
            post_attention_layernorm=object(),
            is_last_layer=False,
            qkv_latent_func=object(),
        )

        with (
            mock.patch.object(
                v32, "LayerCommunicator", return_value=generic_communicator
            ),
            mock.patch.object(
                v32, "DSACPLayerCommunicator", return_value=dsa_communicator
            ),
            mock.patch.object(
                v32,
                "get_parallel",
                return_value=SimpleNamespace(enable_prefill_cp=False),
            ),
        ):
            self.assertIs(
                mixin.create_layer_communicator(**kwargs), generic_communicator
            )

        with (
            mock.patch.object(v32, "LayerCommunicator"),
            mock.patch.object(
                v32, "DSACPLayerCommunicator", return_value=dsa_communicator
            ) as communicator_cls,
            mock.patch.object(
                v32,
                "get_parallel",
                return_value=SimpleNamespace(enable_prefill_cp=True),
            ),
        ):
            self.assertIs(mixin.create_layer_communicator(**kwargs), dsa_communicator)
            self.assertTrue(communicator_cls.call_args.kwargs["allow_reduce_scatter"])

    def test_dsa_forward_topk_uses_primary_backend(self):
        mixin = v32.DeepseekV32Mixin()
        mixin.use_dsa = True
        backend = SimpleNamespace(primary=SimpleNamespace(use_mha=False))

        with mock.patch.object(v32, "get_attn_backend", return_value=backend):
            self.assertTrue(mixin.dsa_forward_uses_topk())

        backend.primary.use_mha = True
        with mock.patch.object(v32, "get_attn_backend", return_value=backend):
            self.assertFalse(mixin.dsa_forward_uses_topk())

    def test_empty_topk_and_attention_context_keep_existing_shapes(self):
        mixin = v32.DeepseekV32Mixin()
        mixin.config = SimpleNamespace()
        mixin.use_dsa = True
        context = mock.Mock()

        with mock.patch.object(v32, "get_dsa_index_topk", return_value=2048):
            empty = mixin.empty_dsa_topk_indices(torch.empty((3, 8)))

        with mock.patch.object(v32, "get_attn_tp_context", return_value=context):
            mixin.init_v32_attn_tp_context(SimpleNamespace(q_lora_rank=1536))

        self.assertEqual(empty.shape, (0, 2048))
        self.assertEqual(empty.dtype, torch.int32)
        context.init_context.assert_called_once_with(1536, True)


if __name__ == "__main__":
    unittest.main()
