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
    def test_init_attention_builds_dsa_indexer_and_cli_reuse_policy(self):
        mixin = v32.DeepseekV32Mixin()
        config = SimpleNamespace(
            cli_factor=2,
            indexer_rope_interleave=True,
        )
        indexer = object()

        with (
            mock.patch.object(v32, "is_deepseek_dsa", return_value=True),
            mock.patch.object(v32, "get_dsa_index_n_heads", return_value=64),
            mock.patch.object(v32, "get_dsa_index_head_dim", return_value=128),
            mock.patch.object(v32, "get_dsa_index_topk", return_value=2048),
            mock.patch.object(
                v32, "get_parallel", return_value=SimpleNamespace(attn_cp_size=4)
            ),
            mock.patch.object(v32, "Indexer", return_value=indexer) as indexer_cls,
        ):
            mixin.init_v32_attention(
                config=config,
                hidden_size=7168,
                qk_rope_head_dim=64,
                q_lora_rank=1536,
                max_position_embeddings=163840,
                rope_theta=10000,
                rope_scaling={"rope_type": "deepseek_yarn"},
                quant_config=None,
                layer_id=1,
                alt_stream=None,
                prefix="model.layers.1.self_attn.indexer",
                is_nextn=False,
                dsa_enable_prefill_cp=True,
                mla_enable_prefill_cp=False,
            )

        self.assertTrue(mixin.use_dsa)
        self.assertEqual(mixin.cp_size, 4)
        self.assertIs(mixin.indexer, indexer)
        self.assertTrue(mixin.skip_topk)
        self.assertFalse(mixin.next_skip_topk)
        indexer_kwargs = indexer_cls.call_args.kwargs
        self.assertFalse(indexer_kwargs["is_neox_style"])
        self.assertEqual(indexer_kwargs["prefix"], "model.layers.1.self_attn.indexer")
        self.assertIs(indexer_kwargs["config"], config)

    def test_prefill_cp_v1_is_disabled_when_cp_v2_is_active(self):
        mixin = v32.DeepseekV32Mixin()
        mixin.dsa_enable_prefill_cp = True
        mixin.mla_enable_prefill_cp = False
        forward_batch = object()

        with (
            mock.patch.object(v32, "dsa_use_prefill_cp", return_value=True),
            mock.patch.object(v32, "mla_use_prefill_cp", return_value=False),
            mock.patch.object(v32, "is_cp_v2_active", return_value=True),
        ):
            self.assertFalse(mixin.use_prefill_cp_v1(forward_batch))

    def test_prepare_dsa_cp_metadata_keeps_extend_lengths(self):
        mixin = v32.DeepseekV32Mixin()
        mixin.use_dsa = True
        mixin.dsa_enable_prefill_cp = True
        mixin.mla_enable_prefill_cp = False
        mixin.cp_rank = 1
        mixin.cp_size = 4
        forward_batch = SimpleNamespace(
            attn_cp_metadata=None,
            seq_lens_cpu=torch.tensor([128]),
            extend_seq_lens_cpu=[96],
        )
        metadata = object()

        with (
            mock.patch.object(v32, "is_cp_v2_active", return_value=False),
            mock.patch.object(v32, "can_dsa_cp_split", return_value=True),
            mock.patch.object(
                v32,
                "prepare_context_parallel_metadata",
                return_value=metadata,
            ) as prepare,
        ):
            mixin.maybe_prepare_cp_metadata(96, forward_batch)

        self.assertIs(forward_batch.attn_cp_metadata, metadata)
        prepare.assert_called_once_with(
            96,
            1,
            4,
            [128],
            extend_seqs_len=[96],
        )

    def test_prepare_dense_mla_cp_metadata_uses_generic_split_check(self):
        mixin = v32.DeepseekV32Mixin()
        mixin.use_dsa = False
        mixin.dsa_enable_prefill_cp = False
        mixin.mla_enable_prefill_cp = True
        mixin.cp_rank = 0
        mixin.cp_size = 2
        forward_batch = SimpleNamespace(
            attn_cp_metadata=None,
            seq_lens_cpu=torch.tensor([64]),
            extend_seq_lens_cpu=[64],
        )
        metadata = object()

        with (
            mock.patch.object(v32, "is_cp_v2_active", return_value=False),
            mock.patch.object(v32, "can_cp_split", return_value=True) as can_split,
            mock.patch.object(
                v32,
                "prepare_context_parallel_metadata",
                return_value=metadata,
            ),
        ):
            mixin.maybe_prepare_cp_metadata(64, forward_batch)

        can_split.assert_called_once_with(64, 2, forward_batch)
        self.assertIs(forward_batch.attn_cp_metadata, metadata)

    def test_prepare_cp_metadata_does_nothing_for_cp_v2(self):
        mixin = v32.DeepseekV32Mixin()
        mixin.use_dsa = True
        mixin.dsa_enable_prefill_cp = True
        mixin.mla_enable_prefill_cp = False
        mixin.cp_rank = 0
        mixin.cp_size = 2
        forward_batch = SimpleNamespace(attn_cp_metadata=None)

        with (
            mock.patch.object(v32, "is_cp_v2_active", return_value=True),
            mock.patch.object(v32, "can_dsa_cp_split") as can_split,
        ):
            mixin.maybe_prepare_cp_metadata(64, forward_batch)

        can_split.assert_not_called()
        self.assertIsNone(forward_batch.attn_cp_metadata)

    def test_round_robin_topk_gather_pads_missing_rows(self):
        topk_indices = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

        with (
            mock.patch.object(
                v32, "is_dsa_prefill_cp_round_robin_split", return_value=True
            ),
            mock.patch.object(
                v32,
                "cp_all_gather_rerange_output",
                side_effect=lambda tensor, *_args: tensor,
            ),
        ):
            gathered = v32.DeepseekV32Mixin.gather_dsa_topk_indices_for_cp(
                topk_indices,
                local_num_tokens=4,
                cp_size=2,
                forward_batch=object(),
                stream=object(),
            )

        self.assertEqual(tuple(gathered.shape), (4, 2))
        torch.testing.assert_close(gathered[:2], topk_indices)
        torch.testing.assert_close(
            gathered[2:], torch.full((2, 2), -1, dtype=torch.int32)
        )


if __name__ == "__main__":
    unittest.main()
