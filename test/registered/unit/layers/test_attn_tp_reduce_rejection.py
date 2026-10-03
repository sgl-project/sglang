"""Layers that shard over attention TP but all-reduce over the full TP group
refuse the layouts where the two groups differ."""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.dp_attention import (
    initialize_dp_attention_flags,
    reject_attn_tp_shard_with_tp_reduce,
)
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestAttnTpReduceRejection(CustomTestCase):
    def publish(self, **fields):
        reset_context()
        self.addCleanup(reset_context)
        server_args = ServerArgs(model_path="dummy", device="cpu", **fields)
        publish(server_args, role="test", ranks=SpawnRanks(world_rank=0))
        initialize_dp_attention_flags(server_args)

    def test_only_a_narrower_wider_than_one_shard_with_a_tp_reduce_is_refused(self):
        self.publish(tp_size=4)
        with get_parallel().override(tp_size=4):
            with self.assertRaisesRegex(ValueError, "Layer shards over the attention"):
                reject_attn_tp_shard_with_tp_reduce(
                    "Layer", shard_tp_size=2, reduces_over_attn_tp=False
                )
            for shard_tp_size, reduces_over_attn_tp in (
                (2, True),
                (1, False),
                (4, False),
            ):
                with self.subTest(shard=shard_tp_size, reduces=reduces_over_attn_tp):
                    reject_attn_tp_shard_with_tp_reduce(
                        "Layer",
                        shard_tp_size=shard_tp_size,
                        reduces_over_attn_tp=reduces_over_attn_tp,
                    )

    def test_an_attention_that_always_reduces_over_tp(self):
        from sglang.srt.models.clip import CLIPAttention

        config = SimpleNamespace(
            hidden_size=16, num_attention_heads=4, attention_dropout=0.0
        )
        self.publish(tp_size=4, attn_dp_size=2)
        with self.assertRaisesRegex(ValueError, "CLIPAttention shards over"):
            CLIPAttention(config)
        # A one-rank attention-TP shard does not reduce at all.
        self.publish(tp_size=4, attn_dp_size=4)
        CLIPAttention(config)
        self.publish(tp_size=4)
        CLIPAttention(config)

    def test_a_layer_that_reduces_over_attention_tp_only_with_attention_dp(self):
        from sglang.srt.models.qwen3_vl import Qwen3_VisionMLP

        def build():
            return Qwen3_VisionMLP(16, 32, bias=False, hidden_act="relu")

        self.publish(tp_size=4, attn_cp_size=2)
        with self.assertRaisesRegex(ValueError, "Qwen3_VisionMLP shards over"):
            build()
        for fields in (
            dict(tp_size=4, attn_dp_size=2),
            dict(tp_size=8, attn_dp_size=2, attn_cp_size=2),
            dict(tp_size=4),
        ):
            with self.subTest(**fields):
                self.publish(**fields)
                build()
        # The data-parallel encoder runs one-rank shards.
        self.publish(tp_size=4, attn_cp_size=2)
        Qwen3_VisionMLP(16, 32, bias=False, hidden_act="relu", use_data_parallel=True)

    def test_a_worker_that_never_forwards_its_tower(self):
        """Most models build their multimodal tower unconditionally. Where the
        tower is not forwarded, the layout it would need must not be refused."""
        from sglang.srt.models.clip import CLIPAttention

        config = SimpleNamespace(
            hidden_size=16, num_attention_heads=4, attention_dropout=0.0
        )
        self.publish(tp_size=4, attn_dp_size=2)
        with self.assertRaisesRegex(ValueError, "CLIPAttention shards over"):
            CLIPAttention(config)
        for fields in (
            # The encoder instance forwards the tower instead.
            dict(language_only=True),
            # Multimodal requests are rejected outright.
            dict(language_model_only=True),
            # A decode instance embeds nothing.
            dict(disaggregation_mode="decode"),
        ):
            with self.subTest(**fields):
                self.publish(tp_size=4, attn_dp_size=2, **fields)
                CLIPAttention(config)

    def test_a_text_layer_is_checked_on_every_worker(self):
        """The exemption belongs to towers that do not run, not to the worker:
        a text attention still reduces on a language-only instance."""
        from sglang.srt.models.phimoe import PhiMoEAttention

        for fields in (
            {},
            dict(language_only=True),
            dict(language_model_only=True),
            dict(disaggregation_mode="decode"),
        ):
            with self.subTest(**fields):
                self.publish(tp_size=4, attn_dp_size=2, **fields)
                with self.assertRaisesRegex(ValueError, "PhiMoEAttention shards over"):
                    PhiMoEAttention(hidden_size=16, num_heads=4, num_kv_heads=4)

    def test_the_other_gated_attentions_refuse_the_same_layout(self):
        """These models have no checkpoint in CI, so state the gate on the
        layer classes themselves. Only PhiMoE builds from a stub config, so it
        carries the accepted-layout half."""
        from sglang.srt.models.exaone_moe import ExaoneMoEAttention
        from sglang.srt.models.phimoe import PhiMoEAttention

        def phimoe():
            return PhiMoEAttention(hidden_size=16, num_heads=4, num_kv_heads=4)

        def exaone():
            return ExaoneMoEAttention(
                config=SimpleNamespace(
                    attention_bias=False, head_dim=4, rms_norm_eps=1e-6
                ),
                hidden_size=16,
                num_heads=4,
                num_kv_heads=4,
            )

        for name, build in (
            ("PhiMoEAttention", phimoe),
            ("ExaoneMoEAttention", exaone),
        ):
            with self.subTest(layer=name):
                self.publish(tp_size=4, attn_dp_size=2)
                with self.assertRaisesRegex(ValueError, f"{name} shards over"):
                    build()

        self.publish(tp_size=4)
        phimoe()

    def test_a_replicated_branch_is_not_gated(self):
        """MoonViT's tensor-parallel MLP is refused, while its ModelSlim branch
        builds replicated layers that neither shard nor reduce."""
        from sglang.srt.models.kimi_vl_moonvit import MLP2

        self.publish(tp_size=4, attn_dp_size=2)
        with self.assertRaisesRegex(ValueError, "MLP2 shards over"):
            MLP2(dims=[16, 32, 16], activation=None, use_tensor_parallel=True)
        # Without tensor parallelism the shard is one rank wide.
        MLP2(dims=[16, 32, 16], activation=None)


if __name__ == "__main__":
    unittest.main()
