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


if __name__ == "__main__":
    unittest.main()
