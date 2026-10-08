"""A Qwen4-exp PLE layer reads its input on full attention rows."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_declared_decoder_boundary as fixture
import torch
from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.layer_boundary import layer_stack
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.layout import TokenAxis
from sglang.srt.layers.layer_boundary.residual.gated import GatedResidualState
from sglang.srt.models.qwen4_exp import Qwen4ExpPLELayer, _build_qwen4_exp_stages
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _residual_ops():
    def unused(*args, **kwargs):
        raise AssertionError("construction only")

    return GatedResidualState(
        expand=unused,
        attn_mix=unused,
        ffn_mix=unused,
        attn_combine=unused,
        ffn_combine=unused,
    ).residual_ops()


def _sliced_over_attention_tp(layout):
    return TokenAxis.ATTN_TP in layout.sharded


class TestQwen4ExpPleRows(CustomTestCase):
    def test_the_ffn_before_a_ple_layer_hands_on_full_rows(self):
        # An all-to-all MoE with attention TP keeps the residual on this
        # rank's slice of the rows between layers; the PLE embedding is
        # computed for every row, so its layer must read the full rows.
        config = SimpleNamespace(num_hidden_layers=4, ple_layer_ids=[3])
        with (
            fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2), a2a=True),
            layer_stack(),
        ):
            layers = [
                _build_qwen4_exp_stages(
                    _residual_ops(), sparse=True, layer_id=layer_id, config=config
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        ordinary = BatchVariant.ORDINARY
        ffn_hands_on = [
            _sliced_over_attention_tp(ffn.plan.edges[ordinary].outgoing.residual_to)
            for _, ffn in layers
        ]
        attn_reads = [
            _sliced_over_attention_tp(attn.plan.edges[ordinary].incoming.residual)
            for attn, _ in layers
        ]
        # Layer 2 has the PLE: layer 1's FFN gathers the rows back, the last
        # layer's FFN ends the stack on attention rows anyway.
        self.assertEqual(ffn_hands_on, [True, False, True, False])
        self.assertEqual(attn_reads, [False, True, False, True])

    def test_ple_exit_rows_follow_the_active_sequence_parallel_layout(self):
        config = SimpleNamespace(num_hidden_layers=2, ple_layer_ids=[2])
        with (
            fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2), a2a=True),
            patch.object(layernorm_sp, "layernorm_sp_enabled", return_value=True),
            layer_stack(),
        ):
            layers = [
                _build_qwen4_exp_stages(
                    _residual_ops(), sparse=True, layer_id=layer_id, config=config
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        sp = BatchVariant.SEQUENCE_PARALLEL
        self.assertTrue(
            _sliced_over_attention_tp(layers[0][1].plan.edges[sp].outgoing.residual_to)
        )


class _FakeTPGroup:
    world_size = 4

    def __init__(self, rank_in_group=0):
        self.rank_in_group = rank_in_group

    def all_gather_into_tensor(self, output, local):
        output.copy_(torch.cat([local] * self.world_size))


class _Embedding:
    enable_ple_fusion = False

    def __init__(self, embedding_rows):
        self.embedding_rows = embedding_rows

    def __call__(self, batch, forward_batch):
        return torch.arange(self.embedding_rows, dtype=torch.float32).unsqueeze(-1)


class TestQwen4ExpPleSequenceParallelRows(CustomTestCase):
    def _layer(self, physical_tokens, conv_rows, projected_embeddings=None):
        layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
        torch.nn.Module.__init__(layer)
        layer.hidden_size = 1
        layer.hc_count = 1
        layer._prefetch_state = None
        layer.ple_embedding = _Embedding(physical_tokens)

        def key_proj(value):
            if projected_embeddings is not None:
                projected_embeddings.append(value.clone())
            return value, None

        layer.key_proj = key_proj
        layer.value_proj = lambda value: (value, None)
        layer.norm_key = layer.norm_query = layer.norm_conv = None
        layer._apply_ple_norm = lambda norm, value: value

        def short_conv(value, forward_batch, batch):
            conv_rows.append(value.shape[0])
            return torch.zeros_like(value)

        layer._short_conv = short_conv
        return layer

    @staticmethod
    def _batch(processed_tokens, physical_tokens):
        return SimpleNamespace(
            processed_tokens=processed_tokens,
            physical_tokens=physical_tokens,
            mode=SimpleNamespace(is_target_verify=lambda: False),
            use_decode_fast_path=False,
            valid_tokens=torch.ones(processed_tokens, dtype=torch.bool),
        )

    @staticmethod
    def _forward_batch():
        return SimpleNamespace(
            _original_forward_mode=None,
            forward_mode=SimpleNamespace(),
        )

    def test_processed_rows_smaller_than_local_physical_shard(self):
        conv_rows = []
        layer = self._layer(physical_tokens=17, conv_rows=conv_rows)
        hidden_states = torch.ones(5, 1)
        with (
            patch(
                "sglang.srt.models.qwen4_exp.get_parallel",
                return_value=SimpleNamespace(tp_group=_FakeTPGroup()),
            ),
            patch(
                "sglang.srt.models.qwen4_exp.layernorm_sp.runs_sp", return_value=True
            ),
        ):
            output = layer(
                hidden_states,
                self._forward_batch(),
                self._batch(processed_tokens=2, physical_tokens=17),
            )
        self.assertEqual(output.shape, (5, 1))
        self.assertEqual(conv_rows, [2])

    def test_non_divisible_physical_rows_keep_ceil_divided_local_shape(self):
        conv_rows = []
        layer = self._layer(physical_tokens=17, conv_rows=conv_rows)
        with (
            patch(
                "sglang.srt.models.qwen4_exp.get_parallel",
                return_value=SimpleNamespace(tp_group=_FakeTPGroup()),
            ),
            patch(
                "sglang.srt.models.qwen4_exp.layernorm_sp.runs_sp", return_value=True
            ),
        ):
            output = layer(
                torch.ones(5, 1),
                self._forward_batch(),
                self._batch(processed_tokens=17, physical_tokens=17),
            )
        self.assertEqual(output.shape, (5, 1))
        self.assertEqual(conv_rows, [17])

    def test_graph_replay_may_grow_processed_rows_in_the_same_bucket(self):
        conv_rows = []
        layer = self._layer(physical_tokens=17, conv_rows=conv_rows)
        parallel = SimpleNamespace(tp_group=_FakeTPGroup())
        with (
            patch("sglang.srt.models.qwen4_exp.get_parallel", return_value=parallel),
            patch(
                "sglang.srt.models.qwen4_exp.layernorm_sp.runs_sp", return_value=True
            ),
        ):
            for processed_tokens in (2, 7):
                output = layer(
                    torch.ones(5, 1),
                    self._forward_batch(),
                    self._batch(processed_tokens=processed_tokens, physical_tokens=17),
                )
                self.assertEqual(output.shape, (5, 1))
        self.assertEqual(conv_rows, [2, 7])

    def _assert_pad_then_shard_layout(self, *, embedding_rows, physical_tokens, rank):
        conv_rows = []
        projected_embeddings = []
        group = _FakeTPGroup(rank)
        local_rows = (physical_tokens + group.world_size - 1) // group.world_size
        layer = self._layer(embedding_rows, conv_rows, projected_embeddings)
        with (
            patch(
                "sglang.srt.models.qwen4_exp.get_parallel",
                return_value=SimpleNamespace(tp_group=group),
            ),
            patch(
                "sglang.srt.models.qwen4_exp.layernorm_sp.runs_sp", return_value=True
            ),
        ):
            output = layer(
                torch.ones(local_rows, 1),
                self._forward_batch(),
                self._batch(
                    processed_tokens=embedding_rows,
                    physical_tokens=physical_tokens,
                ),
            )

        full = torch.arange(embedding_rows, dtype=torch.float32).unsqueeze(-1)
        full = torch.nn.functional.pad(
            full, (0, 0, 0, physical_tokens - embedding_rows)
        )
        expected = layernorm_sp.shard_token_rows(full, group=group)
        torch.testing.assert_close(projected_embeddings[0], expected)
        self.assertEqual(output.shape, (local_rows, 1))

    def test_ple_embeddings_pad_5624_rows_to_8192_before_tp4_sharding(self):
        for rank in range(4):
            with self.subTest(rank=rank):
                self._assert_pad_then_shard_layout(
                    embedding_rows=5624,
                    physical_tokens=8192,
                    rank=rank,
                )

    def test_ple_embeddings_pad_non_divisible_physical_rows_before_tp4_sharding(
        self,
    ):
        for rank in range(4):
            with self.subTest(rank=rank):
                self._assert_pad_then_shard_layout(
                    embedding_rows=17,
                    physical_tokens=22,
                    rank=rank,
                )


if __name__ == "__main__":
    unittest.main()
