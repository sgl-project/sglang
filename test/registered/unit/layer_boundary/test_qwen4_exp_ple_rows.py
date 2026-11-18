"""A Qwen4-exp PLE layer reads its input on full attention rows."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import test_declared_decoder_boundary as fixture
import torch

from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.layer_boundary import layer_stack
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.layout import TokenAxis
from sglang.srt.layers.layer_boundary.residual.gated import GatedResidualState
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import qwen4_exp
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpPerTokenEmbedding,
    Qwen4ExpPLELayer,
    Qwen4ExpVLModel,
    _build_qwen4_exp_stages,
    _check_per_token_table,
    _has_ple,
    _is_ngram_only_ple_tensor,
    _note_per_token_tables,
    _prepare_ple_batch,
    _sp_moe_keeps_local_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _residual_ops(*, attn_reads_every_row):
    def unused(*args, **kwargs):
        raise AssertionError("construction only")

    return GatedResidualState(
        expand=unused,
        attn_mix=unused,
        ffn_mix=unused,
        attn_combine=unused,
        ffn_combine=unused,
        attn_reads_every_row=attn_reads_every_row,
    ).residual_ops()


def _sliced_over_attention_tp(layout):
    return TokenAxis.ATTN_TP in layout.sharded


class TestQwen4ExpPleRows(CustomTestCase):
    def test_the_ffn_before_a_ple_layer_hands_on_full_rows(self):
        # An all-to-all MoE with attention TP keeps the residual on this
        # rank's slice of the rows between layers; the PLE embedding is
        # computed for every row, so its layer must read the full rows.
        config = SimpleNamespace(
            model_type="qwen4_exp_text", num_hidden_layers=4, ple_layer_ids=[3]
        )
        with (
            fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2), a2a=True),
            layer_stack(),
        ):
            layers = [
                _build_qwen4_exp_stages(
                    config,
                    layer_id,
                    _residual_ops(attn_reads_every_row=_has_ple(layer_id, config)),
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
        config = SimpleNamespace(
            model_type="qwen4_exp_text", num_hidden_layers=2, ple_layer_ids=[2]
        )
        with (
            fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2), a2a=True),
            patch.object(layernorm_sp, "layernorm_sp_enabled", return_value=True),
            layer_stack(),
        ):
            layers = [
                _build_qwen4_exp_stages(
                    config,
                    layer_id,
                    _residual_ops(attn_reads_every_row=_has_ple(layer_id, config)),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        sp = BatchVariant.SEQUENCE_PARALLEL
        self.assertTrue(
            _sliced_over_attention_tp(layers[0][1].plan.edges[sp].outgoing.residual_to)
        )


class _FakeTPGroup:
    world_size = 4
    rank_in_group = 3

    def all_gather_into_tensor(self, output, local):
        output.copy_(torch.cat([local] * self.world_size))


class TestQwen4ExpPleSequenceParallelRows(CustomTestCase):
    @staticmethod
    def _layer(embedding_rows):
        layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
        torch.nn.Module.__init__(layer)
        layer.hidden_size = 1
        layer.hc_count = 1
        layer._prefetch_state = None
        layer.ple_embedding = MagicMock(
            return_value=torch.arange(embedding_rows, dtype=torch.float32).unsqueeze(-1)
        )
        layer.ple_embedding.enable_ple_fusion = False
        layer.key_proj = MagicMock(side_effect=lambda value: (value, None))
        layer.value_proj = lambda value: (value, None)
        layer.norm_key = layer.norm_query = layer.norm_conv = None
        layer._apply_ple_norm = lambda norm, value: value
        layer._short_conv = lambda value, forward_batch, batch: torch.zeros_like(value)
        return layer

    @staticmethod
    def _batch(embedding_rows, physical_tokens):
        return SimpleNamespace(
            processed_tokens=embedding_rows,
            physical_tokens=physical_tokens,
            mode=SimpleNamespace(is_target_verify=lambda: False),
            use_decode_fast_path=False,
            valid_tokens=torch.ones(embedding_rows, dtype=torch.bool),
        )

    def test_ple_embeddings_pad_before_tp4_sharding(self):
        embedding_rows, physical_tokens = 17, 22
        group = _FakeTPGroup()
        local_rows = (physical_tokens + group.world_size - 1) // group.world_size
        layer = self._layer(embedding_rows)
        with (
            patch(
                "sglang.srt.models.qwen4_exp.get_parallel",
                return_value=SimpleNamespace(tp_group=group),
            ),
            patch(
                "sglang.srt.models.qwen4_exp.layernorm_sp.runs_sp",
                return_value=True,
            ),
        ):
            output = layer(
                torch.ones(local_rows, 1),
                SimpleNamespace(
                    _original_forward_mode=None,
                    forward_mode=SimpleNamespace(),
                ),
                self._batch(embedding_rows, physical_tokens),
            )

        full = torch.arange(embedding_rows, dtype=torch.float32).unsqueeze(-1)
        full = torch.nn.functional.pad(
            full, (0, 0, 0, physical_tokens - embedding_rows)
        )
        expected = layernorm_sp.shard_token_rows(full, group=group)
        torch.testing.assert_close(layer.key_proj.call_args.args[0], expected)
        self.assertEqual(output.shape, (local_rows, 1))

    def test_returned_hc_hidden_states_gather_all_tp4_rows(self):
        local_hc = torch.arange(2, dtype=torch.float32).unsqueeze(-1)
        forward_batch = SimpleNamespace(
            forward_mode=object(),
            capture_hidden_mode=SimpleNamespace(need_capture=lambda: True),
        )
        with (
            patch.object(layernorm_sp, "runs_sp", return_value=True),
            patch.object(
                layernorm_sp,
                "get_parallel",
                return_value=SimpleNamespace(tp_group=_FakeTPGroup()),
            ),
        ):
            hidden_states = Qwen4ExpVLModel._captured_hc_hidden_states(
                local_hc, torch.arange(8), forward_batch
            )

        torch.testing.assert_close(hidden_states, local_hc.repeat(4, 1))

    def test_complete_a2a_moe_keeps_sequence_parallel_rows_local(self):
        for backend_name in ("deepep", "megamoe"):
            backend = SimpleNamespace(
                is_deepep=lambda: backend_name == "deepep",
                is_megamoe=lambda: backend_name == "megamoe",
            )
            with (
                self.subTest(backend=backend_name),
                patch(
                    "sglang.srt.models.qwen4_exp.get_moe_a2a_backend",
                    return_value=backend,
                ),
            ):
                self.assertTrue(_sp_moe_keeps_local_rows())


_TABLE = "model.layers.1.ple.ple_embedding.per_token_table.weight"
# models/qwen4_exp.py logs through sglang.srt.utils.logger.
_MODEL_LOGGER = "sglang.srt.utils.common"


class _LoadTarget(torch.nn.Module):
    """What Qwen4ExpForConditionalGeneration.load_weights reads, around one
    per-token PLE layer and one ordinary parameter."""

    load_weights = Qwen4ExpForConditionalGeneration.load_weights
    _load_qwen4_exp_ple_buffer = (
        Qwen4ExpForConditionalGeneration._load_qwen4_exp_ple_buffer
    )

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False, encoder_only=False)
        self.pp_group = SimpleNamespace(is_last_rank=True)
        self.start_layer, self.end_layer = 0, 2
        self.language_model_only = True
        self.other = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(torch.nn.Module() for _ in range(2))
        self.model.layers[1].ple = torch.nn.Module()
        self.model.layers[1].ple.ple_embedding = Qwen4ExpPerTokenEmbedding(
            SimpleNamespace(vocab_size=8), embedding_dim=4
        )


class TestQwen4ExpPerTokenPle(CustomTestCase):
    @staticmethod
    def _embedding() -> Qwen4ExpPerTokenEmbedding:
        return Qwen4ExpPerTokenEmbedding(SimpleNamespace(vocab_size=8), embedding_dim=4)

    def test_lookup_reads_one_row_per_token_in_row_order(self):
        embedding = self._embedding()
        weight = embedding.per_token_table.weight
        # Zeros until a load, and not trainable.
        self.assertFalse(weight.any() or weight.requires_grad)
        table = torch.arange(32.0).view(8, 4)
        weight.data.copy_(table)
        # Ids outside the table read its first or last row: 99 and -3 here.
        batch = SimpleNamespace(token_ids=torch.tensor([[5, 1], [7, 99], [-3, 0]]))

        rows = embedding(batch, forward_batch=None)

        self.assertTrue(torch.equal(rows, table[[5, 1, 7, 7, 0, 0]]))

    def test_the_batch_carries_the_processed_tokens_and_no_ngram_history(self):
        # Six physical rows, four of them real, as a padded decode batch.
        forward_batch = SimpleNamespace(
            tbo_parent_token_range=None,
            spec_algorithm=None,
            spec_info=None,
            _original_forward_mode=None,
            forward_mode=ForwardMode.DECODE,
            global_num_token_non_padded_cpu=4,
            extend_seq_lens_cpu=None,
            extend_seq_lens=None,
            out_cache_loc=None,
        )
        backend = SimpleNamespace(
            forward_state_slots=lambda count, _: SimpleNamespace(
                working=torch.arange(count)
            )
        )
        with (
            patch.object(qwen4_exp, "get_req_to_token_pool", SimpleNamespace),
            patch.object(qwen4_exp, "get_attn_backend", return_value=backend),
            patch.object(qwen4_exp, "get_token_to_kv_pool", side_effect=AssertionError),
        ):
            batch = _prepare_ple_batch(
                torch.tensor([4, 3, 2, 1, 7, 7]),
                forward_batch,
                ngram_size=None,
                ngram_eos_token_id=None,
            )
        self.assertEqual(batch.token_ids.tolist(), [4, 3, 2, 1])
        self.assertIsNone(batch.ngram_context)

    def test_table_takes_the_model_dtype_and_a_plain_float_checkpoint(self):
        embedding = self._embedding()
        tables = {_TABLE: embedding}
        table = embedding.per_token_table.weight
        self.assertEqual(table.dtype, torch.get_default_dtype())
        exported = torch.zeros(8, 4, dtype=torch.bfloat16)

        for good in (exported, exported.half(), exported.float()):
            _check_per_token_table(
                name=_TABLE, loaded_weight=good, per_token_tables=tables
            )
        # Half the rows, integers, doubles, and fp8 codes without a scale.
        for bad in (
            exported[:4],
            exported.long(),
            exported.double(),
            exported.to(torch.float8_e4m3fn),
        ):
            with self.subTest(dtype=bad.dtype, shape=tuple(bad.shape)):
                with self.assertRaises(ValueError):
                    _check_per_token_table(
                        name=_TABLE, loaded_weight=bad, per_token_tables=tables
                    )
        # An n-gram model would otherwise skip the table with a warning.
        with self.assertRaises(ValueError):
            _check_per_token_table(
                name=_TABLE, loaded_weight=exported, per_token_tables={}
            )

    def test_only_the_ngram_tensors_of_a_ple_layer_are_set_aside(self):
        prefix = "model.layers.1.ple.ple_embedding."
        for leaf in (
            "ngram_embedding.shard_0.weight",
            "ngram_embedding.shard_37.weight",
            "layer_multipliers",
            "ngram_heads_offsets",
            "ngram_heads_vocab_sizes",
        ):
            self.assertTrue(_is_ngram_only_ple_tensor(prefix + leaf), leaf)
        for name in (
            _TABLE,
            "model.layers.1.ple.conv1d.weight",
            "model.layers.1.ngram_embedding.shard_0.weight",
        ):
            self.assertFalse(_is_ngram_only_ple_tensor(name), name)

    def test_an_absent_table_is_a_warning_until_one_load_has_copied_it(self):
        embedding = self._embedding()
        tables = {_TABLE: embedding}

        with self.assertLogs(_MODEL_LOGGER, level="WARNING") as logs:
            _note_per_token_tables(per_token_tables=tables, loaded_params={"other"})
        self.assertIn(_TABLE, logs.output[0])
        self.assertFalse(embedding.table_loaded)
        with self.assertNoLogs(_MODEL_LOGGER, level="WARNING"):
            _note_per_token_tables(per_token_tables=tables, loaded_params={_TABLE})
            self.assertTrue(embedding.table_loaded)
            _note_per_token_tables(per_token_tables=tables, loaded_params={"other"})

    def test_load_weights_takes_the_table_and_sets_the_ngram_tensors_aside(self):
        table = torch.arange(32.0).view(8, 4).to(torch.bfloat16)
        # The checkpoint spells the prefix "model.language_model.".
        table_bucket = [(_TABLE.replace("model.", "model.language_model.", 1), table)]
        prefix = "model.layers.1.ple.ple_embedding."
        # A checkpoint may still carry the n-gram tensors.
        ngram_bucket = [
            (prefix + "ngram_embedding.shard_37.weight", torch.zeros(4, 3)),
            (prefix + "layer_multipliers", torch.zeros(2)),
        ]
        other = lambda value: [("other", torch.tensor([value]))]  # noqa: E731

        # A staged load brings its buckets in any order; none of them is an error.
        for first, second in (
            (table_bucket, ngram_bucket),
            (ngram_bucket, table_bucket),
        ):
            with self.subTest(table_first=first is table_bucket):
                target = _LoadTarget()
                embedding = target.model.layers[1].ple.ple_embedding
                with self.assertLogs(_MODEL_LOGGER, level="WARNING"):
                    target.load_weights(other(5.0))
                if first is ngram_bucket:
                    with self.assertLogs(_MODEL_LOGGER, level="WARNING"):
                        self.assertEqual(target.load_weights(first), set())
                    self.assertFalse(embedding.table_loaded)
                    self.assertEqual(target.load_weights(second), {_TABLE})
                else:
                    with self.assertNoLogs(_MODEL_LOGGER, level="WARNING"):
                        self.assertEqual(target.load_weights(first), {_TABLE})
                        self.assertEqual(target.load_weights(second), set())
                with self.assertNoLogs(_MODEL_LOGGER, level="WARNING"):
                    target.load_weights(other(9.0))
                self.assertTrue(
                    torch.equal(embedding.per_token_table.weight, table.float())
                )
                self.assertEqual(target.other.item(), 9.0)

        target = _LoadTarget()
        with self.assertRaisesRegex(ValueError, "expected a bf16, fp16 or fp32"):
            target.load_weights([(_TABLE, table.to(torch.float8_e4m3fn))])
        self.assertFalse(target.model.layers[1].ple.ple_embedding.table_loaded)


if __name__ == "__main__":
    unittest.main()
