"""CPU regressions for DFlash draft checkpoint configuration."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models import dflash
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative import dflash_worker_v2
from sglang.srt.speculative.dflash_utils import (
    get_dflash_attention_sliding_window_size,
)
from sglang.srt.speculative.dflash_worker_v2 import _get_dflash_embedding_module


class _FakeDraftLayer(nn.Module):
    calls = []

    def __init__(self, **kwargs):
        super().__init__()
        self.calls.append(kwargs)


class _FakeAttention(nn.Module):
    calls = []

    def __init__(self, **kwargs):
        super().__init__()
        self.calls.append(kwargs)


class _FakeMLP(nn.Module):
    calls = []

    def __init__(self, **kwargs):
        super().__init__()
        self.calls.append(kwargs)


class _FakeNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _FakeEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs


class _FakeReplicatedLinear(nn.Module):
    calls = []

    def __init__(
        self,
        input_size,
        output_size,
        *,
        bias,
        quant_config,
        prefix,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.in_features = input_size
        self.calls.append(
            {
                "input_size": input_size,
                "output_size": output_size,
                "bias": bias,
                "quant_config": quant_config,
                "prefix": prefix,
            }
        )


class _FakeColumnParallelLinear(_FakeReplicatedLinear):
    pass


class _FakeQKVParallelLinear(nn.Module):
    calls = []

    def __init__(self, *args, prefix, **kwargs):
        super().__init__()
        self.calls.append({"prefix": prefix})
        self.register_parameter(
            "weight", nn.Parameter(torch.empty(0), requires_grad=False)
        )


class _FakeRowParallelLinear(_FakeReplicatedLinear):
    calls = []

    def __init__(self, *args, prefix, **kwargs):
        super().__init__(*args, prefix=prefix, **kwargs)
        self.register_parameter(
            "weight", nn.Parameter(torch.empty(0), requires_grad=False)
        )


class _FakeGatedColumnParallelLinear(_FakeReplicatedLinear):
    calls = []

    def __init__(self, *args, prefix, **kwargs):
        super().__init__(*args, prefix=prefix, **kwargs)
        self.register_parameter(
            "weight", nn.Parameter(torch.empty(0), requires_grad=False)
        )


class _FakeRadixAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _MinimalDraftModel(DFlashDraftModel):
    decoder_layer_cls = _FakeDraftLayer


class _MinimalDecoderLayer(dflash.DFlashDecoderLayer):
    attention_cls = _FakeAttention


class _WeightLoadingDraft(nn.Module):
    load_weights = DFlashDraftModel.load_weights

    def __init__(self, weight, *, hidden_size=4, num_context_features=6):
        super().__init__()
        self.fc = nn.Module()
        self.fc.register_parameter("weight", weight)
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.num_context_features = num_context_features


class _LagunaWeightLoadingDraft(nn.Module):
    load_weights = DFlashDraftModel.load_weights

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].self_attn = nn.Module()
        self.events = []

        qkv_proj = nn.Module()
        qkv_weight = nn.Parameter(torch.empty((4, 4)), requires_grad=False)

        def load_qkv(param, loaded_weight, shard_id):
            self.events.append(("qkv_proj", shard_id, tuple(loaded_weight.shape)))

        qkv_weight.weight_loader = load_qkv
        qkv_proj.register_parameter("weight", qkv_weight)
        self.layers[0].self_attn.qkv_proj = qkv_proj

        for projection_name in ("o_proj", "g_proj"):
            projection = nn.Module()
            weight = nn.Parameter(torch.empty((4, 4)), requires_grad=False)

            def load_projection(param, loaded_weight, name=projection_name):
                self.events.append((name, None, tuple(loaded_weight.shape)))

            weight.weight_loader = load_projection
            projection.register_parameter("weight", weight)
            setattr(self.layers[0].self_attn, projection_name, projection)


def _draft_config(*, has_embed_tokens):
    return SimpleNamespace(
        hidden_size=4,
        num_hidden_layers=6,
        rms_norm_eps=1e-6,
        vocab_size=128,
        has_embed_tokens=has_embed_tokens,
        dflash_config={
            "block_size": 8,
            "target_layer_ids": [1, 5, 19, 29, 41, 51],
        },
    )


class TestDFlashDraftModel(unittest.TestCase):
    @patch.object(dflash, "VocabParallelEmbedding", _FakeEmbedding)
    @patch.object(dflash, "RMSNorm", _FakeNorm)
    @patch.object(dflash, "ReplicatedLinear", _FakeReplicatedLinear)
    def test_explicit_target_ids_are_not_validated_against_draft_depth(self):
        _FakeDraftLayer.calls.clear()
        _FakeReplicatedLinear.calls.clear()
        quant_config = object()
        model = _MinimalDraftModel(
            _draft_config(has_embed_tokens=True),
            quant_config=quant_config,
            prefix="draft",
        )

        self.assertEqual(model.num_context_features, 6)
        self.assertEqual(model.fc.in_features, 24)
        self.assertEqual(model.block_size, 8)
        self.assertIs(model.get_input_embeddings(), model.embed_tokens)
        self.assertEqual(model.embed_tokens.kwargs["prefix"], "draft.embed_tokens")
        self.assertEqual(
            [call["prefix"] for call in _FakeDraftLayer.calls],
            [f"draft.layers.{i}" for i in range(6)],
        )
        self.assertEqual(_FakeReplicatedLinear.calls[0]["prefix"], "draft.fc")
        self.assertIs(_FakeReplicatedLinear.calls[0]["quant_config"], quant_config)

    @patch.object(dflash, "RMSNorm", _FakeNorm)
    @patch.object(dflash, "ReplicatedLinear", _FakeReplicatedLinear)
    def test_checkpoint_without_embedding_uses_fallback_contract(self):
        model = _MinimalDraftModel(_draft_config(has_embed_tokens=False))

        self.assertIsNone(model.get_input_embeddings())


class TestDFlashDecoderLayer(unittest.TestCase):
    def test_sliding_window_falls_back_to_published_dflash_config(self):
        config = SimpleNamespace(
            layer_types=["sliding_attention"],
            dflash_config={"swa_window_size": 1024},
        )

        self.assertEqual(get_dflash_attention_sliding_window_size(config), 1023)

    @patch.object(dflash, "DFlashMLP", _FakeMLP)
    @patch.object(dflash, "RMSNorm", _FakeNorm)
    def test_attention_receives_quant_config_and_full_prefix(self):
        _FakeAttention.calls.clear()
        _FakeMLP.calls.clear()
        quant_config = object()

        _MinimalDecoderLayer(
            SimpleNamespace(hidden_size=4, rms_norm_eps=1e-6),
            layer_id=2,
            quant_config=quant_config,
            prefix="draft.layers.2",
        )

        self.assertEqual(_FakeAttention.calls[0]["prefix"], "draft.layers.2.self_attn")
        self.assertIs(_FakeAttention.calls[0]["quant_config"], quant_config)
        self.assertEqual(_FakeMLP.calls[0]["prefix"], "draft.layers.2.mlp")

    @patch.object(dflash, "ColumnParallelLinear", _FakeColumnParallelLinear)
    @patch.object(dflash, "DFlashMLP", _FakeMLP)
    @patch.object(dflash, "RMSNorm", _FakeNorm)
    def test_laguna_attention_propagates_nested_prefix(self):
        _FakeColumnParallelLinear.calls.clear()
        seen = {}

        def fake_attention_init(
            attention, config, layer_id, quant_config=None, prefix=""
        ):
            nn.Module.__init__(attention)
            seen.update(
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=prefix,
            )
            attention.total_num_heads = 2
            attention.num_heads = 2
            attention.head_dim = 2

        config = SimpleNamespace(
            hidden_size=4,
            rms_norm_eps=1e-6,
            gating=True,
        )
        quant_config = object()
        with patch.object(dflash.DFlashAttention, "__init__", fake_attention_init):
            dflash.DFlashLagunaDecoderLayer(
                config,
                layer_id=2,
                quant_config=quant_config,
                prefix="draft.layers.2",
            )

        self.assertEqual(seen["prefix"], "draft.layers.2.self_attn")
        self.assertIs(seen["quant_config"], quant_config)
        self.assertEqual(
            _FakeColumnParallelLinear.calls[0]["prefix"],
            "draft.layers.2.self_attn.g_proj",
        )

    @patch.object(dflash, "ColumnParallelLinear", _FakeColumnParallelLinear)
    def test_laguna_attention_preserves_root_g_proj_prefix(self):
        _FakeColumnParallelLinear.calls.clear()

        def fake_attention_init(
            attention, config, layer_id, quant_config=None, prefix=""
        ):
            nn.Module.__init__(attention)
            attention.total_num_heads = 2
            attention.num_heads = 2
            attention.head_dim = 2

        with patch.object(dflash.DFlashAttention, "__init__", fake_attention_init):
            dflash.DFlashLagunaAttention(
                SimpleNamespace(hidden_size=4, gating=True), layer_id=0
            )

        self.assertEqual(_FakeColumnParallelLinear.calls[0]["prefix"], "g_proj")

    @patch.object(dflash, "QKVParallelLinear", _FakeQKVParallelLinear)
    @patch.object(dflash, "RowParallelLinear", _FakeRowParallelLinear)
    @patch.object(dflash, "ColumnParallelLinear", _FakeGatedColumnParallelLinear)
    @patch.object(dflash, "RMSNorm", _FakeNorm)
    @patch.object(dflash, "RadixAttention", _FakeRadixAttention)
    @patch.object(dflash, "get_parallel", return_value=SimpleNamespace(tp_size=1))
    @patch.object(dflash, "get_rope_config", return_value=(10000.0, None))
    @patch.object(
        dflash,
        "get_rope",
        return_value=SimpleNamespace(
            cos_sin_cache=None,
            rotary_dim=2,
            is_neox_style=False,
        ),
    )
    def test_laguna_projection_prefixes_and_state_dict_names(
        self, _get_rope, _get_rope_config, _get_parallel
    ):
        _FakeQKVParallelLinear.calls.clear()
        _FakeRowParallelLinear.calls.clear()
        _FakeGatedColumnParallelLinear.calls.clear()

        attention = dflash.DFlashLagunaAttention(
            SimpleNamespace(
                hidden_size=4,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=2,
                max_position_embeddings=128,
                gating=True,
            ),
            layer_id=0,
            prefix="draft.layers.0.self_attn",
        )

        self.assertEqual(
            _FakeQKVParallelLinear.calls[0]["prefix"],
            "draft.layers.0.self_attn.qkv_proj",
        )
        self.assertEqual(
            _FakeRowParallelLinear.calls[0]["prefix"],
            "draft.layers.0.self_attn.o_proj",
        )
        self.assertEqual(
            _FakeGatedColumnParallelLinear.calls[0]["prefix"],
            "draft.layers.0.self_attn.g_proj",
        )
        self.assertEqual(
            set(attention.state_dict()),
            {"qkv_proj.weight", "o_proj.weight", "g_proj.weight"},
        )

    @patch.object(dflash, "VocabParallelEmbedding", _FakeEmbedding)
    @patch.object(dflash, "ReplicatedLinear", _FakeReplicatedLinear)
    @patch.object(dflash, "ColumnParallelLinear", _FakeColumnParallelLinear)
    @patch.object(dflash, "DFlashMLP", _FakeMLP)
    @patch.object(dflash, "RMSNorm", _FakeNorm)
    def test_registered_laguna_entry_class_constructs_with_nested_prefix(self):
        _FakeColumnParallelLinear.calls.clear()
        seen_prefixes = []

        def fake_attention_init(
            attention, config, layer_id, quant_config=None, prefix=""
        ):
            nn.Module.__init__(attention)
            seen_prefixes.append(prefix)
            attention.total_num_heads = 2
            attention.num_heads = 2
            attention.head_dim = 2

        config = _draft_config(has_embed_tokens=True)
        config.gating = True
        self.assertIn(dflash.DFlashLagunaForCausalLM, dflash.EntryClass)

        with patch.object(dflash.DFlashAttention, "__init__", fake_attention_init):
            dflash.DFlashLagunaForCausalLM(config, prefix="draft")

        self.assertEqual(
            seen_prefixes,
            [f"draft.layers.{i}.self_attn" for i in range(6)],
        )
        self.assertEqual(
            [
                call["prefix"]
                for call in _FakeColumnParallelLinear.calls
                if call["prefix"].endswith("g_proj")
            ],
            [f"draft.layers.{i}.self_attn.g_proj" for i in range(6)],
        )


class TestDFlashWeightValidation(unittest.TestCase):
    @staticmethod
    def _weight(shape, *, pack_factor=None):
        weight = nn.Parameter(torch.empty(shape), requires_grad=False)
        weight.weight_loader = lambda param, loaded: None
        if pack_factor is not None:
            weight.pack_factor = pack_factor
        return weight

    def test_valid_unquantized_fc_weight_loads(self):
        model = _WeightLoadingDraft(self._weight((4, 24)))

        model.load_weights([("fc.weight", torch.empty((4, 24)))])

    def test_invalid_unquantized_fc_weight_reports_logical_shape(self):
        model = _WeightLoadingDraft(self._weight((4, 24)))

        with self.assertRaisesRegex(
            ValueError, r"Expected logical fc\.weight\.shape=\(4, 24\)"
        ):
            model.load_weights([("fc.weight", torch.empty((4, 20)))])

    def test_valid_packed_fc_weight_loads(self):
        model = _WeightLoadingDraft(self._weight((48,), pack_factor=2))

        model.load_weights([("fc.weight", torch.empty((48,)))])

    def test_valid_logical_checkpoint_loads_into_packed_fc_weight(self):
        model = _WeightLoadingDraft(self._weight((48,), pack_factor=2))

        model.load_weights([("fc.weight", torch.empty((4, 24)))])

    def test_invalid_packed_fc_weight_still_fails(self):
        model = _WeightLoadingDraft(self._weight((40,), pack_factor=2))

        with self.assertRaisesRegex(ValueError, r"logical shape=\(4, 20\)"):
            model.load_weights([("fc.weight", torch.empty((40,)))])

    def test_valid_modelopt_packed_on_disk_shape_loads(self):
        model = _WeightLoadingDraft(self._weight((4, 12)))

        model.load_weights([("fc.weight", torch.empty((4, 12)))])

    def test_laguna_checkpoint_projection_names_reach_expected_loaders(self):
        model = _LagunaWeightLoadingDraft()

        model.load_weights(
            [
                ("model.layers.0.self_attn.q_proj.weight", torch.empty((2, 4))),
                ("model.layers.0.self_attn.k_proj.weight", torch.empty((1, 4))),
                ("model.layers.0.self_attn.v_proj.weight", torch.empty((1, 4))),
                ("model.layers.0.self_attn.o_proj.weight", torch.empty((4, 4))),
                ("model.layers.0.self_attn.g_proj.weight", torch.empty((4, 4))),
            ]
        )

        self.assertEqual(
            model.events,
            [
                ("qkv_proj", "q", (2, 4)),
                ("qkv_proj", "k", (1, 4)),
                ("qkv_proj", "v", (1, 4)),
                ("o_proj", None, (4, 4)),
                ("g_proj", None, (4, 4)),
            ],
        )


class TestDFlashEmbeddingSelection(unittest.TestCase):
    def test_draft_embedding_is_preferred(self):
        draft_embedding = object()
        target_embedding = object()
        draft_model = SimpleNamespace(get_input_embeddings=lambda: draft_embedding)
        target_model = SimpleNamespace(get_input_embeddings=lambda: target_embedding)

        self.assertIs(
            _get_dflash_embedding_module(draft_model, target_model), draft_embedding
        )

    def test_target_embedding_is_the_fallback(self):
        target_embedding = object()
        draft_model = SimpleNamespace(get_input_embeddings=lambda: None)
        target_model = SimpleNamespace(get_input_embeddings=lambda: target_embedding)

        self.assertIs(
            _get_dflash_embedding_module(draft_model, target_model), target_embedding
        )


class _FakeW4A16HeadMethod:
    def __init__(self, dense_weight):
        self.dense_weight = dense_weight
        self.calls = 0

    def apply(self, _layer, hidden_states, _bias):
        self.calls += 1
        return torch.matmul(hidden_states, self.dense_weight.T)


_FakeW4A16HeadMethod.__name__ = "ModelOptNvFp4A16LinearMethod"


class TestDFlashGreedyHead(unittest.TestCase):
    @staticmethod
    def _worker_stub():
        worker = object.__new__(dflash_worker_v2.DFlashWorkerV2)
        worker._draft_greedy_local_cap = 0
        worker._draft_greedy_local_max_buf = None
        worker._draft_greedy_local_arg_buf = None
        return worker

    @staticmethod
    def _shard_indices(*, num_org, num_org_padded, num_added):
        return SimpleNamespace(
            num_org_elements=num_org,
            num_org_elements_padded=num_org_padded,
            num_added_elements=num_added,
            org_vocab_start_index=0,
            added_vocab_start_index=100,
        )

    @patch.object(
        dflash_worker_v2,
        "get_tp_group",
        return_value=SimpleNamespace(world_size=1),
    )
    def test_quantized_head_uses_quant_method_instead_of_packed_weight(self, _tp_group):
        hidden_states = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
        dense_weight = torch.tensor(
            [
                [0.5, 1.0, -1.0, 0.0],
                [1.0, 0.0, 1.0, 1.0],
                [-1.0, 0.0, 0.0, -1.0],
            ]
        )
        lm_head = SimpleNamespace(
            weight=torch.zeros((3, 1), dtype=torch.int32),
            quant_method=_FakeW4A16HeadMethod(dense_weight),
            weight_scale=torch.ones(1),
            weight_global_scale=torch.ones(1),
            workspace=torch.empty(1),
            input_size_per_partition=4,
            output_size_per_partition=3,
        )

        actual = (
            dflash_worker_v2.DFlashWorkerV2._greedy_sample_from_vocab_parallel_head(
                self._worker_stub(), hidden_states=hidden_states, lm_head=lm_head
            )
        )
        expected = torch.argmax(hidden_states @ dense_weight.T, dim=-1)
        torch.testing.assert_close(actual, expected)

    @patch.object(
        dflash_worker_v2,
        "get_tp_group",
        return_value=SimpleNamespace(world_size=1),
    )
    def test_unquantized_fast_path_multiplies_only_valid_base_vocab(self, _tp_group):
        weight = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
            ]
        )
        lm_head = SimpleNamespace(
            weight=weight,
            quant_method=None,
            shard_indices=self._shard_indices(num_org=3, num_org_padded=6, num_added=0),
        )
        matmul_rhs_shapes = []
        torch_matmul = torch.matmul

        def recording_matmul(left, right):
            matmul_rhs_shapes.append(tuple(right.shape))
            return torch_matmul(left, right)

        with patch.object(
            dflash_worker_v2.torch, "matmul", side_effect=recording_matmul
        ):
            actual = (
                dflash_worker_v2.DFlashWorkerV2._greedy_sample_from_vocab_parallel_head(
                    self._worker_stub(),
                    hidden_states=torch.tensor([[1.0, 2.0]]),
                    lm_head=lm_head,
                )
            )

        torch.testing.assert_close(actual, torch.tensor([2]))
        self.assertEqual(matmul_rhs_shapes, [(2, 3)])

    @patch.object(
        dflash_worker_v2,
        "get_tp_group",
        return_value=SimpleNamespace(world_size=1),
    )
    def test_unquantized_added_vocab_path_uses_two_sliced_matmuls(self, _tp_group):
        weight = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [100.0, 100.0],
                [5.0, 0.0],
                [0.0, 4.0],
            ]
        )
        lm_head = SimpleNamespace(
            weight=weight,
            quant_method=None,
            shard_indices=self._shard_indices(num_org=2, num_org_padded=3, num_added=2),
        )
        matmul_rhs_shapes = []
        torch_matmul = torch.matmul

        def recording_matmul(left, right):
            matmul_rhs_shapes.append(tuple(right.shape))
            return torch_matmul(left, right)

        with patch.object(
            dflash_worker_v2.torch, "matmul", side_effect=recording_matmul
        ):
            actual = (
                dflash_worker_v2.DFlashWorkerV2._greedy_sample_from_vocab_parallel_head(
                    self._worker_stub(),
                    hidden_states=torch.tensor([[1.0, 0.0]]),
                    lm_head=lm_head,
                )
            )

        torch.testing.assert_close(actual, torch.tensor([100]))
        self.assertEqual(matmul_rhs_shapes, [(2, 2), (2, 2)])

    @patch.object(
        dflash_worker_v2,
        "get_tp_group",
        return_value=SimpleNamespace(world_size=1),
    )
    def test_quantized_head_path_applies_once_and_excludes_padding(self, _tp_group):
        dense_weight = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [100.0, 100.0],
                [5.0, 0.0],
                [0.0, 4.0],
            ]
        )
        quant_method = _FakeW4A16HeadMethod(dense_weight)
        lm_head = SimpleNamespace(
            weight=torch.zeros((5, 1), dtype=torch.int32),
            quant_method=quant_method,
            weight_scale=torch.ones(1),
            weight_global_scale=torch.ones(1),
            workspace=torch.empty(1),
            input_size_per_partition=2,
            output_size_per_partition=5,
            org_vocab_size=2,
        )

        actual = (
            dflash_worker_v2.DFlashWorkerV2._greedy_sample_from_vocab_parallel_head(
                self._worker_stub(),
                hidden_states=torch.tensor([[1.0, 0.0]]),
                lm_head=lm_head,
            )
        )

        torch.testing.assert_close(actual, torch.tensor([0]))
        self.assertEqual(quant_method.calls, 1)


if __name__ == "__main__":
    unittest.main()
