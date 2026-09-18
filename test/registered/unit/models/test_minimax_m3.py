"""Unit tests for the MiniMax-M3 model implementation."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.layers.attention.minimax_sparse_backend import (
    _require_chain_speculation,
)
from sglang.srt.models.minimax_m3 import MiniMaxM3DecoderLayer
from sglang.srt.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration
from sglang.test.test_utils import CustomTestCase


class TestMiniMaxM3DecoderLayer(CustomTestCase):
    def _run_decoder_layer(self, *, is_layer_sparse):
        hidden_states = SimpleNamespace(shape=(1, 6144))
        attention_output = SimpleNamespace(shape=(1, 6144))
        mlp_input = SimpleNamespace(shape=(1, 6144))
        mlp_output = SimpleNamespace(shape=(1, 6144))
        residual = object()
        forward_batch = SimpleNamespace(num_token_non_padded=1)

        communicator = Mock()
        communicator.prepare_attn_and_capture_last_layer_outputs.return_value = (
            hidden_states,
            residual,
        )
        communicator.prepare_mlp.return_value = mlp_input, residual
        communicator.should_fuse_mlp_allreduce_with_next_layer.return_value = False
        communicator.should_use_reduce_scatter.return_value = True
        communicator.postprocess_layer.return_value = mlp_output, residual

        mlp = Mock(return_value=mlp_output)
        layer = SimpleNamespace(
            is_layer_sparse=is_layer_sparse,
            layer_communicator=communicator,
            self_attn=Mock(return_value=attention_output),
            mlp=mlp,
        )

        with patch(
            "sglang.srt.models.minimax_m3.get_parallel",
            return_value=SimpleNamespace(tp_size=1),
        ):
            output = MiniMaxM3DecoderLayer.forward(
                layer,
                positions=object(),
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )

        self.assertEqual(output, (mlp_output, residual))
        return mlp, mlp_input, forward_batch

    def test_sparse_mlp_receives_forward_batch(self):
        """Sparse MLP calls must not bind all-reduce flags as ForwardBatch."""
        mlp, mlp_input, forward_batch = self._run_decoder_layer(is_layer_sparse=True)

        mlp.assert_called_once_with(
            hidden_states=mlp_input,
            forward_batch=forward_batch,
            should_allreduce_fusion=False,
            use_reduce_scatter=True,
        )

    def test_dense_mlp_uses_its_input_parameter_name(self):
        """Dense MLP calls must use ``x`` rather than the sparse MLP API."""
        mlp, mlp_input, _ = self._run_decoder_layer(is_layer_sparse=False)

        mlp.assert_called_once_with(
            x=mlp_input,
            should_allreduce_fusion=False,
            use_reduce_scatter=True,
        )


class TestMiniMaxSparseTargetVerify(CustomTestCase):
    def test_chain_verify_is_supported(self):
        _require_chain_speculation(None)
        _require_chain_speculation(1)

    def test_tree_verify_is_rejected(self):
        for tree_topk in (-1, 2):
            with self.subTest(tree_topk=tree_topk):
                with self.assertRaisesRegex(
                    NotImplementedError, "supports only chain target verification"
                ):
                    _require_chain_speculation(tree_topk)


class TestMiniMaxM3VlEagle3Capture(CustomTestCase):
    @staticmethod
    def _model(num_layers=60):
        layers = [SimpleNamespace() for _ in range(num_layers)]
        return SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=True),
            capture_aux_hidden_states=False,
            config=SimpleNamespace(
                text_config=SimpleNamespace(num_hidden_layers=num_layers)
            ),
            model=SimpleNamespace(layers=layers, layers_to_capture=[]),
        )

    def test_default_capture_layers_keep_legacy_indices(self):
        model = self._model()

        MiniMaxM3SparseForConditionalGeneration.set_eagle3_layers_to_capture(model)

        self.assertEqual(model.model.layers_to_capture, [2, 30, 57])
        self.assertEqual(
            [
                i
                for i, layer in enumerate(model.model.layers)
                if getattr(layer, "_is_layer_to_capture", False)
            ],
            [2, 30, 57],
        )

    def test_explicit_capture_layers_apply_output_offset(self):
        model = self._model()

        MiniMaxM3SparseForConditionalGeneration.set_eagle3_layers_to_capture(
            model, [1, 29, 56]
        )

        self.assertEqual(model.model.layers_to_capture, [2, 30, 57])


if __name__ == "__main__":
    unittest.main()
