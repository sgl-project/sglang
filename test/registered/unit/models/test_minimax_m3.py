"""Unit tests for the MiniMax-M3 model implementation."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from sglang.srt.models.minimax_m3 import MiniMaxM3DecoderLayer
from sglang.srt.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration
from sglang.test.test_utils import CustomTestCase


class TestMiniMaxM3DecoderLayer(CustomTestCase):
    def _run_decoder_layer(self, *, is_layer_sparse):
        hidden_states = SimpleNamespace(shape=(1, 6144))
        attention_output = SimpleNamespace(shape=(1, 6144))
        attn_output = SimpleNamespace(shape=(1, 6144))
        mlp_input = SimpleNamespace(shape=(1, 6144))
        mlp_output = SimpleNamespace(shape=(1, 6144))
        forward_batch = SimpleNamespace(num_token_non_padded=1)

        # HEAD drives the decoder layer through the layer-boundary stages rather
        # than the removed ``layer_communicator`` API: the attention boundary
        # prepares/finishes the mixer, and the FFN boundary prepares the MLP
        # input and hands back an ``exit`` context manager carrying the
        # all-reduce fusion decisions plus the ``finish`` completion.
        attn_boundary = Mock()
        attn_boundary.prepare.return_value = hidden_states
        attn_boundary.finish.return_value = attn_output

        ffn_boundary = MagicMock()
        ffn_boundary.prepare.return_value = mlp_input
        ffn_exit = SimpleNamespace(
            fuse_mlp_allreduce=False,
            mlp_reduce_scatter=True,
            finish=Mock(return_value=mlp_output),
        )
        ffn_boundary.exit.return_value.__enter__.return_value = ffn_exit
        ffn_boundary.exit.return_value.__exit__.return_value = False

        mlp = Mock(return_value=mlp_output)
        layer = SimpleNamespace(
            is_layer_sparse=is_layer_sparse,
            attn_boundary=attn_boundary,
            ffn_boundary=ffn_boundary,
            self_attn=Mock(return_value=attention_output),
            mlp=mlp,
        )

        output = MiniMaxM3DecoderLayer.forward(
            layer,
            positions=object(),
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        self.assertEqual(output, mlp_output)
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
