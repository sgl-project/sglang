import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.srt.models.minimax_m3 import MiniMaxM3DecoderLayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMiniMaxM3DecoderLayer(CustomTestCase):
    def _make_layer(self, *, is_sparse, hidden_states, mlp_output):
        layer = MiniMaxM3DecoderLayer.__new__(MiniMaxM3DecoderLayer)
        nn.Module.__init__(layer)
        layer.layer_id = 0
        layer.is_layer_sparse = is_sparse
        layer.self_attn = SimpleNamespace(is_sparse_attention_layer=False)
        layer.mlp = MagicMock(return_value=mlp_output)
        communicator = MagicMock()
        communicator.prepare_attn_and_capture_last_layer_outputs.return_value = (
            hidden_states,
            None,
        )
        communicator.prepare_mlp.return_value = (hidden_states, None)
        communicator.should_fuse_mlp_allreduce_with_next_layer.return_value = False
        communicator.should_use_reduce_scatter.return_value = True
        communicator.postprocess_layer.return_value = (mlp_output, None)
        layer.layer_communicator = communicator
        return layer

    @patch("sglang.srt.models.minimax_m3.get_parallel")
    def test_sparse_mlp_receives_forward_batch(self, get_parallel):
        get_parallel.return_value = SimpleNamespace(tp_size=1)
        hidden_states = torch.empty((0, 4))
        mlp_output = torch.empty_like(hidden_states)
        forward_batch = SimpleNamespace(num_token_non_padded=0)
        layer = self._make_layer(
            is_sparse=True,
            hidden_states=hidden_states,
            mlp_output=mlp_output,
        )

        layer(
            positions=torch.empty(0),
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=None,
        )

        layer.mlp.assert_called_once_with(
            hidden_states,
            forward_batch=forward_batch,
            should_allreduce_fusion=False,
            use_reduce_scatter=True,
        )

    @patch("sglang.srt.models.minimax_m3.get_parallel")
    def test_dense_mlp_keeps_dense_signature(self, get_parallel):
        get_parallel.return_value = SimpleNamespace(tp_size=1)
        hidden_states = torch.empty((1, 4))
        mlp_output = torch.empty_like(hidden_states)
        forward_batch = SimpleNamespace(num_token_non_padded=1)
        layer = self._make_layer(
            is_sparse=False,
            hidden_states=hidden_states,
            mlp_output=mlp_output,
        )
        layer.self_attn = MagicMock(return_value=hidden_states)
        layer.self_attn.is_sparse_attention_layer = False

        layer(
            positions=torch.empty(1),
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=None,
        )

        layer.mlp.assert_called_once_with(
            hidden_states,
            should_allreduce_fusion=False,
            use_reduce_scatter=True,
        )


if __name__ == "__main__":
    unittest.main()
