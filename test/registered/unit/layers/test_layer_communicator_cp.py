import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_communicator():
    return types.SimpleNamespace(
        _sp_variant=None,
        _speculative_algo=None,
        layer_scatter_modes=LayerScatterModes(
            layer_input_mode=ScatterMode.TP_ATTN_FULL,
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=ScatterMode.FULL,
            middle_residual_mode=ScatterMode.TP_ATTN_FULL,
            layer_output_mode=ScatterMode.TP_ATTN_FULL,
        ),
        is_last_layer=False,
        allow_reduce_scatter=False,
        post_attention_layernorm=object(),
        _context=types.SimpleNamespace(tp_size=4),
        _communicate_with_all_reduce_and_layer_norm_fn=lambda **kwargs: (
            kwargs["hidden_states"],
            kwargs["residual"],
        ),
        _communicate_summable_tensor_pair_fn=lambda **kwargs: (
            kwargs["hidden_states"],
            kwargs["residual"],
        ),
    )


class TestLayerCommunicatorContextParallel(CustomTestCase):
    def setUp(self):
        self.forward_batch = types.SimpleNamespace(
            input_ids=types.SimpleNamespace(shape=(8,))
        )

    def test_prepare_mlp_materializes_global_token_order(self):
        communicator = _fake_communicator()
        hidden_states = torch.tensor([[1.0], [2.0]])
        residual = torch.tensor([[10.0], [20.0]])

        with (
            patch.object(comm, "is_cp_v2_active", return_value=True),
            patch.object(
                comm,
                "cp_gather_after_forward",
                side_effect=lambda tensor, _batch: tensor.flip(0),
            ),
        ):
            hidden_states, residual = LayerCommunicator.prepare_mlp(
                communicator, hidden_states, residual, self.forward_batch
            )

        torch.testing.assert_close(hidden_states, torch.tensor([[2.0], [1.0]]))
        torch.testing.assert_close(residual, torch.tensor([[10.0], [20.0]]))

    def test_postprocess_layer_restores_cp_local_token_order(self):
        communicator = _fake_communicator()
        hidden_states = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        residual = torch.tensor([[10.0], [20.0], [30.0], [40.0]])

        with (
            patch.object(comm, "is_cp_v2_active", return_value=True),
            patch.object(
                comm,
                "cp_shard_hidden_states",
                side_effect=lambda tensor, _batch: tensor[::2],
            ),
        ):
            hidden_states, residual = LayerCommunicator.postprocess_layer(
                communicator, hidden_states, residual, self.forward_batch
            )

        torch.testing.assert_close(hidden_states, torch.tensor([[1.0], [3.0]]))
        torch.testing.assert_close(
            residual, torch.tensor([[10.0], [20.0], [30.0], [40.0]])
        )

    def test_sparse_full_mlp_keeps_existing_cp_communication(self):
        communicator = _fake_communicator()
        communicator.layer_scatter_modes.is_layer_sparse = True
        hidden_states = torch.tensor([[1.0], [2.0]])
        residual = torch.tensor([[10.0], [20.0]])

        with (
            patch.object(comm, "is_cp_v2_active", return_value=True),
            patch.object(
                comm,
                "cp_gather_after_forward",
                side_effect=lambda tensor, _batch: tensor.flip(0),
            ),
            patch.object(
                comm,
                "cp_shard_hidden_states",
                side_effect=lambda tensor, _batch: tensor[::2],
            ),
        ):
            prepared_hidden_states, prepared_residual = LayerCommunicator.prepare_mlp(
                communicator, hidden_states, residual, self.forward_batch
            )
            hidden_states, residual = LayerCommunicator.postprocess_layer(
                communicator,
                prepared_hidden_states,
                prepared_residual,
                self.forward_batch,
            )

        torch.testing.assert_close(hidden_states, torch.tensor([[1.0], [2.0]]))
        torch.testing.assert_close(residual, torch.tensor([[10.0], [20.0]]))

    def test_full_tp_mlp_does_not_fuse_while_cp_is_active(self):
        communicator = _fake_communicator()

        with (
            patch.object(comm, "is_cp_v2_active", return_value=True),
            patch.object(comm, "is_enable_moe_cp_allgather", return_value=False),
            patch.object(comm, "apply_flashinfer_allreduce_fusion", return_value=True),
            patch.object(
                comm,
                "get_attn_tp_context",
                return_value=types.SimpleNamespace(input_scattered=False),
            ),
            get_parallel().override(moe_ep_size=1, moe_tp_size=4, tp_size=4),
        ):
            should_fuse = LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer(
                communicator, self.forward_batch
            )

        self.assertFalse(should_fuse)

    def test_inactive_cp_leaves_mlp_boundary_unchanged(self):
        communicator = _fake_communicator()
        hidden_states = torch.tensor([[1.0], [2.0]])
        residual = torch.tensor([[10.0], [20.0]])

        with patch.object(comm, "is_cp_v2_active", return_value=False):
            prepared_hidden_states, prepared_residual = LayerCommunicator.prepare_mlp(
                communicator, hidden_states, residual, self.forward_batch
            )
            output_hidden_states, output_residual = LayerCommunicator.postprocess_layer(
                communicator,
                prepared_hidden_states,
                prepared_residual,
                self.forward_batch,
            )

        torch.testing.assert_close(output_hidden_states, hidden_states)
        torch.testing.assert_close(output_residual, residual)


if __name__ == "__main__":
    unittest.main()
