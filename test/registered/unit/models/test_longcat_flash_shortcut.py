import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from sglang.srt.layers import communicator as comm
from sglang.srt.models.longcat_flash import LongcatFlashDecoderLayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestLongcatShortcut(CustomTestCase):
    def test_shortcut_does_not_add_the_residual_again(self):
        """On the last layer with scattered MoE tokens, the shortcut output must
        not add the residual a second time; the dense branch already carries it."""
        tp, rows = 2, 4
        # The MoE runs on each attention-TP rank's slice; the last layer hands on
        # the attention's rows, where the dense branch ends.
        local = comm.Layout(frozenset({comm.TokenAxis.ATTN_TP_SCATTER}))
        attention = comm.Layout(frozenset())

        class Communicator(SimpleNamespace):
            branch_output = comm.LayerCommunicator.branch_output
            merge_branch = comm.LayerCommunicator.merge_branch

        fork_hidden = torch.full((rows // tp, 3), 2.0)
        fork_residual = torch.full_like(fork_hidden, 5.0)
        moe_communicator = Communicator(
            prepare_attn=lambda h, r, batch: (h, r),
            prepare_mlp=lambda h, r, batch: (fork_hidden, fork_residual),
            _branch_rows=lambda batch: (local, local, attention),
        )
        dense_communicator = Communicator(
            _branch_rows=lambda batch: (attention, attention, attention)
        )
        layer = LongcatFlashDecoderLayer.__new__(LongcatFlashDecoderLayer)
        nn.Module.__init__(layer)
        layer.moe_layer_communicator = moe_communicator
        layer.mlp_layer_communicator = [None, dense_communicator]
        layer.self_attn = [lambda **kw: kw["hidden_states"]]
        layer.mlp = nn.Identity()
        layer.forward_mlp = Mock(
            return_value=(
                torch.full((rows, 3), 3.0),
                torch.full((rows, 3), 11.0),
                None,
            )
        )
        with (
            patch_communicator(
                "get_local_dp_buffer",
                side_effect=lambda group, hidden_size=None: torch.empty(rows, 3),
            ),
            patch_communicator(
                "attn_tp_all_gather_into_tensor",
                side_effect=lambda out, x: out.copy_(x.repeat(tp, 1)),
            ),
            patch_communicator(
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_group=object()),
            ),
        ):
            hidden, residual, _ = layer(
                torch.arange(rows), torch.zeros(rows, 3), None, None, None, None
            )
        torch.testing.assert_close(hidden, torch.full((rows, 3), 5.0))
        torch.testing.assert_close(hidden + residual, torch.full((rows, 3), 16.0))
        torch.testing.assert_close(fork_residual, torch.full_like(fork_residual, 5.0))


if __name__ == "__main__":
    unittest.main()
