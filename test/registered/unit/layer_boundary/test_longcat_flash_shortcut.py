import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers import layer_boundary as comm
from sglang.srt.layers.layer_boundary.adapters import branch
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.layers.layer_boundary.stage import StageBoundary
from sglang.srt.models.longcat_flash import LongcatFlashDecoderLayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


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
            branch_output = branch.branch_output
            merge_branch = branch.merge_branch

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
        batch = SimpleNamespace(residual_stream=ResidualStream())
        layer.attn_boundary = SimpleNamespace(
            prepare=lambda h, fb: h, finish=lambda h, fb: h
        )
        moe_communicator.post_attention_layernorm = None
        dense_communicator.post_attention_layernorm = None
        layer.moe_boundary = StageBoundary(
            moe_communicator, declaration=comm.declare_ffn()
        )
        layer.moe_boundary.prepare = lambda h, fb: fork_hidden
        layer.second_ffn_boundary = StageBoundary(
            dense_communicator, declaration=comm.declare_ffn()
        )
        layer.self_attn = [lambda **kw: kw["hidden_states"]]
        layer.mlp = nn.Identity()

        def dense_branch(*args):
            stream = ResidualStream(torch.full((rows, 3), 11.0))
            hidden = stream.leave(torch.full((rows, 3), 3.0), comm.ADD)
            batch.residual_stream = stream
            return hidden, None

        layer.forward_mlp = dense_branch
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
            hidden, _ = layer(
                torch.arange(rows), torch.zeros(rows, 3), batch, None, None
            )
        hidden, residual = batch.residual_stream.finish(hidden)
        torch.testing.assert_close(hidden, torch.full((rows, 3), 5.0))
        torch.testing.assert_close(hidden + residual, torch.full((rows, 3), 16.0))
        torch.testing.assert_close(fork_residual, torch.full_like(fork_residual, 5.0))


if __name__ == "__main__":
    unittest.main()
