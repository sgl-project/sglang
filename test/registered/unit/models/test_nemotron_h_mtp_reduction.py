import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.layers import communicator as comm
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import nemotron_h_mtp
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Norm(nn.Module):
    def forward(self, x, residual=None):
        if residual is None:
            return x
        return x + residual, x + residual


class TestNemotronMTPReduction(CustomTestCase):
    def test_attention_partial_is_reduced_once(self):
        """Under DP attention, the MTP MoE layer sums the attention output over the
        attention TP group exactly once before the residual add."""
        for tp in (1, 2):
            with self.subTest(tp=tp):
                reduce = Mock(side_effect=lambda x: x * tp)
                group = SimpleNamespace(all_reduce=reduce)
                with (
                    get_context().override_server_args(
                        tp_size=tp, enable_dp_attention=True
                    ),
                    get_flags().dp.override(enabled=True),
                    get_parallel().override(
                        attn_tp_group=group,
                        launch_world_rank=0,
                        tp_rank=0,
                        tp_size=tp,
                        attn_tp_rank=0,
                        attn_tp_size=tp,
                        attn_dp_rank=0,
                        attn_dp_size=1,
                        attn_cp_rank=0,
                        attn_cp_size=1,
                        moe_tp_rank=0,
                        moe_tp_size=tp,
                        moe_ep_rank=0,
                        moe_ep_size=1,
                        moe_dp_rank=0,
                        moe_dp_size=1,
                    ),
                    patch.object(comm, "get_moe_cp_size", return_value=1),
                    patch.object(
                        comm, "apply_flashinfer_allreduce_fusion", return_value=False
                    ),
                    patch.object(
                        comm, "apply_aiter_all_reduce_fusion", return_value=False
                    ),
                ):
                    layer = nemotron_h_mtp.NemotronHMTPMoEDecoderLayer.__new__(
                        nemotron_h_mtp.NemotronHMTPMoEDecoderLayer
                    )
                    nn.Module.__init__(layer)
                    layer.has_start_projections = False
                    layer.has_end_norm = False
                    layer.mixer = nn.Identity()
                    layer.norm = _Norm()
                    layer._init_layer_communicator(
                        SimpleNamespace(hybrid_override_pattern="*E"),
                        1,
                        is_sparse=False,
                    )
                    partial = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                    residual = torch.tensor([[7.0, 3.0], [5.0, 9.0]])
                    expected = partial * tp + residual
                    hidden, output_residual = layer(
                        inputs_embeds=torch.zeros_like(partial),
                        hidden_states=partial,
                        residual=residual,
                        forward_batch=SimpleNamespace(
                            forward_mode=ForwardMode.DECODE,
                            input_ids=torch.zeros(2, dtype=torch.long),
                        ),
                    )
                    torch.testing.assert_close(hidden, expected)
                    torch.testing.assert_close(output_residual, expected)
                    self.assertEqual(reduce.call_count, int(tp > 1))


if __name__ == "__main__":
    unittest.main()
