"""Nemotron-H's custom communicator must keep FP4 dispatch inputs DP-local."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.models import nemotron_h, nemotron_h_utils
from sglang.srt.models.nemotron_h import NemotronHMoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestNemotronHFP4Dispatch(unittest.TestCase):
    def test_only_sparse_fp4_layers_keep_dp_local_rows(self):
        for fp4_dispatch in (False, True):
            for sparse in (False, True):
                with (
                    self.subTest(fp4_dispatch=fp4_dispatch, sparse=sparse),
                    patch.object(
                        nemotron_h_utils,
                        "get_moe_a2a_backend",
                        return_value=MoeA2ABackend.NONE,
                    ),
                    patch.object(
                        nemotron_h_utils,
                        "should_use_flashinfer_moe_fp4_allgather",
                        return_value=fp4_dispatch,
                    ),
                ):
                    modes = nemotron_h_utils._build_layer_scatter_modes(sparse)
                    self.assertEqual(
                        modes.mlp_mode,
                        ScatterMode.SCATTERED
                        if sparse and fp4_dispatch
                        else ScatterMode.FULL,
                    )
                    self.assertEqual(modes.layer_output_mode, ScatterMode.TP_ATTN_FULL)

    def test_empty_rank_dispatches_with_expert_hidden_width(self):
        hidden = torch.empty((0, 64), dtype=torch.bfloat16)
        for latent in (False, True):
            with (
                self.subTest(latent=latent),
                patch.object(
                    nemotron_h,
                    "should_use_flashinfer_moe_fp4_allgather",
                    return_value=True,
                ),
            ):
                model = SimpleNamespace(
                    use_latent_moe=latent,
                    moe_hidden_size=32 if latent else 64,
                    topk=Mock(),
                    experts=Mock(side_effect=lambda hidden, topk: hidden),
                )
                output, shared = NemotronHMoE._forward_core(model, hidden)
                model.topk.empty_topk_output.assert_called_once_with(hidden.device)
                model.experts.assert_called_once()
                self.assertEqual(output.shape, (0, model.moe_hidden_size))
                self.assertIsNone(shared)

    def test_empty_latent_output_skips_projection_after_combine(self):
        hidden = torch.empty((0, 64), dtype=torch.bfloat16)
        model = SimpleNamespace(
            use_latent_moe=True,
            tp_size=1,
            _forward_core=Mock(return_value=(torch.empty((0, 32)), None)),
            fc2_latent_proj=Mock(),
        )
        with patch.object(
            nemotron_h, "should_use_flashinfer_moe_fp4_allgather", return_value=True
        ):
            output = NemotronHMoE.forward(model, hidden)
        model._forward_core.assert_called_once_with(hidden)
        model.fc2_latent_proj.assert_not_called()
        self.assertEqual(output.shape, hidden.shape)
        self.assertEqual(output.dtype, hidden.dtype)

    def test_disabled_dispatch_keeps_existing_empty_core(self):
        hidden = torch.empty((0, 64), dtype=torch.bfloat16)
        expected = (hidden, None)
        model = SimpleNamespace(_forward_core_normal=Mock(return_value=expected))
        with (
            patch.object(
                nemotron_h,
                "should_use_flashinfer_moe_fp4_allgather",
                return_value=False,
            ),
            patch.object(nemotron_h, "_is_cuda", False),
        ):
            self.assertIs(NemotronHMoE._forward_core(model, hidden), expected)
        model._forward_core_normal.assert_called_once_with(hidden)


if __name__ == "__main__":
    unittest.main()
