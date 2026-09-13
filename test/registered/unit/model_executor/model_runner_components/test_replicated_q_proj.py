"""Unit tests for refreshing DCP full-head Q projection weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_executor.model_runner_components.replicated_q_proj import (
    prepare_replicated_q_proj,
)
from sglang.srt.model_executor.model_runner_components.startup_weight_load import (
    ModelStorageManifest,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_attention(*, dtype=torch.float16, has_q_b_proj=True):
    attention = object.__new__(DeepseekV2AttentionMLA)
    nn.Module.__init__(attention)
    attention.w_kc = torch.ones(2, 3, dtype=dtype)
    attention.w_kc_qrep = None
    attention.q_b_proj_qrep_weight = None
    attention.has_q_b_proj = has_q_b_proj
    q_proj = nn.Linear(3, 2, bias=False, dtype=dtype)
    q_proj.quant_method = UnquantizedLinearMethod()
    setattr(attention, "q_b_proj" if has_q_b_proj else "q_proj", q_proj)
    return attention, q_proj


def _make_group(world_size=2):
    return SimpleNamespace(
        world_size=world_size,
        all_gather=Mock(side_effect=lambda tensor, dim: torch.cat([tensor] * 2, dim)),
    )


class TestReplicatedQProj(CustomTestCase):
    def test_refresh_preserves_storage_and_updates_both_head_layouts(self):
        for has_q_b_proj in (True, False):
            with self.subTest(has_q_b_proj=has_q_b_proj):
                attention, q_proj = _make_attention(has_q_b_proj=has_q_b_proj)
                group = _make_group()
                prepare_replicated_q_proj(model=attention, dcp_group=group)
                manifest = ModelStorageManifest.capture(attention)
                derived = dict(attention.named_startup_weight_load_derived_tensors())
                self.assertIn("w_kc_qrep", derived)
                self.assertIn("q_b_proj_qrep_weight", derived)

                attention.w_kc.fill_(3)
                with torch.no_grad():
                    q_proj.weight.fill_(5)
                prepare_replicated_q_proj(model=attention, dcp_group=group)

                self.assertEqual(manifest.changed_names(attention), ())
                torch.testing.assert_close(
                    attention.w_kc_qrep, torch.full((4, 3), 3, dtype=torch.float16)
                )
                torch.testing.assert_close(
                    attention.q_b_proj_qrep_weight,
                    torch.full((4, 3), 5, dtype=torch.float16),
                )
                self.assertEqual(group.all_gather.call_count, 4)

    def test_skipped_paths_do_not_gather(self):
        for case in ("single_rank", "non_16_bit", "quantized", "missing_absorb"):
            with self.subTest(case=case):
                attention, q_proj = _make_attention(
                    dtype=torch.float32 if case == "non_16_bit" else torch.float16
                )
                group = _make_group(world_size=1 if case == "single_rank" else 2)
                if case == "quantized":
                    q_proj.quant_method = object()
                elif case == "missing_absorb":
                    attention.w_kc = None

                prepare_replicated_q_proj(model=attention, dcp_group=group)

                group.all_gather.assert_not_called()
                self.assertIsNone(attention.w_kc_qrep)
                self.assertIsNone(attention.q_b_proj_qrep_weight)

    def test_refresh_rejects_layout_change_without_replacing_storage(self):
        attention, _ = _make_attention()
        group = _make_group()
        prepare_replicated_q_proj(model=attention, dcp_group=group)
        original = attention.w_kc_qrep
        group.all_gather.side_effect = lambda tensor, dim: tensor

        with self.assertRaisesRegex(RuntimeError, "layout changed: w_kc_qrep"):
            prepare_replicated_q_proj(model=attention, dcp_group=group)

        self.assertIs(attention.w_kc_qrep, original)


if __name__ == "__main__":
    unittest.main()
