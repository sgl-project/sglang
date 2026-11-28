"""Kimi's real model entrypoints must schedule and invoke NPU KV prefetch."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestKimiK3Prefetch(CustomTestCase):
    def _backbone(self, full_layers, start, end):
        config = SimpleNamespace(
            num_hidden_layers=20,
            full_attention_layer_ids=full_layers,
            attn_res_block_size=None,
        )
        # Restrict the stack to the local PP span: crossing a PP boundary
        # raises, rather than silently scheduling a remote stage's layer.
        layers = {
            i: SimpleNamespace(next_full_attention_layer_id=None)
            for i in range(start, end)
        }
        with (
            patch.object(
                kimi_k3,
                "get_parallel",
                return_value=SimpleNamespace(
                    pp_group=SimpleNamespace(is_first_rank=False, is_last_rank=False),
                    kda_dwdp_size=None,
                ),
            ),
            patch.object(kimi_k3, "is_dp_attention_enabled", return_value=False),
            patch.object(kimi_k3.k3_sp_collective, "enabled", return_value=False),
            patch.object(kimi_k3, "_carries_bank_slices", return_value=False),
            patch.object(kimi_k3.torch.cuda, "Stream"),
            patch.object(kimi_k3, "_is_gfx95", False),
            patch.object(kimi_k3, "make_pp_layers", return_value=(layers, start, end)),
        ):
            return kimi_k3.KimiK3LinearModel(config)

    def test_backbone_schedules_only_local_full_attention_layers(self):
        cases = [
            ([3, 7, 11, 15, 19], 8, 20, {8: 11, 11: 15, 15: 19}),
            ([3, 7, 11, 15, 19], 7, 16, {7: 11, 11: 15}),
            ([3, 7, 11, 15, 19], 4, 7, {}),
            ([3, 7, 11, 15, 19], 8, 12, {8: 11}),
            ([0, 1, 2], 0, 3, {0: 1, 1: 2}),
        ]
        for full_layers, start, end, expected in cases:
            with self.subTest(full_layers=full_layers, start=start, end=end):
                model = self._backbone(full_layers, start, end)
                self.assertEqual(
                    {
                        i: layer.next_full_attention_layer_id
                        for i, layer in model.layers.items()
                    },
                    {i: expected.get(i) for i in range(start, end)},
                )

    def test_attention_entrypoint_prefetches_after_attention_completes(self):
        for returns_tuple in (False, True):
            with self.subTest(returns_tuple=returns_tuple):
                events = []
                hidden, output = torch.zeros(2, 8), torch.ones(2, 8)
                positions = torch.arange(2)
                batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
                allocator = object()

                def attention(**kwargs):
                    self.assertIs(kwargs["hidden_states"], hidden)
                    self.assertIs(kwargs["positions"], positions)
                    self.assertIs(kwargs["forward_batch"], batch)
                    self.assertIs(kwargs["zero_allocator"], allocator)
                    events.append("attention")
                    return (output, None) if returns_tuple else output

                layer = SimpleNamespace(
                    self_attn=attention, next_full_attention_layer_id=11
                )
                with patch.object(
                    kimi_k3,
                    "maybe_prefetch_next_kimi_full_attention_kv",
                    side_effect=lambda *args: events.append("prefetch"),
                ) as prefetch:
                    result = kimi_k3.KimiK3DecoderLayer._attend(
                        layer, hidden, positions, batch, allocator
                    )
                self.assertIs(result, output)
                prefetch.assert_called_once_with(batch, 11)
                self.assertEqual(events, ["attention", "prefetch"])

    def test_idle_attention_does_not_prefetch(self):
        layer = SimpleNamespace(self_attn=Mock(), next_full_attention_layer_id=11)
        hidden = torch.zeros(2, 8)
        with patch.object(
            kimi_k3, "maybe_prefetch_next_kimi_full_attention_kv"
        ) as prefetch:
            result = kimi_k3.KimiK3DecoderLayer._attend(
                layer,
                hidden,
                torch.arange(2),
                SimpleNamespace(forward_mode=ForwardMode.IDLE),
                object(),
            )
        self.assertIs(result, hidden)
        layer.self_attn.assert_not_called()
        prefetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
