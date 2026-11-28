"""Compare tiled local expert calls with a complete weighted expert reference."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe import prefill_moe_tile as tile
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPrefillMoETiling(CustomTestCase):
    def test_uneven_tiles_keep_routes_logits_and_values(self):
        generator = torch.Generator().manual_seed(52)
        inputs = torch.randn(13, 5, generator=generator)
        ids = torch.randint(0, 7, (13, 2), generator=generator)
        weights = torch.rand(13, 2, generator=generator)
        logits = torch.randn(13, 7, generator=generator)
        routes = StandardTopKOutput(weights, ids, logits)
        seen = []

        def expert(values, selected):
            seen.append((len(values), selected.router_logits.clone()))
            scale = (selected.topk_weights * (selected.topk_ids + 1)).sum(1)
            return values * scale[:, None]

        expected = expert(inputs, routes)
        seen.clear()
        actual = tile.tiled_experts(expert, inputs, routes, 4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual([size for size, _ in seen], [4, 4, 4, 1])
        torch.testing.assert_close(torch.cat([value for _, value in seen]), logits)

    def test_absent_logits_and_empty_input(self):
        for count in (0, 1, 8):
            with self.subTest(count=count):
                inputs = torch.arange(count * 3, dtype=torch.float32).view(count, 3)
                routes = StandardTopKOutput(
                    torch.ones(count, 1), torch.zeros(count, 1, dtype=torch.int64), None
                )
                actual = tile.tiled_experts(
                    lambda values, _: values + 1, inputs, routes, 4
                )
                torch.testing.assert_close(actual, inputs + 1)

    def test_bad_output_and_tile_size_are_rejected(self):
        inputs = torch.ones(3, 2)
        routes = StandardTopKOutput(
            torch.ones(3, 1), torch.zeros(3, 1, dtype=torch.int64), None
        )
        with self.assertRaises(ValueError):
            tile.tiled_experts(lambda *_: inputs, inputs, routes, 0)
        with self.assertRaises(RuntimeError):
            tile.tiled_experts(lambda values, _: values.double(), inputs, routes, 2)

    def test_only_local_prefill_is_eligible(self):
        experts = SimpleNamespace(
            dispatcher=Mock(spec=tile.StandardDispatcher),
            moe_ep_size=1,
            reduce_results=False,
            _dwdp_bound=False,
        )
        layer = SimpleNamespace(experts=experts, _fuse_shared_experts_inside_sbo=False)
        inputs = SimpleNamespace(shape=(8, 2), device=SimpleNamespace(type="npu"))
        routes = StandardTopKOutput(torch.ones(8, 1), torch.zeros(8, 1), None)
        with (
            patch.object(
                tile,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="prefill"),
            ) as mode,
            patch.object(
                tile,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: True),
            ),
            patch.object(
                tile,
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=False,
            ),
        ):
            self.assertTrue(tile.can_tile_prefill_moe(layer, inputs, routes, 4))
            for name, value in (
                ("moe_ep_size", 2),
                ("reduce_results", True),
                ("_dwdp_bound", True),
                ("dispatcher", object()),
            ):
                with self.subTest(name=name), patch.object(experts, name, value):
                    self.assertFalse(
                        tile.can_tile_prefill_moe(layer, inputs, routes, 4)
                    )
            mode.return_value.disaggregation_mode = "decode"
            self.assertFalse(tile.can_tile_prefill_moe(layer, inputs, routes, 4))


if __name__ == "__main__":
    unittest.main()
