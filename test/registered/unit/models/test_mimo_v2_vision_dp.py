"""MiMo's DP vision encoder must not materialize a whole clip on every rank."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.mimo_v2 import MiMoV2ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _PackedVisual:
    def __init__(self):
        self.use_data_parallel = False
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.calls = []

    def __call__(self, pixel_values, grid_thw):
        self.calls.append((pixel_values.clone(), grid_thw.clone()))
        return pixel_values


def _model(visual):
    model = MiMoV2ForCausalLM.__new__(MiMoV2ForCausalLM)
    model.visual = visual
    return model


def _item(feature, **grids):
    return SimpleNamespace(feature=feature, **grids)


class TestMiMoVisionDpSharding(CustomTestCase):
    def test_non_dp_keeps_the_packed_video_grid(self):
        visual = _PackedVisual()
        feature = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        item = _item(feature, video_grid_thw=torch.tensor([[2, 1, 3]]))

        out = _model(visual).get_video_feature([item])

        self.assertEqual(len(visual.calls), 1)
        pixels, grid = visual.calls[0]
        self.assertTrue(torch.equal(pixels, feature))
        self.assertTrue(torch.equal(grid, torch.tensor([[2, 1, 3]])))
        self.assertTrue(torch.equal(out, feature))

    def test_dp_splits_a_video_into_frames_and_loads_only_those_rows(self):
        visual = SimpleNamespace(
            use_data_parallel=True,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        frame0 = torch.full((4, 3), 1.0)
        frame1 = torch.full((4, 3), 2.0)
        item = _item(
            torch.cat([frame0, frame1], dim=0),
            video_grid_thw=torch.tensor([[2, 2, 2]]),
        )
        captured = {}

        def fake_run(vision_model, pixel_values, grid_thw_list, **kwargs):
            captured["vision_model"] = vision_model
            captured["pixel_values"] = pixel_values
            captured["grids"] = grid_thw_list
            captured["loader"] = kwargs["load_local_pixel_values"]
            captured["device"] = kwargs["pixel_values_device"]
            captured["dtype"] = kwargs["pixel_values_dtype"]
            return torch.zeros(8, 4)

        import sglang.srt.models.mimo_v2 as mimo_v2

        original = mimo_v2.run_dp_sharded_mrope_vision_model
        mimo_v2.run_dp_sharded_mrope_vision_model = fake_run
        try:
            out = _model(visual).get_video_feature([item])
        finally:
            mimo_v2.run_dp_sharded_mrope_vision_model = original

        self.assertIs(captured["vision_model"], visual)
        self.assertIsNone(captured["pixel_values"])
        self.assertEqual(captured["grids"], [[1, 2, 2], [1, 2, 2]])
        self.assertEqual(captured["device"], visual.device)
        self.assertEqual(captured["dtype"], visual.dtype)
        self.assertTrue(torch.equal(captured["loader"]([0]), frame0))
        self.assertTrue(torch.equal(captured["loader"]([1]), frame1))
        self.assertTrue(
            torch.equal(
                captured["loader"]([1, 0]),
                torch.cat([frame1, frame0], dim=0),
            )
        )
        self.assertEqual(captured["loader"]([]).shape, (0, 3))
        self.assertEqual(tuple(out.shape), (8, 4))

    def test_dp_image_loader_returns_the_requested_image_only(self):
        visual = SimpleNamespace(
            use_data_parallel=True,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        image0 = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        image1 = torch.arange(4, dtype=torch.float32).reshape(2, 2) + 10
        items = [
            _item(image0, image_grid_thw=torch.tensor([[1, 1, 3]])),
            _item(image1, image_grid_thw=torch.tensor([[1, 1, 2]])),
        ]
        captured = {}

        def fake_run(vision_model, pixel_values, grid_thw_list, **kwargs):
            captured["grids"] = grid_thw_list
            captured["loader"] = kwargs["load_local_pixel_values"]
            return torch.zeros(1, 1)

        import sglang.srt.models.mimo_v2 as mimo_v2

        original = mimo_v2.run_dp_sharded_mrope_vision_model
        mimo_v2.run_dp_sharded_mrope_vision_model = fake_run
        try:
            _model(visual).get_image_feature(items)
        finally:
            mimo_v2.run_dp_sharded_mrope_vision_model = original

        self.assertEqual(captured["grids"], [[1, 1, 3], [1, 1, 2]])
        loaded = captured["loader"]([1])
        self.assertTrue(torch.equal(loaded, image1.to(torch.bfloat16)))
        self.assertEqual(loaded.dtype, torch.bfloat16)

    def test_patch_count_must_match_the_grid(self):
        visual = _PackedVisual()
        item = _item(
            torch.zeros(3, 2),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )

        with self.assertRaises(RuntimeError):
            _model(visual).get_image_feature([item])


if __name__ == "__main__":
    unittest.main()
