"""Resolved MM specs stay compatible with Dynamo's Python parity fixtures."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from sglang.srt.rust_server.multimodal import RUST_MM_FAMILIES
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRustMmSpec(unittest.TestCase):
    def fixture(self, family):
        root = Path(__file__).resolve().parents[6]
        name = {"qwen_vl": "qwen2.5-vl-3b-instruct", "glm5_next": "glm-5.3-flash"}[
            family
        ]
        fixture = (
            root / "rust/sglang-processor/tests/fixtures/mm_parity" / f"{name}.json"
        )
        return json.loads(fixture.read_text())["cases"][0]["spec"]

    def resolve(self, family, *, overrides=None, backend=None, **changes):
        expected = self.fixture(family)
        image = dict(expected)
        image.update(
            do_resize=True,
            do_rescale=True,
            do_normalize=True,
            rescale_factor=1 / 255,
            resample=Image.Resampling.BICUBIC,
        )
        image.update(changes)
        name = (
            backend
            or {
                "qwen_vl": "Qwen2VLImageProcessor",
                "glm5_next": "Glm5NextImageProcessor",
            }[family]
        )
        processor = type(name, (), image)()
        config = SimpleNamespace(
            image_token_id=expected["image_token_id"],
            video_token_id=expected.get("video_token_id", 999),
        )
        entry = next(f for f in RUST_MM_FAMILIES if f.name == family)
        mm = SimpleNamespace(mm_process_config={"image": overrides or {}})
        with patch("sglang.srt.rust_server.multimodal.get_mm", return_value=mm):
            return entry.resolve(
                processor=SimpleNamespace(image_processor=processor),
                hf_config=config,
                feature_shm=True,
            )

    def test_resolved_specs_match_the_native_parity_oracle(self):
        for family in ("qwen_vl", "glm5_next"):
            with self.subTest(family=family):
                spec = self.resolve(family)
                self.assertEqual(json.loads(spec.rust_json()), self.fixture(family))
                self.assertTrue(spec.feature_shm)
                self.assertNotIn("feature_shm", json.loads(spec.rust_json()))
        # GLM's drain adapter rejects video that expands with the image id.
        spec = self.resolve("glm5_next")
        self.assertEqual(spec.video_token_id, spec.image_token_id)

    def test_pil_backend_selects_the_matching_resampler(self):
        for family, backend in (
            ("qwen_vl", "Qwen2VLImageProcessorPil"),
            ("glm5_next", "Glm5NextImageProcessorPil"),
        ):
            with self.subTest(family=family):
                self.assertEqual(
                    self.resolve(family, backend=backend).pipeline["resample"], "pil"
                )

    def test_family_overrides_use_their_own_units(self):
        for family, overrides in (
            ("qwen_vl", {"min_pixels": 4096, "max_pixels": 102400}),
            ("glm5_next", {"min_image_tokens": 4, "max_image_tokens": 64}),
        ):
            with self.subTest(family=family):
                spec = self.resolve(family, overrides=overrides)
                for name, value in overrides.items():
                    self.assertEqual(spec.pipeline[name], value)
                self.assertIsNone(self.resolve(family, overrides={"unrecognized": 1}))

    def test_unsupported_pixel_settings_disable_the_native_path(self):
        for family in ("qwen_vl", "glm5_next"):
            for name, value in (
                ("do_resize", False),
                ("do_rescale", False),
                ("do_normalize", False),
                ("rescale_factor", 1.0),
                ("resample", Image.Resampling.BILINEAR),
                ("resample", None),
            ):
                with self.subTest(family=family, name=name, value=value):
                    self.assertIsNone(self.resolve(family, **{name: value}))
            self.assertIsNone(self.resolve(family, backend="UnknownImageProcessor"))


if __name__ == "__main__":
    unittest.main()
