import unittest

from sglang.multimodal_gen.runtime.models.vaes.autoencoder_dc import AutoencoderDC


class _FakeInnerModel:
    def __init__(self):
        self.tiling_enabled = False
        self.enable_tiling_kwargs = None
        self.disable_tiling_called = False

    def enable_tiling(self, **kwargs):
        self.tiling_enabled = True
        self.enable_tiling_kwargs = kwargs

    def disable_tiling(self):
        self.tiling_enabled = False
        self.disable_tiling_called = True


class TestAutoencoderDCTiling(unittest.TestCase):
    def setUp(self) -> None:
        # Bypass _ensure_inner_model's real diffusers construction: once
        # _inner_model is set, _ensure_inner_model becomes a no-op.
        self.wrapper = AutoencoderDC()
        self.fake_inner = _FakeInnerModel()
        self.wrapper._inner_model = self.fake_inner

    def test_enable_tiling_forwards_to_inner_model(self) -> None:
        self.wrapper.enable_tiling()

        self.assertTrue(self.fake_inner.tiling_enabled)

    def test_enable_tiling_forwards_kwargs_to_inner_model(self) -> None:
        self.wrapper.enable_tiling(
            tile_sample_min_height=512, tile_sample_min_width=512
        )

        self.assertEqual(
            self.fake_inner.enable_tiling_kwargs,
            {"tile_sample_min_height": 512, "tile_sample_min_width": 512},
        )

    def test_disable_tiling_forwards_to_inner_model(self) -> None:
        self.wrapper.enable_tiling()
        self.wrapper.disable_tiling()

        self.assertFalse(self.fake_inner.tiling_enabled)
        self.assertTrue(self.fake_inner.disable_tiling_called)


if __name__ == "__main__":
    unittest.main()
