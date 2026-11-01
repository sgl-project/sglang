import unittest

from sglang.multimodal_gen.runtime.models.vaes.autoencoder_dc import AutoencoderDC


class _FakeInnerModel:
    def __init__(self):
        self.tiling_enabled = False
        self.enable_tiling_kwargs = None

    def enable_tiling(self, **kwargs):
        self.tiling_enabled = True
        self.enable_tiling_kwargs = kwargs

    def disable_tiling(self):
        self.tiling_enabled = False


class TestAutoencoderDCTiling(unittest.TestCase):
    def setUp(self) -> None:
        # Once _inner_model is set, _ensure_inner_model() is a no-op, so no
        # real diffusers model is constructed.
        self.wrapper = AutoencoderDC()
        self.fake_inner = _FakeInnerModel()
        self.wrapper._inner_model = self.fake_inner

    def test_tiling_calls_forward_to_inner_model(self) -> None:
        self.wrapper.enable_tiling(tile_sample_min_height=512)
        self.assertTrue(self.fake_inner.tiling_enabled)
        self.assertEqual(
            self.fake_inner.enable_tiling_kwargs, {"tile_sample_min_height": 512}
        )

        self.wrapper.disable_tiling()
        self.assertFalse(self.fake_inner.tiling_enabled)

    def test_enable_tiling_is_noop_with_spatial_shard_decode(self) -> None:
        self.wrapper._spatial_parallel_decode_enabled = True
        self.wrapper.enable_tiling()
        self.assertFalse(self.fake_inner.tiling_enabled)


if __name__ == "__main__":
    unittest.main()
