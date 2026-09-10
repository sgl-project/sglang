import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage

_DECODING_MODULE = "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding"


def _make_server_args(*, vae_tiling: bool) -> SimpleNamespace:
    return SimpleNamespace(
        enable_torch_compile=False,
        disable_autocast=True,
        pipeline_config=SimpleNamespace(
            vae_tiling=vae_tiling,
            get_decode_scale_and_shift=lambda device, dtype, vae: (1.0, None),
            preprocess_decoding=lambda latents, server_args, vae: latents,
        ),
    )


class FakeVAE(nn.Module):
    """Mock VAE whose decode() outcome is fully controlled by `decode_side_effect`."""

    def __init__(
        self,
        decode_side_effect,
        *,
        with_enable_tiling: bool = True,
        with_disable_tiling: bool = True,
    ):
        super().__init__()
        self.decode = MagicMock(side_effect=decode_side_effect)
        if with_enable_tiling:
            self.enable_tiling = MagicMock()
        if with_disable_tiling:
            self.disable_tiling = MagicMock()


def _oom_error() -> RuntimeError:
    return RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


class TestDecodeOomTiledFallback(unittest.TestCase):
    def _decode(
        self, stage: DecodingStage, server_args: SimpleNamespace
    ) -> torch.Tensor:
        latents = torch.zeros(1, 4, 1, 8, 8)
        with (
            patch(
                f"{_DECODING_MODULE}.get_local_torch_device",
                return_value=torch.device("cpu"),
            ),
            patch(f"{_DECODING_MODULE}.current_platform") as mock_platform,
        ):
            mock_platform.is_cpu.return_value = True
            return stage.decode(latents, server_args, vae_dtype=torch.float32)

    def test_oom_then_success_retries_with_tiling(self):
        success = torch.zeros(1, 3, 1, 8, 8)
        vae = FakeVAE(decode_side_effect=[_oom_error(), success])
        manager = MagicMock()
        manager.attach_mock(vae.decode, "decode")
        manager.attach_mock(vae.enable_tiling, "enable_tiling")
        manager.attach_mock(vae.disable_tiling, "disable_tiling")

        stage = DecodingStage(vae)
        server_args = _make_server_args(vae_tiling=False)

        with self.assertLogs(_DECODING_MODULE, level="WARNING") as logs:
            self._decode(stage, server_args)

        self.assertEqual(
            [c[0] for c in manager.mock_calls],
            ["decode", "enable_tiling", "decode", "disable_tiling"],
        )
        self.assertTrue(
            any("falling back to tiled decode" in msg for msg in logs.output)
        )

    def test_no_enable_tiling_support_does_not_retry(self):
        vae = FakeVAE(
            decode_side_effect=[_oom_error()],
            with_enable_tiling=False,
            with_disable_tiling=False,
        )
        stage = DecodingStage(vae)
        server_args = _make_server_args(vae_tiling=False)

        with self.assertRaises(RuntimeError):
            self._decode(stage, server_args)

        self.assertEqual(vae.decode.call_count, 1)

    def test_enable_only_no_disable_tiling_does_not_retry(self):
        vae = FakeVAE(decode_side_effect=[_oom_error()], with_disable_tiling=False)
        stage = DecodingStage(vae)
        server_args = _make_server_args(vae_tiling=False)

        with self.assertRaises(RuntimeError):
            self._decode(stage, server_args)

        self.assertEqual(vae.decode.call_count, 1)
        # condition (c) requires both enable_tiling and disable_tiling; a VAE
        # missing disable_tiling must not retry even though enable_tiling exists.
        vae.enable_tiling.assert_not_called()

    def test_vae_tiling_already_enabled_does_not_retry(self):
        vae = FakeVAE(decode_side_effect=[_oom_error()])
        stage = DecodingStage(vae)
        server_args = _make_server_args(vae_tiling=True)

        with self.assertRaises(RuntimeError):
            self._decode(stage, server_args)

        self.assertEqual(vae.decode.call_count, 1)
        # enable_tiling() here comes from the pre-existing vae_tiling=True
        # gate at the top of decode(), not from the new retry path.
        vae.enable_tiling.assert_called_once()
        vae.disable_tiling.assert_not_called()

    def test_non_oom_exception_propagates_immediately(self):
        vae = FakeVAE(decode_side_effect=[ValueError("bad shape")])
        stage = DecodingStage(vae)
        server_args = _make_server_args(vae_tiling=False)

        with self.assertRaises(ValueError):
            self._decode(stage, server_args)

        self.assertEqual(vae.decode.call_count, 1)
        vae.enable_tiling.assert_not_called()
        vae.disable_tiling.assert_not_called()

    def test_tiled_retry_also_oom_still_restores_tiling_state(self):
        vae = FakeVAE(decode_side_effect=[_oom_error(), _oom_error()])
        stage = DecodingStage(vae)
        server_args = _make_server_args(vae_tiling=False)

        with self.assertRaises(RuntimeError):
            self._decode(stage, server_args)

        self.assertEqual(vae.decode.call_count, 2)
        vae.enable_tiling.assert_called_once()
        vae.disable_tiling.assert_called_once()


if __name__ == "__main__":
    unittest.main()
