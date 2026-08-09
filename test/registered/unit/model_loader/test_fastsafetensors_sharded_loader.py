import sys
import types
from unittest.mock import MagicMock, patch

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.loader import (
    FastSafetensorsShardedStateLoader,
    get_model_loader,
)
from sglang.srt.server_args import LOAD_FORMAT_CHOICES
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _fastsafetensors_module(loader_factory):
    module = types.ModuleType("fastsafetensors")
    module.SafeTensorsFileLoader = loader_factory
    module.SingleGroup = MagicMock(return_value=object())
    return module


class TestFastSafetensorsShardedStateLoader(CustomTestCase):
    def _make_loader(self, **extra_config):
        return FastSafetensorsShardedStateLoader(
            LoadConfig(
                load_format=LoadFormat.FASTSAFETENSORS_SHARDED,
                model_loader_extra_config=extra_config,
            )
        )

    def test_load_format_routes_and_validates_config(self):
        loader = self._make_loader(
            pattern="rank-{rank}-{part}.safetensors",
            nogds=True,
            bbuf_size_kb=4096,
            max_threads=4,
            max_copy_block_size=1024,
            debug_log=True,
        )

        self.assertIn("fastsafetensors_sharded", LOAD_FORMAT_CHOICES)
        self.assertIsInstance(
            get_model_loader(LoadConfig(load_format="fastsafetensors_sharded"), None),
            FastSafetensorsShardedStateLoader,
        )
        self.assertEqual(loader.pattern, "rank-{rank}-{part}.safetensors")
        self.assertTrue(loader.nogds)
        self.assertEqual(loader.bbuf_size_kb, 4096)
        self.assertEqual(loader.max_threads, 4)
        self.assertEqual(loader.max_copy_block_size, 1024)
        self.assertTrue(loader.debug_log)

        with self.assertRaisesRegex(ValueError, "nogds must be a boolean or null"):
            self._make_loader(nogds="true")

    def test_loads_all_rank_files_together_and_closes_resources(self):
        loader = self._make_loader(nogds=True)
        file_buffer = MagicMock()
        file_buffer.key_to_rank_lidx = {"weight": None}
        file_buffer.get_tensor.return_value = torch.ones(1)
        fast_loader = MagicMock()
        fast_loader.copy_files_to_device.return_value = file_buffer
        module = _fastsafetensors_module(MagicMock(return_value=fast_loader))
        stream = MagicMock()

        with (
            patch.dict(sys.modules, {"fastsafetensors": module}),
            patch("torch.cuda.synchronize") as synchronize,
            patch("torch.cuda.current_stream", return_value=stream),
        ):
            tensors = list(
                loader.iterate_over_files(["part-1", "part-0"], torch.device("cuda:0"))
            )

        self.assertEqual(tensors[0][0], "weight")
        fast_loader.add_filenames.assert_called_once_with({0: ["part-0", "part-1"]})
        fast_loader.copy_files_to_device.assert_called_once_with(
            max_copy_block_size=loader.max_copy_block_size
        )
        synchronize.assert_called_once_with(torch.device("cuda:0"))
        stream.synchronize.assert_called_once()
        file_buffer.close.assert_called_once()
        fast_loader.close.assert_called_once()

    def test_retries_gds_failure_before_copy(self):
        loader = self._make_loader()
        file_buffer = MagicMock()
        file_buffer.key_to_rank_lidx = {"weight": None}
        file_buffer.get_tensor.return_value = torch.ones(1)
        loaders = [MagicMock(), MagicMock()]
        loaders[0].copy_files_to_device.side_effect = RuntimeError(
            "cuFile GDS initialization failed"
        )
        loaders[1].copy_files_to_device.return_value = file_buffer
        loader_factory = MagicMock(side_effect=loaders)
        module = _fastsafetensors_module(loader_factory)

        with (
            patch.dict(sys.modules, {"fastsafetensors": module}),
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.current_stream", return_value=MagicMock()),
        ):
            tensors = list(
                loader.iterate_over_files(["part-0"], torch.device("cuda:0"))
            )

        self.assertEqual(tensors[0][0], "weight")
        self.assertTrue(loader.nogds)
        self.assertEqual(
            [call.kwargs["nogds"] for call in loader_factory.call_args_list],
            [False, True],
        )
        for fast_loader in loaders:
            fast_loader.close.assert_called_once()


if __name__ == "__main__":
    import unittest

    unittest.main()
