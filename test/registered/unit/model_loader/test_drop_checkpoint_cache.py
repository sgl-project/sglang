"""Verify checkpoint page-cache release in both shard and model modes."""

import argparse
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import safetensors.torch
import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import (
    buffered_multi_thread_safetensors_weights_iterator,
    drop_checkpoint_cache_after_model_load,
    safetensors_weights_iterator,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DROP = "sglang.srt.model_loader.weight_utils._drop_file_cache_after_load"


class TestDropCheckpointCache(CustomTestCase):
    """Check shard accounting, model-level rank partitioning and CLI parsing."""

    def test_drop_after_each_shard(self):
        """Each consumed shard is advised once while tensor values remain intact."""
        with tempfile.TemporaryDirectory() as directory:
            files = []
            expected = {}
            for i in range(3):
                name = f"layer{i}.weight"
                expected[name] = torch.arange(16).reshape(4, 4) + i
                path = os.path.join(directory, f"model-{i}.safetensors")
                safetensors.torch.save_file({name: expected[name]}, path)
                files.append(path)
            for iterator in (
                safetensors_weights_iterator,
                buffered_multi_thread_safetensors_weights_iterator,
            ):
                for disable_mmap in (False, True):
                    for drop_cache in (False, True):
                        with self.subTest(
                            iterator=iterator.__name__,
                            mmap=not disable_mmap,
                            drop=drop_cache,
                        ):
                            kwargs = dict(
                                disable_mmap=disable_mmap,
                                drop_cache_after_load=drop_cache,
                            )
                            if iterator is not safetensors_weights_iterator:
                                kwargs["max_workers"] = 2
                            with patch(DROP) as drop:
                                actual = dict(iterator(files, **kwargs))
                            self.assertEqual(set(actual), set(expected))
                            for name in expected:
                                self.assertTrue(
                                    torch.equal(actual[name], expected[name])
                                )
                            self.assertCountEqual(
                                [c.args[0] for c in drop.call_args_list],
                                files if drop_cache else [],
                            )

    def test_model_mode_without_distributed_advises_every_file_once(self):
        """Deduplicate file paths when loading outside a distributed engine."""
        files = ["/ckpt/b.safetensors", "/ckpt/a.safetensors", "/ckpt/b.safetensors"]
        with (
            patch(DROP) as drop,
            patch("torch.distributed.is_initialized", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            drop_checkpoint_cache_after_model_load(files)
        self.assertEqual(
            [c.args[0] for c in drop.call_args_list],
            ["/ckpt/a.safetensors", "/ckpt/b.safetensors"],
        )

    def test_model_mode_waits_for_all_ranks_and_partitions_files(self):
        """Cover every file per node and order copies, barriers and eviction."""
        files = [f"/ckpt/{i}.safetensors" for i in range(5)]
        for node in range(2):
            advised = []
            for local_rank in range(2):
                events = []
                group = Mock(
                    rank=node * 2 + local_rank,
                    world_size=4,
                    local_rank=4 + local_rank * 2,
                    local_size=0,
                )
                group.barrier.side_effect = lambda: events.append("barrier")
                with (
                    patch(DROP, side_effect=lambda path: events.append(path)),
                    patch("torch.distributed.is_initialized", return_value=True),
                    patch(
                        "sglang.srt.model_loader.weight_utils.get_parallel",
                        return_value=SimpleNamespace(world_group=group),
                    ),
                    patch("torch.cuda.is_available", return_value=True),
                    patch(
                        "torch.cuda.synchronize",
                        side_effect=lambda: events.append("sync"),
                    ),
                ):
                    drop_checkpoint_cache_after_model_load(
                        list(reversed(files)), nnodes=2
                    )
                expected = files[local_rank::2]
                self.assertEqual(events, ["sync", "barrier", *expected, "barrier"])
                advised.extend(expected)
            self.assertCountEqual(advised, files)

    def test_default_loader_modes_and_reuse(self):
        """Exercise actual loader construction, iteration and post-load release."""
        module = "sglang.srt.model_loader.loader"
        with tempfile.TemporaryDirectory() as directory:
            files = [os.path.join(directory, f"{i}.safetensors") for i in range(2)]
            expected = torch.arange(4)
            for path in files:
                safetensors.torch.save_file({"weight": expected}, path)
            config = SimpleNamespace(
                model_path=directory,
                revision=None,
                dtype=torch.float32,
                hf_config=SimpleNamespace(),
            )
            for multithread in (False, True):
                for mode in (None, "shard", "model"):
                    with self.subTest(multithread=multithread, mode=mode):
                        loader = DefaultModelLoader(
                            LoadConfig(
                                model_loader_extra_config={
                                    "enable_multithread_load": multithread,
                                    "num_threads": 2,
                                }
                            )
                        )
                        args = SimpleNamespace(
                            weight_loader_disable_mmap=False,
                            weight_loader_prefetch_checkpoints=False,
                            weight_loader_prefetch_num_threads=4,
                            weight_loader_drop_cache_after_load=mode,
                            nnodes=1,
                        )
                        model = torch.nn.Module()
                        events = []

                        def consume(weights):
                            for name, tensor in weights:
                                self.assertEqual(name, "weight")
                                self.assertTrue(torch.equal(tensor, expected))
                            events.append("copied")

                        model.load_weights = consume
                        for path in files:
                            events.clear()
                            with (
                                patch.object(
                                    loader,
                                    "_prepare_weights",
                                    return_value=(directory, [path], True),
                                ),
                                patch(f"{module}.get_model", return_value=args),
                                patch(f"{module}.get_parallel", return_value=args),
                                patch(
                                    f"{module}._initialize_model", return_value=model
                                ),
                                patch(
                                    f"{module}._get_quantization_config",
                                    return_value=None,
                                ),
                                patch(DROP) as shard_drop,
                                patch(
                                    f"{module}.drop_checkpoint_cache_after_model_load",
                                    side_effect=lambda paths, nnodes: events.append(
                                        list(paths)
                                    ),
                                ) as model_drop,
                            ):
                                self.assertIs(
                                    loader.load_model(
                                        model_config=config,
                                        device_config=SimpleNamespace(device="cpu"),
                                    ),
                                    model,
                                )
                            self.assertEqual(
                                shard_drop.call_count, int(mode == "shard")
                            )
                            self.assertEqual(
                                model_drop.call_count, int(mode == "model")
                            )
                            self.assertEqual(
                                events,
                                ["copied", [path]] if mode == "model" else ["copied"],
                            )

    def test_cli_modes(self):
        """Preserve the bare flag and accept explicit shard/model modes."""
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        def parse(*extra):
            args = parser.parse_args(["--model-path", "m", *extra])
            return args.weight_loader_drop_cache_after_load

        self.assertIsNone(parse())
        self.assertEqual(parse("--weight-loader-drop-cache-after-load"), "shard")
        self.assertEqual(
            parse("--weight-loader-drop-cache-after-load", "shard"), "shard"
        )
        self.assertEqual(
            parse("--weight-loader-drop-cache-after-load", "model"), "model"
        )


if __name__ == "__main__":
    unittest.main()
