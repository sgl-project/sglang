# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

from sglang.srt.checkpoint_engine.update import (
    check_sglang_ready,
    req_inference,
    run_with_torchrun,
    split_checkpoint_files,
    split_tensors,
)


class TestSplitCheckpointFiles(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.checkpoint_path = tmp.name
        self.names = [f"shard-{i}.safetensors" for i in range(5)]
        for name in self.names:
            open(os.path.join(self.checkpoint_path, name), "w").close()
        # Non-safetensors entries must be ignored by the splitter.
        open(os.path.join(self.checkpoint_path, "index.json"), "w").close()
        open(os.path.join(self.checkpoint_path, "notes.bin"), "w").close()

    def _all_ranks(self, world_size):
        return [
            split_checkpoint_files(self.checkpoint_path, rank, world_size)
            for rank in range(world_size)
        ]

    def test_counts_use_ceiling_division(self):
        # 5 files over 2 ranks: ceil(5/2) = 3 on rank 0, 2 on rank 1.
        rank0, rank1 = self._all_ranks(2)
        self.assertEqual(len(rank0), 3)
        self.assertEqual(len(rank1), 2)

    def test_extra_ranks_get_empty_slice(self):
        # 5 files over 7 ranks: one file per rank for ranks 0-4, empty for 5-6.
        counts = [len(files) for files in self._all_ranks(7)]
        self.assertEqual(counts, [1, 1, 1, 1, 1, 0, 0])

    def test_only_safetensors_files_are_split(self):
        for files in self._all_ranks(2):
            for path in files:
                self.assertTrue(path.endswith(".safetensors"))

    def test_union_is_complete_and_disjoint(self):
        rank0, rank1 = self._all_ranks(2)
        names0 = {os.path.basename(path) for path in rank0}
        names1 = {os.path.basename(path) for path in rank1}
        self.assertEqual(names0 | names1, set(self.names))
        self.assertFalse(names0 & names1)


class TestSplitTensors(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.checkpoint_path = tmp.name
        self.weights = {
            "w0": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "w1": torch.full((3,), 1.5),
            "w2": torch.zeros(2, 2),
            "w3": torch.ones(4),
            "w4": torch.full((2, 3), 2.0),
        }
        shard1 = dict(list(self.weights.items())[:3])
        shard2 = dict(list(self.weights.items())[3:])
        save_file(shard1, os.path.join(self.checkpoint_path, "shard1.safetensors"))
        save_file(shard2, os.path.join(self.checkpoint_path, "shard2.safetensors"))
        weight_map = {name: "shard1.safetensors" for name in shard1}
        weight_map.update({name: "shard2.safetensors" for name in shard2})
        with open(
            os.path.join(self.checkpoint_path, "model.safetensors.index.json"), "w"
        ) as f:
            json.dump({"weight_map": weight_map}, f)

    def test_per_rank_split_uses_ceiling_division(self):
        # 5 weights over 2 ranks: rank 0 gets the first 3, rank 1 the last 2.
        self.assertEqual(
            set(split_tensors(self.checkpoint_path, 0, 2)), {"w0", "w1", "w2"}
        )
        self.assertEqual(set(split_tensors(self.checkpoint_path, 1, 2)), {"w3", "w4"})

    def test_tensor_values_roundtrip(self):
        for rank in (0, 1):
            for name, tensor in split_tensors(self.checkpoint_path, rank, 2).items():
                self.assertTrue(torch.equal(tensor, self.weights[name]))

    def test_union_is_complete(self):
        names = set()
        for rank in range(2):
            names |= set(split_tensors(self.checkpoint_path, rank, 2))
        self.assertEqual(names, set(self.weights))


class TestCheckSglangReady(unittest.TestCase):
    def test_non_head_rank_returns_without_http(self):
        # rank 1 with inference_parallel_size 2 is not the head rank
        # (rank // ips * ips == 0 != 1), so readiness polling is skipped.
        with mock.patch.dict(os.environ, {"RANK": "1"}):
            with mock.patch("httpx.Client") as client_cls:
                check_sglang_ready("http://localhost:1", 2)
                client_cls.assert_not_called()


class TestReqInference(unittest.TestCase):
    def test_non_src_rank_closure_is_noop(self):
        with mock.patch.dict(os.environ, {"RANK": "1"}):
            with mock.patch("httpx.Client") as client_cls:
                req = req_inference("http://localhost:1", 2)
                req([("s0", "p0"), ("s1", "p1")])
                client_cls.assert_not_called()

    def test_src_rank_posts_ipc_payload(self):
        with mock.patch.dict(os.environ, {"RANK": "0"}):
            with mock.patch("httpx.Client") as client_cls:
                client = client_cls.return_value.__enter__.return_value
                req = req_inference("http://localhost:1", 2, weight_version="v42")
                req([("s0", "p0"), ("s1", "p1"), ("s2", "p2")])
                client.post.assert_called_once()
                args, kwargs = client.post.call_args
                self.assertEqual(args[0], "http://localhost:1/update_weights_from_ipc")
                self.assertEqual(
                    kwargs["json"]["zmq_handles"], {"s0": "p0", "s1": "p1"}
                )
                self.assertTrue(kwargs["json"]["flush_cache"])
                self.assertEqual(kwargs["json"]["weight_version"], "v42")


class TestRunWithTorchrun(unittest.TestCase):
    def _run(self, argv):
        with mock.patch.object(sys, "argv", ["update.py"] + argv):
            with mock.patch("subprocess.run") as run:
                run.return_value = SimpleNamespace(returncode=0)
                with self.assertRaises(SystemExit) as ctx:
                    run_with_torchrun()
                self.assertEqual(ctx.exception.code, 0)
                return run.call_args[0][0]

    def test_space_form(self):
        cmd = self._run(["--inference-parallel-size", "4", "--endpoint", "http://x"])
        self.assertEqual(cmd[0], "torchrun")
        self.assertEqual(cmd[1], "--nproc-per-node=4")
        self.assertEqual(cmd[-2:], ["--endpoint", "http://x"])

    def test_equals_form(self):
        cmd = self._run(["--inference-parallel-size=2"])
        self.assertEqual(cmd[1], "--nproc-per-node=2")

    def test_default_is_eight(self):
        cmd = self._run(["--other"])
        self.assertEqual(cmd[1], "--nproc-per-node=8")
        self.assertIn("--other", cmd)

    def test_missing_torchrun_exits_one(self):
        with mock.patch.object(sys, "argv", ["update.py"]):
            with mock.patch("subprocess.run", side_effect=FileNotFoundError):
                with self.assertRaises(SystemExit) as ctx:
                    run_with_torchrun()
                self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
