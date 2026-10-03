"""Exercise SiMM NUMA discovery without RDMA or a running storage service."""

import builtins
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# The topology helpers do not use the optional native SiMM bindings.
native = types.ModuleType("simm.kv")
for name in ("BlockView", "Store", "register_mr", "set_flag"):
    setattr(native, name, Mock())
with patch.dict(sys.modules, {"simm.kv": native}):
    from sglang.srt.mem_cache.storage.simm import hicache_simm


class TestSiMMNuma(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cpu = self.root / "sys/devices/system/cpu/cpu88"
        self.cpu.mkdir(parents=True)
        (self.root / "proc/self").mkdir(parents=True)
        self.write_stat("python")
        real_open = builtins.open

        def mapped(path):
            return self.root / str(path).lstrip("/")

        def fixture_open(path, *args, **kwargs):
            return real_open(mapped(path), *args, **kwargs)

        fixture_os = types.SimpleNamespace(
            path=types.SimpleNamespace(
                exists=lambda path: os.path.exists(mapped(path)),
                islink=lambda path: os.path.islink(mapped(path)),
                join=os.path.join,
            ),
            listdir=lambda path: os.listdir(mapped(path)),
            readlink=lambda path: os.readlink(mapped(path)),
            environ={},
            getpid=os.getpid,
        )
        for patched in (
            patch.object(hicache_simm, "os", fixture_os),
            patch.object(hicache_simm, "open", fixture_open, create=True),
        ):
            patched.start()
            self.addCleanup(patched.stop)

    def write_stat(self, comm):
        # Fields 3 through 52; processor is field 39, index 36 after comm.
        fields = ["S", *(["0"] * 49)]
        fields[36] = "88"
        (self.root / "proc/self/stat").write_text(f"123 ({comm}) " + " ".join(fields))

    def add_node(self, node):
        target = self.root / f"sys/devices/system/node/node{node}"
        target.mkdir(parents=True)
        (self.cpu / f"node{node}").symlink_to(target)

    def test_discovers_zero_and_nonzero_node_links(self):
        for node in (0, 1, 12):
            with self.subTest(node=node):
                self.add_node(node)
                try:
                    self.assertEqual(hicache_simm.get_current_process_numa(), node)
                finally:
                    (self.cpu / f"node{node}").unlink()

    def test_parenthesized_command_does_not_shift_processor_field(self):
        self.add_node(0)
        for comm in ("sglang worker", "sglang) worker", "(worker)"):
            with self.subTest(comm=comm):
                self.write_stat(comm)
                self.assertEqual(hicache_simm.get_current_process_numa(), 0)

    def test_absent_node_and_non_node_links_return_unknown(self):
        (self.cpu / "topology").mkdir()
        (self.cpu / "firmware_node").symlink_to(self.cpu / "topology")
        (self.cpu / "node1").mkdir()
        self.assertEqual(hicache_simm.get_current_process_numa(), -1)

    def test_missing_or_malformed_stat_returns_unknown(self):
        self.add_node(0)
        stat = self.root / "proc/self/stat"
        for content in ("", "123 (python) S", "123 python S " + "0 " * 50):
            with self.subTest(content=content):
                stat.write_text(content)
                self.assertEqual(hicache_simm.get_current_process_numa(), -1)
        stat.unlink()
        self.assertEqual(hicache_simm.get_current_process_numa(), -1)

    def test_missing_cpu_directory_returns_unknown(self):
        self.cpu.rmdir()
        self.assertEqual(hicache_simm.get_current_process_numa(), -1)

    def test_constructor_selects_nics_for_detected_nonzero_node(self):
        self.add_node(2)
        for nic, node in (("mlx5_0", 0), ("mlx5_2", 2), ("mlx5_3", 2)):
            device = self.root / f"sys/class/infiniband/{nic}/device"
            device.mkdir(parents=True)
            (device / "numa_node").write_text(str(node))
        config = types.SimpleNamespace(
            extra_config={"manager_address": "127.0.0.1:12345"},
            model_name="model",
            is_mla_model=False,
            tp_rank=0,
            pp_rank=0,
            pp_size=1,
        )
        with (
            patch.object(hicache_simm, "Store"),
            patch.object(hicache_simm, "set_flag"),
            patch.object(hicache_simm.HiCacheSiMM, "warmup"),
        ):
            hicache_simm.HiCacheSiMM(config)
        self.assertEqual(
            set(hicache_simm.os.environ.get("SICL_NET_DEVICES", "").split(",")),
            {"mlx5_2", "mlx5_3"},
        )


if __name__ == "__main__":
    unittest.main()
