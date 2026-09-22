"""CPU protocol checks for real-IO HiSparse scheduling across device capabilities."""

import ast
import logging
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRocmHiSparseOptimizations(CustomTestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[4] / (
            "python/sglang/srt/managers/hisparse_coordinator.py"
        )
        tree = ast.parse(source.read_text())
        coordinator = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "HiSparseCoordinator"
        )
        names = {
            "_init_shared_index_prefetch",
            "_run_swap_in_kernel",
            "_run_copy_only_kernel",
            "swap_in_selected_pages",
            "map_last_loc_to_buffer",
        }
        coordinator.body = [
            n
            for n in coordinator.body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        functions = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name in {"resolve_shared_index_layers", "_build_prefetch_groups"}
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *functions,
                coordinator,
            ],
            type_ignores=[],
        )
        self.disabled = Mock(return_value=False)
        self.events = []
        self.copy_blocks = []
        self.stream = SimpleNamespace(
            wait_stream=lambda stream: self.events.append("fork")
        )
        device = SimpleNamespace(
            Stream=lambda: self.stream,
            Event=lambda: SimpleNamespace(
                record=lambda stream: self.events.append("record"),
                wait=lambda stream: self.events.append("wait"),
            ),
            current_stream=lambda: "main",
            stream=lambda stream: nullcontext(),
        )
        self.aiter = Mock(return_value=True)
        self.gfx95 = Mock(return_value=True)
        self.scope = scope = {
            "torch": torch,
            "device_module": device,
            "_is_hip": True,
            "is_gfx95_supported": self.gfx95,
            "logger": logging.getLogger(__name__),
            "envs": SimpleNamespace(
                SGLANG_USE_AITER=SimpleNamespace(get=self.aiter),
                SGLANG_DISABLE_HISPARSE_PREFETCH=SimpleNamespace(get=self.disabled),
            ),
            "is_deepseek_dsa": lambda config: True,
            "load_cache_to_device_buffer_mla": self.plan,
            "load_cache_to_device_buffer_dsv4_mla": self.plan,
            "copy_cache_planned_mla": self.copy,
        }
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
        self.cls = scope["HiSparseCoordinator"]
        self.resolve = scope["resolve_shared_index_layers"]

    def plan(self, **kw):
        self.last_plan_kwargs = kw
        self.events.append(("plan", kw["skip_io"]))
        kw["top_k_device_locs"].fill_(1)
        if "miss_count" in kw:
            kw["miss_src"].fill_(2)
            kw["miss_dst"].fill_(1)
            kw["miss_count"].fill_(1)
        if not kw["skip_io"]:
            kw["device_buffer"][1].copy_(kw["host_cache"][2])

    def copy(self, **kw):
        self.events.append(("copy", kw["skip_io"]))
        self.copy_blocks.append(kw["num_blocks"])
        self.assertFalse(kw["skip_io"])
        kw["device_buffer"][1].copy_(kw["host_cache"][2])

    def fixture(
        self,
        *,
        hip=True,
        aiter=True,
        gfx95=True,
        dsv4=False,
        shared=(False, True, False, True),
    ):
        self.scope["_is_hip"] = hip
        self.aiter.return_value = aiter
        self.gfx95.return_value = gfx95
        c = self.cls()
        c.enable_batched_prefix = False
        c.is_dsv4_hisparse = dsv4
        c._separate_copy = hip and aiter and gfx95 and not dsv4
        c.skip_io = False
        c.device, c.top_k, c.device_buffer_size = "cpu", 2, 4
        c.swap_in_block_size, c.item_size_bytes = 1024, 8
        c.mem_pool_host = SimpleNamespace(kv_buffer=torch.arange(64).reshape(4, 8, 2))
        c.mem_pool_device = SimpleNamespace(
            kv_buffer=torch.zeros(4, 8, 2, dtype=torch.int64)
        )
        c.req_device_buffer_tokens = torch.zeros(4, 2, 5, dtype=torch.int32)
        c.req_device_buffer_token_locs = torch.zeros_like(c.req_device_buffer_tokens)
        c.req_to_host_pool = torch.zeros(2, 16, dtype=torch.int64)
        c.lru_slots = torch.zeros(4, 2, 4, dtype=torch.int16)
        c.num_real_reqs = torch.tensor([1], dtype=torch.int32)
        c.top_k_device_locs_buffer = torch.full((2, 2), -1, dtype=torch.int32)
        c._init_shared_index_prefetch(list(shared) if shared else None, 4, 2)
        return c

    def run_layer(self, c, layer):
        return c.swap_in_selected_pages(
            torch.tensor([0]), torch.tensor([6]), torch.tensor([[2, 3]]), layer
        )

    def test_unsupported_paths_keep_original_fused_swap(self):
        for unsupported in (
            {"hip": False},
            {"aiter": False},
            {"gfx95": False},
            {"dsv4": True},
        ):
            with self.subTest(**unsupported):
                self.events.clear()
                c = self.fixture(shared=None, **unsupported)
                self.run_layer(c, 0)
                self.assertEqual(self.events, [("plan", False)])
                self.assertFalse(hasattr(c, "_miss_count"))

    def test_split_plan_is_immediately_followed_by_real_copy(self):
        c = self.fixture(shared=None)
        self.run_layer(c, 0)
        self.assertEqual(self.events, [("plan", True), ("copy", False)])
        self.assertEqual(self.copy_blocks, [16])
        self.assertFalse(c.skip_io)
        torch.testing.assert_close(
            c.mem_pool_device.kv_buffer[0, 1], c.mem_pool_host.kv_buffer[0, 2]
        )

    def test_batched_prefix_keeps_real_copy_after_planning(self):
        c = self.fixture(shared=None)
        c.enable_batched_prefix = True
        self.run_layer(c, 0)
        self.assertTrue(self.last_plan_kwargs["batched_prefix"])
        self.assertEqual(self.events, [("plan", True), ("copy", False)])
        self.assertFalse(c.skip_io)
        torch.testing.assert_close(
            c.mem_pool_device.kv_buffer[0, 1], c.mem_pool_host.kv_buffer[0, 2]
        )

    def test_synchronous_shared_layer_reuses_plan_without_streams(self):
        self.disabled.return_value = True
        c = self.fixture()
        self.run_layer(c, 0)
        self.run_layer(c, 1)
        self.assertEqual(
            self.events, [("plan", True), ("copy", False), ("copy", False)]
        )
        self.assertFalse(hasattr(c, "prefetch_stream"))
        torch.testing.assert_close(
            c.mem_pool_device.kv_buffer[1, 1], c.mem_pool_host.kv_buffer[1, 2]
        )

    def test_prefetch_preserves_fork_copy_record_wait_order(self):
        c = self.fixture()
        self.run_layer(c, 0)
        self.run_layer(c, 1)
        torch.testing.assert_close(
            c.mem_pool_device.kv_buffer[:2, 1], c.mem_pool_host.kv_buffer[:2, 2]
        )
        self.assertEqual(
            self.events,
            [
                ("plan", True),
                ("copy", False),
                "fork",
                ("copy", False),
                "record",
                "wait",
            ],
        )
        self.assertEqual(self.copy_blocks, [16, 16])

    def test_invalid_pattern_does_not_reuse_a_plan(self):
        self.disabled.return_value = True
        c = self.fixture(shared=(False, True))
        self.run_layer(c, 0)
        self.run_layer(c, 1)
        self.assertFalse(c._sync_shared)
        self.assertEqual(self.events, [("plan", True), ("copy", False)] * 2)
        self.assertEqual(self.copy_blocks, [16, 16])

    def test_prefetch_kill_switch_retains_sharing_only_when_opted_in(self):
        self.disabled.return_value = True
        args = dict(
            hf_text_config=SimpleNamespace(num_hidden_layers=4, cli_factor=2),
            pp_size=1,
            is_speculative=False,
        )
        self.assertIsNone(self.resolve(**args))
        self.assertEqual(
            self.resolve(**args, allow_synchronous_shared=True),
            [False, True, False, True],
        )
        for changed in ({"pp_size": 2}, {"is_speculative": True}):
            self.assertIsNone(
                self.resolve(**{**args, **changed}, allow_synchronous_shared=True)
            )


if __name__ == "__main__":
    unittest.main()
