"""Exercise real MLA PF/LF routing and layer ownership without an NPU runtime."""

from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]


class Buffer:
    def __init__(self, shape, dtype="bf16"):
        self.shape, self.dtype = tuple(shape), dtype
        self.ndim = len(self.shape)
        self.selection = None

    def narrow(self, axis, start, length):
        if start < 0 or start + length > self.shape[axis]:
            raise IndexError("invalid layer range")
        shape = list(self.shape)
        shape[axis] = length
        view = Buffer(shape, self.dtype)
        view.selection = (self, axis, start, length)
        return view

    def unsqueeze(self, axis):
        shape = list(self.shape)
        shape.insert(axis if axis >= 0 else axis + self.ndim + 1, 1)
        view = Buffer(shape, self.dtype)
        view.selection = self.selection
        return view

    def transpose(self, a, b):
        shape = list(self.shape)
        shape[a], shape[b] = shape[b], shape[a]
        return Buffer(shape, self.dtype)

    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result


class TestRegisteredNPUHostAllocation(unittest.TestCase):
    def test_pin_memory_allocator_selects_configured_backend(self):
        path = ROOT / "python/sglang/srt/mem_cache/pool_host/common.py"
        tree = ast.parse(path.read_text())
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("alloc_with_pin_memory", "alloc_with_pin_memory_npu")
        ]
        module = ast.Module(body=functions, type_ignores=[])
        native = Mock(return_value="native")
        memfabric = Mock(return_value="memfabric")
        acl = Mock(return_value="acl")
        get_backend = Mock(return_value=None)
        backend = SimpleNamespace(get_host_memory_backend=get_backend)
        namespace = {
            "torch": SimpleNamespace(Tensor=object, dtype=object, empty=native)
        }
        with patch.dict(
            sys.modules,
            {"sglang.srt.mem_cache.pool_host.npu_host_memory": backend},
        ):
            exec(compile(module, str(path), "exec"), namespace)
            allocate = namespace["alloc_with_pin_memory_npu"]
            native_allocate = namespace["alloc_with_pin_memory"]
            kwargs = dict(
                dims=(2, 3),
                dtype="uint8",
                device="cpu",
                pin_memory=True,
                allocator=None,
            )

            self.assertEqual(allocate(**kwargs), "native")
            native.assert_called_once_with(
                (2, 3), dtype="uint8", device="cpu", pin_memory=True
            )

            get_backend.return_value = SimpleNamespace(alloc=memfabric)
            self.assertEqual(allocate(**kwargs), "memfabric")
            memfabric.assert_called_once_with((2, 3), "uint8", "cpu", True, None)
            self.assertEqual(native_allocate(**kwargs), "native")

            get_backend.return_value = SimpleNamespace(alloc=acl)
            self.assertEqual(allocate(**kwargs), "acl")
            acl.assert_called_once_with((2, 3), "uint8", "cpu", True, None)


class TestNPUSparseDispatch(unittest.TestCase):
    def setUp(self):
        path = ROOT / "python/sglang/srt/mem_cache/pool_host/mla.py"
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MLATokenToKVPoolHost"
        )
        names = {
            "load_to_device_per_layer",
            "backup_from_device_all_layer",
            "_init_npu_transfer_bindings",
            "_npu_draft_pool_compatible",
            "_warn_draft_indexer_skipped",
            "init_kv_buffer",
            "_init_kv_split_buffer",
        }
        methods = ast.Module(
            body=[
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name in names
            ],
            type_ignores=[],
        )
        backend_tree = ast.parse(path.with_name("npu_host_memory.py").read_text())
        methods.body.append(
            next(
                n
                for n in backend_tree.body
                if isinstance(n, ast.FunctionDef)
                and n.name == "reserve_mla_split_host_memory"
            )
        )
        self.ops = {
            name: Mock(name=name)
            for name in (
                "transfer_kv_per_layer_kv_split_pf_lf",
                "transfer_kv_all_layer_kv_split_lf_pf",
                "transfer_kv_per_layer_kv_split",
                "transfer_kv_all_layer_kv_split",
            )
        }
        self.localize = Mock(side_effect=lambda indices, *_: indices)
        self.namespace = dict(
            self.ops,
            _is_npu=True,
            maybe_dcp_kernel_indices=self.localize,
            logger=Mock(),
            torch=SimpleNamespace(
                distributed=SimpleNamespace(is_initialized=lambda: False)
            ),
            get_host_memory_backend=lambda: None,
        )
        exec(compile(methods, str(path), "exec"), self.namespace)
        host_class = type("Host", (), {name: self.namespace[name] for name in names})
        self.host = h = host_class()
        h._is_dummy = False
        h.dcp_size, h.dcp_rank = 1, 0
        h._target_layer_indices = (0, 1, 2)
        h._host_layer_by_controller = {0: 0, 1: 1, 2: 2, 3: 3}
        h._device_layer_num = h.layer_num = 3
        h._layer_shard_enabled = False
        h.layout, h.page_size = "page_first_kv_split", 128
        h.k_buffer = Buffer((8, 3, 128, 1, 512))
        h.v_buffer = Buffer((8, 3, 128, 1, 64))
        self.native = SimpleNamespace(
            k=Buffer((3, 8, 128, 1, 512)),
            v=Buffer((3, 8, 128, 1, 64)),
            index_k=Buffer((2, 8, 128, 1, 128)),
            index_scales=Buffer((2, 8, 128, 1)),
            index_layer_indices=(0, 2),
            index_width=128,
            k_width=512,
            v_width=64,
        )
        h._native_infos = [SimpleNamespace(native_buffers=lambda: self.native)]
        h._native_buffers = self.native
        h.index_k_buffer = Buffer((8, 2, 128, 1, 128))
        h.index_k_scale_buffer = Buffer((8, 2, 128, 1, 1))
        h._init_npu_transfer_bindings()

    def test_h2d_transfers_each_layer_and_only_its_indexer_slot(self):
        for layout, name in (
            ("page_first_kv_split", "transfer_kv_per_layer_kv_split_pf_lf"),
            ("layer_first_kv_split", "transfer_kv_per_layer_kv_split"),
        ):
            with self.subTest(layout=layout):
                if self.host.layout != layout:
                    for attr in (
                        "k_buffer",
                        "v_buffer",
                        "index_k_buffer",
                        "index_k_scale_buffer",
                    ):
                        setattr(
                            self.host, attr, getattr(self.host, attr).transpose(0, 1)
                        )
                self.host.layout = layout
                self.host._init_npu_transfer_bindings()
                op = self.ops[name]
                for layer in range(3):
                    self.host.load_to_device_per_layer(
                        None, "host", "device", layer, "kernel_ascend"
                    )
                self.assertEqual(op.call_count, 3)
                for layer, call in enumerate(op.call_args_list):
                    kw = call.kwargs
                    self.assertEqual(
                        (kw["device_layer_id"], kw["host_layer_id"]), (0, 0)
                    )
                    axis = 0 if layout == "layer_first_kv_split" else 1
                    self.assertEqual(
                        kw["device_k"].selection, (self.native.k, 0, layer, 1)
                    )
                    self.assertEqual(
                        kw["host_k"].selection, (self.host.k_buffer, axis, layer, 1)
                    )
                    self.assertEqual(
                        kw["device_v"].selection, (self.native.v, 0, layer, 1)
                    )
                    self.assertEqual(
                        kw["host_v"].selection, (self.host.v_buffer, axis, layer, 1)
                    )
                    slot = {0: 0, 2: 1}.get(layer)
                    selected_slot = 0 if slot is not None else None
                    self.assertEqual(kw["device_index_layer_id"], selected_slot)
                    self.assertEqual(kw["host_index_layer_id"], selected_slot)
                    if slot is None:
                        self.assertIsNone(kw["device_index_k"])
                        self.assertIsNone(kw["host_index_k"])
                        self.assertIsNone(kw["device_index_k_scale"])
                        self.assertIsNone(kw["host_index_k_scale"])
                    else:
                        self.assertEqual(
                            kw["device_index_k"].selection,
                            (self.native.index_k, 0, slot, 1),
                        )
                        self.assertEqual(
                            kw["host_index_k"].selection,
                            (self.host.index_k_buffer, axis, slot, 1),
                        )
                        self.assertEqual(
                            kw["device_index_k_scale"].selection,
                            (self.native.index_scales, 0, slot, 1),
                        )
                        self.assertEqual(kw["device_index_k_scale"].ndim, 5)
                        self.assertEqual(
                            kw["host_index_k_scale"].selection,
                            (self.host.index_k_scale_buffer, axis, slot, 1),
                        )
                    self.assertEqual(kw["impl"], "auto")

    def add_draft(self, width=512):
        draft = SimpleNamespace(
            k=Buffer((1, 8, 128, 1, width)),
            v=Buffer((1, 8, 128, 1, 64)),
            index_k=Buffer((1, 8, 128, 1, 128)),
            index_scales=None,
            index_layer_indices=(0,),
        )
        self.host._native_infos.append(SimpleNamespace(native_buffers=lambda: draft))
        layer_id = self.host._device_layer_num + len(self.host._native_infos) - 2
        self.host._host_layer_by_controller[layer_id] = layer_id
        axis = 0 if self.host.layout == "layer_first_kv_split" else 1
        for attr in ("k_buffer", "v_buffer"):
            buffer = getattr(self.host, attr)
            shape = list(buffer.shape)
            shape[axis] = layer_id + 1
            setattr(self.host, attr, Buffer(shape, buffer.dtype))
        self.host._init_npu_transfer_bindings()
        return draft

    def test_draft_h2d_selects_only_draft_and_independent_host_layer(self):
        draft = self.add_draft()
        self.host.load_to_device_per_layer(
            None, "host", "device", 3, "kernel_ascend", is_draft=True
        )
        op = self.ops["transfer_kv_per_layer_kv_split_pf_lf"]
        op.assert_called_once()
        kw = op.call_args.kwargs
        self.assertIs(kw["device_k"], draft.k)
        self.assertEqual((kw["device_layer_id"], kw["host_layer_id"]), (0, 0))
        self.assertEqual(kw["host_k"].selection, (self.host.k_buffer, 1, 3, 1))
        self.assertIsNone(kw["host_index_k"])
        self.assertIsNone(kw["device_index_layer_id"])

    def test_d2h_passes_pool_views_and_target_only_indexer(self):
        draft = self.add_draft()
        for layout, name in (
            ("page_first_kv_split", "transfer_kv_all_layer_kv_split_lf_pf"),
            ("layer_first_kv_split", "transfer_kv_all_layer_kv_split"),
        ):
            with self.subTest(layout=layout):
                if self.host.layout != layout:
                    for attr in (
                        "k_buffer",
                        "v_buffer",
                        "index_k_buffer",
                        "index_k_scale_buffer",
                    ):
                        setattr(
                            self.host, attr, getattr(self.host, attr).transpose(0, 1)
                        )
                    self.host.layout = layout
                    self.host._init_npu_transfer_bindings()
                axis = 0 if layout == "layer_first_kv_split" else 1
                self.host.backup_from_device_all_layer(
                    None, "host", "device", "kernel_ascend"
                )
                calls = self.ops[name].call_args_list
                self.assertEqual(len(calls), 2)
                for call, (start, count) in zip(calls, ((0, 3), (3, 1))):
                    self.assertEqual(call.kwargs.get("host_layer_start", 0), 0)
                    self.assertEqual(
                        call.kwargs["host_k"].selection,
                        (self.host.k_buffer, axis, start, count),
                    )
                    self.assertEqual(
                        call.kwargs["host_v"].selection,
                        (self.host.v_buffer, axis, start, count),
                    )
                self.assertIs(calls[0].kwargs["host_index_k"], self.host.index_k_buffer)
                self.assertIsNone(calls[1].kwargs["host_index_k"])
                self.assertIs(calls[1].kwargs["device_k"], draft.k)

    def test_skipped_draft_preserves_next_pool_host_offset(self):
        self.add_draft(width=999)
        self.add_draft()
        pools = self.host._npu_pool_bindings
        self.assertEqual([host_k.selection[2] for _, host_k, _, _ in pools], [0, 4])

    def test_incompatible_draft_skips_both_directions_and_preserves_next_binding(self):
        self.add_draft(width=999)
        draft = self.add_draft()
        self.host.load_to_device_per_layer(
            None, "host", "device", 3, "kernel_ascend", is_draft=True
        )
        op = self.ops["transfer_kv_per_layer_kv_split_pf_lf"]
        op.assert_not_called()
        self.host.load_to_device_per_layer(
            None, "host", "device", 4, "kernel_ascend", is_draft=True
        )
        kw = op.call_args.kwargs
        self.assertIs(kw["device_k"], draft.k)
        self.assertEqual((kw["device_layer_id"], kw["host_layer_id"]), (0, 0))
        self.assertEqual(kw["host_k"].selection, (self.host.k_buffer, 1, 4, 1))
        self.host.backup_from_device_all_layer(None, "host", "device", "kernel_ascend")
        calls = self.ops["transfer_kv_all_layer_kv_split_lf_pf"].call_args_list
        self.assertEqual([c.kwargs["host_k"].selection[2] for c in calls], [0, 4])

    def test_transfers_reuse_bindings_without_resolving_pools_or_checking_drafts(self):
        self.add_draft()
        for info in self.host._native_infos:
            info.native_buffers = Mock(
                side_effect=AssertionError("resolved at runtime")
            )
        self.host._npu_draft_pool_compatible = Mock(
            side_effect=AssertionError("checked at runtime")
        )
        with (
            patch.object(
                Buffer, "narrow", side_effect=AssertionError("view created at runtime")
            ),
            patch.object(
                Buffer,
                "unsqueeze",
                side_effect=AssertionError("view created at runtime"),
            ),
        ):
            self.host.load_to_device_per_layer(
                None, "host", "device", 2, "kernel_ascend"
            )
            self.host.load_to_device_per_layer(
                None, "host", "device", 3, "kernel_ascend", is_draft=True
            )
            self.host.backup_from_device_all_layer(
                None, "host", "device", "kernel_ascend"
            )
        self.assertEqual(self.ops["transfer_kv_per_layer_kv_split_pf_lf"].call_count, 2)
        self.assertEqual(self.ops["transfer_kv_all_layer_kv_split_lf_pf"].call_count, 2)

    def test_host_layer_mapping_is_not_assumed_equal_to_device_layer(self):
        self.host._host_layer_by_controller[2] = 1
        self.host._init_npu_transfer_bindings()
        self.host.load_to_device_per_layer(None, "host", "device", 2, "kernel_ascend")
        kw = self.ops["transfer_kv_per_layer_kv_split_pf_lf"].call_args.kwargs
        self.assertEqual((kw["device_layer_id"], kw["host_layer_id"]), (0, 0))
        self.assertEqual(kw["device_k"].selection, (self.native.k, 0, 2, 1))
        self.assertEqual(kw["host_k"].selection, (self.host.k_buffer, 1, 1, 1))

    def test_dcp_localization_is_preserved(self):
        self.host.dcp_size, self.host.dcp_rank = 2, 1
        self.localize.side_effect = lambda indices, *_: f"local-{indices}"
        self.host.backup_from_device_all_layer(None, "host", "device", "kernel_ascend")
        kw = self.ops["transfer_kv_all_layer_kv_split_lf_pf"].call_args.kwargs
        self.assertEqual(
            (kw["host_indices"], kw["device_indices"]), ("local-host", "local-device")
        )

    def test_pf_and_lf_allocation_mirror_native_component_shapes(self):
        h = self.host
        h.page_num, h.layer_num, h.page_size = 8, 3, 128
        h._transfer_device = SimpleNamespace(type="npu")
        h.device, h.pin_memory, h.allocator = "cpu", True, None
        h.kv_lora_rank, h.qk_rope_head_dim = 512, 64
        allocate = Mock(side_effect=lambda shape, **_: Buffer(shape))
        reserve = Mock()
        empty = Mock(side_effect=lambda shape, **_: Buffer(shape))
        self.namespace.update(
            ALLOC_MEMORY_FUNCS={"npu": allocate},
            get_host_memory_backend=lambda: SimpleNamespace(ensure_capacity=reserve),
            torch=SimpleNamespace(
                float32="fp32",
                empty=empty,
                npu=SimpleNamespace(current_device=lambda: 0),
            ),
        )
        for layout, kv_dims, index_dims in (
            ("page_first_kv_split", (8, 3, 128, 1), (8, 2, 128, 1)),
            ("layer_first_kv_split", (3, 8, 128, 1), (2, 8, 128, 1)),
        ):
            for packed in (False, True):
                with self.subTest(layout=layout, packed=packed):
                    allocate.reset_mock()
                    reserve.reset_mock()
                    empty.reset_mock()
                    h.layout = layout
                    h.dsa_kv_cache_store_fp8 = packed
                    h.dtype = SimpleNamespace(itemsize=1 if packed else 2)
                    self.native.k_width = 656 if packed else 512
                    v_width = 0 if packed else 64
                    self.assertIs(h.init_kv_buffer(), h.k_buffer)
                    self.assertEqual(h.k_buffer.shape, (*kv_dims, self.native.k_width))
                    self.assertEqual(h.v_buffer.shape, (*kv_dims, v_width))
                    self.assertEqual(h.index_k_buffer.shape, (*index_dims, 128))
                    self.assertEqual(h.index_k_scale_buffer.shape, (*index_dims, 1))
                    expected_blocks = [
                        8 * 128 * 3 * self.native.k_width * h.dtype.itemsize
                    ]
                    if v_width:
                        expected_blocks.append(8 * 128 * 3 * v_width * h.dtype.itemsize)
                    expected_blocks.extend(
                        [8 * 128 * 2 * 128 * h.dtype.itemsize, 8 * 128 * 2 * 4]
                    )
                    reserve.assert_called_once_with(sum(expected_blocks), 0)
                    self.assertEqual(allocate.call_count, 3 if v_width == 0 else 4)
                    self.assertEqual(empty.call_count, 1 if v_width == 0 else 0)

    def test_acl_allocation_mirrors_component_budget(self):
        """The ACL backend receives the same component budget as MemFabric."""
        h = self.host
        h.page_num, h.layer_num, h.page_size = 8, 3, 128
        h._transfer_device = SimpleNamespace(type="npu")
        h.device, h.pin_memory, h.allocator = "cpu", True, None
        h.kv_lora_rank, h.qk_rope_head_dim = 512, 64
        allocate = Mock(side_effect=lambda shape, **_: Buffer(shape))
        reserve = Mock()
        empty = Mock(side_effect=lambda shape, **_: Buffer(shape))
        self.namespace.update(
            ALLOC_MEMORY_FUNCS={"npu": allocate},
            get_host_memory_backend=lambda: SimpleNamespace(ensure_capacity=reserve),
            torch=SimpleNamespace(
                float32="fp32",
                empty=empty,
                npu=SimpleNamespace(current_device=lambda: 0),
            ),
        )
        for layout, kv_dims, index_dims in (
            ("page_first_kv_split", (8, 3, 128, 1), (8, 2, 128, 1)),
            ("layer_first_kv_split", (3, 8, 128, 1), (2, 8, 128, 1)),
        ):
            for packed in (False, True):
                with self.subTest(layout=layout, packed=packed):
                    allocate.reset_mock()
                    reserve.reset_mock()
                    empty.reset_mock()
                    h.layout = layout
                    h.dsa_kv_cache_store_fp8 = packed
                    h.dtype = SimpleNamespace(itemsize=1 if packed else 2)
                    self.native.k_width = 656 if packed else 512
                    v_width = 0 if packed else 64
                    self.assertIs(h.init_kv_buffer(), h.k_buffer)
                    self.assertEqual(h.k_buffer.shape, (*kv_dims, self.native.k_width))
                    self.assertEqual(h.v_buffer.shape, (*kv_dims, v_width))
                    self.assertEqual(h.index_k_buffer.shape, (*index_dims, 128))
                    self.assertEqual(h.index_k_scale_buffer.shape, (*index_dims, 1))
                    expected_blocks = [
                        8 * 128 * 3 * self.native.k_width * h.dtype.itemsize
                    ]
                    if v_width:
                        expected_blocks.append(8 * 128 * 3 * v_width * h.dtype.itemsize)
                    expected_blocks.extend(
                        [8 * 128 * 2 * 128 * h.dtype.itemsize, 8 * 128 * 2 * 4]
                    )
                    reserve.assert_called_once_with(sum(expected_blocks), 0)
                    self.assertEqual(allocate.call_count, 3 if v_width == 0 else 4)
                    self.assertEqual(empty.call_count, 1 if v_width == 0 else 0)

    @staticmethod
    def _npu_hicache_resolution_code():
        path = ROOT / "python/sglang/srt/hardware_backend/npu/utils.py"
        tree = ast.parse(path.read_text())
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "set_default_server_args"
        )
        block = next(
            n
            for n in function.body
            if isinstance(n, ast.If)
            and any(
                isinstance(child, ast.keyword) and child.arg == "hicache_io_backend"
                for child in ast.walk(n)
            )
        )
        return compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec")

    def test_npu_layout_resolution_separates_dsv4_from_mla(self):
        code = self._npu_hicache_resolution_code()
        for dsv4, mla in ((False, False), (False, True), (True, False), (True, True)):
            for layout in (
                "layer_first",
                "page_first",
                "page_first_direct",
                "layer_first_kv_split",
                "page_first_kv_split",
            ):
                with self.subTest(dsv4=dsv4, mla=mla, layout=layout):
                    resolved = {}
                    cfg = SimpleNamespace(
                        enable_hierarchical_cache=True, hicache_mem_layout=layout
                    )
                    namespace = dict(
                        os=os,
                        cfg=cfg,
                        args=cfg,
                        use_mla_backend=lambda _: mla,
                        model_config_of=lambda _: SimpleNamespace(hf_config=None),
                        declare_resolution=lambda args, source, **kw: resolved.update(
                            kw
                        ),
                    )
                    with (
                        patch.dict(os.environ, {}, clear=True),
                        patch.dict(
                            sys.modules,
                            {
                                "sglang.srt.configs.model_config": SimpleNamespace(
                                    is_deepseek_v4=lambda _: dsv4
                                )
                            },
                        ),
                    ):
                        exec(code, namespace)
                        self.assertEqual(
                            os.environ.get("SGLANG_HICACHE_HOST_MEM_BACKEND"),
                            "acl" if dsv4 or mla else None,
                        )
                    if dsv4:
                        expected = (
                            "layer_first" if layout == "layer_first" else "page_first"
                        )
                    elif mla:
                        expected = (
                            "layer_first_kv_split"
                            if layout in ("layer_first", "layer_first_kv_split")
                            else "page_first_kv_split"
                        )
                    else:
                        expected = "page_first_direct"
                    self.assertEqual(resolved["hicache_mem_layout"], expected)
                    self.assertEqual(resolved["hicache_io_backend"], "kernel_ascend")

    def test_npu_host_backend_default_respects_explicit_values_and_cache_triggers(self):
        code = self._npu_hicache_resolution_code()
        env_name = "SGLANG_HICACHE_HOST_MEM_BACKEND"
        for dsv4, mla in ((False, False), (False, True), (True, False), (True, True)):
            for backend in (None, "acl", "memfabric", ""):
                for trigger in ("disabled", "hicache", "offload", "retraction"):
                    with self.subTest(
                        dsv4=dsv4, mla=mla, backend=backend, trigger=trigger
                    ):
                        cfg = SimpleNamespace(
                            enable_hierarchical_cache=trigger == "hicache",
                            disaggregation_decode_enable_offload_kvcache=trigger
                            == "offload",
                            disaggregation_mode=(
                                "decode" if trigger == "retraction" else None
                            ),
                            disaggregation_decode_retraction_backup=None,
                            hicache_mem_layout="page_first",
                        )
                        namespace = dict(
                            os=os,
                            cfg=cfg,
                            args=cfg,
                            use_mla_backend=lambda _: mla,
                            model_config_of=lambda _: SimpleNamespace(hf_config=None),
                            declare_resolution=Mock(),
                        )
                        environment = {} if backend is None else {env_name: backend}
                        with (
                            patch.dict(os.environ, environment, clear=True),
                            patch.dict(
                                sys.modules,
                                {
                                    "sglang.srt.configs.model_config": SimpleNamespace(
                                        is_deepseek_v4=lambda _: dsv4
                                    ),
                                },
                            ),
                        ):
                            exec(code, namespace)
                            expected = backend
                            if (
                                backend is None
                                and (dsv4 or mla)
                                and trigger != "disabled"
                            ):
                                expected = "acl"
                            self.assertEqual(os.environ.get(env_name), expected)
                            if trigger == "disabled":
                                namespace["declare_resolution"].assert_not_called()

    def test_controller_keeps_pf_and_lf_indices_on_npu(self):
        path = ROOT / "python/sglang/srt/managers/cache_controller.py"
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "HiCacheController"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "move_indices"
        )
        namespace = {"torch": SimpleNamespace(Tensor=object)}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        backend_path = ROOT / "python/sglang/srt/mem_cache/pool_host/npu_host_memory.py"
        backend_functions = [
            node
            for node in ast.parse(backend_path.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("host_memory_backend", "ascendc_io_enabled")
        ]
        backend_namespace = {"os": os}
        exec(
            compile(
                ast.Module(body=backend_functions, type_ignores=[]),
                str(backend_path),
                "exec",
            ),
            backend_namespace,
        )
        hi = SimpleNamespace(device="npu")
        cpu_hi = SimpleNamespace(device="cpu")
        cpu_di = object()
        for backend in ("acl", "memfabric", None):
            layouts = (
                (
                    "page_first_kv_split",
                    "layer_first_kv_split",
                    "page_first",
                    "layer_first",
                    "page_first_direct",
                )
                if backend
                else ("page_first_direct",)
            )
            for layout in layouts:
                with self.subTest(backend=backend, layout=layout):
                    di = SimpleNamespace(cpu=Mock(return_value=cpu_di))
                    upload = Mock(return_value=hi)
                    environment = (
                        {"SGLANG_HICACHE_HOST_MEM_BACKEND": backend} if backend else {}
                    )
                    with (
                        patch.dict(os.environ, environment, clear=True),
                        patch.dict(
                            sys.modules,
                            {
                                "sglang.srt.mem_cache.pool_host.npu_host_memory": SimpleNamespace(
                                    ascendc_io_enabled=backend_namespace[
                                        "ascendc_io_enabled"
                                    ],
                                    to_device_no_sync=upload,
                                ),
                            },
                        ),
                    ):
                        ctl = SimpleNamespace(
                            io_backend="kernel_ascend",
                            device="npu",
                            mem_pool_host=SimpleNamespace(layout=layout),
                        )
                        move = namespace["move_indices"]
                        if backend:
                            self.assertEqual(move(ctl, hi, di), (hi, di))
                            upload.assert_not_called()
                            self.assertEqual(move(ctl, cpu_hi, di), (hi, di))
                            upload.assert_called_once_with(cpu_hi, "npu")
                            di.cpu.assert_not_called()
                        else:
                            self.assertEqual(move(ctl, cpu_hi, di), (cpu_hi, cpu_di))
                            di.cpu.assert_called_once_with()
                            upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
