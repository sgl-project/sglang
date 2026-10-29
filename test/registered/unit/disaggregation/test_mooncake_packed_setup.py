"""Execute the generic setup hook and Mooncake serializer without GPU imports."""

import ast
import json
import struct
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt/disaggregation"


def extract(path, names, class_name=None, namespace=None):
    nodes = ast.parse(path.read_text()).body
    if class_name:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name
        ).body
    selected = [n for n in nodes if getattr(n, "name", None) in names]
    assert len(selected) == len(names)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    namespace = {} if namespace is None else namespace
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *selected], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace


class Tensor:
    ndim = 3
    shape = (128, 4, 64)
    dtype = "torch.bfloat16"

    def __init__(self, pointer):
        self.pointer = pointer

    def is_contiguous(self):
        return True

    def element_size(self):
        return 2

    def data_ptr(self):
        return self.pointer


def storage(hybrid):
    def pool(offset):
        tensors = [Tensor(offset + i * 65536) for i in range(4)]
        p = SimpleNamespace(
            kv_cache_layout="nhd",
            dtype="torch.bfloat16",
            page_size=16,
            k_buffer=tensors[:2],
            v_buffer=tensors[2:],
            model_layer_ids=(0, 2),
        )
        p.get_contiguous_buf_infos = lambda: (
            [t.data_ptr() for t in tensors],
            [65536] * 4,
            [8192] * 4,
        )
        return p

    full, swa = pool(65536), pool(1048576)

    def region(p):
        pointers, spans, items = p.get_contiguous_buf_infos()
        return SimpleNamespace(
            ptrs=pointers,
            lens=spans,
            item_lens=items,
            layer_ids=[],
            dim_per_tensor=[],
            conv_shard_groups=[],
            slice_outer_counts=[],
            kv_layout="nhd",
        )

    regions = dict(full=region(full), swa=region(swa))
    views = dict(full=SimpleNamespace(full_kv_pool=full) if hybrid else full, swa=swa)
    return SimpleNamespace(
        owner="full",
        view=views.__getitem__,
        regions=regions.__getitem__,
        wire_entries=lambda: (
            [SimpleNamespace(sub_pool="swa", state_type="swa")] if hybrid else []
        ),
    )


class TestGenericSetup(unittest.TestCase):
    def setUp(self):
        fake_modules = {}
        for module_name, class_name in (
            ("sglang.srt.hardware_backend.npu.memory_pool_npu", "NPUMLATokenToKVPool"),
            ("sglang.srt.mem_cache.deepseek_v4_memory_pool", "DeepSeekV4TokenToKVPool"),
        ):
            module = ModuleType(module_name)
            setattr(module, class_name, type(class_name, (), {}))
            fake_modules[module_name] = module
        module = ModuleType("sglang.srt.disaggregation.base.conn")
        module.StateType = SimpleNamespace(SWA="swa", MAMBA="mamba")
        fake_modules[module.__name__] = module
        self.patch = patch.dict(sys.modules, fake_modules)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        ns = extract(
            ROOT / "utils.py",
            ["append_state_component", "setup_state_kv_args"],
            namespace={
                "_wire_state_type": lambda value: value,
                "is_mla_backend": lambda view: False,
            },
        )
        self.setup = ns["setup_state_kv_args"]

    def test_full_and_hybrid_actual_proof_tracks_wire_regions(self):
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid):
                args = SimpleNamespace()
                s = storage(hybrid)
                self.setup(args, s)
                self.assertEqual(args.kv_pack_layout["version"], 2)
                self.assertEqual(
                    args.kv_pack_layout["full"]["page_capacities"], [8] * 4
                )
                self.assertEqual(args.state_types, ["swa"] if hybrid else [])
                if hybrid:
                    self.assertEqual(args.state_data_ptrs, [s.regions("swa").ptrs])
                    self.assertEqual(args.kv_pack_layout["swa"]["heads"], 4)
                self.assertFalse(hasattr(args, "mori_pack_layout"))

    def test_draft_clears_previous_proof_and_preserves_state_registration(self):
        args = SimpleNamespace(kv_pack_layout={"stale": True})
        self.setup(args, storage(True), draft_token_to_kv_pool=object())
        self.assertIsNone(args.kv_pack_layout)
        self.assertEqual(args.state_types, ["swa"])
        self.assertEqual(len(args.state_data_ptrs[0]), 4)

    def test_unsupported_geometry_falls_back_without_changing_wire_state(self):
        args = SimpleNamespace()
        s = storage(True)
        s.view("swa").kv_cache_layout = "hnd"
        self.setup(args, s)
        self.assertIsNone(args.kv_pack_layout)
        self.assertEqual(args.state_item_lens, [[8192] * 4])

    def test_setup_to_mooncake_frame20_uses_neutral_field(self):
        args = SimpleNamespace(
            engine_rank=0,
            kv_data_ptrs=[11, 12],
            aux_data_ptrs=[],
            kv_item_lens=[8192, 8192],
            kv_layer_ids=[],
        )
        self.setup(args, storage(True))
        ns = extract(
            ROOT / "mooncake/conn.py",
            ["build_receiver_registration_message"],
            class_name="MooncakeKVManager",
            namespace={
                "struct": struct,
                "json": json,
                "pack_int_lists": lambda *args: b"",
            },
        )
        manager = SimpleNamespace(
            kv_args=args,
            get_session_id=lambda: "peer",
            attn_tp_size=4,
            dcp_size=1,
            dcp_rank=0,
            enable_staging=False,
            local_ip="localhost",
            rank_port=1,
        )
        fields = ns["build_receiver_registration_message"](manager)
        self.assertEqual(len(fields), 21)
        self.assertEqual(fields[14:16], [b"", b""])
        self.assertEqual(fields[18], b"")
        self.assertEqual(list(struct.unpack("2Q", fields[19])), [8192, 8192])
        self.assertEqual(json.loads(fields[20]), args.kv_pack_layout)

    def test_base_args_has_neutral_optional_default(self):
        ns = extract(ROOT / "base/conn.py", ["KVArgs"])
        self.assertIsNone(ns["KVArgs"]().kv_pack_layout)
        self.assertFalse(hasattr(ns["KVArgs"](), "mori_pack_layout"))


if __name__ == "__main__":
    unittest.main()
