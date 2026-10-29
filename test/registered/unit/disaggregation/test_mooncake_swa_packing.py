"""CPU proof, optional wire compatibility and uncertain-completion ownership."""

import ast
import dataclasses
import importlib.util
import json
import struct
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sglang.srt.disaggregation.packed_layout import (
    PackedSlice,
    decode_optional_pack_layout,
    prove_packed_transfer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "_moon_packed_cpu", ROOT / "python/sglang/srt/disaggregation/mooncake/packed.py"
)
packed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packed)


def geometry(heads):
    return dict(
        version=2,
        swa=dict(
            page_size=16,
            heads=heads,
            head_dim=64,
            element_bytes=2,
            storage_dtype="torch.bfloat16",
            logical_dtype="torch.bfloat16",
            page_bytes=16 * heads * 128,
            page_capacities=[8, 8],
        ),
    )


def proof(**changes):
    arguments = dict(
        src_tp=4,
        dst_tp=8,
        src_rank=0,
        dst_rank=1,
        total_heads=16,
        src_ptrs=[100, 200],
        dst_ptrs=[300, 400],
        src_item_lens=[8192] * 2,
        dst_item_lens=[4096] * 2,
        src_layer_ids=[],
        dst_layer_ids=[],
        src_pages=np.array([1, 3], dtype=np.int32),
        dst_pages=np.array([4, 2], dtype=np.int32),
    )
    arguments.update(changes)
    return prove_packed_transfer(geometry(4), geometry(2), "swa", **arguments)


class TestProof(unittest.TestCase):
    def load_method(self, name):
        path = ROOT / "python/sglang/srt/disaggregation/mooncake/conn.py"
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MooncakeKVManager"
        )
        method = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        namespace = {"StateType": SimpleNamespace(SWA="swa")}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[future, method], type_ignores=[])
                ),
                str(path),
                "exec",
            ),
            namespace,
        )
        return namespace[name]

    def test_default_off_and_scope_fallback_before_import_or_gather(self):
        method = self.load_method("_try_send_swa_packed")
        args = (
            SimpleNamespace(mooncake_session_id="peer"),
            [],
            [],
            [],
            [],
            [],
            [],
            None,
            [],
            [],
        )
        self.assertIsNone(method(SimpleNamespace(_swa_packing_enabled=False), *args))
        peer = SimpleNamespace(
            requires_dcp_relayout=False,
            dst_dcp_size=1,
            dst_attn_tp_size=8,
            dst_state_data_ptrs=[[]],
        )
        manager = SimpleNamespace(
            _swa_packing_enabled=True,
            decode_kv_args_table={"peer": peer},
            enable_deferred_decode_kv_release=True,
            is_mla_backend=False,
            is_hybrid_mla_backend=False,
            pp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            kv_args=SimpleNamespace(
                num_draft_entries=0, state_types=["swa"], state_data_ptrs=[[]]
            ),
        )
        for field in ("pp_size", "attn_cp_size", "dcp_size"):
            setattr(manager, field, 2)
            self.assertIsNone(method(manager, *args))
            setattr(manager, field, 1)
        manager.kv_args.num_draft_entries = 1
        self.assertIsNone(method(manager, *args))

    def test_packed_success_or_failure_never_reaches_direct_fallback(self):
        method = self.load_method("_send_swa_state_slice")
        direct = Mock(side_effect=AssertionError("direct fallback entered"))
        manager = SimpleNamespace(
            _try_send_swa_packed=Mock(return_value=0), _get_mha_head_slice_blocks=direct
        )
        self.assertEqual(method(manager, None, [], [], [], [], [], [], 1, 8, None), 0)
        manager._try_send_swa_packed.side_effect = RuntimeError("unknown submitted")
        with self.assertRaisesRegex(RuntimeError, "unknown submitted"):
            method(manager, None, [], [], [], [], [], [], 1, 8, None)
        direct.assert_not_called()

    def test_nonzero_slice_and_invalid_bounds(self):
        self.assertEqual(proof().head_offset_bytes, 256)
        for pages in (
            np.array([-1], dtype=np.int64),
            np.array([8], dtype=np.int64),
            np.array([1], dtype=np.uint64),
        ):
            with self.subTest(pages=pages), self.assertRaises(ValueError):
                proof(src_pages=pages, dst_pages=np.array([1], dtype=np.int32))

    def test_item_order_dtype_legacy_and_malformed_fallback(self):
        self.assertIsNone(proof(dst_item_lens=[8192] * 2))
        self.assertIsNone(proof(src_layer_ids=[1, 2], dst_layer_ids=[2, 1]))
        self.assertIsNone(proof(src_tp=8, dst_tp=4))
        for frame in (None, b"", b"invalid", b"[]", b"null"):
            self.assertIsNone(decode_optional_pack_layout(frame))
        self.assertEqual(decode_optional_pack_layout(b'{"version":2}'), {"version": 2})
        for remote in (
            None,
            {},
            dict(version=2, swa={}),
            dict(
                version=2,
                swa=geometry(2)["swa"] | {"logical_dtype": "torch.float8_e4m3fn"},
            ),
        ):
            self.assertIsNone(
                prove_packed_transfer(
                    geometry(4),
                    remote,
                    "swa",
                    src_tp=4,
                    dst_tp=8,
                    src_rank=0,
                    dst_rank=1,
                    total_heads=16,
                    src_ptrs=[1],
                    dst_ptrs=[2],
                    src_item_lens=[8192],
                    dst_item_lens=[4096],
                    src_layer_ids=[],
                    dst_layer_ids=[],
                    src_pages=[1],
                    dst_pages=[1],
                )
            )

    def test_optional_wire20_preserves_staging19_and_legacy(self):
        path = ROOT / "python/sglang/srt/disaggregation/mooncake/conn.py"
        tree = ast.parse(path.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "KVArgsRegisterInfo"
        )
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        marker = object()
        ns = dict(
            dataclasses=dataclasses,
            struct=struct,
            unpack_int_lists=lambda *args: [],
            StagingRegisterInfo=SimpleNamespace(
                from_zmq_fields=lambda msg, *args, **kwargs: marker
            ),
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[future, node], type_ignores=[])
                ),
                str(path),
                "exec",
            ),
            ns,
        )
        fields = [
            b"room",
            b"host",
            b"1",
            b"session",
            b"",
            b"",
            b"",
            b"1",
            b"8",
            b"4096",
            b"",
            b"",
            b"",
            b"",
            b"",
            b"",
            b"1",
            b"0",
            b"",
            struct.pack("Q", 4096),
        ]
        cls = ns["KVArgsRegisterInfo"]
        old = cls.from_zmq(fields)
        self.assertIsNone(old.pack_layout)
        self.assertEqual(old.dst_kv_item_lens, [4096])
        self.assertIs(old.staging, marker)
        new = cls.from_zmq(fields + [json.dumps(geometry(2)).encode()])
        self.assertEqual(new.pack_layout, geometry(2))
        self.assertEqual(new.dst_kv_item_lens, [4096])
        self.assertIsNone(cls.from_zmq(fields + [b"not-json"]).pack_layout)


class TestLifetime(unittest.TestCase):
    def setUp(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(self.executor.shutdown)
        self.registered = []
        self.deregistered = []
        self.calls = []
        self.callback = lambda *args: 0
        engine = SimpleNamespace(
            register=lambda *args: self.registered.append(args),
            deregister=lambda ptr: self.deregistered.append(ptr),
            batch_transfer_sync=self.transfer,
        )
        self.gather_calls = []
        self.sender = packed.MooncakePackedSwaSender(
            engine,
            self.executor,
            0,
            16384,
            buffer=SimpleNamespace(get_ptr=lambda: 100000),
            gather=lambda *args: self.gather_calls.append(args),
        )
        self.layout = PackedSlice(16, 8192, 4096, 256)

    def transfer(self, *args):
        self.calls.append(args)
        return self.callback(*args)

    def send(self):
        return self.sender.send(
            "peer", [100, 200], [300, 400], [1, 3], [4, 2], self.layout
        )

    def test_success_reuses_registration_and_close_once(self):
        self.assertEqual(self.send(), 0)
        self.assertEqual(self.send(), 0)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(len(self.registered), 1)
        self.assertEqual(self.calls[0][3], [4096, 4096])
        self.assertTrue(self.sender.close())
        self.assertTrue(self.sender.close())
        self.assertEqual(self.deregistered, [100000])

    def test_native_done_failure_is_permanent_quarantine(self):
        self.callback = lambda *args: -1
        with self.assertRaises(RuntimeError):
            self.send()
        self.assertTrue(all(f.done() for f in self.sender._pending))
        self.assertFalse(self.sender.close())
        self.assertFalse(self.deregistered)
        with self.assertRaises(RuntimeError):
            self.send()
        self.assertIn(self.sender, packed._QUARANTINED_SENDERS)

    def test_native_exception_retains_future_and_engine(self):
        def fail(*args):
            raise RuntimeError("unknown")

        self.callback = fail
        with self.assertRaises(RuntimeError):
            self.send()
        self.assertEqual(len(self.sender._pending), 2)
        self.assertFalse(self.sender.close())

    def test_gpu_sync_failure_retains_scratch(self):
        def fail(*args):
            raise RuntimeError("sync unknown")

        self.sender._gather_override = fail
        with self.assertRaises(RuntimeError):
            self.send()
        self.assertTrue(self.sender._gpu_unsettled)
        self.assertFalse(self.calls)
        self.assertFalse(self.sender.close())

    def test_partial_submit_exception_retains_prior_future(self):
        real = self.sender.executor
        count = 0

        def submit(*args):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("executor rejected")
            return real.submit(*args)

        self.sender.executor = SimpleNamespace(submit=submit)
        with self.assertRaises(RuntimeError):
            self.send()
        self.assertEqual(len(self.sender._pending), 1)
        self.assertFalse(self.sender.close())


if __name__ == "__main__":
    unittest.main()
