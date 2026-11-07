"""CPU tests of actual packing geometry, eligibility and optional peer frame."""

import ast
import dataclasses
import importlib.util
import json
import struct
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.srt.disaggregation.packed_layout import describe_packable_storage
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


ROOT = Path(__file__).resolve().parents[4]
PACKED_PATH = ROOT / "python/sglang/srt/disaggregation/mori/packed.py"
CONN_PATH = ROOT / "python/sglang/srt/disaggregation/mori/conn.py"
SPEC = importlib.util.spec_from_file_location("_mori_packed_contract", PACKED_PATH)
packed = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = packed
SPEC.loader.exec_module(packed)


def contract(heads, *, head_dim=64, page_size=1, dtype="torch.bfloat16"):
    return {
        "version": packed.PACKED_LAYOUT_VERSION,
        "full": {
            "heads": heads,
            "head_dim": head_dim,
            "element_bytes": 2,
            "storage_dtype": dtype,
            "logical_dtype": dtype,
            "page_size": page_size,
            "page_bytes": heads * head_dim * 2 * page_size,
            "page_capacities": [32, 32],
        },
        "swa": None,
    }


def load_conn_objects():
    tree = ast.parse(CONN_PATH.read_text())
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    method_names = (
        "_try_send_packed",
        "send_kvcache",
        "_send_swa_dsa_state",
        "_get_proven_gqa_transfer_layout",
        "_try_send_direct_swa_slice",
        "_build_tp_slice_transfer_plan",
        "_submit_batch_transfer_plan",
    )
    methods = [
        n
        for n in classes["MoriKVManager"].body
        if isinstance(n, ast.FunctionDef) and n.name in method_names
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ns = {
        "dataclasses": dataclasses,
        "struct": struct,
        "EngineDesc": SimpleNamespace(unpack=lambda b: SimpleNamespace(key="peer")),
        "_unpack_mem_desc_list": lambda b: [],
        "_unpack_mem_desc_lists": lambda b: [],
        "unpack_int_lists": lambda b, fmt: [],
        "msgspec": SimpleNamespace(msgpack=SimpleNamespace(decode=json.loads)),
        "envs": SimpleNamespace(
            SGLANG_MORI_KV_PACK_BUFFER_SIZE_MB=SimpleNamespace(get=lambda: 1),
        ),
        "GroupedIndexPlan": SimpleNamespace(from_groups=lambda *args: None),
        "group_concurrent_contiguous": lambda *args: ([], []),
        "KVPoll": SimpleNamespace(Failed=0, Success=4),
        "np": np,
    }
    module = ast.Module(
        body=[
            future,
            classes["KVArgsRegisterInfo"],
            classes["TPSliceConfig"],
            classes["BatchTransferPlan"],
            *methods,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(CONN_PATH), "exec"), ns)
    return tuple(
        ns[name]
        for name in (
            "KVArgsRegisterInfo",
            "_try_send_packed",
            "send_kvcache",
            "_send_swa_dsa_state",
            "_get_proven_gqa_transfer_layout",
            "_try_send_direct_swa_slice",
            "_build_tp_slice_transfer_plan",
            "_submit_batch_transfer_plan",
        )
    )


(
    PeerInfo,
    try_send,
    send_full,
    send_swa,
    proven_layout,
    direct_swa_slice,
    build_slice_plan,
    submit_plan,
) = load_conn_objects()


class FakeTensor:
    def __init__(self, ptr, shape=(32, 4, 64), dtype="bf16", contiguous=True):
        self.ptr, self.shape, self.dtype, self.contiguous = (
            ptr,
            shape,
            dtype,
            contiguous,
        )
        self.ndim = len(shape)

    def data_ptr(self):
        return self.ptr

    def is_contiguous(self):
        return self.contiguous

    def element_size(self):
        return 2


class TestMoriPackedContract(unittest.TestCase):
    def layout(self, local=None, remote=None, **kw):
        options = dict(src_tp=4, dst_tp=8, src_rank=0, dst_rank=1, total_heads=16)
        options.update(kw)
        return packed.negotiate_packed_slice(
            local or contract(4), remote or contract(2), "full", **options
        )

    def test_head_partition_selects_complete_destination_row(self):
        layout = self.layout()
        self.assertEqual(
            (layout.src_page_bytes, layout.dst_page_bytes, layout.head_offset_bytes),
            (512, 256, 256),
        )
        self.assertIsNone(self.layout(dst_rank=2))

    def test_unsupported_tp_and_replication_fall_back(self):
        for kw in (
            dict(src_tp=8, dst_tp=4),
            dict(src_tp=4, dst_tp=4),
            dict(total_heads=4),
            dict(src_tp=3),
        ):
            self.assertIsNone(self.layout(**kw))

    def test_contract_version_dtype_shape_and_page_mismatch_fall_back(self):
        self.assertIsNone(
            packed.negotiate_packed_slice(
                contract(4),
                None,
                "full",
                src_tp=4,
                dst_tp=8,
                src_rank=0,
                dst_rank=0,
                total_heads=16,
            )
        )
        for remote in (
            contract(2, head_dim=128),
            contract(2, page_size=16),
            contract(2, dtype="fp8"),
            dict(contract(2), version=1),
        ):
            self.assertIsNone(self.layout(remote=remote))

    def test_offsets_preserve_random_page_destination_and_tensor_order(self):
        local, remote, lengths = packed.packed_page_offsets(2, [7, 1, 9], 256)
        self.assertEqual(local, [[0, 256, 512], [768, 1024, 1280]])
        self.assertEqual(remote, [[1792, 256, 2304]] * 2)
        self.assertEqual(lengths, [[256] * 3] * 2)
        with self.assertRaises(ValueError):
            packed.packed_page_offsets(2, [-1], 256)

    def test_uint8_storage_does_not_hide_incompatible_logical_fp8(self):
        local = contract(4, dtype="torch.uint8")
        remote = contract(2, dtype="torch.uint8")
        local["full"]["logical_dtype"] = "torch.float8_e4m3fnuz"
        remote["full"]["logical_dtype"] = "torch.float8_e4m3fn"
        self.assertIsNone(self.layout(local=local, remote=remote))
        remote["full"]["logical_dtype"] = "torch.float8_e4m3fnuz"
        self.assertIsNotNone(self.layout(local=local, remote=remote))
        del remote["full"]["logical_dtype"]
        self.assertIsNone(self.layout(local=local, remote=remote))

    def test_pool_metadata_requires_real_contiguous_uniform_registerable_nhd(self):
        k, v = FakeTensor(1000), FakeTensor(2000)
        pool = SimpleNamespace(
            k_buffer=[k], v_buffer=[v], page_size=1, kv_cache_layout="nhd", dtype="bf16"
        )
        pool.get_contiguous_buf_infos = lambda: (
            [1000, 2000],
            [16384, 16384],
            [512, 512],
        )
        self.assertEqual(packed.describe_packable_pool(pool)["heads"], 4)
        v.contiguous = False
        self.assertIsNone(packed.describe_packable_pool(pool))
        v.contiguous = True
        pool.kv_cache_layout = "hnd"
        self.assertIsNone(packed.describe_packable_pool(pool))
        pool.kv_cache_layout = "nhd"
        pool.get_contiguous_buf_infos = lambda: (
            [999, 2000],
            [16384, 16384],
            [512, 512],
        )
        self.assertIsNone(packed.describe_packable_pool(pool))

    def test_legacy_registration_has_no_packing_contract(self):
        payload = [
            b"",
            b"host",
            b"123",
            b"engine",
            b"",
            b"",
            b"",
            b"0",
            b"8",
            b"1",
            b"256",
            b"",
            b"",
            b"1",
            b"0",
            b"",
            b"",
        ]
        self.assertIsNone(PeerInfo.from_zmq(payload).pack_layout)
        layout = contract(2)
        self.assertEqual(
            PeerInfo.from_zmq(payload + [json.dumps(layout).encode()]).pack_layout,
            layout,
        )

    def test_current_storage_api_proves_actual_owner_and_swa_regions(self):
        def component(ptr, layer):
            pool = SimpleNamespace(
                k_buffer=[FakeTensor(ptr)],
                v_buffer=[FakeTensor(ptr + 1000)],
                dtype="bf16",
                page_size=1,
                kv_cache_layout="nhd",
                model_layer_ids=(layer,),
            )
            pool.get_contiguous_buf_infos = lambda: (
                [ptr, ptr + 1000],
                [16384, 16384],
                [512, 512],
            )
            region = SimpleNamespace(
                ptrs=(ptr, ptr + 1000),
                lens=(16384, 16384),
                item_lens=(512, 512),
                layer_ids=(layer, layer),
            )
            return pool, region

        full, full_region = component(1000, 0)
        swa, swa_region = component(3000, 1)
        views, regions = (
            {"owner": full, "window": swa},
            {
                "owner": full_region,
                "window": swa_region,
            },
        )
        entries = [SimpleNamespace(sub_pool="window", state_type="swa")]
        storage = SimpleNamespace(
            owner="owner",
            view=views.__getitem__,
            regions=regions.__getitem__,
            wire_entries=lambda: entries,
        )
        result = describe_packable_storage(storage)
        self.assertEqual(result["full"]["heads"], 4)
        self.assertEqual(result["swa"]["page_capacities"], [32, 32])
        # Metadata must describe the exact region registered on the wire.
        for field, wrong in (
            ("ptrs", (999, 4000)),
            ("lens", (8192, 16384)),
            ("item_lens", (256, 512)),
            ("layer_ids", (2, 2)),
        ):
            before = getattr(swa_region, field)
            setattr(swa_region, field, wrong)
            self.assertIsNone(describe_packable_storage(storage))
            setattr(swa_region, field, before)
        swa.k_scale_buffer = object()
        self.assertIsNone(describe_packable_storage(storage))
        swa.k_scale_buffer = None
        full.kv_cache_layout = "page_major"
        self.assertIsNone(describe_packable_storage(storage))
        full.kv_cache_layout = "nhd"
        entries.append(SimpleNamespace(sub_pool="window", state_type="block_scale_swa"))
        self.assertIsNone(describe_packable_storage(storage))
        entries.clear()
        self.assertIsNone(describe_packable_storage(storage)["swa"])

    def manager(self):
        calls = []
        sender = SimpleNamespace(send=lambda **kw: calls.append(kw) or ["status"])
        manager = SimpleNamespace(
            _packing_enabled=True,
            enable_deferred_decode_kv_release=True,
            is_mla_backend=False,
            is_hybrid_mla_backend=False,
            pp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            attn_tp_size=4,
            attn_tp_rank=0,
            kv_args=SimpleNamespace(
                mori_pack_layout=contract(4),
                num_draft_entries=0,
                state_types=[],
                total_kv_head_num=16,
            ),
            _packed_local=SimpleNamespace(sender=sender),
        )
        peer = SimpleNamespace(
            pack_layout=contract(2),
            requires_dcp_relayout=False,
            dst_dcp_size=1,
            decode_tp_size=8,
            decode_tp_rank=1,
        )

        def get_layout(*args):
            with patch.dict(
                sys.modules, {"sglang.srt.disaggregation.mori.packed": packed}
            ):
                return proven_layout(manager, *args)

        manager._get_proven_gqa_transfer_layout = get_layout
        manager._try_send_direct_swa_slice = lambda *args: direct_swa_slice(
            manager, *args
        )
        manager._build_tp_slice_transfer_plan = lambda *args: build_slice_plan(
            manager, *args
        )
        return manager, peer, calls

    def invoke(self, manager, peer, **overrides):
        kw = dict(
            component="full",
            src_ptrs=[1000, 2000],
            dst_descs=["k", "v"],
            src_pages=[7, 1],
            dst_pages=[2, 9],
            src_item_lens=[512, 512],
            dst_item_lens=[256, 256],
            src_layer_ids=[0, 0],
            dst_layer_ids=[0, 0],
        )
        kw.update(overrides)
        with patch.dict(sys.modules, {"sglang.srt.disaggregation.mori.packed": packed}):
            return try_send(manager, peer, **kw)

    def test_manager_routes_only_proven_eligible_transfer(self):
        manager, peer, calls = self.manager()
        self.assertEqual(self.invoke(manager, peer), ["status"])
        self.assertEqual(calls[0]["head_offset_bytes"], 256)
        for attr, value in (
            ("_packing_enabled", False),
            ("enable_deferred_decode_kv_release", False),
            ("is_mla_backend", True),
            ("pp_size", 2),
            ("attn_cp_size", 2),
            ("dcp_size", 2),
        ):
            previous = getattr(manager, attr)
            setattr(manager, attr, value)
            self.assertIsNone(self.invoke(manager, peer))
            setattr(manager, attr, previous)
        peer.pack_layout = None
        self.assertIsNone(self.invoke(manager, peer))
        self.assertEqual(len(calls), 1)

    def test_manager_rejects_bad_indices_before_gather_or_network(self):
        manager, peer, calls = self.manager()
        for kw in (
            dict(src_pages=[32, 1]),
            dict(dst_pages=[2, -1]),
            dict(dst_pages=[2]),
        ):
            with self.assertRaises(ValueError):
                self.invoke(manager, peer, **kw)
        self.assertEqual(calls, [])

    def test_manager_normalizes_global_decode_rank_to_attention_tp(self):
        manager, peer, calls = self.manager()
        peer.decode_tp_rank = 9
        self.assertEqual(self.invoke(manager, peer), ["status"])
        self.assertEqual(calls[0]["head_offset_bytes"], 256)

    def test_swa_uses_independent_page_metadata_and_map(self):
        manager, peer, calls = self.manager()
        manager.kv_args.state_types = ["swa"]
        manager.kv_args.mori_pack_layout["swa"] = contract(4, page_size=16)["full"]
        peer.pack_layout["swa"] = contract(2, page_size=16)["full"]
        self.assertEqual(
            self.invoke(
                manager,
                peer,
                component="swa",
                src_pages=[1],
                dst_pages=[3],
                src_item_lens=[8192] * 2,
                dst_item_lens=[4096] * 2,
            ),
            ["status"],
        )
        self.assertEqual(calls[0]["page_size"], 16)
        self.assertEqual(calls[0]["dst_pages"], [3])

    def test_full_transfer_calls_packer_and_default_off_keeps_direct_path(self):
        manager, peer, calls = self.manager()
        manager.kv_args.kv_item_lens = [512, 512]
        manager.kv_args.kv_data_ptrs = [1000, 2000]
        manager.kv_args.kv_layer_ids = [0, 0]
        peer.dst_kv_mem_descs = ["dk", "dv"]
        peer.dst_kv_layer_ids = [0, 0]
        peer.dst_kv_item_len = 256
        manager._get_mha_mem_desc_slices = lambda *args: (
            ["sk"],
            ["sv"],
            ["dk"],
            ["dv"],
            1,
        )
        manager._try_send_packed = lambda *args: ["packed"]
        self.assertEqual(send_full(manager, peer, [1], [2]), ["packed"])
        manager._packing_enabled = False
        manager._build_tp_slice_config = lambda *args: None
        manager._build_tp_slice_transfer_plan = lambda *args: "slice"
        manager._submit_batch_transfer_plan = lambda *args: ["direct"]
        self.assertEqual(send_full(manager, peer, [1], [2]), ["direct", "direct"])

    def test_swa_transfer_calls_packer_but_legacy_peer_still_rejects(self):
        manager, peer, calls = self.manager()
        manager.kv_args.state_types = ["swa"]
        manager.kv_args.state_data_ptrs = [[1000, 2000]]
        manager._try_send_packed = lambda *args: ["packed-swa"]
        args = (
            peer,
            [1],
            [2],
            ["sk", "sv"],
            [512, 512],
            ["dk", "dv"],
            [256, 256],
            "swa",
            [0, 0],
            [0, 0],
        )
        self.assertEqual(send_swa(manager, *args), ["packed-swa"])
        manager._try_send_packed = lambda *args: None
        with self.assertRaisesRegex(RuntimeError, "TP-mismatched"):
            send_swa(manager, *args)

    def swa_direct_manager(self):
        manager, peer, calls = self.manager()
        manager._packing_enabled = False
        manager.kv_args.state_types = ["swa"]
        manager.kv_args.state_data_ptrs = [[1000, 2000]]
        manager.kv_args.mori_pack_layout["swa"] = contract(4, page_size=2)["full"]
        peer.pack_layout["swa"] = contract(2, page_size=2)["full"]
        source = {
            "sk": (np.arange(32768, dtype=np.int64) % 251).astype(np.uint8),
            "sv": ((np.arange(32768, dtype=np.int64) + 19) % 251).astype(np.uint8),
        }
        destination = {
            "dk": np.zeros(16384, dtype=np.uint8),
            "dv": np.zeros(16384, dtype=np.uint8),
        }
        submitted, recorded = [], []

        class Engine:
            uid = 0

            def allocate_transfer_uid(self):
                self.uid += 1
                return self.uid

            def batch_write(
                self, src_descs, src_offsets, dst_descs, dst_offsets, sizes, uids
            ):
                submitted.append(
                    (src_descs, src_offsets, dst_descs, dst_offsets, sizes)
                )
                for src, so, dst, do, length in zip(
                    src_descs, src_offsets, dst_descs, dst_offsets, sizes
                ):
                    for a, b, n in zip(so, do, length):
                        destination[dst][b : b + n] = source[src][a : a + n]
                return [f"status-{uids[0]}"]

        manager.engine = Engine()
        manager._record_submitted_statuses = recorded.extend
        manager._submit_batch_transfer_plan = lambda *args: submit_plan(manager, *args)
        args = (
            peer,
            [7, 1],
            [2, 9],
            ["sk", "sv"],
            [1024, 1024],
            ["dk", "dv"],
            [512, 512],
            "swa",
            [1, 1],
            [1, 1],
        )
        return manager, peer, args, source, destination, submitted, recorded

    def test_packing_off_swa_heterogeneous_direct_reference_is_byte_exact(self):
        manager, peer, args, source, destination, submitted, recorded = (
            self.swa_direct_manager()
        )
        result = send_swa(manager, *args)
        self.assertEqual(result, ["status-1", "status-2"])
        self.assertEqual(recorded, result)
        self.assertEqual(len(submitted), 2)
        self.assertEqual(submitted[0][1], [[7424, 7936, 1280, 1792]])
        self.assertEqual(submitted[0][3], [[1024, 1280, 4608, 4864]])
        self.assertEqual(submitted[0][4], [[256] * 4])
        for src, dst in (("sk", "dk"), ("sv", "dv")):
            expected = (
                source[src].reshape(32, 2, 4, 64, 2)[[7, 1], :, 2:4].reshape(2, 512)
            )
            np.testing.assert_array_equal(
                destination[dst].reshape(32, 512)[[2, 9]], expected
            )
            untouched = np.delete(destination[dst].reshape(32, 512), [2, 9], axis=0)
            self.assertFalse(np.any(untouched))

    def test_direct_swa_reference_preserves_unsupported_scope_and_bounds(self):
        manager, peer, args, source, destination, submitted, recorded = (
            self.swa_direct_manager()
        )
        direct_args = (peer, *args[1:7], *args[8:])
        for attr, value in (
            ("pp_size", 2),
            ("attn_cp_size", 2),
            ("dcp_size", 2),
            ("enable_deferred_decode_kv_release", False),
            ("is_hybrid_mla_backend", True),
        ):
            before = getattr(manager, attr)
            setattr(manager, attr, value)
            self.assertIsNone(direct_swa_slice(manager, *direct_args))
            setattr(manager, attr, before)
        peer.pack_layout = None
        self.assertIsNone(direct_swa_slice(manager, *direct_args))
        with self.assertRaisesRegex(RuntimeError, "TP-mismatched"):
            send_swa(manager, *args)
        peer.pack_layout = contract(2)
        peer.pack_layout["swa"] = contract(2, page_size=2)["full"]
        peer.pack_layout["swa"]["logical_dtype"] = "torch.float8_e4m3fnuz"
        self.assertIsNone(direct_swa_slice(manager, *direct_args))
        peer.pack_layout["swa"]["logical_dtype"] = "torch.bfloat16"
        bad = (peer, [32, 1], *direct_args[2:])
        with self.assertRaises(ValueError):
            direct_swa_slice(manager, *bad)
        self.assertEqual(submitted, [])


if __name__ == "__main__":
    unittest.main()
