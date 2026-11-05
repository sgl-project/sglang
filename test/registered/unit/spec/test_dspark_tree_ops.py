"""Contract tests for the NPU tree-op adapter layer (dspark_tree_ops).

These tests pin the public surface and the invariants the module must preserve,
on CPU torch only. The four ``sgl_kernel_npu.tree`` operators are never executed
here: their absence is exactly what the no-fallback tests exercise.
"""

import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(
    0,
    str(
        _ROOT
        / "python"
        / "sglang"
        / "srt"
        / "speculative"
        / "dspark_components"
    ),
)

try:
    from sglang.srt.speculative.dspark_components import dspark_tree_ops as tree_ops
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase

    register_cpu_ci(est_time=11, suite="base-a-test-cpu")
except Exception:
    import dspark_tree_ops as tree_ops

    def register_cpu_ci(**kwargs):
        return None

    CustomTestCase = unittest.TestCase

REQUIRED_PUBLIC_NAMES = (
    "NodePlan",
    "build_node_plan",
    "get_node_plan",
    "compress_nodes",
    "tree_indexer",
    "quantize_indexer_nodes",
    "tree_indexer_select_all",
    "build_tree_attn_kwargs",
    "build_tree_attn_metadata",
    "attend_nodes",
    "record_compressed_nodes",
    "prepare_nodes",
    "commit_nodes",
    "move_swa_kv",
)

REQUIRED_BACKEND_TOKENS = (
    "_C_ascend",
    "SGLANG_DSPARK_TREE_BACKEND",
    "SGLANG_DSPARK_TREE_OPS",
)


def _module_source() -> str:
    return Path(tree_ops.__file__).read_text(encoding="utf-8")


def _make_plan(device: str = "cpu") -> tree_ops.NodePlan:
    """Two requests, query_width=3 each (root + two chain nodes)."""
    return tree_ops.NodePlan(
        n=6,
        max_q=3,
        start=torch.tensor([5, 5, 5, 8, 8, 8], dtype=torch.int32, device=device),
        depths=torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64, device=device),
        # Globalized: rows 0-2 request 0, rows 3-5 request 1 (offset 3).
        ancestors=torch.tensor(
            [[0, 0, 0], [0, 1, 0], [0, 1, 2], [3, 3, 3], [3, 4, 3], [3, 4, 5]],
            dtype=torch.int64,
            device=device,
        ),
        req_ids=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64, device=device),
        query_cu=torch.arange(7, dtype=torch.int32, device=device),
        hidden_ids=torch.arange(6, dtype=torch.int64, device=device),
        token_index=torch.arange(6, dtype=torch.int64, device=device),
    )


def _make_pre(device: str = "cpu"):
    """A TreePrePlan stand-in with the metadata ``build_node_plan`` reads."""
    prefix = torch.tensor([5, 8], dtype=torch.int32, device=device)
    depths = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64, device=device)
    node_req_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device)
    prefix_flat = prefix.repeat_interleave(3)
    # Local ancestors: root=[0], node1=[0,1], node2=[0,1,2], -1 padded.
    ancestors = torch.tensor(
        [[0, -1, -1], [0, 1, -1], [0, 1, 2], [0, -1, -1], [0, 1, -1], [0, 1, 2]],
        dtype=torch.int64,
        device=device,
    )
    metadata = SimpleNamespace(
        verify_ids=torch.zeros(6, dtype=torch.int64, device=device),
        tree_ancestors=ancestors,
        node_req_ids=node_req_ids,
        query_width=3,
        tree_prefix=prefix_flat,
        positions=prefix_flat + depths,
        node_cu_seqlens_q=torch.arange(7, dtype=torch.int32, device=device),
    )
    return SimpleNamespace(metadata=metadata, max_prefix=8)


class PublicSurfaceTest(CustomTestCase):
    def test_required_public_names_are_defined(self) -> None:
        for name in REQUIRED_PUBLIC_NAMES:
            self.assertTrue(
                hasattr(tree_ops, name), f"missing public name {name!r}"
            )

    def test_source_has_vendor_and_triton_backends(self) -> None:
        """The module owns both vendor bindings and the in-tree Triton path."""
        src = _module_source()
        for token in REQUIRED_BACKEND_TOKENS:
            self.assertIn(token, src, f"missing backend token {token!r}")

    def test_default_backend_is_sgl_kernel_npu(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(tree_ops._tree_backend(), "sgl_kernel_npu")

    def test_source_uses_the_four_direct_sgl_kernel_imports(self) -> None:
        src = _module_source()
        for op in (
            "compressor_tree",
            "compressor_tree_project",
            "quant_lightning_indexer_tree",
            "sparse_attn_sharedkv_tree",
        ):
            self.assertIn(
                f"from sgl_kernel_npu.tree import {op}", src, f"missing {op!r}"
            )

    def test_compressor_loader_returns_project_then_tree(self) -> None:
        fake_tree = types.ModuleType("sgl_kernel_npu.tree")
        fake_tree.compressor_tree = object()
        fake_tree.compressor_tree_project = object()
        fake_package = types.ModuleType("sgl_kernel_npu")
        fake_package.tree = fake_tree

        with mock.patch.dict(
            sys.modules,
            {
                "sgl_kernel_npu": fake_package,
                "sgl_kernel_npu.tree": fake_tree,
            },
        ), mock.patch.dict(
            os.environ, {"SGLANG_DSPARK_TREE_BACKEND": "sgl_kernel_npu"}
        ):
            project, tree = tree_ops._load_compressor_ops()

        self.assertIs(project, fake_tree.compressor_tree_project)
        self.assertIs(tree, fake_tree.compressor_tree)


class NodePlanTest(CustomTestCase):
    def test_build_node_plan_globalizes_ancestors_and_pads_with_request_root(self):
        plan = tree_ops.build_node_plan(_make_pre())
        self.assertEqual(plan.n, 6)
        self.assertEqual(plan.max_q, 3)
        self.assertTrue(torch.equal(plan.start, torch.tensor([5, 5, 5, 8, 8, 8], dtype=torch.int32)))
        self.assertTrue(torch.equal(plan.depths, torch.tensor([0, 1, 2, 0, 1, 2])))
        # -1 padding must be replaced by the request root offset, not left -1.
        expected_ancestors = torch.tensor(
            [[0, 0, 0], [0, 1, 0], [0, 1, 2], [3, 3, 3], [3, 4, 3], [3, 4, 5]]
        )
        self.assertTrue(torch.equal(plan.ancestors, expected_ancestors))
        self.assertTrue(torch.equal(plan.req_ids, torch.tensor([0, 0, 0, 1, 1, 1])))
        self.assertEqual(plan.start.dtype, torch.int32)
        self.assertEqual(plan.query_cu.dtype, torch.int32)
        self.assertEqual(plan.req_ids.dtype, torch.int64)
        # node id != KV slot: each node consumes its own hidden row.
        self.assertTrue(torch.equal(plan.hidden_ids, torch.arange(6)))
        self.assertTrue(torch.equal(plan.token_index, torch.arange(6)))
        self.assertEqual(plan.max_prefix, 8)

    def test_get_node_plan_caches_per_step(self):
        pre = _make_pre()
        first = tree_ops.get_node_plan(pre)
        second = tree_ops.get_node_plan(pre)
        self.assertIs(first, second)

    def test_cached_max_kv_uses_host_prefix_without_device_readback(self):
        plan = _make_plan()
        plan.max_prefix = 8
        with mock.patch.object(
            torch.Tensor, "item", side_effect=AssertionError("device readback")
        ):
            self.assertEqual(tree_ops._cached_max_kv(plan), 11)

    def test_cached_max_kv_falls_back_to_device_max(self):
        plan = _make_plan()
        self.assertEqual(tree_ops._cached_max_kv(plan), 11)


class IndexerTest(CustomTestCase):
    def test_tree_indexer_select_all_pads_out_of_range_positions(self):
        plan = _make_plan()
        out = tree_ops.tree_indexer_select_all(plan, 4, plan.start.device)
        self.assertEqual(out.shape, (6, 1, 4))
        # Req 0: lengths = (start+depth+1)//4 = (5+0+1)//4=1 -> only pos 0 valid.
        row0 = out[0, 0].tolist()
        self.assertEqual(row0, [0, -1, -1, -1])
        # Req 1 root: (8+0+1)//4=2 -> positions 0,1 valid.
        self.assertEqual(out[3, 0].tolist(), [0, 1, -1, -1])


class AttentionKwargsTest(CustomTestCase):
    def _base(self, **overrides):
        kw = dict(
            q=torch.zeros(1, 1, 8),
            ori_kv=torch.zeros(1, 1, 8),
            ori_sparse_indices=torch.zeros(1, 1, 128, dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            seqused_kv=torch.tensor([1], dtype=torch.int32),
            tree_ancestors=torch.tensor([[0]], dtype=torch.int64),
            tree_prefix=torch.tensor([0], dtype=torch.int32),
            cmp_ratio=4,
            softmax_scale=0.5,
        )
        kw.update(overrides)
        return kw

    def test_missing_required_input_raises(self):
        kw = self._base()
        kw["ori_kv"] = None
        with self.assertRaisesRegex(ValueError, "ori_kv"):
            tree_ops.build_tree_attn_kwargs(**kw)

    def test_cmp_tree_kv_requires_ancestors_and_prefix(self):
        kw = self._base(cmp_tree_kv=torch.zeros(1, 8))
        kw["tree_ancestors"] = None
        with self.assertRaisesRegex(ValueError, "tree_ancestors"):
            tree_ops.build_tree_attn_kwargs(**kw)

    def test_all_or_none_tree_fields(self):
        # No cmp_tree_kv -> tree_ancestors/tree_prefix are forced to None.
        kw = self._base()
        out = tree_ops.build_tree_attn_kwargs(**kw)
        self.assertIsNone(out["tree_ancestors"])
        self.assertIsNone(out["tree_prefix"])
        self.assertIsNone(out["cmp_tree_kv"])
        self.assertEqual(out["ori_mask_mode"], 4)
        self.assertEqual(out["cmp_mask_mode"], 3)
        self.assertEqual(out["layout_q"], "TND")
        self.assertEqual(out["layout_kv"], "PA_ND")

    def test_cmp_tree_kv_present_keeps_tree_fields(self):
        kw = self._base(cmp_tree_kv=torch.zeros(1, 8))
        out = tree_ops.build_tree_attn_kwargs(**kw)
        self.assertIsNotNone(out["cmp_tree_kv"])
        self.assertIsNotNone(out["tree_ancestors"])
        self.assertIsNotNone(out["tree_prefix"])


class RecordCompressedNodesTest(CustomTestCase):
    def test_rejects_unsupported_ratio(self):
        with self.assertRaisesRegex(ValueError, "1/4/128"):
            tree_ops.record_compressed_nodes(
                _make_plan(),
                original=torch.zeros(4, 4, 2),
                node_data=torch.zeros(6, 2),
                table=torch.ones(2, 2, dtype=torch.int32),
                ratio=2,
            )

    def test_rejects_noncontiguous_node_data(self):
        node_data = torch.zeros(4, 6, 2).permute(1, 0, 2)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            tree_ops.record_compressed_nodes(
                _make_plan(),
                original=torch.zeros(4, 4, 2),
                node_data=node_data,
                table=torch.ones(2, 2, dtype=torch.int32),
                ratio=4,
            )

    def test_rejects_row_bytes_mismatch(self):
        with self.assertRaisesRegex(ValueError, "row bytes"):
            tree_ops.record_compressed_nodes(
                _make_plan(),
                original=torch.zeros(4, 4, 2),
                node_data=torch.zeros(6, 3),
                table=torch.ones(2, 2, dtype=torch.int32),
                ratio=4,
            )

    def test_records_pages_and_ratio(self):
        record = tree_ops.record_compressed_nodes(
            _make_plan(),
            original=torch.zeros(4, 4, 2),
            node_data=torch.zeros(6, 2),
            table=torch.ones(2, 2, dtype=torch.int32),
            ratio=4,
        )
        self.assertEqual(record.ratio, 4)
        self.assertEqual(record.pages, (1,))
        self.assertIs(record.original.shape[1], 4)


class WindowPagesCacheTest(CustomTestCase):
    def test_repeated_window_pages_reuses_cache(self):
        plan = _make_plan()
        table = torch.ones(2, 2, dtype=torch.int32)
        with mock.patch.object(
            tree_ops.torch, "unique", wraps=torch.unique
        ) as unique:
            first = tree_ops._window_pages(plan, table, ratio=4, rows=4)
            second = tree_ops._window_pages(plan, table, ratio=4, rows=4)
        self.assertEqual(first, second)
        self.assertEqual(unique.call_count, 1)

    def test_window_pages_cache_key_separates_table_and_ratio(self):
        plan = _make_plan()
        table_a = torch.ones(2, 2, dtype=torch.int32)
        table_b = torch.full((2, 2), 2, dtype=torch.int32)
        with mock.patch.object(
            tree_ops.torch, "unique", wraps=torch.unique
        ) as unique:
            tree_ops._window_pages(plan, table_a, ratio=4, rows=4)
            tree_ops._window_pages(plan, table_b, ratio=4, rows=4)
            tree_ops._window_pages(plan, table_a, ratio=1, rows=4)
        self.assertEqual(unique.call_count, 3)

    def test_window_pages_cache_is_per_plan(self):
        table = torch.ones(2, 2, dtype=torch.int32)
        with mock.patch.object(
            tree_ops.torch, "unique", wraps=torch.unique
        ) as unique:
            tree_ops._window_pages(_make_plan(), table, ratio=4, rows=4)
            tree_ops._window_pages(_make_plan(), table, ratio=4, rows=4)
        self.assertEqual(unique.call_count, 2)


class CommitTest(CustomTestCase):

    def test_move_swa_kv_copies_rows(self):
        buf = torch.arange(24, dtype=torch.float32).reshape(2, 12, 1)
        pool = SimpleNamespace(
            layer_num=2,
            translate_loc_from_full_to_swa=lambda loc: loc,
            get_swa_buffer=lambda layer_id: buf[layer_id],
        )
        plan = SimpleNamespace(
            source_rows=torch.tensor([[2], [3]]),
            dest_positions=torch.tensor([[10], [11]]),
            dest_slots=torch.tensor([[10], [11]]),
            commit_lens=torch.tensor([1, 1]),
            new_seq_lens=torch.tensor([6, 9]),
        )
        tree_ops.move_swa_kv(pool, plan)
        self.assertEqual(float(buf[0, 10, 0]), 2.0)
        self.assertEqual(float(buf[1, 11, 0]), 15.0)

    def test_move_swa_kv_gathers_before_overlapping_writes(self):
        buf = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
        pool = SimpleNamespace(
            layer_num=1,
            translate_loc_from_full_to_swa=lambda loc: loc,
            get_swa_buffer=lambda layer_id: buf[layer_id],
        )
        plan = SimpleNamespace(
            source_rows=torch.tensor([[1], [2]]),
            dest_positions=torch.tensor([[2], [3]]),
            dest_slots=torch.tensor([[2], [3]]),
            commit_lens=torch.tensor([2]),
            new_seq_lens=torch.tensor([7]),
        )
        tree_ops.move_swa_kv(pool, plan)
        self.assertEqual(buf[0, :4, 0].tolist(), [0.0, 1.0, 1.0, 2.0])


class NoFallbackTest(CustomTestCase):
    def test_compress_nodes_raises_runtime_error_without_sgl_kernel_npu(self):
        module = SimpleNamespace(
            _fused_wkv_w=torch.ones(1),
            _fused_wgate_w=torch.ones(1),
            _fused_norm_weight_fp32=torch.ones(1),
            ratio=4,
            overlap=0,
            head_dim=8,
            ape=torch.ones(4, 8),
            rope_head_dim=64,
            norm=SimpleNamespace(variance_epsilon=1e-5),
        )
        plan = _make_plan()
        with mock.patch.dict(sys.modules, {"sgl_kernel_npu": None}), mock.patch.dict(
            os.environ, {"SGLANG_DSPARK_TREE_BACKEND": "sgl_kernel_npu"}
        ):
            with self.assertRaises(RuntimeError):
                tree_ops.compress_nodes(
                    module,
                    hidden=torch.zeros(6, 8),
                    state=torch.zeros(4, 4, 16),
                    table=torch.ones(2, 2, dtype=torch.int32),
                    cos=torch.ones(16, 64),
                    sin=torch.ones(16, 64),
                    plan=plan,
                )

    def test_op_loaders_raise_runtime_error_not_import_error(self):
        with mock.patch.dict(sys.modules, {"sgl_kernel_npu": None}), mock.patch.dict(
            os.environ, {"SGLANG_DSPARK_TREE_BACKEND": "sgl_kernel_npu"}
        ):
            with self.assertRaises(RuntimeError):
                tree_ops._load_compressor_ops()
            with self.assertRaises(RuntimeError):
                tree_ops._load_sgl_kernel_indexer_op()
            with self.assertRaises(RuntimeError):
                tree_ops._load_attention_op()


class TreeAttentionContractTest(CustomTestCase):
    def _base_kwargs(self):
        return dict(
            q=torch.zeros(9, 1, 8),
            ori_kv=torch.zeros(4, 1, 8),
            ori_sparse_indices=torch.zeros(9, 1, 128, dtype=torch.int32),
            cu_seqlens_q=torch.arange(10, dtype=torch.int32),
            seqused_kv=torch.full((9,), 3, dtype=torch.int32),
            tree_ancestors=torch.zeros(9, 2, dtype=torch.long),
            tree_prefix=torch.zeros(9, dtype=torch.int32),
            cmp_ratio=1,
            softmax_scale=0.1,
        )

    def test_swa_only_drops_tree_fields_without_cmp_tree_kv(self):
        out = tree_ops.build_tree_attn_kwargs(**self._base_kwargs())
        self.assertIsNone(out["cmp_tree_kv"])
        self.assertIsNone(out["tree_ancestors"])
        self.assertIsNone(out["tree_prefix"])

    def test_cmp_tree_kv_requires_ancestors_and_prefix(self):
        kwargs = self._base_kwargs()
        kwargs["cmp_tree_kv"] = torch.zeros(9, 8)
        kwargs["tree_ancestors"] = None
        with self.assertRaisesRegex(ValueError, "tree_ancestors"):
            tree_ops.build_tree_attn_kwargs(**kwargs)

    def test_cmp_tree_kv_keeps_explicit_ancestors_and_prefix(self):
        kwargs = self._base_kwargs()
        kwargs["cmp_tree_kv"] = torch.zeros(9, 8)
        out = tree_ops.build_tree_attn_kwargs(**kwargs)
        self.assertIsNotNone(out["tree_ancestors"])
        self.assertIsNotNone(out["tree_prefix"])

    def test_indexed_block_tables_are_cached_per_plan(self):
        plan = _make_plan()
        ori = torch.arange(12, dtype=torch.int32).reshape(3, 4)
        cmp = torch.arange(100, 112, dtype=torch.int32).reshape(3, 4)

        first = tree_ops._cached_indexed_tables(plan, ori, cmp)
        second = tree_ops._cached_indexed_tables(plan, ori, cmp)

        self.assertIs(first, second)
        self.assertTrue(torch.equal(first[0], ori.index_select(0, plan.req_ids)))
        self.assertTrue(torch.equal(first[1], cmp.index_select(0, plan.req_ids)))


class NodeCommitLaunchTest(CustomTestCase):
    def _plan(self):
        return tree_ops.NodePlan(
            n=2,
            max_q=2,
            start=torch.zeros(2, dtype=torch.int32),
            depths=torch.tensor([0, 1], dtype=torch.int64),
            ancestors=torch.zeros((2, 2), dtype=torch.int64),
            req_ids=torch.zeros(2, dtype=torch.int64),
            query_cu=torch.arange(3, dtype=torch.int32),
            hidden_ids=torch.arange(2, dtype=torch.int64),
            token_index=torch.arange(2, dtype=torch.int64),
        )

    def test_prepare_commit_use_one_program_per_job(self):
        calls = {}

        class FakeLauncher:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, grid):
                def launch(*args):
                    calls[self.name] = (grid, args[-1])

                return launch

        fake_freeze = FakeLauncher("freeze")
        fake_commit = FakeLauncher("commit")
        descriptors = [(0, 1, 2, 4, 5, 6, 7, 8, 0, 1)] * 2
        with mock.patch.object(
            tree_ops,
            "build_node_descriptors",
            return_value=(descriptors, 16, 4),
        ), mock.patch.object(
            tree_ops,
            "_node_commit_kernels",
            return_value=(fake_freeze, fake_commit),
        ):
            prepared = tree_ops.prepare_nodes(
                self._plan(),
                [object()],
                torch.zeros(2, dtype=torch.int64),
                torch.ones(2, dtype=torch.bool),
                torch.device("cpu"),
            )
            tree_ops.commit_nodes(prepared)

        self.assertEqual(calls["freeze"], ((2,), 512))
        self.assertEqual(calls["commit"], ((2,), 512))


if __name__ == "__main__":
    unittest.main()
