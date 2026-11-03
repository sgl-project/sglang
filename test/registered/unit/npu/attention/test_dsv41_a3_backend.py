"""CPU call-contract tests for the real A3 DSV4.1 backend methods.

Only device operators/imports are mocked. These tests check layouts, dispatch,
lengths and persistent-buffer updates; they do not prove NPU graph support or
the numerical correctness of the external kernels.
"""

import ast
import functools
import importlib.metadata
import os
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[5]
BACKEND = (
    ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py"
)
METADATA = ROOT / "python/sglang/srt/layers/attention/dsv4/metadata.py"
MODEL = ROOT / "python/sglang/srt/models/deepseek_v4.py"


def _definitions(path, names, namespace, *, operator_namespace=None):
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    if operator_namespace is not None:
        # Keep the production function body and mutation declaration, but do
        # not replace/register a production operator in a shared CI process.
        for node in nodes:
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "custom_op"
                    and isinstance(decorator.args[0], ast.Constant)
                ):
                    op_name = decorator.args[0].value.split("::", 1)[1]
                    decorator.args[0].value = f"{operator_namespace}::{op_name}"
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102
    return namespace


class TestA3Backend(unittest.TestCase):
    def assertTensorEqual(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.equal(actual, expected))

    def setUp(self):
        self.a3 = SimpleNamespace(
            attention=Mock(side_effect=lambda q, **kw: (torch.ones_like(q), None)),
            metadata=Mock(
                side_effect=lambda *a, **kw: torch.full(
                    (2048,), 17 + kw["cmp_ratio"], dtype=torch.int32
                )
            ),
            indexer=Mock(),
            candidate=Mock(),
            sparse_indexer=Mock(),
        )
        self.a5 = SimpleNamespace(
            mixed_quant_sparse_flash_mla=Mock(
                side_effect=lambda q, **kw: (torch.ones_like(q), None)
            ),
            quant_lightning_indexer=Mock(),
            mixed_quant_sparse_flash_mla_metadata=Mock(
                return_value=torch.ones(1024, dtype=torch.int32)
            ),
            quant_lightning_indexer_metadata=Mock(
                return_value=torch.ones(1024, dtype=torch.int32)
            ),
        )
        self.ns = _definitions(
            BACKEND,
            {"LowRatioCompressAscendBackendMixin"},
            {
                "torch": torch,
                "F": F,
                "os": os,
                "SimpleNamespace": SimpleNamespace,
                "is_npu_arch35": lambda: False,
                "_dsv41_a3_ops": lambda: self.a3,
                "_ds41_ops": Mock(side_effect=AssertionError("A3 entered A5 ops")),
            },
        )
        self.backend = self.ns["LowRatioCompressAscendBackendMixin"]()
        b = self.backend
        b.page_size = 128
        b._dsv4_unique_compress_ratios = (0, 1, 2)
        b._dsv4_q_head_num = 32
        b._dsv4_head_dim = 512
        b._dsv4_index_n_heads = 32
        b._dsv4_index_head_dim = 128
        b._dsv4_index_topk = 512
        b._dsv4_sliding_window_size = 128
        b.graph_metadata = {}
        self.fm = SimpleNamespace(
            actual_seq_lengths_q_pa=torch.arange(5, dtype=torch.int32),
            actual_seq_lengths_kv=torch.tensor([129, 128, 1, 0]),
            block_tables=torch.ones(4, 2, dtype=torch.int32),
            swa_page_table=torch.ones(4, 2, dtype=torch.int32),
            c1_page_table=torch.ones(4, 2, dtype=torch.int32),
            c2_page_table=torch.ones(4, 2, dtype=torch.int32),
            c1_index_page_table=torch.ones(4, 4, dtype=torch.int32),
            c2_index_page_table=torch.ones(4, 2, dtype=torch.int32),
            c1_topk_indices=torch.zeros(4, 512, dtype=torch.int32),
            c2_topk_indices=torch.zeros(4, 512, dtype=torch.int32),
            kernel_metadata={
                key: torch.ones(2048, dtype=torch.int32)
                for key in ("win_metadata", "c1a_metadata", "c2a_metadata")
            },
        )
        b.forward_metadata = self.fm
        self.fb = SimpleNamespace(
            seq_lens=self.fm.actual_seq_lengths_kv,
            out_cache_loc=torch.tensor([256, 383, 384, 0]),
            batch_size=4,
        )
        self.pool = SimpleNamespace(
            get_swa_buffer=Mock(
                return_value=torch.zeros(5, 128, 1, 512, dtype=torch.bfloat16)
            ),
            get_compress_buffer=Mock(
                return_value=torch.zeros(5, 64, 1, 512, dtype=torch.bfloat16)
            ),
            get_index_k=Mock(
                return_value=torch.zeros(5, 64, 1, 128, dtype=torch.bfloat16)
            ),
            get_index_scale=Mock(side_effect=AssertionError("A3 requested FP4 scale")),
        )
        b.token_to_kv_pool = self.pool
        self.layer = SimpleNamespace(
            layer_id=20,
            scaling=512**-0.5,
            compress_ratio=1,
            indexer=SimpleNamespace(
                index_head_dim=128,
                candidate_topk_blocks=2048,
                candidate_block_size=8,
            ),
        )
        self.q = torch.zeros(4, 32, 512, dtype=torch.bfloat16)
        self.index_q = torch.zeros(4, 32, 128, dtype=torch.bfloat16)
        self.w = torch.ones(4, 32, dtype=torch.float32)
        b._dsv41_index_q = Mock(return_value=(self.index_q, None, self.w))
        self.indices = torch.zeros(4, 1, 512, dtype=torch.int32)
        self.blocks = torch.zeros(4, 1, 2048, dtype=torch.int32)
        self.lengths = torch.empty(0, dtype=torch.int32)
        self.a3.indexer.return_value = (self.indices, None)
        self.a3.candidate.return_value = (self.indices, None, self.blocks, self.lengths)
        self.a3.sparse_indexer.return_value = (self.indices, None)

        expansion = _definitions(
            METADATA, {"expand_index_page_table"}, {"torch": torch}
        )
        metadata_module = ModuleType("sglang.srt.layers.attention.dsv4.metadata")
        metadata_module.expand_index_page_table = expansion["expand_index_page_table"]
        pool_module = ModuleType("sglang.srt.mem_cache.deepseek_v4_memory_pool")
        pool_module.DSV41_INDEX_PAGE_SIZE = 64
        patcher = patch.dict(
            "sys.modules",
            {
                metadata_module.__name__: metadata_module,
                pool_module.__name__: pool_module,
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a3_attention_ratios_use_bf16_normal_window_and_no_a5_lengths(self):
        for ratio in (0, 1, 2):
            out = self.backend._forward_low_ratio_attention(
                self.q, self.layer, self.fb, torch.ones(32), ratio
            )
            args, kw = self.a3.attention.call_args
            self.assertIs(args[0], self.q)
            self.assertEqual(kw["ori_kv"].dtype, torch.bfloat16)
            self.assertEqual(kw["layout_q"], "TND")
            self.assertEqual(kw["layout_kv"], "PA_BBND")
            self.assertEqual((kw["ori_mask_mode"], kw["ori_win_left"]), (4, 127))
            self.assertEqual(kw["cmp_mask_mode"], 3 if ratio else 0)
            self.assertEqual(kw["sinks"].dtype, torch.float32)
            for name in (
                "ori_topk_length",
                "cmp_topk_length",
                "seqused_q",
                "quant_mode",
            ):
                self.assertNotIn(name, kw)
            if ratio:
                self.assertEqual(kw["cmp_kv"].dtype, torch.bfloat16)
                self.assertEqual(kw["cmp_sparse_indices"].shape, (4, 1, 512))
                self.assertTensorEqual(
                    kw["seqused_cmp_kv"], (self.fb.seq_lens // ratio).int()
                )
                if ratio == 2:
                    self.assertTensorEqual(
                        kw["cmp_residual_kv"], (self.fb.seq_lens % ratio).int()
                    )
                else:
                    self.assertIsNone(kw["cmp_residual_kv"])
            else:
                self.assertNotIn("cmp_kv", kw)
            self.assertTrue(torch.all(out[:3] == 1))
            self.assertTrue(torch.all(out[3] == 0))

    def test_a3_index_default_and_candidate_keep_bf16_and_residual_lengths(self):
        for ratio in (1, 2):
            self.layer.compress_ratio = ratio
            for candidate in (False, True):
                topk, blocks = self.backend._dsv41_index_full_topk(
                    self.layer,
                    self.q,
                    self.index_q,
                    torch.arange(4),
                    self.fb,
                    candidate=candidate,
                )
                op = self.a3.candidate if candidate else self.a3.indexer
                args, kw = op.call_args
                self.assertEqual(args[0].dtype, torch.bfloat16)
                self.assertEqual(args[1].dtype, torch.bfloat16)
                self.assertEqual(args[2].dtype, torch.float32)
                self.assertEqual(args[3], 512)
                self.assertEqual(topk.shape, (4, 512))
                self.assertEqual((kw["layout_q"], kw["layout_k"]), ("TND", "PA_BBND"))
                self.assertTensorEqual(
                    kw["seqused_k"], (self.fb.seq_lens // ratio).int()
                )
                if ratio == 2:
                    self.assertTensorEqual(
                        kw["cmp_residual_k"], (self.fb.seq_lens % ratio).int()
                    )
                else:
                    self.assertIsNone(kw["cmp_residual_k"])
                self.assertNotIn("metadata", kw)
                self.assertNotIn("descale_q", kw)
                if candidate:
                    self.assertIs(blocks[0], self.blocks)
                    self.assertIs(blocks[1], self.lengths)
                    self.assertEqual(kw["candidate_topk_blocks"], 2048)
                else:
                    self.assertIsNone(blocks)

    def test_a3_sparse_indexer_consumes_candidate_ids_not_empty_lengths(self):
        self.fm.dsv41_candidate_blocks = (self.blocks, self.lengths)
        topk = self.backend._dsv41_index_qsli_topk(
            self.layer, self.q, self.index_q, torch.arange(4), self.fb
        )
        args, kw = self.a3.sparse_indexer.call_args
        self.assertIs(args[4], self.blocks)
        self.assertEqual(kw["candidate_block_size"], 8)
        self.assertNotIn("candidate_block_length", kw)
        self.assertNotIn("descale_k", kw)
        self.assertIsNone(kw["cmp_residual_k"])
        self.assertEqual(topk.shape, (4, 512))
        self.assertEqual(self.backend._dsv41_index_kernel_metadata(self.fb), {})

    def test_a3_query_and_key_prologues_do_not_call_fp4_or_nz_ops(self):
        rope = SimpleNamespace(
            for_freqs=Mock(
                return_value=SimpleNamespace(
                    get_cos_sin=Mock(
                        return_value=(torch.ones(4, 64), torch.zeros(4, 64))
                    )
                )
            ),
            apply_rotary_mul_inplace=Mock(),
        )
        rope_module = ModuleType("sglang.srt.hardware_backend.npu.dsv4.dsv4_rope")
        rope_module.Dsv4NpuRoPE = rope
        # Any attribute access for quantization/format casting now fails.
        torch_npu_module = ModuleType("torch_npu")
        idx = self.layer.indexer
        idx.n_heads = 32
        idx.rope_head_dim = 64
        idx.wq_b = Mock(return_value=(self.index_q.flatten(1), None))
        idx.head_weights = Mock(return_value=self.w.to(torch.bfloat16))
        idx.wk = Mock(return_value=torch.ones(4, 128, dtype=torch.bfloat16))
        idx.k_norm = Mock(side_effect=lambda value: value)
        self.layer.freqs_cis = torch.ones(1024, 32)
        self.fm.c1_loc = self.fb.out_cache_loc
        self.fm.positions_cmp_padding_c1 = torch.arange(4)
        self.pool.set_compress_buffer = Mock()
        with patch.dict(
            "sys.modules",
            {rope_module.__name__: rope_module, "torch_npu": torch_npu_module},
        ):
            query_method = type(self.backend)._dsv41_index_q
            q, scale, w = query_method(
                self.backend, self.layer, self.q, self.index_q, torch.arange(4)
            )
            self.backend._dsv41_index_k_store(self.layer, self.q[:, 0])
        self.assertEqual(q.dtype, torch.bfloat16)
        self.assertEqual(q.shape, (4, 32, 128))
        self.assertIsNone(scale)
        self.assertEqual(w.dtype, torch.float32)
        args = self.pool.set_compress_buffer.call_args.args
        self.assertEqual(args[0], self.layer.layer_id)
        self.assertIs(args[1], self.fm.c1_loc)
        self.assertEqual(args[2].dtype, torch.bfloat16)
        self.assertEqual(args[2].shape, (4, 128))
        self.assertIsNone(args[3])
        self.assertTrue(args[4])
        self.assertEqual(rope.apply_rotary_mul_inplace.call_count, 2)

    def _init_graph(self, bs=8):
        fm = SimpleNamespace(
            block_tables=torch.ones(bs, 2, dtype=torch.int32),
            actual_seq_lengths_q_pa=torch.arange(bs + 1, dtype=torch.int32),
            actual_seq_lengths_kv=torch.ones(bs, dtype=torch.int32),
        )
        self.backend._init_dsv41_graph_metadata(fm, bs, 1, "cpu")
        self.backend.forward_metadata = fm
        return fm

    def test_a3_graph_initialization_builds_metadata_before_capture(self):
        fm = self._init_graph()
        self.assertEqual(self.a3.metadata.call_count, 3)
        for key in ("win_metadata", "c1a_metadata", "c2a_metadata"):
            self.assertEqual(fm.kernel_metadata[key].shape, (2048,))
            self.assertTrue(torch.all(fm.kernel_metadata[key] > 0))
        self.assertEqual(fm.c1_index_page_table.shape, (8, 4))
        self.assertEqual(fm.c2_index_page_table.shape, (8, 2))
        self.assertNotIn("c1a_qli_metadata", fm.kernel_metadata)

    def test_a3_capture_uses_actual_batch_lengths_instead_of_probe_ones(self):
        fm = self._init_graph()
        self.a3.metadata.reset_mock()
        fb = SimpleNamespace(seq_lens=torch.zeros(8, dtype=torch.int32))
        self.backend._init_dsv41_graph_metadata(fm, 8, 1, "cpu", forward_batch=fb)
        self.assertTrue(torch.all(fm.actual_seq_lengths_kv == 1))
        self.assertEqual(self.a3.metadata.call_count, 3)
        for call in self.a3.metadata.call_args_list:
            self.assertTensorEqual(call.kwargs["seqused_ori_kv"], fb.seq_lens)
            if call.kwargs["has_cmp_kv"]:
                self.assertTensorEqual(call.kwargs["seqused_cmp_kv"], fb.seq_lens)

    def test_a3_replay_padding_and_updates_preserve_captured_addresses(self):
        fm = self._init_graph()
        tracked = (
            "c1_loc",
            "c2_loc",
            "positions_cmp_padding_c1",
            "positions_cmp_padding_c2",
            "c1_index_page_table",
            "c2_index_page_table",
        )
        pointers = {name: getattr(fm, name).data_ptr() for name in tracked}
        kernel_pointers = {
            name: value.data_ptr() for name, value in fm.kernel_metadata.items()
        }
        for seq, out_loc in (
            ([129, 130, 128], [256, 385, 511]),
            ([132], [259]),
            ([], []),
        ):
            padded_seq = torch.tensor(seq + [0] * (8 - len(seq)), dtype=torch.int32)
            fb = SimpleNamespace(
                out_cache_loc=torch.tensor(out_loc, dtype=torch.int64),
                seq_lens=padded_seq,
                batch_size=8,
            )
            self.backend._refresh_graph_dsv41_1d(
                SimpleNamespace(fm=fm, bs=8, live_seq_lens=padded_seq, forward_batch=fb)
            )
            self.assertTensorEqual(
                fm.c1_loc,
                torch.tensor(out_loc + [-1] * (8 - len(seq)), dtype=torch.int32),
            )
            c2 = [slot // 2 for length, slot in zip(seq, out_loc) if length % 2 == 0]
            self.assertTensorEqual(
                fm.c2_loc, torch.tensor(c2 + [-1] * (8 - len(c2)), dtype=torch.int32)
            )
            self.assertTensorEqual(
                fm.c1_index_page_table[0], torch.tensor([2, 3, 2, 3], dtype=torch.int32)
            )
            self.backend._dsv41_attention_kernel_metadata(fb)
            self.assertEqual(
                pointers, {name: getattr(fm, name).data_ptr() for name in tracked}
            )
            self.assertEqual(
                kernel_pointers,
                {name: value.data_ptr() for name, value in fm.kernel_metadata.items()},
            )
            for call in self.a3.metadata.call_args_list[-3:]:
                self.assertTensorEqual(call.kwargs["seqused_ori_kv"], padded_seq)

    def _switch_to_a5(self):
        self.ns["is_npu_arch35"] = lambda: True
        self.ns["_ds41_ops"] = lambda: self.a5
        self.ns["_dsv41_a3_ops"] = Mock(side_effect=AssertionError("A5 entered A3 ops"))

    def test_a5_attention_dispatch_remains_mixed_quant(self):
        self._switch_to_a5()
        b = self.backend
        b._dsv41_window_indices = Mock(
            return_value=(
                torch.zeros(4, 1, 128, dtype=torch.int32),
                torch.ones(4, 1, dtype=torch.int32),
            )
        )
        b._dsv41_attention_sinks = Mock(return_value=torch.ones(32))
        with patch.dict(os.environ, {}, clear=True):
            b._forward_low_ratio_attention(self.q, self.layer, self.fb, None, 0)
        kw = self.a5.mixed_quant_sparse_flash_mla.call_args.kwargs
        self.assertEqual(kw["ori_kv"].dtype, torch.uint8)
        self.assertEqual(kw["quant_mode"], 1)
        self.assertIn("ori_topk_length", kw)

    def test_a5_graph_metadata_still_uses_mixed_quant_and_qli(self):
        self._switch_to_a5()
        fm = self._init_graph()
        self.assertEqual(self.a5.mixed_quant_sparse_flash_mla_metadata.call_count, 3)
        self.assertEqual(self.a5.quant_lightning_indexer_metadata.call_count, 2)
        self.assertIn("c1a_qli_metadata", fm.kernel_metadata)
        self.assertEqual(fm.kernel_metadata["win_metadata"].numel(), 1024)

    def test_a5_indexer_still_consumes_packed_values_and_scales(self):
        self._switch_to_a5()
        self.backend._dsv41_index_q.return_value = (
            torch.zeros(4, 32, 64, dtype=torch.uint8),
            torch.ones(4, 32, 2, 2, dtype=torch.uint8),
            self.w,
        )
        self.pool.get_index_k.return_value = torch.zeros(
            5, 64, 1, 64, dtype=torch.uint8
        )
        self.pool.get_index_scale = Mock(
            return_value=torch.ones(5, 64, 1, 4, dtype=torch.uint8)
        )
        self.fm.kernel_metadata["c1a_qli_metadata"] = torch.ones(
            1024, dtype=torch.int32
        )
        self.a5.quant_lightning_indexer.return_value = (
            self.indices,
            None,
            self.blocks,
            self.lengths,
        )
        self.backend._dsv41_index_full_topk(
            self.layer, self.q, self.index_q, torch.arange(4), self.fb
        )
        kw = self.a5.quant_lightning_indexer.call_args.kwargs
        self.assertEqual(kw["q"].dtype, torch.uint8)
        self.assertEqual(kw["descale_k"].shape, (5, 64, 1, 2, 2))
        self.assertEqual(kw["quant_mode"], 1)

    def test_model_only_removes_head_padding_for_a3_v41(self):
        model_ast = ast.parse(MODEL.read_text())
        mqa = next(
            node
            for node in model_ast.body
            if isinstance(node, ast.ClassDef) and node.name == "MqaAttentionBase"
        )
        method = next(
            node
            for node in mqa.body
            if isinstance(node, ast.FunctionDef) and node.name == "_kernel_num_heads"
        )
        module = ast.Module(body=[method], type_ignores=[])
        namespace = {
            "_is_npu": True,
            "get_platform": lambda: SimpleNamespace(is_sm120=False),
        }
        exec(compile(ast.fix_missing_locations(module), str(MODEL), "exec"), namespace)  # noqa: S102
        npu_module = ModuleType("sglang.srt.hardware_backend.npu.utils")
        layer = SimpleNamespace(
            attn_tp_size=2, n_local_heads=32, n_heads=64, is_dsv41=True
        )
        with patch.dict("sys.modules", {npu_module.__name__: npu_module}):
            for arch35, is_v41, expected in (
                (False, True, 32),
                (True, True, 64),
                (False, False, 64),
            ):
                npu_module.is_npu_arch35 = lambda arch35=arch35: arch35
                layer.is_dsv41 = is_v41
                self.assertEqual(namespace["_kernel_num_heads"](layer, 1), expected)


class TestA3OpLoading(unittest.TestCase):
    def setUp(self):
        self.ns = _definitions(
            BACKEND,
            {"_dsv41_a3_op_module", "_dsv41_a3_compressor_kernel"},
            {"functools": functools},
        )
        self.load_module = self.ns["_dsv41_a3_op_module"]

    @staticmethod
    def _provider(name, package="cann_ops_transformer_testvendor"):
        return importlib.metadata.EntryPoint(
            name=name,
            value=f"{package}.ops.attention.{name}:{name}",
            group="cann_ops_transformer.ops",
        )

    def test_loads_base_registration_before_arbitrary_vendor_module(self):
        name = "sparse_lightning_indexer"
        provider = self._provider(name)
        module = ModuleType(provider.module)
        with (
            patch.object(importlib, "import_module", return_value=module) as imported,
            patch.object(
                importlib.metadata, "entry_points", return_value=[provider]
            ) as entries,
        ):
            self.assertIs(self.load_module(name), module)
        entries.assert_called_once_with(group="cann_ops_transformer.ops", name=name)
        self.assertEqual(
            imported.call_args_list,
            [call("cann_ops_transformer.ops"), call(provider.module)],
        )

    def test_full_package_without_vendor_falls_back_to_base_module(self):
        name = "lightning_indexer_v2"
        module = ModuleType(f"cann_ops_transformer.ops.attention.{name}")
        with (
            patch.object(importlib, "import_module", return_value=module) as imported,
            patch.object(importlib.metadata, "entry_points", return_value=[]),
        ):
            self.assertIs(self.load_module(name), module)
        self.assertEqual(
            imported.call_args_list,
            [call("cann_ops_transformer.ops"), call(module.__name__)],
        )

    def test_module_discovery_and_import_are_cached(self):
        name = "sparse_flash_mla"
        provider = self._provider(name)
        module = ModuleType(provider.module)
        with (
            patch.object(importlib, "import_module", return_value=module) as imported,
            patch.object(
                importlib.metadata, "entry_points", return_value=[provider]
            ) as entries,
        ):
            self.assertIs(self.load_module(name), module)
            self.assertIs(self.load_module(name), module)
        self.assertEqual(imported.call_count, 2)
        entries.assert_called_once()

    def test_broken_vendor_import_is_not_hidden_by_base_fallback(self):
        name = "sparse_lightning_indexer"
        provider = self._provider(name)
        failure = ImportError("vendor binding dependency missing")

        def import_module(module_name):
            if module_name == provider.module:
                raise failure
            return ModuleType(module_name)

        with (
            patch.object(
                importlib, "import_module", side_effect=import_module
            ) as imported,
            patch.object(importlib.metadata, "entry_points", return_value=[provider]),
        ):
            with self.assertRaises(ImportError) as raised:
                self.load_module(name)
        self.assertIs(raised.exception, failure)
        self.assertEqual(
            imported.call_args_list,
            [call("cann_ops_transformer.ops"), call(provider.module)],
        )

    def test_multiple_vendor_providers_fail_instead_of_selecting_arbitrarily(self):
        name = "lightning_indexer_v2"
        providers = [
            self._provider(name, package) for package in ("vendor_a", "vendor_b")
        ]
        with (
            patch.object(importlib, "import_module") as imported,
            patch.object(importlib.metadata, "entry_points", return_value=providers),
        ):
            with self.assertRaisesRegex(RuntimeError, name):
                self.load_module(name)
        imported.assert_called_once_with("cann_ops_transformer.ops")

    def test_missing_candidate_overload_is_reported_during_initialization(self):
        ops = SimpleNamespace(
            sparse_flash_mla=Mock(),
            sparse_flash_mla_metadata=Mock(),
            lightning_indexer=SimpleNamespace(default=Mock()),
            sparse_lightning_indexer=Mock(),
        )
        loader = Mock()
        namespace = _definitions(
            BACKEND,
            {"_dsv41_a3_ops"},
            {
                "functools": functools,
                "SimpleNamespace": SimpleNamespace,
                "torch": SimpleNamespace(ops=SimpleNamespace(cann_ops_transformer=ops)),
                "_dsv41_a3_op_module": loader,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "candidate"):
            namespace["_dsv41_a3_ops"]()
        self.assertEqual(
            loader.call_args_list,
            [
                call("lightning_indexer_v2"),
                call("sparse_flash_mla"),
                call("sparse_lightning_indexer"),
            ],
        )

    def test_compressor_uses_selected_vendor_builder_and_caches_loaded_kernel(self):
        package = ModuleType(
            "cann_ops_transformer_othercmp.ops.attention.compressor_v2"
        )
        kernel = Mock()
        builder = SimpleNamespace(
            load=Mock(return_value=SimpleNamespace(compressor=kernel))
        )
        module = ModuleType(f"{package.__name__}.compressor")
        module.compressor_op_builder = builder
        loader = Mock(return_value=package)
        self.ns["_dsv41_a3_op_module"] = loader
        with patch.object(importlib, "import_module", return_value=module) as imported:
            self.assertIs(self.ns["_dsv41_a3_compressor_kernel"](), kernel)
            self.assertIs(self.ns["_dsv41_a3_compressor_kernel"](), kernel)
        loader.assert_called_once_with("compressor_v2")
        imported.assert_called_once_with(module.__name__)
        builder.load.assert_called_once()


class TestA3CompressorContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = _definitions(
            BACKEND,
            {"_dsv41_a3_compressor", "_dsv41_a3_compressor_fake"},
            {"torch": torch, "__name__": __name__},
            operator_namespace="sglang_test_a3",
        )
        cls.namespace = namespace
        cls.op = namespace["_dsv41_a3_compressor"]

    @staticmethod
    def _inputs(device="cpu"):
        return {
            "x": torch.zeros(5, 16, dtype=torch.bfloat16, device=device),
            "wkv": torch.zeros(512, 16, dtype=torch.bfloat16, device=device),
            "wgate": torch.zeros(512, 16, dtype=torch.bfloat16, device=device),
            "state_cache": torch.zeros(3, 2, 1024, dtype=torch.float32, device=device),
            "state_block_table": torch.tensor([0, 1], dtype=torch.int32, device=device),
            "cu_seqlens": torch.tensor([0, 2, 5], dtype=torch.int32, device=device),
            "seqused": torch.tensor([2, 3], dtype=torch.int32, device=device),
            "start_pos": torch.zeros(2, dtype=torch.int32, device=device),
            "cmp_ratio": 2,
        }

    def test_schema_marks_only_state_as_mutable(self):
        args = self.op._opoverload._schema.arguments
        self.assertEqual(
            [
                arg.name
                for arg in args
                if arg.alias_info is not None and arg.alias_info.is_write
            ],
            ["state_cache"],
        )

    def test_wrapper_forwards_existing_binding_and_exposes_state_update(self):
        inputs = self._inputs()
        calls = []

        def compressor(x, wkv, wgate, state, ratio, block_table, cu, used, start):
            calls.append((x, wkv, wgate, state, ratio, block_table, cu, used, start))
            state.add_(1)
            return x.new_zeros((4, wkv.shape[0]))

        with patch.dict(
            self.namespace, {"_dsv41_a3_compressor_kernel": lambda: compressor}
        ):
            result = self.op(**inputs)
        self.assertEqual(result.shape, (4, 512))
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertTrue(torch.all(inputs["state_cache"] == 1))
        self.assertIs(calls[0][3], inputs["state_cache"])
        self.assertEqual(calls[0][4], 2)
        self.assertIs(calls[0][5], inputs["state_block_table"])

    def test_meta_registration_has_fixed_capacity_and_no_storage_alias(self):
        inputs = self._inputs(device="meta")
        out = self.op(**inputs)
        self.assertEqual(out.shape, (4, 512))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.device.type, "meta")


if __name__ == "__main__":
    unittest.main()
