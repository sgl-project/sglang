"""CPU contract checks for A3 DSpark target verification.

Execute production metadata methods with CPU tensors and mock only imports of
the device kernels. These tests do not replace on-device graph/accuracy tests.
"""

import ast
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

ROOT = Path(__file__).resolve().parents[5]
BACKEND = (
    ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py"
)


def load_class(path, name, namespace):
    node = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def mode(verify=True):
    return SimpleNamespace(
        is_target_verify=lambda: verify,
        is_decode=lambda: not verify,
        is_draft_extend_v2=lambda: False,
        is_idle=lambda: False,
        is_extend_without_speculative=lambda: False,
    )


class TestA3DsparkMetadata(unittest.TestCase):
    def setUp(self):
        self.metadata_op = Mock(return_value=torch.ones(2048, dtype=torch.int32))
        self.namespace = dict(
            torch=torch,
            F=F,
            SimpleNamespace=SimpleNamespace,
            is_npu_arch35=lambda: False,
            _dsv41_a3_ops=lambda: SimpleNamespace(metadata=self.metadata_op),
        )
        cls = load_class(BACKEND, "LowRatioCompressAscendBackendMixin", self.namespace)
        backend_class = next(
            node
            for node in ast.parse(BACKEND.read_text()).body
            if isinstance(node, ast.ClassDef)
            and node.name == "DeepseekV4AscendAttnBackend"
        )
        refresh = next(
            node
            for node in backend_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_refresh_graph_verify_rows"
        )
        exec(
            compile(ast.Module(body=[refresh], type_ignores=[]), str(BACKEND), "exec"),
            self.namespace,
        )
        cls._refresh_graph_verify_rows = self.namespace[refresh.name]
        self.backend = cls()
        self.backend.page_size = 128
        self.backend._dsv4_unique_compress_ratios = (0, 1, 2)
        self.backend._dsv4_has_low_ratio = True
        self.backend._dsv4_q_head_num = 32
        self.backend._dsv4_head_dim = 512
        self.backend._dsv4_index_topk = 512
        self.backend._dsv4_sliding_window_size = 128
        self.backend._dsv4_graph_tokens_per_req = 5
        self.backend.graph_metadata = {}

        metadata_module = ModuleType("sglang.srt.layers.attention.dsv4.metadata")
        metadata_module.expand_index_page_table = (
            lambda table, *, full_page_size, compress_ratio, index_page_size: (
                table[:, :, None]
                * (full_page_size // compress_ratio // index_page_size)
                + torch.arange(full_page_size // compress_ratio // index_page_size)
            ).reshape(table.shape[0], -1)
        )
        pool_module = ModuleType("sglang.srt.mem_cache.deepseek_v4_memory_pool")
        pool_module.DSV41_INDEX_PAGE_SIZE = 64
        imports = patch.dict(
            "sys.modules",
            {
                metadata_module.__name__: metadata_module,
                pool_module.__name__: pool_module,
            },
        )
        imports.start()
        self.addCleanup(imports.stop)

    def make_batch(self, prefixes, queries, slots, *, raw_bs=None, ragged=False):
        bs = len(prefixes)
        lengths = torch.tensor(queries, dtype=torch.int32)
        positions = torch.cat(
            [
                torch.arange(prefix, prefix + count)
                for prefix, count in zip(prefixes, queries)
            ]
        )
        fb = SimpleNamespace(
            batch_size=bs,
            forward_mode=mode(),
            positions=positions,
            out_cache_loc=torch.tensor(slots, dtype=torch.int64),
            seq_lens=torch.tensor(prefixes, dtype=torch.int32),
            spec_info=SimpleNamespace(draft_token_num=max(queries)),
            ragged_verify_layout=(
                SimpleNamespace(graph_num_tokens=positions.numel(), verify_lens=lengths)
                if ragged
                else None
            ),
        )
        fm = SimpleNamespace(
            actual_seq_lengths_q_pa=torch.cat(
                (torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int())
            ),
            actual_seq_lengths_kv=fb.seq_lens + lengths,
            start_pos=fb.seq_lens.clone(),
            seqused=lengths.clone(),
            block_tables=torch.tensor(
                [[2 + row, 9 + row] for row in range(bs)], dtype=torch.int32
            ),
            kernel_metadata={},
        )
        self.backend.forward_metadata = fm
        for ratio in (1, 2):
            self.backend.graph_metadata[f"dsv41_index_page_table_c{ratio}"] = (
                torch.zeros(bs, 4 // ratio, dtype=torch.int32)
            )
        ctx = SimpleNamespace(
            fm=fm,
            forward_batch=fb,
            live_seq_lens=fb.seq_lens,
            bs=bs,
            raw_bs=bs if raw_bs is None else raw_bs,
            tokens_per_bs=max(queries),
            ragged_layout=fb.ragged_verify_layout,
            verify_lens=lengths if ragged else None,
            device="cpu",
            is_idle_replay=False,
            has_compress=False,
        )
        return fb, fm, ctx

    def test_eager_verify_compresses_cross_boundary_groups_in_request_order(self):
        fb, fm, _ = self.make_batch(
            [128, 129], [5, 5], list(range(256, 261)) + list(range(385, 390))
        )
        self.backend._build_low_ratio_compress_metadata(fb)
        self.assertEqual(fm.c1_loc.tolist(), fb.out_cache_loc.tolist())
        self.assertEqual(fm.c2_loc.tolist(), [128, 129, 192, 193, 194, -1, -1])
        self.assertEqual(
            fm.positions_cmp_padding_c2.tolist(), [128, 130, 128, 130, 132, 0, 0]
        )
        self.assertEqual(fm.c2_index_page_table.tolist(), fm.block_tables.tolist())

    def test_attention_and_indexer_use_final_kv_lengths_not_committed_lengths(self):
        fb, fm, _ = self.make_batch([128, 129], [5, 5], list(range(256, 266)))
        self.backend._build_low_ratio_compress_metadata(fb)
        kw = self.backend._dsv41_a3_attention_kwargs(fb, 2)
        self.assertEqual(kw["seqused_ori_kv"].tolist(), [133, 134])
        self.assertEqual(kw["seqused_cmp_kv"].tolist(), [66, 67])
        self.assertEqual(kw["cmp_residual_kv"].tolist(), [1, 0])
        self.assertEqual(kw["ori_mask_mode"], 4)
        self.assertEqual(kw["cmp_mask_mode"], 3)
        index_kw = self.backend._dsv41_a3_index_kwargs(fb, 2)
        self.assertEqual(index_kw["seqused_k"].tolist(), [66, 67])
        self.assertEqual(index_kw["cmp_residual_k"].tolist(), [1, 0])
        self.assertEqual(fb.seq_lens.tolist(), [128, 129])

    def test_a5_eager_verify_keeps_upstream_row_support(self):
        fb, fm, _ = self.make_batch([128], [5], list(range(256, 261)))
        self.namespace["is_npu_arch35"] = lambda: True
        self.backend._build_low_ratio_compress_metadata(fb)
        self.assertEqual(fm.c1_loc.tolist(), list(range(256, 261)))
        self.assertEqual(fm.c2_loc.tolist(), [128, 129, -1])

    def test_tp8_verify_width6_trims_all_operator_inputs_and_restores_output(self):
        fb, fm, _ = self.make_batch([128], [6], list(range(256, 262)) + [999, 999])
        fb.positions = F.pad(fb.positions, (0, 2))
        fb.encoder_swa_replay = False
        self.backend._build_low_ratio_compress_metadata(fb)
        self.assertEqual(fm.c1_loc.tolist(), list(range(256, 262)))
        self.assertEqual(fm.c2_loc.tolist(), [128, 129, 130, -1])
        self.assertEqual(fm.positions_cmp_padding_c1.shape, (6,))
        self.assertEqual(fm.positions_cmp_padding_c2.shape, (4,))
        x, q_lora = torch.randn(8, 8), torch.randn(8, 4)
        layer = SimpleNamespace(layer_id=0, compressor=object(), indexer=object())
        compressor = Mock(side_effect=lambda layer, x, pos, fb: x.clone())
        indexer = Mock()
        self.backend._low_ratio_compress = compressor
        self.backend._low_ratio_index_topk = indexer
        self.backend.forward_low_ratio_sources(
            layer=layer, x=x, q_lora=q_lora, positions=fb.positions, forward_batch=fb
        )
        self.assertEqual(compressor.call_args.args[1].shape[0], 6)
        self.assertEqual(compressor.call_args.args[2].shape[0], 6)
        self.assertEqual([a.shape[0] for a in indexer.call_args.args[1:5]], [6] * 4)
        self.assertEqual(fb.positions.shape[0], 8)

        # Compressor-only overlap preparation deliberately has no Q projection.
        self.backend.forward_low_ratio_sources(
            layer=layer,
            x=x,
            q_lora=None,
            positions=fb.positions,
            forward_batch=fb,
            run_indexer=False,
        )
        self.assertEqual(compressor.call_args.args[1].shape[0], 6)
        public = Mock(side_effect=lambda q, **kw: (q + 1, None))
        self.namespace["_dsv41_a3_ops"] = lambda: SimpleNamespace(attention=public)
        self.backend.token_to_kv_pool = SimpleNamespace(
            get_swa_buffer=lambda layer_id: torch.empty(1),
            get_compress_buffer=lambda layer_id, index: torch.empty(1),
        )
        fm.swa_page_table = fm.block_tables
        fm.kernel_metadata["c2a_metadata"] = torch.ones(2048, dtype=torch.int32)
        fm.c2_topk_indices = torch.zeros(6, 512, dtype=torch.int32)
        q = torch.randn(8, 8, 512)
        output = self.backend._forward_low_ratio_attention(
            q, SimpleNamespace(layer_id=0, scaling=0.125), fb, torch.zeros(8), 2
        )
        args, kwargs = public.call_args
        self.assertEqual(args[0].shape[0], 6)
        self.assertEqual(kwargs["cmp_sparse_indices"].shape[0], 6)
        self.assertEqual(kwargs["cu_seqlens_q"][-1].item(), 6)
        self.assertTrue(torch.equal(output[:6], q[:6] + 1))
        self.assertEqual(output.shape, q.shape)
        self.assertTrue(torch.all(output[6:] == 0))

    def test_tp8_verify_width6_graph_metadata_ignores_tp_tail_and_keeps_addresses(self):
        fb, fm, ctx = self.make_batch([128], [6], list(range(256, 262)) + [999, 999])
        fb.positions = F.pad(fb.positions, (0, 2))
        self.backend._dsv4_graph_tokens_per_req = 6
        self.backend._init_dsv41_graph_metadata(fm, 1, 6, "cpu", forward_batch=fb)
        fields = (
            "c1_loc",
            "c2_loc",
            "positions_cmp_padding_c1",
            "positions_cmp_padding_c2",
        )
        pointers = {field: getattr(fm, field).data_ptr() for field in fields}
        self.backend._refresh_graph_verify_rows(ctx)
        self.assertEqual(fm.c1_loc.tolist(), list(range(256, 262)))
        self.assertEqual(fm.c2_loc.tolist(), [128, 129, 130, -1])
        # An idle replay may have no runtime spec_info. Capture geometry remains.
        fb.spec_info = None
        ctx.is_idle_replay = True
        self.backend._refresh_graph_verify_rows(ctx)
        self.assertTrue(torch.all(fm.c1_loc == -1))
        self.assertTrue(torch.all(fm.c2_loc == -1))
        self.assertEqual(
            pointers, {field: getattr(fm, field).data_ptr() for field in fields}
        )

    def test_a5_verify_source_tensors_keep_tp_padding(self):
        fb, _, _ = self.make_batch([128], [6], list(range(256, 264)))
        fb.positions = F.pad(fb.positions, (0, 2))
        fb.encoder_swa_replay = False
        self.namespace["is_npu_arch35"] = lambda: True
        self.backend._low_ratio_compress = Mock(return_value=torch.zeros(8, 1))
        self.backend._low_ratio_index_topk = Mock()
        x, q_lora = torch.randn(8, 8), torch.randn(8, 4)
        self.backend.forward_low_ratio_sources(
            layer=SimpleNamespace(layer_id=0, compressor=object(), indexer=object()),
            x=x,
            q_lora=q_lora,
            positions=fb.positions,
            forward_batch=fb,
        )
        self.assertIs(self.backend._low_ratio_compress.call_args.args[1], x)
        self.assertIs(self.backend._low_ratio_index_topk.call_args.args[3], q_lora)

    def test_actual_idle_replay_retains_captured_verify_geometry(self):
        # build_replay_fb_view keeps the capture mode/bucket on forward_batch,
        # and carries the original scheduler mode in actual_forward_mode.
        fb, fm, _ = self.make_batch([128, 129], [6, 6], [])
        fb.positions = F.pad(fb.positions, (0, 4))
        fb.actual_forward_mode = SimpleNamespace(is_idle=lambda: True)
        fb.num_padding = fb.batch_size
        fb.seq_lens_cpu = fb.seq_lens.clone()
        fb.spec_info = None
        self.backend._dsv4_graph_tokens_per_req = 6
        self.backend._dsv4_has_c4 = self.backend._dsv4_has_c128 = False
        self.backend._dspark_ragged_verify_layout = lambda _: None
        node = next(
            node
            for node in ast.parse(BACKEND.read_text()).body
            if isinstance(node, ast.ClassDef)
            and node.name == "DeepseekV4AscendAttnBackend"
        )
        method = next(
            method
            for method in node.body
            if isinstance(method, ast.FunctionDef)
            and method.name == "_build_dsv4_graph_replay_ctx"
        )
        namespace = dict(
            torch=torch, SimpleNamespace=SimpleNamespace, ForwardBatch=object
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                str(BACKEND),
                "exec",
            ),
            namespace,
        )
        ctx = namespace[method.name](self.backend, fb)
        self.assertTrue(ctx.graph_mode.is_target_verify())
        self.assertTrue(ctx.is_idle_replay)
        self.assertEqual((ctx.bs, ctx.raw_bs, ctx.tokens_per_bs), (2, 0, 6))
        self.backend._init_dsv41_graph_metadata(fm, 2, 6, "cpu", forward_batch=fb)
        pointers = (fm.c1_loc.data_ptr(), fm.c2_loc.data_ptr())
        self.backend._refresh_graph_verify_rows(ctx)
        self.assertEqual(fm.c1_loc.shape, (12,))
        self.assertEqual(fm.c2_loc.shape, (8,))
        self.assertTrue(torch.all(fm.c1_loc == -1))
        self.assertTrue(torch.all(fm.c2_loc == -1))
        self.assertEqual(pointers, (fm.c1_loc.data_ptr(), fm.c2_loc.data_ptr()))

    def test_graph_uniform_verify_padding_and_idle_preserve_buffers(self):
        fb, fm, ctx = self.make_batch(
            [128, 129, 0],
            [5, 5, 5],
            list(range(256, 261)) + list(range(385, 390)),
            raw_bs=2,
        )
        self.backend._init_dsv41_graph_metadata(fm, 3, 5, "cpu", forward_batch=fb)
        fields = (
            "c1_loc",
            "c2_loc",
            "positions_cmp_padding_c1",
            "positions_cmp_padding_c2",
            "c1_index_page_table",
            "c2_index_page_table",
        )
        pointers = {field: getattr(fm, field).data_ptr() for field in fields}
        self.backend._refresh_graph_verify_rows(ctx)
        self.assertEqual(fm.seqused.tolist(), [5, 5, 0])
        self.assertEqual(fm.start_pos.tolist(), [128, 129, 0])
        self.assertEqual(fm.c1_loc.tolist()[-5:], [-1] * 5)
        self.assertEqual(fm.c2_loc.tolist(), [128, 129, 192, 193, 194] + [-1] * 5)
        ctx.is_idle_replay = True
        self.backend._refresh_graph_verify_rows(ctx)
        self.assertEqual(fm.seqused.tolist(), [0, 0, 0])
        for ratio in (1, 2):
            self.assertTrue(torch.all(getattr(fm, f"c{ratio}_loc") == -1))
            self.assertTrue(
                torch.all(getattr(fm, f"positions_cmp_padding_c{ratio}") == 0)
            )
        self.assertEqual(
            pointers, {field: getattr(fm, field).data_ptr() for field in fields}
        )

    def test_compact_guard_is_shared_and_static_modes_remain_supported(self):
        guard = next(
            node
            for node in ast.parse(BACKEND.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_ensure_dsv41_static_verify_mode"
        )
        module = ast.Module(body=[guard], type_ignores=[])
        code = compile(ast.fix_missing_locations(module), str(BACKEND), "exec")
        value = Mock(return_value="compact")
        self.namespace["read_ragged_verify_mode"] = value
        self.namespace["RaggedVerifyMode"] = SimpleNamespace(COMPACT="compact")
        exec(code, self.namespace)
        for arch35 in (False, True):
            self.namespace["is_npu_arch35"] = lambda: arch35
            value.return_value = "compact"
            with self.assertRaisesRegex(RuntimeError, "STATIC request-major"):
                self.namespace[guard.name]()
            for accepted in ("static", "cap-accept"):
                value.return_value = accepted
                self.namespace[guard.name]()


if __name__ == "__main__":
    unittest.main()
