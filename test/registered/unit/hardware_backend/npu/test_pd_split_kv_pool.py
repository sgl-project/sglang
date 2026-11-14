import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


def _make_pool():
    pool = NPUMLATokenToKVPool.__new__(NPUMLATokenToKVPool)
    pool.start_layer = 0
    pool.layer_num = 2
    pool.device = "cpu"
    pool.page_size = 128
    pool.cpu_offloading_chunk_size = 2
    pool.dcp_size = 1
    pool.is_draft_worker = False
    pool.use_combined_kv_buffer = False
    pool.dsa_kv_cache_store_fp8 = False
    pool.index_head_dim = 2
    pool.indexer_layer_id_to_slot = {0: 0, 1: 1}
    pool.k_buffer = torch.zeros(2, 2, 128, 1, 512, dtype=torch.bfloat16)
    pool.v_buffer = torch.zeros(2, 2, 128, 1, 64, dtype=torch.bfloat16)
    pool.index_k_buffer = torch.zeros(2, 2, 128, 1, 2, dtype=torch.bfloat16)
    pool.index_k_scale_buffer = None
    return pool


class TestPDSplitKVPool(CustomTestCase):
    def test_pool_allocation_follows_fused_role(self):
        from sglang.srt.environ import envs

        prefix = "sglang.srt.hardware_backend.npu.memory_pool_npu."
        for fused, mode, draft, native in (
            (True, "decode", False, False),
            (True, "null", False, False),
            (True, "prefill", False, True),
            (True, "decode", True, True),
            (False, "prefill", False, False),
            (False, "decode", True, False),
        ):
            with (
                self.subTest(fused=fused, mode=mode, draft=draft),
                envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
                envs.SGLANG_NPU_ENABLE_FUSED_LIM_COPYSFA.override(fused),
                get_context().override_server_args(disaggregation_mode=mode),
                get_parallel().override(attn_dcp_rank=0),
                patch(prefix + "get_bool_env_var", return_value=False),
                patch.object(NPUMLATokenToKVPool, "_finalize_allocation_log"),
            ):
                pool = NPUMLATokenToKVPool(
                    size=128,
                    page_size=128,
                    dtype=torch.bfloat16,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    layer_num=1,
                    device="cpu",
                    enable_memory_saver=False,
                    index_head_dim=128,
                    is_draft_worker=draft,
                )
                self.assertEqual(pool.k_buffer is not None, native)
                self.assertEqual(pool.v_buffer is not None, native)

    def test_native_combined_export_uses_upstream_layout(self):
        pool = _make_pool()
        pool.use_combined_kv_buffer = True
        pool.kv_buffer = torch.zeros(2, 2, 128, 1, 576, dtype=torch.bfloat16)

        ptrs, lengths, page_bytes = pool.get_contiguous_buf_infos()

        self.assertEqual(ptrs, [buffer.data_ptr() for buffer in pool.kv_buffer])
        self.assertEqual(lengths, [buffer.nbytes for buffer in pool.kv_buffer])
        self.assertEqual(page_bytes, [147456, 147456])

    def test_export_matches_prefill_page_order(self):
        prefill, decode = _make_pool(), _make_pool()
        k, v = list(decode.k_buffer), list(decode.v_buffer)
        decode.sparse_kv_manager = SimpleNamespace(
            host_pool=SimpleNamespace(buffers=(k, v), supports_pd_export=True)
        )
        decode.k_buffer = decode.v_buffer = None

        ptrs, lens, page_bytes = decode.get_contiguous_buf_infos()
        self.assertEqual(
            ptrs, [t.data_ptr() for t in k + v + list(decode.index_k_buffer)]
        )
        self.assertEqual((lens, page_bytes), prefill.get_contiguous_buf_infos()[1:])
        self.assertEqual(page_bytes, [131072] * 2 + [16384] * 2 + [512] * 2)

    def test_colocated_shm_is_not_exported_as_pd_memory(self):
        pool = _make_pool()
        pool.sparse_kv_manager = SimpleNamespace(
            host_pool=SimpleNamespace(supports_pd_export=False, buffers=([], []))
        )
        expected = (
            [t.data_ptr() for t in pool.k_buffer]
            + [t.data_ptr() for t in pool.v_buffer]
            + [t.data_ptr() for t in pool.index_k_buffer]
        )
        self.assertEqual(pool.get_contiguous_buf_infos()[0], expected)

    def test_shm_split_copy_rows_and_attention_pointers(self):
        # Run the real manager/pool methods with only the NPU allocation and
        # row-copy boundaries emulated. The fake device pointer is deliberately
        # different from the CPU address so accidental address substitution fails.
        path = Path(__file__).resolve().parents[5] / (
            "python/sglang/srt/hardware_backend/npu/"
            "sparsity_driven_kv_offload/fused_manager.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        shm = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "ShmSplitKVPool"
        )
        manager = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "FusedSparseKVCacheManager"
        )
        attention = next(
            n
            for n in manager.body
            if isinstance(n, ast.FunctionDef) and n.name == "copy_sfa_attention"
        )
        pointers = {}
        observed = {}

        def copy_sfa(*args, **kwargs):
            observed["ckv"] = kwargs["dram_kv_dev_ptr"]
            observed["kpe"] = kwargs["dram_rope_dev_ptr"]

        def allocate(*, shape, dtype, device_id, name):
            tensor = torch.zeros(shape, dtype=dtype)
            ptr = 100000 + len(pointers) * 4096
            pointers[ptr] = tensor.data_ptr()
            return tensor, tensor.data_ptr(), ptr

        def copy(src, dst, si, di, valid, sd, dd, **kw):
            for tensor, key in ((src, "src_ptr"), (dst, "dst_ptr")):
                base = tensor.untyped_storage().data_ptr()
                matched = [device for device, cpu in pointers.items() if cpu == base]
                if matched:
                    self.assertEqual(kw[key], matched[0] + tensor.data_ptr() - base)
                else:
                    self.assertNotIn(key, kw)
            self.assertEqual(sd, src.ndim - 2)
            self.assertEqual(dd, dst.ndim - 2)
            src = src.reshape(-1, *src.shape[-2:])
            dst = dst.reshape(-1, *dst.shape[-2:])
            dst[di[valid]] = src[si[valid]]

        ns = {
            "torch": torch,
            "create_shm_tensor": allocate,
            "unidex_copy_inplace": copy,
            "fused_scatter_copy_sparse_flash_attention": copy_sfa,
        }
        code = ast.Module(
            body=[
                shm,
                ast.ClassDef(
                    name="Manager",
                    bases=[],
                    keywords=[],
                    body=[attention],
                    decorator_list=[],
                ),
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(code), str(path), "exec"), ns)
        host = ns["ShmSplitKVPool"](
            shape=(2, 128, 1, 576),
            widths=(512, 64),
            layers=2,
            dtype=torch.bfloat16,
            device_id=0,
            name="test",
        )
        self.assertEqual(
            [x[0].shape for x in host.buffers], [(2, 128, 1, 512), (2, 128, 1, 64)]
        )
        m = ns["Manager"]()
        m.kv_lora_rank, m.qk_rope_head_dim = 512, 64
        m.head_num, m.start_layer = 1, 0
        m.dev_ptr_list, m.rope_dev_ptr_list = host.dev_ptrs
        m.fused_attn_out = torch.zeros(1, 1, 512)
        m.device_kv_buffer = [torch.zeros(128, 1, 512)]
        m.device_rope_buffer = [torch.zeros(128, 1, 64)]
        m.lim_adapter = SimpleNamespace(
            prepare_copy_sfa_metadata=lambda *_: None,
            _pre_cache_sizes=torch.zeros(1, dtype=torch.int32),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
        )
        m.planner_query_ends = torch.ones(1, dtype=torch.int32)
        for name in (
            "planner_actual_seq_lengths_kv",
            "planner_hbm_block_table",
            "resident_begin",
            "resident_end",
        ):
            setattr(m, name, torch.zeros(1, 1, dtype=torch.long))
        m.req_to_token_pool = SimpleNamespace(req_to_token=torch.zeros(1, 1))
        m.host_token_capacity, m._copy_sfa_operator = 256, object()
        plan = SimpleNamespace(
            request_safe_slots=torch.zeros(1, dtype=torch.long),
            query_width=1,
            source_ids=torch.zeros(1),
            device_slots=torch.zeros(1),
            topk_misses=None,
            miss_sources=None,
            miss_slots=None,
            miss_counts=None,
            valid_rows=torch.ones(1, dtype=torch.bool),
        )
        m.copy_sfa_attention(
            SimpleNamespace(layer_id=0, scaling=1.0),
            torch.zeros(1, 1, 512),
            torch.zeros(1, 1, 64),
            plan,
            (None, None, None),
        )
        self.assertEqual(
            observed, {"ckv": host.dev_ptrs[0][0], "kpe": host.dev_ptrs[1][0]}
        )
        payload = [
            torch.full((2, 1, width), 17, dtype=torch.bfloat16) for width in (512, 64)
        ]
        host.copy_rows(
            payload,
            [c[0] for c in host.buffers],
            torch.tensor([0, 1]),
            torch.tensor([3, 4]),
            torch.tensor([True, False]),
        )
        for component in host.buffers:
            self.assertTrue(torch.all(component[0].flatten(0, 1)[3] == 17))
            self.assertTrue(torch.all(component[0].flatten(0, 1)[4] == 0))
        offset = [component[0][1] for component in host.buffers]
        host.copy_rows(
            payload,
            offset,
            torch.tensor([0, 1]),
            torch.tensor([3, 4]),
            torch.tensor([True, False]),
        )
        for component in host.buffers:
            self.assertTrue(torch.all(component[0][1, 3] == 17))


if __name__ == "__main__":
    unittest.main()
