import ast
import contextlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PATH = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/block_tables.py"
)
spec = importlib.util.spec_from_file_location("prefill_test_block_tables", PATH)
module = importlib.util.module_from_spec(spec)

sys.modules[spec.name] = module
spec.loader.exec_module(module)


class TestPhysicalPrefillPlan(unittest.TestCase):
    def check_case(
        self,
        request_map,
        req,
        lengths,
        prefixes,
        extends,
        locations,
        values,
        valid,
        dtype,
    ):
        class Manager:
            size = 8
            max_context_len = 8
            device = torch.device("cpu")

            def host_source_indices(self, rows, positions):
                return request_map[rows, positions]

        batch = SimpleNamespace(
            req_pool_indices=torch.tensor(req),
            req_pool_indices_cpu=torch.tensor(req),
            seq_lens=torch.tensor(lengths),
            seq_lens_cpu=torch.tensor(lengths),
            extend_seq_lens_cpu=extends,
            extend_prefix_lens_cpu=prefixes,
            out_cache_loc=torch.tensor(locations),
        )
        host = torch.arange(32, dtype=torch.float32).to(dtype)
        payload = torch.tensor(values, dtype=dtype)
        for loc, value, is_valid in zip(locations, payload, valid):
            if is_valid and 0 <= loc < len(host):
                host[loc] = value
        reference = torch.cat(
            [
                host[request_map[r, :length]]
                for r, length in zip(req, lengths)
                if r >= 0 and length > 0
            ]
        )
        plan = module.make_fused_prefill_plan(batch, Manager(), "stream")
        mapping = module.fused_prefill_current_mapping(
            plan, batch, torch.tensor(valid), len(host)
        )
        if mapping is None:
            result = host[plan.source].clone()
        else:
            src, dst, mask, host_mask = mapping
            result = torch.empty_like(reference)
            result[host_mask] = host[plan.source[host_mask]]
            result[dst[mask]] = payload[src[mask]]
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        return mapping

    def test_physical_aliases_and_invalid_rows(self):
        table = torch.zeros((8, 8), dtype=torch.long)
        table[1, :4] = torch.tensor([4, 7, 10, 11])
        table[2, :4] = torch.tensor([4, 9, 11, 13])
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                # Prefix aliases another request's current physical row 11.
                mapping = self.check_case(
                    table,
                    [1, 2],
                    [4, 4],
                    [2, 2],
                    [2, 2],
                    [10, 11, 11, 13],
                    [100, 101, 102, 103],
                    [True] * 4,
                    dtype,
                )
                self.assertEqual(mapping[2].tolist(), [True, False, False, True])
                # A mismatched mapping and invalid row must retain Host gather.
                mapping = self.check_case(
                    table, [1], [4], [2], [2], [12, -1], [120, 0], [True, False], dtype
                )
                self.assertEqual(mapping[2].tolist(), [False, False])
                # Padded current rows use fixed shape without a dynamic boolean index.
                mapping = self.check_case(
                    table,
                    [1, 2],
                    [3, 4],
                    [2, 2],
                    [1, 2],
                    [10, -1, 11, 13],
                    [11, 0, 12, 13],
                    [True, False, True, True],
                    dtype,
                )
                self.assertEqual(mapping[2].tolist(), [True, False, True, True])

    def test_zero_prefix_full_current_skips_all_host_rows(self):
        table = torch.zeros((8, 8), dtype=torch.long)
        table[1, :2] = torch.tensor([7, 3])
        table[2, :2] = torch.tensor([11, 5])
        mapping = self.check_case(
            table,
            [1, 2],
            [2, 2],
            [0, 0],
            [2, 2],
            [7, 3, 11, 5],
            [13, 17, 19, 23],
            [True] * 4,
            torch.bfloat16,
        )
        self.assertEqual(int(mapping[2].sum()), 4)
        self.assertEqual(int(mapping[3].sum()), 0)
        print(
            "zero-prefix: D2D valid rows=4; Host valid copy rows=0; Host kernel still submitted"
        )

    def test_duplicate_request_falls_back(self):
        table = torch.arange(64).reshape(8, 8) % 32
        mapping = self.check_case(
            table,
            [1, 1],
            [2, 2],
            [1, 1],
            [1, 1],
            [9, 9],
            [20, 30],
            [True, True],
            torch.float32,
        )
        self.assertIsNone(mapping)


class TestWorkspaceLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manager_path = PATH.parent / "fused_manager.py"
        tree = ast.parse(manager_path.read_text(encoding="utf-8"))
        original = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "FusedSparseKVCacheManager"
        )
        methods = [
            node
            for node in original.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("prepare_forward_kv_plan", "get_forward_kv")
        ]
        cls.copy_calls = []

        def copy(source, dest, src, dst, mask, *args, **kwargs):
            cls.copy_calls.append((src.clone(), dst.clone(), mask.clone()))
            source = source.reshape(-1, 1, 576)
            dest[dst[mask]] = source[src[mask]]

        namespace = {
            "torch": torch,
            "Optional": object,
            "Union": object,
            "RadixAttention": object,
            "ForwardBatch": object,
            "prepare_prefill_gather_rows": module.prepare_prefill_gather_rows,
            "make_fused_prefill_plan": module.make_fused_prefill_plan,
            "fused_prefill_current_mapping": module.fused_prefill_current_mapping,
            "unidex_copy_inplace": copy,
        }
        code = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                ast.ClassDef(
                    name="Manager",
                    bases=[],
                    keywords=[],
                    body=methods,
                    decorator_list=[],
                ),
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(code)
        exec(compile(code, str(manager_path), "exec"), namespace)
        cls.manager_type = namespace["Manager"]

    def test_reuse_capacity_and_other_stream_fallback(self):
        old_npu = getattr(torch, "npu", None)
        torch.npu = SimpleNamespace(
            stream=lambda _: contextlib.nullcontext(), current_stream=lambda: "A"
        )
        self.addCleanup(
            lambda: (
                setattr(torch, "npu", old_npu)
                if old_npu is not None
                else delattr(torch, "npu")
            )
        )
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            m = self.manager_type()
            m.host_pool = None
            m.budget = SimpleNamespace(prefill_workspace_tokens=8)
            m._prefill_workspace_stream = None
            m.size, m.max_context_len, m.start_layer, m.layer_num = 4, 4, 0, 2
            m.device, m.head_num, m.kv_lora_rank, m.qk_rope_head_dim = (
                torch.device("cpu"),
                1,
                512,
                64,
            )
            m.head_dim, m.store_dtype, m.host_token_capacity = 576, dtype, 16
            m.prefill_packed = torch.empty((8, 1, 576), dtype=dtype)
            m.prefill_nope = torch.empty((8, 1, 512), dtype=dtype)
            m.prefill_rope = torch.empty((8, 1, 64), dtype=dtype)
            table = torch.tensor(
                [[0, 1, 2, 3], [9, 3, 12, 5], [3, 7, 2, 8], [4, 6, 10, 11]]
            )
            m.host_source_indices = lambda rows, pos: table[rows, pos]
            host = (
                torch.arange(16 * 576, dtype=torch.float32)
                .reshape(1, 16, 1, 576)
                .to(dtype)
            )
            m.host_kv_buffer, m.dev_ptr_list = [host, host + 1], [0, 0]

            def batch(ids, lens):
                ids, lens = torch.tensor(ids), torch.tensor(lens)
                return SimpleNamespace(
                    req_pool_indices_cpu=ids,
                    seq_lens_cpu=lens,
                    req_pool_indices=ids,
                    seq_lens=lens,
                )

            first, second = batch([1, 2], [3, 2]), batch([2, 1], [2, 3])
            self.assertIsNone(m.prepare_forward_kv_plan(first, "B"))
            first_plan = m.prepare_forward_kv_plan(first, None)
            self.assertEqual(first_plan.stream, "A")
            a, b = m.get_forward_kv(0, first, "A", plan=first_plan)
            saved = torch.cat((a.clone(), b.clone()), -1)
            reference = torch.cat([host[0, table[r, :n]] for r, n in ([1, 3], [2, 2])])
            torch.testing.assert_close(saved, reference, rtol=0, atol=0)
            second_plan = m.prepare_forward_kv_plan(second, "A")
            c, d = m.get_forward_kv(1, second, "A", plan=second_plan)
            self.assertEqual(a.data_ptr(), c.data_ptr())
            reference2 = torch.cat(
                [(host + 1)[0, table[r, :n]] for r, n in ([2, 2], [1, 3])]
            )
            torch.testing.assert_close(
                torch.cat((c, d), -1), reference2, rtol=0, atol=0
            )
            self.assertIsNone(m.prepare_forward_kv_plan(batch([1], [0]), "A"))
            self.assertIsNone(m.prepare_forward_kv_plan(first, "B"))
            self.assertIsNone(
                m.prepare_forward_kv_plan(batch([1, 2, 3], [4, 4, 4]), "A")
            )
            e, f = m.get_forward_kv(0, first, "B", plan=first_plan)
            self.assertNotEqual(e.data_ptr(), a.data_ptr())
            torch.testing.assert_close(torch.cat((e, f), -1), reference, rtol=0, atol=0)
            current_batch = batch([1, 2], [3, 2])
            current_batch.extend_prefix_lens_cpu = [2, 1]
            current_batch.extend_seq_lens_cpu = [1, 1]
            current_batch.out_cache_loc = torch.tensor([12, 7])
            current = torch.full((2, 1, 576), 3, dtype=dtype)
            host[0, 12] = current[0]
            host[0, 7] = current[1]
            plan = m.prepare_forward_kv_plan(current_batch, "A")
            n, r = m.get_forward_kv(
                0,
                current_batch,
                "A",
                plan=plan,
                current=(current, None, torch.tensor([True, True])),
            )
            current_reference = torch.cat(
                [host[0, table[1, :3]], host[0, table[2, :2]]]
            )
            torch.testing.assert_close(
                torch.cat((n, r), -1), current_reference, rtol=0, atol=0
            )
            self.assertEqual(plan.current_mapping[2].tolist(), [True, True])


if __name__ == "__main__":
    unittest.main()
