import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

PATH = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/prefill_kv_plan.py"
)
spec = importlib.util.spec_from_file_location("prefill_test_kv_plan", PATH)
module = importlib.util.module_from_spec(spec)

sys.modules[spec.name] = module
spec.loader.exec_module(module)


class PrefillReferenceTest(unittest.TestCase):
    def setUp(self):
        # Exercise framework planning on CPU; device kernel parity is tested by
        # test_sparse_kv_row_metadata.py and the companion NPU kernel tests.
        adapter = ModuleType("row_metadata_test_adapter")

        def prefill_plan(
            batch,
            manager,
            total,
            *,
            current_metadata=None,
            current_valid=None,
            max_current_width=0,
        ):
            if (current_metadata is None) != (current_valid is None):
                raise ValueError("current metadata and validity must be paired")
            rows, positions = [], []
            for req, length in zip(batch.req_pool_indices_cpu, batch.seq_lens_cpu):
                if req >= 0 and length > 0:
                    rows.extend([int(req)] * int(length))
                    positions.extend(range(int(length)))
            source = manager.host_source_indices(
                torch.tensor(rows), torch.tensor(positions)
            )
            current_mapping = None
            if current_metadata is not None and current_valid is not None:
                n = batch.req_pool_indices_cpu.numel()
                starts, prefixes, extends, widths = current_metadata.split(n)
                dst = torch.zeros(current_valid.numel(), dtype=torch.long)
                eligible = torch.zeros_like(current_valid, dtype=torch.bool)
                for i, (req, start, prefix, extend, width) in enumerate(
                    zip(
                        batch.req_pool_indices_cpu.tolist(),
                        starts.tolist(),
                        prefixes.tolist(),
                        extends.tolist(),
                        widths.tolist(),
                    )
                ):
                    for col in range(extend):
                        row = start + col
                        if req > 0:
                            dst[row] = (
                                sum(int(x) for x in batch.seq_lens_cpu[:i])
                                + prefix
                                + col
                            )
                            eligible[row] = True
                mask = eligible & current_valid.bool()
                host_mask = torch.ones(total, dtype=torch.bool)
                host_mask[dst[mask]] = False
                current_mapping = (
                    torch.arange(current_valid.numel()),
                    dst,
                    mask,
                    host_mask,
                )
            return (
                source,
                torch.arange(total),
                torch.ones(total, dtype=torch.bool),
                current_mapping,
            )

        adapter.prefill_plan = prefill_plan
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.row_metadata": adapter
                },
            )
        )


class TestLogicalPrefillPlan(PrefillReferenceTest):
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
        # Independently assemble request KV in logical order. Current payload
        # row IDs are allocator slots; prefix/current placement follows only
        # prefix and extend lengths, never physical ID equality.
        reference = torch.cat(
            [host[request_map[r, :length]] for r, length in zip(req, lengths)]
        )
        current_row = 0
        output_offset = 0
        slots = (
            extends
            if sum(extends) == len(valid)
            else [len(valid) // len(req)] * len(req)
        )
        for request, length, prefix, extend, width in zip(
            req, lengths, prefixes, extends, slots
        ):
            for col in range(width):
                if col < extend and request > 0 and valid[current_row + col]:
                    reference[output_offset + prefix + col] = payload[current_row + col]
            current_row += width
            output_offset += length
        plan = module.make_fused_prefill_plan(
            batch, Manager(), current_valid=torch.tensor(valid)
        )
        mapping = plan.current_mapping
        if mapping is None:
            result = host[plan.source].clone()
        else:
            src, dst, mask, host_mask = mapping
            result = torch.empty_like(reference)
            result[host_mask] = host[plan.source[host_mask]]
            result[dst[mask]] = payload[src[mask]]
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        return mapping

    def test_ragged_and_padded_current_rows(self):
        table = torch.zeros((8, 8), dtype=torch.long)
        table[1, :4] = torch.tensor([4, 7, 10, 11])
        table[2, :4] = torch.tensor([4, 9, 11, 13])
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                # Prefix physical rows may alias; current rows remain unique.
                mapping = self.check_case(
                    table,
                    [1, 2],
                    [4, 4],
                    [2, 2],
                    [2, 2],
                    [14, 15, 16, 17],
                    [100, 101, 102, 103],
                    [True] * 4,
                    dtype,
                )
                self.assertEqual(mapping[2].tolist(), [True, True, True, True])
                # Invalid current rows retain the Host value at that logical slot.
                mapping = self.check_case(
                    table, [1], [4], [2], [2], [18, 19], [120, 0], [True, False], dtype
                )
                self.assertEqual(mapping[2].tolist(), [True, False])
                # Padded current rows use fixed shape without a dynamic boolean index.
                mapping = self.check_case(
                    table,
                    [1, 2],
                    [3, 4],
                    [2, 2],
                    [1, 2],
                    [20, 21, 22, 23],
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

    def test_request_zero_current_stays_on_host(self):
        table = torch.zeros((8, 8), dtype=torch.long)
        table[0, :2] = torch.tensor([2, 3])
        table[2, :2] = torch.tensor([5, 7])
        mapping = self.check_case(
            table,
            [0, 2],
            [2, 2],
            [1, 1],
            [1, 1],
            [20, 21],
            [31, 37],
            [True, True],
            torch.float32,
        )
        self.assertEqual(mapping[2].tolist(), [False, True])


class TestWorkspaceLifecycle(PrefillReferenceTest):
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
            source = source.reshape(-1, 1, source.shape[-1])
            dest[dst[mask]] = source[src[mask]]

        namespace = {
            "torch": torch,
            "Optional": object,
            "Union": object,
            "RadixAttention": object,
            "ForwardBatch": object,
            "prepare_prefill_gather_rows": module.prepare_prefill_gather_rows,
            "make_fused_prefill_plan": module.make_fused_prefill_plan,
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

    def test_reuse_capacity_and_host_fallback(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            m = self.manager_type()

            class SplitPool:
                def copy_rows(self, sources, destinations, src, dst, mask):
                    for source, dest in zip(sources, destinations):
                        source = source.reshape(-1, 1, source.shape[-1])
                        dest.reshape(-1, 1, dest.shape[-1])[dst[mask]] = source[
                            src[mask]
                        ]

            m.host_pool = SplitPool()
            m.budget = SimpleNamespace(prefill_workspace_tokens=8)
            m.size, m.max_context_len, m.start_layer, m.layer_num = 4, 4, 0, 2
            m.device, m.head_num, m.kv_lora_rank, m.qk_rope_head_dim = (
                torch.device("cpu"),
                1,
                512,
                64,
            )
            m.head_dim, m.store_dtype, m.host_token_capacity = 576, dtype, 16
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
            hosts = [host, host + 1]
            m.host_k_buffer = [h[..., :512].clone() for h in hosts]
            m.host_v_buffer = [h[..., 512:].clone() for h in hosts]

            def batch(ids, lens):
                ids, lens = torch.tensor(ids), torch.tensor(lens)
                return SimpleNamespace(
                    req_pool_indices_cpu=ids,
                    seq_lens_cpu=lens,
                    req_pool_indices=ids,
                    seq_lens=lens,
                )

            first, second = batch([1, 2], [3, 2]), batch([2, 1], [2, 3])
            first_plan = m.prepare_forward_kv_plan(first)
            a, b = m.get_forward_kv(0, first, plan=first_plan)
            saved = torch.cat((a.clone(), b.clone()), -1)
            reference = torch.cat([host[0, table[r, :n]] for r, n in ([1, 3], [2, 2])])
            torch.testing.assert_close(saved, reference, rtol=0, atol=0)
            second_plan = m.prepare_forward_kv_plan(second)
            c, d = m.get_forward_kv(1, second, plan=second_plan)
            self.assertEqual(a.data_ptr(), c.data_ptr())
            reference2 = torch.cat(
                [(host + 1)[0, table[r, :n]] for r, n in ([2, 2], [1, 3])]
            )
            torch.testing.assert_close(
                torch.cat((c, d), -1), reference2, rtol=0, atol=0
            )
            self.assertIsNone(m.prepare_forward_kv_plan(batch([1], [0])))
            self.assertIsNone(m.prepare_forward_kv_plan(batch([1, 2, 3], [4, 4, 4])))
            e, f = m.get_forward_kv(0, first)
            self.assertNotEqual(e.data_ptr(), a.data_ptr())
            torch.testing.assert_close(torch.cat((e, f), -1), reference, rtol=0, atol=0)
            current_batch = batch([1, 2], [3, 2])
            current_batch.extend_prefix_lens_cpu = [2, 1]
            current_batch.extend_seq_lens_cpu = [1, 1]
            current_batch.out_cache_loc = torch.tensor([12, 7])
            current = torch.full((2, 1, 576), 3, dtype=dtype)
            current_inputs = (
                (current[..., :512].contiguous(), current[..., 512:].contiguous()),
                None,
                torch.tensor([True, True]),
            )
            plan = m.prepare_forward_kv_plan(current_batch, current=current_inputs)
            n, r = m.get_forward_kv(
                0,
                current_batch,
                plan=plan,
                current=current_inputs,
            )
            current_reference = torch.cat(
                [host[0, table[1, :3]], host[0, table[2, :2]]]
            )
            current_reference[2] = current[0, 0]
            current_reference[4] = current[1, 0]
            torch.testing.assert_close(
                torch.cat((n, r), -1), current_reference, rtol=0, atol=0
            )
            self.assertEqual(plan.current_mapping[2].tolist(), [True, True])

            # The cached plan may carry a D2D mapping from an earlier layer,
            # while a later layer omits current payload. That call must gather
            # the complete logical sequence from Host.
            host_only, rope_only = m.get_forward_kv(
                1, current_batch, plan=plan, current=None
            )
            layer_host = hosts[1]
            host_reference = torch.cat(
                [layer_host[0, table[1, :3]], layer_host[0, table[2, :2]]]
            )
            torch.testing.assert_close(
                torch.cat((host_only, rope_only), -1),
                host_reference,
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
