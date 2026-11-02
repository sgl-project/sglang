"""CPU contracts for A3 BF16 DSV4.1 pools, without a torch_npu installation.

Load the production class definitions with mocked base allocations so these
tests exercise the real layout/scatter/budget methods, not Ascend imports.
Device operator execution and ACLGraph replay still require the A3 smoke test.
"""

import ast
import math
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[5]
POOL_FILE = ROOT / "python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py"
BUDGET_FILE = ROOT / "python/sglang/srt/model_executor/pool_configurator.py"


def _load_definitions(path, names, namespace):
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
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


class _IndexerBase:
    def _create_buffer(self):
        self.base_allocation_count = getattr(self, "base_allocation_count", 0) + 1
        self.index_k_with_scale_buffer = [torch.empty(1, dtype=torch.uint8)]

    def contiguous_page_row_buffers(self):
        return self.index_k_with_scale_buffer


def _pool_classes(arch35=False):
    return _load_definitions(
        POOL_FILE,
        {
            "dsv41_quant",
            "dsv41_row_layout",
            "_write_dsv41_bf16_rows",
            "NPUDeepSeekV4SingleKVPool",
            "DSV41NPUSingleKVPool",
            "NPUDeepSeekV4IndexerPool",
            "DSV41NPUIndexerPool",
            "DSV4NPUTokenToKVPool",
        },
        {
            "torch": torch,
            "math": math,
            "is_npu_arch35": lambda: arch35,
            "DeepSeekV4SingleKVPool": object,
            "DeepSeekV4IndexerPool": _IndexerBase,
            "DeepSeekV4TokenToKVPool": object,
            "KVLayout": SimpleNamespace(V4=object()),
            "GPU_MEMORY_TYPE_KV_CACHE": "kv_cache",
            "_DSV41_LOW_RATIOS": (1, 2),
            "_DSV41_ORI_KV_QUANT": ("mxfp8_bf16", 32),
            "_DSV41_CMP_KV_QUANT": ("mxfp4_bf16", 16),
            "_DSV41_ROW_ALIGNMENT": 32,
            "_DSV41_INDEX_SCALE_GROUP": 32,
        },
    )


def _single_pool(classes, ratio=1, compressed=True, size=256):
    pool = object.__new__(classes["DSV41NPUSingleKVPool"])
    pool.qk_nope_head_dim = 448
    pool.qk_rope_head_dim = 64
    pool.is_compressed = compressed
    pool.kernel_page_size = 128 // ratio
    pool.size = size
    pool.device = "cpu"
    pool.kv_buffer = [pool.create_buffer(num_pages=0)]
    return pool


def _index_pool(classes, size=256):
    pool = object.__new__(classes["DSV41NPUIndexerPool"])
    pool.index_head_dim = 128
    pool._kernel_page_size = 64
    pool.size = size
    pool.layer_num = 1
    pool.device = "cpu"
    pool.memory_saver_adapter = SimpleNamespace(region=lambda _: nullcontext())
    pool._create_buffer()
    return pool


class TestA3Cache(unittest.TestCase):
    def test_window_and_both_compressed_pools_are_bf16(self):
        classes = _pool_classes()
        for ratio, compressed in ((1, False), (1, True), (2, True)):
            pool = _single_pool(classes, ratio, compressed)
            buf = pool.kv_buffer[0]
            self.assertEqual(buf.dtype, torch.bfloat16)
            self.assertEqual(tuple(buf.shape[1:]), (128 // ratio, 1, 512))
            self.assertEqual(pool.dsv41_row_bytes, 1024)
            self.assertEqual(pool.bytes_per_page_padded, 128 // ratio * 1024)
            self.assertEqual(buf.shape[0] * buf.shape[1], 256 + 128 // ratio)

    def test_a5_layout_is_unchanged(self):
        classes = _pool_classes(arch35=True)
        for compressed, width, dtype in (
            (False, 544, torch.float8_e4m3fn),
            (True, 320, torch.uint8),
        ):
            pool = _single_pool(classes, compressed=compressed)
            self.assertEqual(pool.kv_buffer[0].shape[-1], width)
            self.assertEqual(pool.kv_buffer[0].dtype, dtype)
            self.assertEqual(pool.dsv41_row_bytes, width)

    def test_bf16_scatter_ignores_negative_and_zero_padding(self):
        write = _pool_classes()["_write_dsv41_bf16_rows"]
        buf = torch.full((3, 64, 1, 512), 7, dtype=torch.bfloat16)
        values = torch.arange(5, dtype=torch.float32)[:, None].expand(5, 512)
        write(buf, torch.tensor([-1, 0, 64, -1, 65]), values)
        rows = buf.view(-1, 1, 512)
        self.assertTrue(torch.equal(rows[0], torch.zeros_like(rows[0])))
        self.assertTrue(torch.equal(rows[64], torch.full_like(rows[64], 2)))
        self.assertTrue(torch.equal(rows[65], torch.full_like(rows[65], 4)))
        self.assertTrue(torch.equal(rows[-1], torch.full_like(rows[-1], 7)))
        write(buf, torch.empty(0, dtype=torch.int64), values[:0])
        with self.assertRaisesRegex(ValueError, "one slot"):
            write(buf, torch.tensor([1]), values)

    def test_a3_indexer_has_no_packed_copy_or_scale_payload(self):
        pool = _index_pool(_pool_classes())
        self.assertFalse(hasattr(pool, "base_allocation_count"))
        self.assertIsNone(pool.index_k_with_scale_buffer)
        self.assertEqual(pool.get_index_k(0).shape, (5, 64, 1, 128))
        self.assertEqual(pool.get_index_k(0).dtype, torch.bfloat16)
        self.assertEqual(pool.get_index_scale(0).numel(), 0)
        self.assertEqual(pool.contiguous_page_row_buffers()[0].shape, (5, 64 * 128 * 2))
        pool.set_index_k_scale(0, torch.tensor([-1, 64]), torch.ones(2, 128), None)
        self.assertEqual(pool.get_index_k(0)[0, 0].sum(), 0)
        self.assertEqual(pool.get_index_k(0)[1, 0].sum(), 128)

    def test_a5_indexer_allocations_are_unchanged(self):
        pool = _index_pool(_pool_classes(arch35=True))
        self.assertEqual(pool.base_allocation_count, 1)
        self.assertEqual(pool.get_index_k(0).shape[-1], 64)
        self.assertEqual(pool.get_index_scale(0).shape[-1], 4)
        self.assertEqual(pool.get_index_k(0).dtype, torch.uint8)

    def test_owner_mapping_routes_a3_writes_without_a5_ops(self):
        classes = _pool_classes()
        pool = _single_pool(classes, ratio=2)
        index_pool = _index_pool(classes)
        wrapper = object.__new__(classes["DSV4NPUTokenToKVPool"])
        # Two consumers point at the same owner; compressed sharing is retained.
        item = SimpleNamespace(
            compress_ratio=2, compress_layer_id=0, compress_kv_pool=pool
        )
        wrapper.layer_mapping = {2: item, 5: item}
        wrapper._indexer_pool = lambda _: index_pool
        wrapper.wait_layer_transfer = lambda _: None
        wrapper.set_compress_buffer(
            2, torch.tensor([64, -1]), torch.ones(2, 512), None, False
        )
        self.assertEqual(wrapper.get_compress_buffer(5)[1, 0].sum(), 512)
        wrapper.set_compress_buffer(
            2, torch.tensor([64, -1]), torch.ones(2, 128), None, True
        )
        self.assertEqual(wrapper.get_index_k(5)[1, 0].sum(), 128)


class TestA3Budget(unittest.TestCase):
    def _budget(self, a3=True):
        ns = _load_definitions(
            BUDGET_FILE,
            {"DSV4PoolConfigurator"},
            {
                "torch": torch,
                "MemoryPoolConfigurator": object,
                "_is_npu": True,
                "ceil_div": lambda a, b: (a + b - 1) // b,
            },
        )
        pool = object.__new__(ns["DSV4PoolConfigurator"])
        pool._npu_low_ratio_bf16 = a3
        pool._unified = False
        pool.kv_bytes = 584
        pool.attn_head_dim = 512
        pool.low_ratio_index_bytes = 256 if a3 else 68
        pool.page_size = 128
        pool.num_layers_total = 40
        pool.stage_owner_layers = {1: [20], 2: [2, 8, 14]}
        pool.ring_sizes = {2: 2}
        pool.disaggregation_mode = "null"
        return pool

    def test_a3_token_bytes(self):
        budget = self._budget()
        for ratio in (0, 1, 2):
            self.assertEqual(budget._get_paged_kv_bytes_per_token(ratio), 1024)
        self.assertEqual(budget._compressed_bytes_per_full_token(1), 1280)
        self.assertEqual(budget._compressed_bytes_per_full_token(2), 640)

    def test_padding_matches_actual_physical_allocations(self):
        budget = self._budget()
        fake_module = ModuleType("sglang.srt.mem_cache.deepseek_v4_memory_pool")
        fake_module.DSV41_INDEX_PAGE_SIZE = 64
        classes = _pool_classes()
        for tokens in (128, 256, 512, 1024):
            allocated = 0
            for ratio, owners in budget.stage_owner_layers.items():
                kv = _single_pool(classes, ratio, size=tokens // ratio)
                index = _index_pool(classes, size=(tokens + 128) // ratio)
                allocated += len(owners) * (
                    kv.kv_buffer[0].nbytes + index.get_index_k(0).nbytes
                )
            allocated += 128 * 1024 * 40  # SWA reserved pages only
            nominal = tokens * (1280 + 3 * 640)
            with patch.dict("sys.modules", {fake_module.__name__: fake_module}):
                self.assertEqual(
                    budget._get_low_ratio_kv_padding_bytes(tokens), allocated - nominal
                )

    def test_pair_state_fp32_and_ring_padding(self):
        budget = self._budget()
        self.assertEqual(budget._get_c2_state_fixed_bytes(2), 10 * 1024 * 4 * 3)

    def test_a5_budget_is_unchanged(self):
        budget = self._budget(a3=False)
        self.assertEqual(budget._get_paged_kv_bytes_per_token(1), 584)
        self.assertEqual(budget._get_low_ratio_kv_padding_bytes(128), 0)


if __name__ == "__main__":
    unittest.main()
