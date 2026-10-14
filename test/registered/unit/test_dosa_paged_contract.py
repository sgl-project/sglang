"""Memory alignment, initial state, causal storage and request lifetime properties."""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'python/sglang').is_dir())
SOURCE = ROOT / 'python/sglang/srt/layers/attention/dsa/dosa.py'
TORCH = NS(float8_e4m3fn='fp8', float32='f32', uint8='u8', int32='i32',
           cuda=NS(is_current_stream_capturing=lambda: False))


class Tensor:
    def __init__(self, shape, dtype='i32', device='cuda', contiguous=True, pointer=4096):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.ndim, self.contiguous = len(shape), contiguous
        self.pointer = pointer

    def __len__(self):
        return self.shape[0]

    def numel(self):
        total = 1
        for size in self.shape:
            total *= size
        return total

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return self.pointer


def function(source, name, namespace):
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual production function>', 'exec'), namespace)
    return namespace[name]


class PagedContractTests(unittest.TestCase):
    def setUp(self):
        self.ns = {'torch': TORCH, 'TOPK': 2048}
        self.valid = function(SOURCE.read_text(), 'valid_paged_inputs', self.ns)

    def inputs(self, rows=32768):
        return [Tensor((rows, 32, 128), 'fp8'), Tensor((rows, 32), 'f32'),
                Tensor((2048, 64, 132), 'u8'), Tensor((1, 2048)), Tensor((1, 131072)),
                Tensor((rows,)), Tensor((rows,)), Tensor((rows, 2048))]

    def test_unprepared_capacity_keeps_stock_without_cuda_work(self):
        namespace = {'_enabled': lambda: True}
        indexer = NS(_dosa_prefill_prepared=False)
        prefill = function(SOURCE.read_text(), 'try_paged_prefill', namespace)
        publish = function(SOURCE.read_text(), 'publish_stock_paged_prior', namespace)
        self.assertFalse(prefill(indexer, *([None] * 10)))
        self.assertFalse(publish(indexer, *([None] * 5), rows=2048))


    def test_borrowed_cache_alignment_matches_uint4_gather(self):
        values = self.inputs(8)
        for offset in range(32):
            values[2] = Tensor((2048, 64, 132), 'u8', pointer=4096 + offset)
            self.assertEqual(self.valid('cuda', *values), offset % 16 == 0)
        values = self.inputs(8)
        # Q/W are staged into owned buffers. Integer tables and output need
        # their scalar dtype alignment, not the cache's uint4 alignment.
        for slot in (0, 1, 3, 4, 5, 6, 7):
            values[slot].pointer += 4
        self.assertTrue(self.valid('cuda', *values))

    def test_planner_and_mapping_start_empty_with_dirty_scratch(self):
        torch = NS(float32='f32', float8_e4m3fn='fp8', uint8='u8', int16='i16',
                   int32='i32', int64='i64', zeros=lambda size, **kw: [0] * size,
                   zeros_like=lambda other: [0] * len(other),
                   full=MagicMock(), arange=MagicMock(),
                   cuda=NS(Event=MagicMock, current_stream=lambda device: None))
        allocate = function(SOURCE.read_text(), '_allocate',
                            dict(torch=torch, SEED=12288, TOPK=2048, RECENT=1536,
                                 _LeafSelection=lambda *args: None))
        for heads in (32, 64):
            owner = NS(max_q=8, max_k=12288, score_heads=heads, max_pages=2048,
                       record_capacity=2048, high_paid=20 if heads == 32 else 28,
                       device='cpu', sms=1, logical_query_capacity=8,
                       modules={'audit': NS(packed_record_words=lambda: 80)},
                       triton=NS(cdiv=lambda n, d: (n + d - 1) // d),
                       _empty=MagicMock(side_effect=lambda *args, **kw: MagicMock()))
            allocate(owner)
            # The planner reads old pair count/sticky error before writing;
            # mapping only reads its status. Neither may consume empty storage.
            self.assertEqual(owner.plan_counts, [0, 0, 0, 0])
            self.assertEqual(owner.backing['map_status'], [0] * owner.max_q)
            self.assertIs(owner.backing['map_status'], owner.zero)


def production(names):
    namespace = dict(SEED=12288, RECENT=1536, FEW_RECORD_CAPACITY=114688)
    for name in names:
        function(SOURCE.read_text(), name, namespace)
    return NS(**namespace)


class QueryStorageTests(unittest.TestCase):
    def test_records_and_causal_ids_share_the_original_memory_budget(self):
        g = production({"causal_id_page_bound", "few_id_pool_pages", "service_query_storage"})
        for logical, expected in ((4096, 1024), (8192, 2048), (16384, 4096), (32768, 4096)):
            for heads in (32, 64):
                for keys in (196608, 262144, 524288, 786432, 1048576):
                    rows, pages = g.service_query_storage(logical, keys, heads)
                    self.assertEqual(rows % 8, 0)
                    self.assertLessEqual(g.causal_id_page_bound(rows, keys), pages)
                    self.assertLessEqual(rows * 114688 * 5 + pages * 132 * 4,
                                         min(logical, 16384) * (65536 * 5 + 256 * 132 * 4))
                    if keys == 1048576:
                        self.assertEqual(rows, expected)

    def test_slabs_cover_every_query_and_the_exact_recent_window(self):
        slabs = production({"query_slabs"}).query_slabs
        for capacity in (8, 512, 1024, 4096, 8192):
            for rows in (1, 7, 8, capacity - 1, capacity, capacity + 1,
                         capacity + 1535, capacity + 1536, 16384, 32761, 32768):
                cursor, carry = 0, []
                for begin, end, physical, overlap in slabs(rows, capacity):
                    self.assertEqual(begin, cursor)
                    self.assertEqual(physical % 8, 0)
                    self.assertLessEqual(end - begin, physical)
                    self.assertLess(physical - (end - begin), 8)
                    self.assertLessEqual(physical, capacity)
                    carry.extend(range(end - overlap, end))
                    cursor = end
                self.assertEqual(cursor, rows)
                self.assertEqual(carry, list(range(max(0, rows - 1536), rows)))

    def test_retiring_a_request_keeps_reserved_layer_storage(self):
        tree = ast.parse(SOURCE.read_text())
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FewBackend")
        release = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "release_request")
        namespace = {}
        exec(compile(ast.Module(body=[release], type_ignores=[]), str(SOURCE), "exec"), namespace)
        votes, hot = object(), object()
        entry = dict(request="a", prior=object(), pending=object(), votes=votes, hot=hot)
        backend = NS(lock=threading.RLock(), layers={3: entry})
        namespace["release_request"](backend, (3, "b"))
        self.assertEqual(entry["request"], "a")
        namespace["release_request"](backend, (3, "a"))
        self.assertIsNone(entry["request"])
        self.assertIsNone(entry["prior"])
        self.assertIsNone(entry["pending"])
        self.assertIs(entry["votes"], votes)
        self.assertIs(entry["hot"], hot)


if __name__ == '__main__':
    unittest.main()
