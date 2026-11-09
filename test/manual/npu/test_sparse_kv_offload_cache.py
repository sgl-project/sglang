"""CPU regression tests for sparse KV planning and the Python stream schedule.

Run directly with Python + CPU PyTorch; no SGLang/NPU installation is required.
Native transfers are emulated, so this does not validate the device kernel ABI,
device overlap or ACLGraph capture.
"""

import ast
import importlib.util
import logging
import random
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import List, Optional, Union
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload"


class FakeStream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def record_event(self, event):
        self.log.append(("record", self.name, event))

    def wait_event(self, event):
        self.log.append(("wait", self.name, event))


class TestTraceReplay(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "sparse_kv_bench_test", ROOT / "benchmark/hisparse/analyze_budget.py"
        )
        self.bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bench)

    def replay(self, rows, **kwargs):
        import json

        with TemporaryDirectory() as directory:
            path = Path(directory) / "topk.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            return self.bench.replay_trace(path, [2, 4], 1152, **kwargs)

    def test_capacity_improves_reuse_on_known_trace(self):
        rows = [
            dict(request_id="r", layer=0, step=i, token_ids=tokens)
            for i, tokens in enumerate(([1, 2], [3, 4], [1, 2]))
        ]
        small, large = self.replay(rows)
        self.assertEqual(small["hits"], 0)
        self.assertEqual(large["hits"], 2)
        self.assertEqual(large["h2d_kv_bytes"], 4 * 1152)

    def test_request_generation_and_warmup(self):
        rows = [
            dict(
                request_id="r",
                generation=generation,
                layer=0,
                step=step,
                token_ids=[1, 2],
            )
            for generation in (0, 1)
            for step in (0, 1)
        ]
        small, _ = self.replay(rows, warmup_steps=1)
        self.assertEqual(small["misses"], 0)
        self.assertEqual(small["hits"], 4)
        small, _ = self.replay(rows)
        self.assertEqual(small["misses"], 4)

    def test_rejects_unordered_steps(self):
        rows = [dict(request_id="r", layer=0, step=0, token_ids=[1, 2])] * 2
        with self.assertRaisesRegex(ValueError, "steps must increase"):
            self.replay(rows)


class TestSparseKVManager(unittest.TestCase):
    def setUp(self):
        # Execute the real manager methods without importing serving/runtime/NPU
        # modules. All changed method bodies come directly from the source file.
        tree = ast.parse((SOURCE / "manager.py").read_text(encoding="utf-8"))
        names = {
            "SparseKVCacheManager",
            "_record_stream_event",
            "_wait_stream_event",
            "normalize_batch_topk_indices",
            "_build_hit_src_dst_index",
            "_build_miss_src_dst_index",
        }
        tree.body = [
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ] + [node for node in tree.body if getattr(node, "name", None) in names]
        scope = dict(
            torch=torch,
            Optional=Optional,
            List=List,
            Union=Union,
            logger=logging.getLogger(__name__),
            slot_map_lookup=self.lookup,
            unidex_copy_inplace=self.copy,
        )
        exec(
            compile(
                ast.fix_missing_locations(tree), str(SOURCE / "manager.py"), "exec"
            ),
            scope,
        )
        self.manager_scope = scope
        self.Manager = scope["SparseKVCacheManager"]
        self.log = []
        self.main = FakeStream("main", self.log)
        self.side = FakeStream("side", self.log)
        self.active = self.main
        self.host_copy_rows = 0
        self.context_patch = patch.object(
            torch,
            "npu",
            SimpleNamespace(
                stream=self.stream, current_stream=lambda *args, **kwargs: self.main
            ),
            create=True,
        )
        self.context_patch.start()
        self.addCleanup(self.context_patch.stop)
        self.record_patch = patch.object(
            torch.Tensor, "record_stream", lambda tensor, stream: None
        )
        self.record_patch.start()
        self.addCleanup(self.record_patch.stop)

    @contextmanager
    def stream(self, stream):
        previous = self.active
        self.active = stream
        try:
            yield
        finally:
            self.active = previous

    @staticmethod
    def lookup(slot_map, requests, tokens):
        slots = slot_map[requests.long().unsqueeze(1), tokens.long()]
        return slots >= 0, slots

    def copy(
        self,
        source,
        destination,
        source_indices,
        destination_indices,
        valid,
        src_dims,
        dst_dims,
        **kwargs,
    ):
        self.log.append(
            ("copy", self.active.name, "host" if "src_ptr" in kwargs else "device")
        )
        if "src_ptr" in kwargs:
            self.host_copy_rows += int(valid.sum())
        src = source.reshape(-1, *source.shape[src_dims:])
        dst = destination.view(-1, *destination.shape[dst_dims:])
        dst[destination_indices[valid].long()] = src[
            source_indices[valid].long()
        ].clone()

    def manager(self):
        m = self.Manager.__new__(self.Manager)
        m.copy_sfa_enabled = False
        m.size, m.max_context_len, m.sparse_context_len = 4, 64, 2
        m.device_cache_size = 2
        m.device, m.start_layer, m.layer_num = "cpu", 0, 1
        m._slot_map_width = 72
        m.device_kv_buffer = [torch.full((4, 2, 1, 3), -999.0)]
        m.selected_kv_buffer = torch.empty((4, 2, 1, 3))
        m.device_slot_map = [torch.full((5, 72), -1, dtype=torch.int32)]
        m.host_kv_buffer = [torch.arange(4 * 64 * 3).float().view(4, 64, 1, 3)]
        m.dev_ptr_list = [123]
        m._device_cache_slot_ids = torch.arange(2)
        m._materialize_h2d_miss_stream = self.side
        m.copy_ready, m.hit_done, m.miss_done, m.refill_done = (
            object(),
            object(),
            object(),
            object(),
        )
        return m

    def test_request_reuse_resets_both_target_and_draft_managers(self):
        target, draft = self.manager(), self.manager()
        pool = SimpleNamespace(alloc=lambda reqs: [1] * len(reqs))
        target._install_req_alloc_hook(pool)
        draft._install_req_alloc_hook(pool)
        for manager in (target, draft):
            manager.device_slot_map[0][1:3].fill_(7)
        pool.alloc([SimpleNamespace(kv=SimpleNamespace(holds_kv=False))])
        for manager in (target, draft):
            self.assertTrue(torch.all(manager.device_slot_map[0][1] == -1))
            self.assertTrue(torch.all(manager.device_slot_map[0][2] == 7))

    def step(self, manager, requests, tokens, lengths=None):
        selected = torch.zeros((len(requests), 2, 1, 3))
        batch = SimpleNamespace(
            req_pool_indices=torch.tensor(requests),
            seq_lens=torch.tensor(lengths or [64] * len(requests)),
        )
        manager.materialize_selected_kv(
            SimpleNamespace(layer_id=0),
            batch,
            torch.tensor(tokens, dtype=torch.int32),
            selected,
            self.main,
        )
        for i, (rid, row) in enumerate(zip(requests, tokens)):
            for j, token in enumerate(row):
                if 0 < rid < manager.size and 0 <= token < 64 and batch.seq_lens[i] > 0:
                    torch.testing.assert_close(
                        selected[i, j], manager.host_kv_buffer[0][rid, token]
                    )
                else:
                    self.assertEqual(float(selected[i, j].abs().sum()), 0.0)
        return selected

    def test_inactive_request_mappings_survive(self):
        manager = self.manager()
        self.step(manager, [1], [[1, 2]])
        saved = manager.device_slot_map[0][1].clone()
        self.step(manager, [2], [[3, 4]])
        self.assertTrue(torch.equal(saved, manager.device_slot_map[0][1]))

    def test_hit_miss_reordering_and_stream_publication(self):
        manager = self.manager()
        self.step(manager, [1], [[1, 2]])
        self.step(manager, [1], [[3, 1]])
        miss_ready = self.log.index(("record", "side", manager.miss_done))
        refill_ready = self.log.index(("record", "side", manager.refill_done))
        self.assertLess(miss_ready, refill_ready)
        self.assertIn(("wait", "side", manager.copy_ready), self.log)
        self.assertIn(("wait", "side", manager.hit_done), self.log)

    def test_padding_and_request_reuse(self):
        manager = self.manager()
        self.step(manager, [1, 2, 0, -1, 9], [[1, -1], [64, 3], [1, 2], [1, 2], [1, 2]])
        self.assertEqual(self.host_copy_rows, 2)
        self.assertTrue(torch.all(manager.device_slot_map[0][0] == -1))
        manager.reset_requests([1])
        self.assertTrue(torch.all(manager.device_slot_map[0][1] == -1))
        self.step(manager, [1], [[1, 2]])
        self.assertEqual(self.host_copy_rows, 4)

    def test_randomized_gather_and_mapping_consistency(self):
        random_generator = random.Random(17392)
        manager = self.manager()
        for _ in range(80):
            requests = random_generator.sample(
                [1, 2, 3], random_generator.randint(1, 3)
            )
            selections = [random_generator.sample(range(16), 2) for _ in requests]
            self.step(manager, requests, selections)

    def test_attention_pads_short_topk_before_gather(self):
        manager = self.manager()
        tree = ast.parse((SOURCE / "attention.py").read_text(encoding="utf-8"))
        tree.body = [
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ] + [
            node
            for node in tree.body
            if getattr(node, "name", None) == "forward_sparsity_driven_kv_offload"
        ]
        observed = []

        def sparse_attention(query, key, value, indices, scale, **kwargs):
            observed.append((key.clone(), kwargs["key_rope"].clone()))
            return torch.full_like(query, 123)

        scope = dict(
            torch=torch,
            torch_npu=SimpleNamespace(npu_sparse_flash_attention=sparse_attention),
            _get_sparse_kv_manager=lambda backend: backend.sparse_kv_manager,
            normalize_batch_topk_indices=self.manager_scope[
                "normalize_batch_topk_indices"
            ],
            _wait_stream_event=self.manager_scope["_wait_stream_event"],
        )
        exec(
            compile(
                ast.fix_missing_locations(tree), str(SOURCE / "attention.py"), "exec"
            ),
            scope,
        )
        backend = SimpleNamespace(
            device="cpu",
            kv_lora_rank=2,
            qk_rope_head_dim=1,
            sparse_kv_manager=manager,
            forward_metadata=SimpleNamespace(
                actual_seq_lengths_q=None,
                actual_seq_lengths_kv=None,
                seq_lens_cpu_int=None,
                seq_lens=torch.tensor([64, 64]),
            ),
        )
        mode = SimpleNamespace(
            is_extend_without_speculative=lambda: False,
            is_decode=lambda: True,
            is_draft_extend_v2=lambda: False,
            is_target_verify=lambda: False,
        )
        batch = SimpleNamespace(
            batch_size=2,
            forward_mode=mode,
            req_pool_indices=torch.tensor([1, 2]),
            seq_lens=torch.tensor([64, 64]),
        )

        def run_attention():
            return scope["forward_sparsity_driven_kv_offload"](
                backend,
                torch.ones(2, 1, 2),
                torch.ones(2, 1, 2),
                None,
                SimpleNamespace(
                    layer_id=0, tp_k_head_num=1, tp_q_head_num=1, scaling=0.5
                ),
                batch,
                save_kv_cache=False,
                q_rope=torch.ones(2, 1, 1),
                k_rope=torch.ones(2, 1, 1),
                topk_indices=torch.tensor([[1], [7]], dtype=torch.int32),
            )

        result = run_attention()
        self.assertEqual(result.shape, (2, 2))
        nope, rope = observed[0]
        for row, (request, token) in enumerate(((1, 1), (2, 7))):
            actual = torch.cat((nope[row, 0], rope[row, 0]), dim=-1)
            torch.testing.assert_close(
                actual, manager.host_kv_buffer[0][request, token]
            )
        self.assertIn(("wait", "main", manager.miss_done), self.log)
        self.assertIn(("wait", "main", manager.refill_done), self.log)
        # A skipped vendor output must be cleared for padding while real rows
        # keep their output. The fake kernel deliberately returns nonzero data.
        batch.seq_lens[1] = 0
        batch.req_pool_indices[1] = 0
        result = run_attention()
        self.assertTrue(torch.all(result[0] == 123))
        self.assertTrue(torch.all(result[1] == 0))


if __name__ == "__main__":
    unittest.main()
