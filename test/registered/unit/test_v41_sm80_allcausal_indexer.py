"""CPU torch regressions for exact SM80 selection; no runtime/CUDA imports."""

from __future__ import annotations

import ast
import os
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
register_cpu_ci = runpy.run_path(
    str(ROOT / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=3, suite="base-a-test-cpu")
BASE = ROOT / "python/sglang/srt/layers/attention/dsv4/v41_indexer"
FLAG = "SGLANG_DSV41_SM80_EXPERIMENTAL"
ENVS = runpy.run_path(str(ROOT / "python/sglang/srt/environ.py"))["envs"]


class PrefillInputs(NS):
    def reset_outputs(self):
        self.out_raw_indices.fill_(-1)
        if self.out_page_indices is not None:
            self.out_page_indices.fill_(-1)


class DecodeInputs(NS):
    def reset_outputs(self):
        self.out_page_indices.fill_(-1)


class Struct:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102


def namespace():
    ns = dict(
        torch=torch,
        envs=ENVS,
        F=F,
        PrefillInputs=PrefillInputs,
        DecodeInputs=DecodeInputs,
        CandidateMetadata=type("CandidateMetadata", (), {}),
        msgspec=NS(Struct=Struct),
    )
    helper = ROOT / "python/sglang/kernels/ops/attention/dsv4/sm80_utils.py"
    gate_ns = dict(
        envs=ENVS,
        torch=NS(
            device=lambda d: NS(type="cuda"),
            version=NS(hip=None),
            cuda=NS(is_available=lambda: True, get_device_capability=lambda d: (8, 0)),
        ),
    )
    extract(helper, {"is_dsv41_sm80_experimental"}, gate_ns)
    ns["is_dsv41_sm80_experimental"] = gate_ns["is_dsv41_sm80_experimental"]
    ns["is_deep_select_supported"] = lambda: False
    # Execute original baseline torch scoring/writes with CPU-only dependencies.
    ns["_TORCH_SCORE_BUDGET_BYTES"] = 1024
    extract(
        BASE / "scoring.py",
        {
            "RequestScores",
            "ChunkScores",
            "DecodeScores",
            "prefill_requests",
            "_score_chunks",
            "write_prefill",
            "write_decode",
        },
        ns,
    )
    extract(
        ROOT / "python/sglang/kernels/ops/attention/dsv4/candidate_blocks.py",
        {"select_candidate_block_ids", "topk_among_blocks"},
        ns,
    )
    extract(
        BASE / "full_topk.py",
        {"_all_causal_bound", "_write_all_causal", "FullTopKIndexer"},
        ns,
    )
    extract(BASE / "dense_blocks.py", {"BlockIds", "DenseBlocksBackend"}, ns)
    extract(BASE / "types.py", {"get_tail_row_indices"}, ns)
    ns["async_h2d"] = lambda values, dtype, device: torch.tensor(
        values, dtype=dtype, device=device
    )
    return ns, gate_ns


def inputs(ratio=1, lengths=(512, 31), rows=(4, 3), pages=True):
    positions = torch.cat([torch.arange(s - n, s) for s, n in zip(lengths, rows)])
    req_rows = torch.cat(
        [torch.full((n,), r, dtype=torch.long) for r, n in enumerate(rows)]
    )
    indexer = NS(
        index_topk=512, scores=Mock(side_effect=AssertionError("shortcut scored"))
    )
    return PrefillInputs(
        indexer=indexer,
        positions=positions,
        req_rows=req_rows,
        req_pool_indices=torch.arange(len(rows)),
        compress_ratio=ratio,
        seq_lens_cpu=list(lengths),
        rows_per_request=list(rows),
        layer_id=0,
        x=torch.zeros(len(positions), 1),
        q_lora=torch.zeros(len(positions), 1),
        freqs_cis=torch.zeros(max(lengths, default=0), 1),
        out_raw_indices=torch.full((len(positions), 512), 99, dtype=torch.int32),
        out_page_indices=torch.full((len(positions), 512), 99, dtype=torch.int32)
        if pages
        else None,
    )


def request_map(width=17000):
    # Disjoint non-contiguous physical slots for two requests; ratio-2 aligned.
    return torch.stack(
        [torch.arange(width) * 6 + 1000, torch.arange(width) * 10 + 200000]
    )


def enable_baseline(i):
    i.indexer.queries = lambda q, f: torch.zeros(len(q), 1, 1)
    i.indexer.head_weights = lambda x: torch.zeros(len(x), 1)
    i.indexer.scores = Mock(
        side_effect=lambda q, k, w: torch.arange(len(k), dtype=torch.float32)[
            None, :
        ].expand(len(q), -1)
    )
    return NS(
        get_low_ratio_index_k_dequant=lambda layer, slots: torch.zeros(len(slots), 1)
    )


class AllCausalTests(unittest.TestCase):
    def setUp(self):
        self.ns, self.gate = namespace()
        self.env = patch.dict(os.environ, {FLAG: "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.mapping = request_map()

    def full(self, pool=None):
        return self.ns["FullTopKIndexer"](
            token_to_kv_pool=pool,
            req_to_token=self.mapping,
            use_deep_gemm_prefill=False,
            use_deep_gemm_decode=False,
        )

    def dense(self, pool=None, top=2048):
        return self.ns["DenseBlocksBackend"](
            token_to_kv_pool=pool,
            req_to_token=self.mapping,
            candidate_topk_blocks=top,
            candidate_block_size=8,
            use_deep_gemm_prefill=False,
        )

    def assert_expected(self, i):
        for row, (pos, req) in enumerate(
            zip(i.positions.tolist(), i.req_rows.tolist())
        ):
            n = (pos + 1) // i.compress_ratio
            raw = torch.full((512,), -1, dtype=torch.int32)
            raw[:n] = torch.arange(n)
            torch.testing.assert_close(i.out_raw_indices[row], raw)
            if i.out_page_indices is not None:
                slots = torch.full_like(raw, -1)
                slots[:n] = (
                    self.mapping[req, torch.arange(n) * i.compress_ratio]
                    // i.compress_ratio
                )
                torch.testing.assert_close(i.out_page_indices[row], slots)

    def test_prefill_prefix_chunks_ratios_and_zero(self):
        for ratio in (1, 2):
            for length in (0, 1, 511, 512, 513):
                for pages in (False, True):
                    # floor length/ratio boundary; wider batch must fallback.
                    i = inputs(ratio, (length, 17), (min(length, 4), 3), pages)
                    if length // ratio > 512:
                        continue
                    self.full().topk_prefill(i)
                    self.assert_expected(i)
                    i.indexer.scores.assert_not_called()
                    # Compare sorted raw and physical picks to original baseline.
                    baseline = inputs(ratio, (length, 17), (min(length, 4), 3), pages)
                    pool = enable_baseline(baseline)
                    self.full(pool)._torch_prefill(baseline)
                    torch.testing.assert_close(
                        i.out_raw_indices, baseline.out_raw_indices
                    )
                    if pages:
                        torch.testing.assert_close(
                            i.out_page_indices, baseline.out_page_indices
                        )

    def test_whole_batch_wide_missing_metadata_and_gates_fallback(self):
        for ratio, length in ((1, 513), (2, 1026)):
            i = inputs(ratio, (31, length), (3, 4))
            full = self.full()
            full._torch_prefill = Mock()
            full.topk_prefill(i)
            full._torch_prefill.assert_called_once_with(i)
        for missing in ("seq_lens_cpu", "rows_per_request", "req_rows"):
            i = inputs()
            setattr(i, missing, None)
            full = self.full()
            full._torch_prefill = Mock()
            full.topk_prefill(i)
            full._torch_prefill.assert_called_once()
        for flag, cap, hip in (
            ("0", (8, 0), None),
            ("true", (8, 0), None),
            ("1", (8, 6), None),
            ("1", (9, 0), None),
            ("1", (8, 0), "6.0"),
        ):
            with patch.dict(os.environ, {FLAG: flag}):
                self.gate["torch"].cuda.get_device_capability = lambda d, cap=cap: cap
                self.gate["torch"].version.hip = hip
                i = inputs()
                full = self.full()
                full._torch_prefill = Mock()
                full.topk_prefill(i)
                full._torch_prefill.assert_called_once()

    def test_dense_publish_consume_baseline_blocks_and_tail(self):
        for ratio in (1, 2):
            i = inputs(ratio)
            dense = self.dense()
            published = dense.publish_prefill(i)
            self.assert_expected(i)
            baseline = inputs(ratio)
            pool = enable_baseline(baseline)
            old = self.dense(pool)._torch_publish_prefill(baseline)
            torch.testing.assert_close(
                published.blocks.sort(-1).values, old.blocks.sort(-1).values
            )
            consumer = inputs(ratio)
            dense.consume_prefill(consumer, published)
            self.assert_expected(consumer)
            consumer.indexer.scores.assert_not_called()
            tail = published.tail([2, 1])
            tail_i = inputs(ratio, rows=(2, 1))
            dense.consume_prefill(tail_i, tail)
            self.assert_expected(tail_i)
            self.assertEqual(tail.rows_per_request, [2, 1])
            # An ordinary candidate subset has no exact-source proof.
            subset = self.ns["BlockIds"](
                blocks=published.blocks, rows_per_request=[4, 3]
            )
            dense._torch_consume_prefill = Mock()
            dense.consume_prefill(consumer, subset)
            dense._torch_consume_prefill.assert_called_once()

    def test_consumer_respects_shorter_source_row_candidate_extent(self):
        source = inputs()
        source.positions[0] = 0
        dense = self.dense()
        published = dense.publish_prefill(source)
        consumer = inputs()
        dense.consume_prefill(consumer, published)
        baseline = inputs()
        pool = enable_baseline(baseline)
        self.dense(pool)._torch_consume_prefill(baseline, published)
        torch.testing.assert_close(consumer.out_raw_indices, baseline.out_raw_indices)
        torch.testing.assert_close(consumer.out_page_indices, baseline.out_page_indices)
        self.assertTrue(torch.all(consumer.out_raw_indices[0, 8:] == -1))

    def test_dense_wide_source_still_scores_top512(self):
        i = inputs(lengths=(513, 31))
        pool = enable_baseline(i)
        dense = self.dense(pool)
        p = dense.publish_prefill(i)
        self.assertTrue(i.indexer.scores.called)
        self.assertEqual(p.blocks.shape[1], 65)
        self.assertEqual(int((i.out_raw_indices[3] >= 0).sum()), 512)
        dense._torch_consume_prefill = Mock()
        dense.consume_prefill(i, p)
        dense._torch_consume_prefill.assert_called_once()
        # Not all blocks fit: leave source entirely on original candidate path.
        dense = self.dense(top=1)
        dense._torch_publish_prefill = Mock()
        dense.publish_prefill(i)
        dense._torch_publish_prefill.assert_called_once()

    def test_decode_shape_capacity_and_candidate_extent(self):
        for ratio in (1, 2):
            i = DecodeInputs(
                indexer=NS(index_topk=512),
                compress_ratio=ratio,
                positions=torch.tensor([511, 0]),
                req_rows=torch.tensor([0, 1]),
                paged_metadata=NS(max_compressed_seq_len=512),
                out_page_indices=torch.full((2, 512), 99, dtype=torch.int32),
            )
            self.full().topk_decode(i)
            n = 512 // ratio
            torch.testing.assert_close(
                i.out_page_indices[0, :n].to(torch.int64),
                self.mapping[0, torch.arange(n) * ratio] // ratio,
            )
            self.assertTrue(torch.all(i.out_page_indices[0, n:] == -1))
            dense = self.dense()
            p = dense.publish_decode(i)
            dense.consume_decode(i, p)
            # Candidate source with shorter extent must not gain new blocks.
            i.positions = torch.tensor([512, 511])
            dense.consume_decode(i, p)
            self.assertTrue(
                torch.all(i.out_page_indices[1, (0 if ratio == 2 else 8) :] == -1)
            )
            i.paged_metadata.max_compressed_seq_len = 1024
            full = self.full()
            full._torch_decode = Mock()
            full.topk_decode(i)
            full._torch_decode.assert_called_once()

    def test_shortcut_has_no_device_to_host_read(self):
        for filename, names in (
            ("full_topk.py", {"_all_causal_bound", "_write_all_causal"}),
            (
                "dense_blocks.py",
                {"_publish_all_blocks", "_all_blocks_bound", "_consume_all_causal"},
            ),
        ):
            tree = ast.parse((BASE / filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in names:
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call) and isinstance(
                            child.func, ast.Attribute
                        ):
                            self.assertNotIn(
                                child.func.attr, {"item", "tolist", "cpu", "numpy"}
                            )


if __name__ == "__main__":
    unittest.main()
