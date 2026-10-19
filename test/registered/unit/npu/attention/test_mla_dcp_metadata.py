"""CPU regressions for dense MLA DCP causal masks and empty query rows.

Load the real Python functions without importing device extensions. Only the
page-table kernel is substituted; its device implementation is outside these
metadata and softmax-identity tests.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "python").is_dir()
)
_PYTHON = _ROOT / "python"


def _load_functions(relative_path, names, namespace=None):
    """Execute unmodified function definitions, omitting device-only imports."""
    path = _PYTHON / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != set(names):
        raise AssertionError(f"Missing functions in {path}: {names}")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *functions,
        ],
        type_ignores=[],
    )
    namespace = {} if namespace is None else namespace
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


# Keep the repository's CPU CI marker without importing the top-level serving
# package and its optional accelerator dependencies.
register_cpu_ci = _load_functions("sglang/test/ci/ci_register.py", {"register_cpu_ci"})[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Mode:
    def __init__(self, verify):
        self.verify = verify

    def is_target_verify(self):
        return self.verify

    def is_decode_or_idle(self):
        return not self.verify


class TestMlaDcpMetadata(unittest.TestCase):
    def setUp(self):
        self.parallel = SimpleNamespace(dcp_size=2, dcp_rank=1)
        self.ns = {"torch": torch, "get_parallel": lambda: self.parallel}
        _load_functions(
            "sglang/srt/layers/dcp/layout.py",
            {"get_dcp_lens", "get_dcp_chain_spec_lens"},
            self.ns,
        )
        _load_functions(
            "sglang/srt/hardware_backend/npu/attention/dcp_metadata.py",
            {
                "build_mla_dcp_mtp_mask",
                "init_mla_dcp_metadata",
                "init_mla_dcp_graph_state",
                "update_mla_dcp_graph_metadata",
            },
            self.ns,
        )
        _load_functions(
            "sglang/srt/hardware_backend/npu/attention/dcp.py",
            {"mask_empty_mla_dcp_shards_npu"},
            self.ns,
        )
        self.ns["build_mla_dcp_local_block_tables"] = self._fake_block_tables

    def _fake_block_tables(
        self,
        req_to_token,
        req_pool_indices,
        seq_lens,
        physical_page_size,
        dcp_size,
        dcp_rank,
        *,
        num_pages=None,
    ):
        local = self.ns["get_dcp_lens"](seq_lens, dcp_size, dcp_rank).int()
        if num_pages is None:
            length = int(local.max()) if local.numel() else 0
            num_pages = max(1, (length + physical_page_size - 1) // physical_page_size)
        pages = local[:, None] * 10 + torch.arange(num_pages, dtype=torch.int32)
        return pages, local

    @staticmethod
    def _expected_mask(prefixes, query_len, size, rank, width):
        # Independent oracle: enumerate causal global tokens owned by this rank.
        expected = torch.ones(len(prefixes), query_len, width, dtype=torch.bool)
        lengths = []
        for request, prefix in enumerate(prefixes):
            for query in range(query_len):
                visible = [
                    token for token in range(prefix + query + 1) if token % size == rank
                ]
                lengths.append(len(visible))
                expected[request, query, : min(width, len(visible))] = False
        return expected, torch.tensor(lengths, dtype=torch.int32)

    def test_masks_match_causal_token_ownership(self):
        for prefixes, query_len, size, rank in (
            ([0], 3, 2, 1),
            ([1], 3, 4, 3),
            ([0, 5], 3, 2, 1),
            ([2, 9], 4, 4, 2),
            ([0, 3], 3, 1, 0),
        ):
            with self.subTest(prefixes=prefixes, size=size, rank=rank):
                width = max(prefixes) + query_len + 2
                expected, lengths = self._expected_mask(
                    prefixes, query_len, size, rank, width
                )
                prefix_tensor = torch.tensor(prefixes, dtype=torch.int64)
                for supplied_lengths in (None, lengths):
                    mask = self.ns["build_mla_dcp_mtp_mask"](
                        prefix_tensor,
                        query_len,
                        size,
                        rank,
                        max_local_kv_len=width,
                        local_query_lens=supplied_lengths,
                    )
                    self.assertTrue(torch.equal(mask, expected))
                    self.assertEqual(mask.dtype, torch.bool)
                    self.assertTrue(mask.is_contiguous())

    def test_zero_width_and_zero_query_remain_fully_masked(self):
        for query_len, width in ((0, 4), (3, 0)):
            with self.subTest(query_len=query_len, width=width):
                mask = self.ns["build_mla_dcp_mtp_mask"](
                    torch.tensor([0, 3]),
                    query_len,
                    2,
                    1,
                    max_local_kv_len=width,
                )
                self.assertEqual(mask.shape, (2, max(1, query_len), max(1, width)))
                self.assertTrue(bool(mask.all()))

    def test_eager_verify_stores_per_query_lengths_and_padded_mask(self):
        metadata = SimpleNamespace(dcp_mtp_attn_mask=None, dcp_spec_seq_lens=None)
        batch = SimpleNamespace(
            forward_mode=_Mode(True),
            spec_info=SimpleNamespace(draft_token_num=3),
            seq_lens=torch.tensor([0, 5]),
            req_pool_indices=torch.tensor([0, 1]),
        )
        local = self.ns["init_mla_dcp_metadata"](
            metadata, batch, torch.zeros(2, 16), 2, 0
        )
        self.assertEqual(local.tolist(), [1, 4])
        expected, lengths = self._expected_mask([0, 5], 3, 2, 1, 4)
        self.assertTrue(torch.equal(metadata.dcp_spec_seq_lens, lengths))
        self.assertTrue(torch.equal(metadata.dcp_mtp_attn_mask, expected))

    def test_eager_decode_uses_request_lengths(self):
        metadata = SimpleNamespace(dcp_mtp_attn_mask=None, dcp_spec_seq_lens=None)
        batch = SimpleNamespace(
            forward_mode=_Mode(False),
            spec_info=None,
            seq_lens=torch.tensor([1, 6]),
            req_pool_indices=torch.tensor([0, 1]),
        )
        local = self.ns["init_mla_dcp_metadata"](
            metadata, batch, torch.zeros(2, 16), 2, 0
        )
        self.assertEqual(local.tolist(), [0, 3])
        self.assertIsNone(metadata.dcp_mtp_attn_mask)

    def test_eager_decode_clears_previous_verify_metadata(self):
        metadata = SimpleNamespace(dcp_mtp_attn_mask=None, dcp_spec_seq_lens=None)
        batch = SimpleNamespace(
            forward_mode=_Mode(True),
            spec_info=SimpleNamespace(draft_token_num=3),
            seq_lens=torch.tensor([0]),
            req_pool_indices=torch.tensor([0]),
        )
        initialize = self.ns["init_mla_dcp_metadata"]
        initialize(metadata, batch, torch.zeros(1, 16), 2, 0)
        self.assertEqual(metadata.dcp_spec_seq_lens.tolist(), [0, 1, 1])
        self.assertIsNotNone(metadata.dcp_mtp_attn_mask)

        batch.forward_mode = _Mode(False)
        batch.spec_info = None
        batch.seq_lens = torch.tensor([4])
        local = initialize(metadata, batch, torch.zeros(1, 16), 2, 0)
        self.assertEqual(local.tolist(), [2])
        self.assertIsNone(metadata.dcp_spec_seq_lens)
        self.assertIsNone(metadata.dcp_mtp_attn_mask)

    def test_empty_queries_replace_nan_without_changing_valid_rows(self):
        lengths = torch.tensor([0, 1, 1, 0, 0, 1], dtype=torch.int32)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                output = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2)
                output = output.to(dtype)
                lse = torch.arange(12, dtype=torch.float32).reshape(6, 2, 1)
                empty = lengths == 0
                output[empty] = float("nan")
                lse[empty] = float("nan")
                normalized, normalized_lse = self.ns["mask_empty_mla_dcp_shards_npu"](
                    output, lse, lengths
                )
                self.assertEqual(normalized.dtype, dtype)
                self.assertTrue(torch.equal(normalized[~empty], output[~empty]))
                self.assertTrue(torch.equal(normalized_lse[~empty], lse[~empty]))
                self.assertTrue(bool((normalized[empty] == 0).all()))
                self.assertTrue(bool(torch.isneginf(normalized_lse[empty]).all()))

    def test_decode_empty_request_and_legacy_request_broadcast(self):
        normalize = self.ns["mask_empty_mla_dcp_shards_npu"]
        for rows, lengths in ((2, [0, 2]), (6, [0, 2])):
            with self.subTest(rows=rows):
                output = torch.ones(rows, 2, 3)
                lse = torch.zeros(rows, 2, 1)
                result, result_lse = normalize(output, lse, torch.tensor(lengths))
                empty_rows = rows // 2
                self.assertTrue(bool((result[:empty_rows] == 0).all()))
                self.assertTrue(bool(torch.isneginf(result_lse[:empty_rows]).all()))
                self.assertTrue(torch.equal(result[empty_rows:], output[empty_rows:]))

    def test_normalized_partials_merge_to_unsharded_causal_attention(self):
        generator = torch.Generator().manual_seed(7)
        q = torch.randn(3, 2, 4, generator=generator, dtype=torch.float64)
        k = torch.randn(3, 4, generator=generator, dtype=torch.float64)
        v = torch.randn(3, 5, generator=generator, dtype=torch.float64)
        scores = torch.einsum("qhd,kd->qhk", q, k) / 2
        causal = torch.arange(3)[None, :] <= torch.arange(3)[:, None]
        scores = scores.masked_fill(~causal[:, None, :], float("-inf"))
        expected = torch.einsum("qhk,kd->qhd", scores.softmax(-1), v)
        outputs, lses = [], []
        for rank in range(2):
            owned = torch.arange(rank, 3, 2)
            local_scores = scores[:, :, owned]
            local_lse = local_scores.logsumexp(-1, keepdim=True)
            local_output = torch.einsum(
                "qhk,kd->qhd", local_scores.softmax(-1), v[owned]
            )
            lengths = torch.tensor(
                [sum(int(token <= query) for token in owned) for query in range(3)]
            )
            # Model an operator with undefined output/statistics for an all-mask row.
            local_output[lengths == 0] = float("nan")
            local_lse[lengths == 0] = float("nan")
            local_output, local_lse = self.ns["mask_empty_mla_dcp_shards_npu"](
                local_output, local_lse, lengths
            )
            outputs.append(local_output)
            lses.append(local_lse)
        stacked_lse = torch.stack(lses)
        weights = (stacked_lse - stacked_lse.logsumexp(0)).exp()
        actual = (torch.stack(outputs) * weights).sum(0)
        torch.testing.assert_close(actual, expected)

    def test_graph_updates_frontiers_and_masks_without_replacing_buffers(self):
        buffers = self.ns["init_mla_dcp_graph_state"](2, 32, 2, 3, torch.device("cpu"))
        metadata = SimpleNamespace(
            block_tables=buffers["block_tables"],
            seq_lens=buffers["dcp_seq_lens"],
            dcp_spec_seq_lens=buffers["dcp_spec_seq_lens"],
            dcp_mtp_attn_mask=buffers["dcp_mtp_attn_mask"],
        )
        tensor_fields = (
            "block_tables",
            "seq_lens",
            "dcp_spec_seq_lens",
            "dcp_mtp_attn_mask",
        )
        addresses = {name: getattr(metadata, name).data_ptr() for name in tensor_fields}
        for prefixes, max_len, in_capture in (([0, 9], 12, True), ([3, 0], 6, False)):
            with self.subTest(prefixes=prefixes, in_capture=in_capture):
                self.ns["update_mla_dcp_graph_metadata"](
                    metadata,
                    torch.zeros(2, 32),
                    bs=2,
                    req_pool_indices=torch.tensor([0, 1]),
                    seq_lens=torch.tensor(prefixes),
                    forward_mode=_Mode(True),
                    spec_info=SimpleNamespace(draft_token_num=3),
                    max_len=max_len,
                    page_size=2,
                    speculative_num_draft_tokens=3,
                    speculative_step_id=0,
                    in_capture=in_capture,
                )
                width = metadata.dcp_mtp_attn_mask.shape[-1]
                expected, lengths = self._expected_mask(prefixes, 3, 2, 1, width)
                self.assertTrue(torch.equal(metadata.dcp_spec_seq_lens, lengths))
                self.assertTrue(torch.equal(metadata.dcp_mtp_attn_mask, expected))
                self.assertEqual(
                    metadata.seq_lens.tolist(), lengths.reshape(2, 3)[:, -1].tolist()
                )
                for name in tensor_fields:
                    self.assertEqual(
                        getattr(metadata, name).data_ptr(), addresses[name]
                    )
        # The replay uses two active pages and clears pages left over from capture.
        self.assertTrue(bool((metadata.block_tables[:, 2:] == 0).all()))


if __name__ == "__main__":
    unittest.main()
