import unittest

import torch

from sglang.srt.layers.attention.dsv4.dcp import (
    build_local_page_table,
    local_c4_topk_candidates,
    local_compressed_lens,
    local_swa_lens,
    localize_compressed_indices,
    localize_full_indices,
    merge_c4_topk_candidates,
    select_dcp_attn_sink,
    validate_dsv4_dcp_topology,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSV4DCPLayout(CustomTestCase):
    def test_topology_accepts_single_request_dcp_subgroups(self):
        for dcp_size in (2, 4, 8):
            for attn_tp_rank in range(8):
                validate_dsv4_dcp_topology(
                    dcp_size=dcp_size,
                    dcp_rank=attn_tp_rank % dcp_size,
                    attn_tp_size=8,
                    attn_tp_rank=attn_tp_rank,
                    attn_dp_size=1,
                )

    def test_topology_rejects_attention_dp(self):
        with self.assertRaisesRegex(NotImplementedError, "data parallel size 1"):
            validate_dsv4_dcp_topology(
                dcp_size=8,
                dcp_rank=0,
                attn_tp_size=4,
                attn_tp_rank=0,
                attn_dp_size=2,
            )

    def test_topology_rejects_unvalidated_comm_backends(self):
        for comm_backend in ("a2a", "fi_a2a"):
            with self.subTest(comm_backend=comm_backend):
                with self.assertRaisesRegex(NotImplementedError, "only the ag_rs"):
                    validate_dsv4_dcp_topology(
                        dcp_size=8,
                        dcp_rank=0,
                        attn_tp_size=8,
                        attn_tp_rank=0,
                        attn_dp_size=1,
                        comm_backend=comm_backend,
                    )

    def test_topology_rejects_disaggregation(self):
        with self.assertRaisesRegex(NotImplementedError, "disaggregated serving"):
            validate_dsv4_dcp_topology(
                dcp_size=8,
                dcp_rank=0,
                attn_tp_size=8,
                attn_tp_rank=0,
                attn_dp_size=1,
                disaggregation_mode="decode",
            )

    def test_topology_rejects_inconsistent_rank_mapping(self):
        with self.assertRaisesRegex(RuntimeError, "rank mapping"):
            validate_dsv4_dcp_topology(
                dcp_size=4,
                dcp_rank=2,
                attn_tp_size=8,
                attn_tp_rank=1,
                attn_dp_size=1,
            )

    def test_sink_slice_matches_dcp_subgroup_heads(self):
        sink = torch.arange(128, dtype=torch.float32)
        for dcp_size in (2, 4, 8):
            local_heads = sink.numel() // 8
            for tp_rank in range(8):
                dcp_rank = tp_rank % dcp_size
                got = select_dcp_attn_sink(
                    sink, local_heads, tp_rank, dcp_size, dcp_rank
                )
                group_start = tp_rank - dcp_rank
                expected = sink[
                    group_start * local_heads : (group_start + dcp_size) * local_heads
                ]
                torch.testing.assert_close(got, expected)

    def test_full_owner_and_local_rows(self):
        global_indices = torch.arange(8192, dtype=torch.int64)
        for dcp_size in (1, 2, 4, 8):
            for rank in range(dcp_size):
                got = localize_full_indices(global_indices, dcp_size, rank)
                expected_owned = global_indices % dcp_size == rank
                torch.testing.assert_close(got.owned, expected_owned)
                torch.testing.assert_close(
                    got.local[got.owned], global_indices[expected_owned] // dcp_size
                )
                self.assertTrue(torch.all(got.local[~got.owned] == -1))

    def test_compressed_owner_divides_before_ranking(self):
        global_indices = torch.arange(8 * 256, dtype=torch.int64)
        for ratio in (4, 128):
            for dcp_size in (2, 4, 8):
                compressed = global_indices // ratio
                for rank in range(dcp_size):
                    got = localize_compressed_indices(
                        global_indices, ratio, dcp_size, rank
                    )
                    expected_owned = compressed % dcp_size == rank
                    torch.testing.assert_close(got.owned, expected_owned)
                    torch.testing.assert_close(
                        got.local[got.owned], compressed[expected_owned] // dcp_size
                    )

                    full_owned = global_indices % dcp_size == rank
                    if ratio > 1:
                        self.assertFalse(torch.equal(expected_owned, full_owned))

    def test_compressed_page_number_is_shared_across_ranks(self):
        physical_page_size = 256
        pages = (0, 1, 7, 31)
        for ratio in (4, 128):
            compressed_page_size = physical_page_size // ratio
            for dcp_size in (1, 2, 4, 8):
                logical_page_size = physical_page_size * dcp_size
                for page in pages:
                    global_indices = torch.arange(
                        page * logical_page_size,
                        (page + 1) * logical_page_size,
                        dtype=torch.int64,
                    )
                    for rank in range(dcp_size):
                        localized = localize_compressed_indices(
                            global_indices, ratio, dcp_size, rank
                        )
                        local_rows = localized.local[localized.owned]
                        self.assertTrue(
                            torch.all(local_rows // compressed_page_size == page)
                        )

    def test_compressed_lengths_match_owner_count(self):
        seq_lens = torch.tensor(
            [0, 1, 3, 4, 5, 127, 128, 129, 2047, 2048, 2049],
            dtype=torch.int64,
        )
        for ratio in (4, 128):
            for dcp_size in (1, 2, 4, 8):
                for rank in range(dcp_size):
                    got = local_compressed_lens(seq_lens, ratio, dcp_size, rank)
                    expected = torch.tensor(
                        [
                            sum(
                                1
                                for index in range(int(length) // ratio)
                                if index % dcp_size == rank
                            )
                            for length in seq_lens
                        ],
                        dtype=torch.int64,
                    )
                    torch.testing.assert_close(got, expected)

    def test_swa_lengths_use_absolute_position_phase(self):
        seq_lens = torch.tensor(
            [1, 7, 127, 128, 129, 255, 256, 257, 2047, 2048, 2049],
            dtype=torch.int64,
        )
        window = 128
        for dcp_size in (1, 2, 4, 8):
            for rank in range(dcp_size):
                got = local_swa_lens(seq_lens, window, dcp_size, rank)
                expected = []
                for length in seq_lens.tolist():
                    start = max(0, length - window)
                    expected.append(
                        sum(
                            1
                            for position in range(start, length)
                            if position % dcp_size == rank
                        )
                    )
                torch.testing.assert_close(got, torch.tensor(expected))

    def test_page_table_uses_widened_full_pages(self):
        physical_page_size = 256
        max_seq_len = 5 * physical_page_size * 8
        req_to_token = torch.zeros((3, max_seq_len), dtype=torch.int64)
        for dcp_size in (1, 2, 4, 8):
            logical_page_size = physical_page_size * dcp_size
            for request in range(req_to_token.shape[0]):
                for page in range(max_seq_len // logical_page_size):
                    start = page * logical_page_size
                    req_to_token[request, start : start + logical_page_size] = (
                        request * 64 + page + 1
                    ) * logical_page_size + torch.arange(logical_page_size)
            got = build_local_page_table(
                req_to_token,
                torch.tensor([0, 2]),
                max_seq_len,
                physical_page_size,
                dcp_size,
            )
            expected = torch.stack(
                (
                    torch.arange(1, got.shape[1] + 1),
                    torch.arange(129, 129 + got.shape[1]),
                )
            ).to(torch.int32)
            torch.testing.assert_close(got, expected)

    def test_distributed_c4_topk_matches_global_selection(self):
        torch.manual_seed(7)
        batch, global_width, topk, dcp_size = 4, 137, 32, 8
        global_lens = torch.tensor([7, 32, 91, global_width], dtype=torch.int64)
        global_scores = torch.randn(batch, global_width)
        global_scores[1, 3:11] = 5.0
        global_scores[2, 80:] += 20.0

        candidate_scores = []
        candidate_ids = []
        page_tables = []
        c4_page_size = 64
        for rank in range(dcp_size):
            local_scores = global_scores[:, rank::dcp_size]
            local_lens = local_compressed_lens(global_lens * 4, 4, dcp_size, rank)
            scores, ids = local_c4_topk_candidates(
                local_scores, local_lens, topk, dcp_size, rank
            )
            candidate_scores.append(scores)
            candidate_ids.append(ids)
            page_tables.append(
                torch.arange(100, 100 + (local_scores.shape[1] + 63) // 64)
                .repeat(batch, 1)
                .to(torch.int32)
            )

        gathered_scores = torch.cat(candidate_scores, dim=1)
        gathered_ids = torch.cat(candidate_ids, dim=1)
        per_rank = [
            merge_c4_topk_candidates(
                gathered_scores,
                gathered_ids,
                topk,
                dcp_size,
                rank,
                page_tables[rank],
                c4_page_size,
            )
            for rank in range(dcp_size)
        ]

        valid = torch.arange(global_width).unsqueeze(0) < global_lens.unsqueeze(1)
        masked = global_scores.masked_fill(~valid, -float("inf"))
        expected = torch.argsort(masked, dim=1, descending=True, stable=True)[:, :topk]
        expected_valid = torch.gather(valid, 1, expected)
        expected = expected.masked_fill(~expected_valid, -1).to(torch.int32)
        torch.testing.assert_close(per_rank[0].global_indices, expected)
        for result in per_rank[1:]:
            torch.testing.assert_close(result.global_indices, expected)

        for row in range(batch):
            reconstructed = []
            for rank, result in enumerate(per_rank):
                count = int(result.local_lens[row])
                local_ids = result.local_raw_indices[row, :count].to(torch.int64)
                reconstructed.extend((local_ids * dcp_size + rank).tolist())
                page_indices = result.page_indices[row, :count]
                expected_pages = (
                    page_tables[rank][row, local_ids // c4_page_size] * c4_page_size
                    + local_ids % c4_page_size
                ).to(torch.int32)
                torch.testing.assert_close(page_indices, expected_pages)
            self.assertCountEqual(
                reconstructed,
                [value for value in expected[row].tolist() if value >= 0],
            )


if __name__ == "__main__":
    unittest.main()
