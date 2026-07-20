# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""GPU unit test for the verify-DCP pass-1 prefix block-table build.

Drives ``create_flashmla_kv_indices_triton`` exactly as the two
``_forward_verify_dcp`` metadata sites in ``trtllm_mla_backend.py`` do:

  * eager  (``init_forward_metadata``): fresh -1-filled table sized
    cdiv(cdiv(prefix_max, eff_page), npb) * npb, PAGED_SIZE = page * dcp_world;
  * replay (``_apply_cuda_graph_metadata``): one persistent full-width captured
    buffer, -1-filled ONCE, rebuilt in place across replays with
    prefix = seq_lens_k - num_draft_tokens.

and pins, against a CPU reference, the two properties the cascade relies on:

  1. every entry the pass-1 consumer can dereference — cdiv(local_prefix_r,
     page_size) LOCAL pages per rank, "local page N == global page N" — is a
     valid (>= 0) global page id equal to req_to_token[req, j*eff_page] //
     eff_page;
  2. the consumer bound never reaches the un-rewritten tail of the persistent
     replay buffer. The triton kernel only stores cdiv(prefix_r, eff_page)
     entries per row (mask_out), so after a long->short replay sequence the
     tail holds STALE page ids from the earlier, longer replay — harmless iff
     property 1's bound holds for every rank.

Shapes cover the cc16 crash regime: bs=16, mixed 50K-scale prefixes,
page_size in {32, 64}, dcp_world = 8.

Usage:
    python -m pytest test_dcp_verify_prefix_table.py -v
    python test_dcp_verify_prefix_table.py
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _local_len(global_len: int, world: int, rank: int) -> int:
    # get_dcp_lens, scalar form (owner rule pos % world == rank).
    return global_len // world + (rank < global_len % world)


def _make_req_to_token(
    bs: int, prefix_lens: list[int], eff_page: int, max_context_len: int, device
) -> tuple[torch.Tensor, list[list[int]]]:
    """Paged req_to_token mirroring the DCP allocator: each request holds
    distinct eff_page-aligned global pages; token t of req r lives at
    page_ids[r][t // eff_page] * eff_page + t % eff_page."""
    g = torch.Generator().manual_seed(1234)
    max_pages_per_req = _cdiv(max(prefix_lens), eff_page)
    total_pages = bs * max_pages_per_req + 8
    perm = torch.randperm(total_pages, generator=g).tolist()
    req_to_token = torch.zeros((bs, max_context_len), dtype=torch.int32, device=device)
    page_ids: list[list[int]] = []
    it = iter(perm)
    for r, plen in enumerate(prefix_lens):
        pages = [next(it) for _ in range(_cdiv(plen, eff_page))]
        page_ids.append(pages)
        for j, pid in enumerate(pages):
            start = j * eff_page
            width = min(eff_page, plen - start)
            req_to_token[r, start : start + width] = torch.arange(
                pid * eff_page,
                pid * eff_page + width,
                dtype=torch.int32,
                device=device,
            )
    return req_to_token, page_ids


def _build_table(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    table: torch.Tensor,
    eff_page: int,
) -> None:
    from sglang.kernels.ops.kvcache.kv_indices import (
        create_flashmla_kv_indices_triton,
        get_num_kv_index_blocks_flashmla,
    )

    bs = req_pool_indices.numel()
    create_flashmla_kv_indices_triton[
        (bs, get_num_kv_index_blocks_flashmla(table.shape[1], eff_page))
    ](
        req_to_token,
        req_pool_indices,
        prefix_lens,
        None,
        table,
        req_to_token.stride(0),
        table.shape[1],
        PAGED_SIZE=eff_page,
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA (triton kernel)")
class TestDcpVerifyPrefixTable(unittest.TestCase):
    DCP_WORLD = 8
    BS = 16
    NUM_DRAFT_TOKENS = 8
    # Mixed cc16-regime prefixes, including eff_page-boundary +-1 neighbors
    # (page 64 * dcp 8 -> eff_page 512; 50176 = 98 * 512).
    PREFIX_LENS = [
        50000,
        49999,
        50007,
        50176,
        50175,
        50177,
        3000,
        3001,
        511,
        512,
        513,
        1,
        65535,
        65536,
        2048,
        50000,
    ]

    def _run_site(self, page_size: int, persistent: bool):
        from sglang.kernels.ops.kvcache.kv_indices import (
            get_num_page_per_block_flashmla,
        )

        device = torch.device("cuda")
        world = self.DCP_WORLD
        eff_page = page_size * world
        npb = get_num_page_per_block_flashmla(eff_page)
        prefix_lens = self.PREFIX_LENS
        max_context_len = max(prefix_lens) + self.NUM_DRAFT_TOKENS + eff_page

        req_to_token, page_ids = _make_req_to_token(
            self.BS, prefix_lens, eff_page, max_context_len, device
        )
        req_pool_indices = torch.arange(self.BS, dtype=torch.int64, device=device)
        prefix_t = torch.tensor(prefix_lens, dtype=torch.int32, device=device)

        if persistent:
            # Replay site: full-width captured buffer, -1-filled once, then
            # dirtied by a longer "previous replay" before the build under test.
            width = _cdiv(_cdiv(max_context_len, eff_page), npb) * npb
            table = torch.full((self.BS, width), -1, dtype=torch.int32, device=device)
            longer = torch.clamp(prefix_t + 3 * eff_page, max=max_context_len).to(
                torch.int32
            )
            _build_table(req_to_token, req_pool_indices, longer, table, eff_page)
            # Replay computes prefix as seq_lens_k - T.
            seq_lens_k = prefix_t + self.NUM_DRAFT_TOKENS
            rebuilt_prefix = (seq_lens_k - self.NUM_DRAFT_TOKENS).to(torch.int32)
            _build_table(
                req_to_token, req_pool_indices, rebuilt_prefix, table, eff_page
            )
        else:
            # Eager site: fresh -1 table sized from the batch prefix max.
            prefix_max = max(prefix_lens)
            width = _cdiv(_cdiv(prefix_max, eff_page), npb) * npb
            table = torch.full((self.BS, width), -1, dtype=torch.int32, device=device)
            _build_table(req_to_token, req_pool_indices, prefix_t, table, eff_page)

        torch.cuda.synchronize()
        table_cpu = table.cpu()

        for r, plen in enumerate(prefix_lens):
            n_valid = _cdiv(plen, eff_page)
            self.assertLessEqual(n_valid, table.shape[1], f"row {r} wider than table")
            expect = torch.tensor(page_ids[r], dtype=torch.int32)
            got = table_cpu[r, :n_valid]
            self.assertTrue(
                torch.equal(got, expect),
                f"row {r} (prefix={plen}): table {got.tolist()[:8]}... != "
                f"expected {expect.tolist()[:8]}...",
            )
            self.assertTrue(
                (got >= 0).all(), f"row {r}: -1 sentinel inside valid region"
            )
            # The pass-1 consumer bound on every rank stays inside the
            # freshly-rewritten region (never touches -1 fill / stale tail).
            for rank in range(world):
                pages_read = _cdiv(_local_len(plen, world, rank), page_size)
                self.assertLessEqual(
                    pages_read,
                    n_valid,
                    f"row {r} rank {rank}: consumer reads {pages_read} pages, "
                    f"only {n_valid} valid",
                )
        return table_cpu

    def test_eager_site_page64(self):
        self._run_site(page_size=64, persistent=False)

    def test_eager_site_page32(self):
        self._run_site(page_size=32, persistent=False)

    def test_replay_site_page64_stale_tail_never_read(self):
        self._run_site(page_size=64, persistent=True)

    def test_replay_site_page32_stale_tail_never_read(self):
        self._run_site(page_size=32, persistent=True)

    def test_eager_and_replay_agree_on_consumer_region(self):
        eager = self._run_site(page_size=64, persistent=False)
        replay = self._run_site(page_size=64, persistent=True)
        eff_page = 64 * self.DCP_WORLD
        for r, plen in enumerate(self.PREFIX_LENS):
            n_valid = _cdiv(plen, eff_page)
            self.assertTrue(
                torch.equal(eager[r, :n_valid], replay[r, :n_valid]),
                f"row {r}: eager vs replay consumer-region mismatch",
            )


if __name__ == "__main__":
    unittest.main()
