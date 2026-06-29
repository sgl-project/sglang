"""Equivalence tests for the optimized Quest sparse attention algorithm.

These tests lock the behavior of the vectorized ``retrieve_topk`` path and the
full-page fast path in ``_compute_page_representations`` against the original
per-request reference implementations. They run on CPU and do not start a
server or call a real attention backend.
"""

import unittest

import torch

from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _Config:
    def __init__(self, page_size, sparsity_ratio, num_recent_pages):
        self.page_size = page_size
        self.sparse_extra_config = {
            "sparsity_ratio": sparsity_ratio,
            "num_recent_pages": num_recent_pages,
        }


class _FakeReqToTokenPool:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token
        self.max_context_len = req_to_token.shape[1]


class _FakeTokenToKVPool:
    def __init__(self, key_buffer):
        self.key_buffer = key_buffer

    def get_key_buffer(self, layer_id):
        return self.key_buffer


class _FakeStates:
    def __init__(self, size, device):
        self.repr_constructed = torch.zeros(size, dtype=torch.bool, device=device)
        self.prompt_lens = torch.zeros(size, dtype=torch.int64, device=device)
        self.last_constructed_page = torch.zeros(size, dtype=torch.int64, device=device)


class _FakeForwardBatch:
    def __init__(self, seq_lens):
        self.seq_lens = seq_lens


def _build_req_to_token(batch_size, seq_lens, page_size, device):
    """Contiguous KV layout: request r owns a block of aligned token ids."""
    max_seq = int(seq_lens.max().item())
    tokens_per_req = ((max_seq + page_size - 1) // page_size) * page_size
    req_to_token = torch.zeros((batch_size, max_seq), dtype=torch.int32, device=device)
    positions = torch.arange(max_seq, dtype=torch.int32, device=device)
    for r in range(batch_size):
        req_to_token[r] = r * tokens_per_req + positions
    total_tokens = batch_size * tokens_per_req
    return req_to_token, total_tokens


def _make_algorithm(
    batch_size,
    seq_lens,
    page_size,
    sparsity_ratio,
    num_recent_pages,
    kv_heads,
    head_dim,
    device,
    seed,
):
    torch.manual_seed(seed)
    req_to_token, total_tokens = _build_req_to_token(
        batch_size, seq_lens, page_size, device
    )
    k_buffer = torch.randn(
        total_tokens, kv_heads, head_dim, dtype=torch.float32, device=device
    )
    config = _Config(page_size, sparsity_ratio, num_recent_pages)
    algo = QuestAlgorithm(config, device)
    algo.initialize_representation_pool(
        start_layer=0,
        end_layer=1,
        token_to_kv_pool=_FakeTokenToKVPool(k_buffer),
        req_to_token_pool=_FakeReqToTokenPool(req_to_token),
        states=_FakeStates(batch_size, device),
    )
    return algo, k_buffer


def _populate_page_reps(algo, batch_size, seq_lens, k_buffer, device):
    """Build page representations for all complete pages of every request."""
    req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
    end_pages = seq_lens // algo.page_size
    algo._compute_page_representations(
        0,
        req_pool_indices,
        seq_lens,
        0,
        end_pages,
        k_buffer,
    )


def _reference_retrieve_topk(
    algo, queries, layer_id, req_pool_indices, sparse_mask, forward_batch
):
    """Original per-request loop implementation (pre-optimization)."""
    bs, device = queries.shape[0], queries.device
    seq_lens = forward_batch.seq_lens.to(device)
    req_to_token = algo.req_to_token_pool.req_to_token
    max_req_tokens = req_to_token.shape[1]

    per_request_indices = []
    per_request_lengths = []
    for i in range(bs):
        if not sparse_mask[i]:
            per_request_indices.append(torch.empty(0, device=device, dtype=torch.int32))
            per_request_lengths.append(0)
            continue
        num_pages = int((seq_lens[i].item() + algo.page_size - 1) // algo.page_size)
        if num_pages <= algo.num_recent_pages:
            per_request_indices.append(torch.empty(0, device=device, dtype=torch.int32))
            per_request_lengths.append(0)
            continue

        page_idx = torch.arange(num_pages, device=device)
        page_start_token = req_to_token[
            req_pool_indices[i],
            (page_idx * algo.page_size).clamp(0, max_req_tokens - 1),
        ]
        phys_pages = (page_start_token // algo.page_size).unsqueeze(0)
        scores = algo._retrieve_page_scores(
            layer_id, phys_pages, req_pool_indices[i : i + 1], queries[i : i + 1]
        )
        recent_start = max(num_pages - algo.num_recent_pages, 0)
        scores = scores.clone()
        scores[:, recent_start:] = float("-inf")
        history_pages = max(recent_start, 1)
        k = max(int(history_pages * algo.sparsity_ratio), 1)
        k = min(k, history_pages)
        topk_idx = torch.topk(scores, k=k, dim=1, sorted=False)[1].squeeze(0)
        recent_idx = torch.arange(
            recent_start, recent_start + algo.num_recent_pages, device=device
        )
        recent_idx = recent_idx[recent_idx < num_pages]
        combined = torch.cat([topk_idx, recent_idx], dim=0).sort()[0].to(torch.int32)
        per_request_indices.append(combined)
        per_request_lengths.append(int(combined.numel()))

    return per_request_indices, per_request_lengths


def _sorted_rows(indices, lengths):
    """Per-request sorted page lists, truncated to each row's valid length.

    Works for both the optimized output (a padded ``[bs, max]`` tensor with a
    tensor of lengths) and the reference output (a list of variable-length
    tensors with a list of lengths).
    """
    rows = []
    for i in range(len(lengths)):
        length = int(lengths[i])
        rows.append(sorted(indices[i][:length].tolist()))
    return rows


class TestQuestRetrieveTopkEquivalence(CustomTestCase):
    device = torch.device("cpu")

    def _run_case(
        self,
        seq_lens_list,
        page_size=16,
        sparsity_ratio=0.5,
        num_recent_pages=4,
        kv_heads=2,
        q_heads=4,
        head_dim=8,
        sparse_mask_list=None,
        seed=0,
    ):
        device = self.device
        batch_size = len(seq_lens_list)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64, device=device)
        algo, k_buffer = _make_algorithm(
            batch_size,
            seq_lens,
            page_size,
            sparsity_ratio,
            num_recent_pages,
            kv_heads,
            head_dim,
            device,
            seed,
        )
        _populate_page_reps(algo, batch_size, seq_lens, k_buffer, device)

        req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
        queries = torch.randn(
            batch_size, q_heads, head_dim, dtype=torch.float32, device=device
        )
        if sparse_mask_list is None:
            sparse_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            sparse_mask = torch.tensor(
                sparse_mask_list, dtype=torch.bool, device=device
            )
        forward_batch = _FakeForwardBatch(seq_lens)

        opt_indices, opt_lengths = algo.retrieve_topk(
            queries,
            0,
            req_pool_indices,
            sparse_mask,
            forward_batch=forward_batch,
        )
        ref_indices, ref_lengths = _reference_retrieve_topk(
            algo, queries, 0, req_pool_indices, sparse_mask, forward_batch
        )

        opt_rows = _sorted_rows(opt_indices, opt_lengths)
        ref_rows = _sorted_rows(ref_indices, ref_lengths)
        self.assertEqual(
            opt_lengths.tolist(),
            [len(r) for r in ref_rows],
            msg=f"length mismatch for seq_lens={seq_lens_list}",
        )
        self.assertEqual(
            opt_rows,
            ref_rows,
            msg=f"selected pages mismatch for seq_lens={seq_lens_list}",
        )

    def test_uniform_aligned(self):
        self._run_case([512, 512, 512, 512])

    def test_ragged_unaligned(self):
        self._run_case([511, 333, 257, 129])

    def test_mixed_sparse_mask(self):
        self._run_case(
            [512, 480, 400, 320],
            sparse_mask_list=[True, False, True, False],
        )

    def test_short_context_no_sparsity(self):
        # All requests have num_pages <= num_recent_pages -> empty selection.
        self._run_case([32, 48, 16, 64], page_size=16, num_recent_pages=4)

    def test_mixed_short_and_long(self):
        self._run_case([48, 512, 32, 400], page_size=16, num_recent_pages=4)

    def test_gqa_grouped_heads(self):
        self._run_case([512, 448], kv_heads=2, q_heads=8, head_dim=8)

    def test_batch_size_one(self):
        self._run_case([777])

    def test_different_page_size(self):
        self._run_case([1024, 800, 640], page_size=32)


def _reference_compute_page_reps_masked(
    algo, layer_id, reqs, seq_lens, end_page, k_buffer
):
    """Original masked min/max page representation (always-fallback path)."""
    device = k_buffer.device
    req_to_token = algo.req_to_token_pool.req_to_token
    n = reqs.shape[0]
    start_page = torch.zeros_like(end_page)
    max_pages = int((end_page - start_page).max().item())
    if max_pages <= 0:
        return

    pg_off = torch.arange(max_pages, device=device).unsqueeze(0)
    pg_id = start_page.unsqueeze(1) + pg_off
    pg_mask = pg_id < end_page.unsqueeze(1)

    tok_start = pg_id * algo.page_size
    tok_off = torch.arange(algo.page_size, device=device).view(1, 1, -1)
    tok_pos = tok_start.unsqueeze(2) + tok_off
    tok_mask = (
        tok_pos
        < (tok_start + algo.page_size).clamp(max=seq_lens.unsqueeze(1)).unsqueeze(2)
    ) & pg_mask.unsqueeze(2)

    phys_tok = req_to_token[
        reqs.view(n, 1, 1).expand(n, max_pages, algo.page_size),
        tok_pos.clamp(0, req_to_token.shape[1] - 1),
    ].clamp(0, k_buffer.shape[0] - 1)
    keys = k_buffer[phys_tok].to(torch.float32)
    mask = tok_mask.unsqueeze(-1).unsqueeze(-1)
    page_min = torch.where(mask, keys, torch.full_like(keys, float("inf"))).amin(dim=2)
    page_max = torch.where(mask, keys, torch.full_like(keys, float("-inf"))).amax(dim=2)

    phys_pg = (
        req_to_token[
            reqs.unsqueeze(1).expand(n, max_pages),
            tok_start.clamp(0, req_to_token.shape[1] - 1),
        ]
        // algo.page_size
    )
    idx = pg_mask.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return
    target_pages = phys_pg[idx[:, 0], idx[:, 1]].clamp(
        0, algo.page_k_min[layer_id].shape[0] - 1
    )
    algo.page_k_min[layer_id][target_pages] = page_min[idx[:, 0], idx[:, 1]]
    algo.page_k_max[layer_id][target_pages] = page_max[idx[:, 0], idx[:, 1]]
    algo.page_valid[layer_id][target_pages] = True


class TestQuestPageRepresentationEquivalence(CustomTestCase):
    device = torch.device("cpu")

    def _run_case(
        self,
        seq_lens_list,
        page_size=16,
        kv_heads=2,
        head_dim=8,
        seed=0,
        end_page_mode="floor",
    ):
        """Compare fast-path vs masked-fallback page representations.

        ``end_page_mode`` controls which branch of
        ``_compute_page_representations`` the new implementation takes:

        - ``"floor"``: ``end_page = seq_lens // page_size`` (only complete
          pages). This is what production callers
          (``construct_representations`` / ``update_representations``) always
          pass, so ``end_page * page_size <= seq_lens`` holds and the new code
          takes the full-page fast path.
        - ``"ceil"``: ``end_page = ceil(seq_lens / page_size)``. The last page
          of an unaligned request is partial, so the new code is forced down
          the masked fallback branch. Production never does this today, but the
          branch exists for safety and we lock its equivalence here.
        """
        device = self.device
        batch_size = len(seq_lens_list)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64, device=device)

        algo_new, k_buffer = _make_algorithm(
            batch_size,
            seq_lens,
            page_size,
            0.5,
            4,
            kv_heads,
            head_dim,
            device,
            seed,
        )
        algo_ref, _ = _make_algorithm(
            batch_size,
            seq_lens,
            page_size,
            0.5,
            4,
            kv_heads,
            head_dim,
            device,
            seed,
        )

        req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
        if end_page_mode == "floor":
            end_pages = seq_lens // page_size
        elif end_page_mode == "ceil":
            end_pages = (seq_lens + page_size - 1) // page_size
        else:
            raise ValueError(f"Unknown end_page_mode: {end_page_mode}")

        algo_new._compute_page_representations(
            0, req_pool_indices, seq_lens, 0, end_pages, k_buffer
        )
        _reference_compute_page_reps_masked(
            algo_ref, 0, req_pool_indices, seq_lens, end_pages, k_buffer
        )

        valid_new = algo_new.page_valid[0]
        valid_ref = algo_ref.page_valid[0]
        self.assertTrue(torch.equal(valid_new, valid_ref))
        torch.testing.assert_close(
            algo_new.page_k_min[0][valid_new],
            algo_ref.page_k_min[0][valid_ref],
        )
        torch.testing.assert_close(
            algo_new.page_k_max[0][valid_new],
            algo_ref.page_k_max[0][valid_ref],
        )

    def test_fast_path_aligned(self):
        # seq_len divisible by page_size -> all pages full -> fast path.
        self._run_case([512, 256, 384], end_page_mode="floor")

    def test_fast_path_production_floor(self):
        # Unaligned seq_lens but floor end_page (production semantics) still
        # only builds complete pages, so the fast path is taken.
        self._run_case([500, 300, 257], end_page_mode="floor")

    def test_fallback_unaligned(self):
        # Force a partial last page via ceil end_page -> masked fallback path.
        self._run_case([500, 300, 257], end_page_mode="ceil")

    def test_fallback_mixed_alignment(self):
        # Some requests aligned, some not, with ceil end_page -> fallback.
        self._run_case([512, 257, 384, 333], end_page_mode="ceil")


if __name__ == "__main__":
    unittest.main()
