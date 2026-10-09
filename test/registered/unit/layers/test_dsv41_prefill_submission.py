"""Request geometry parity and CUDA submission without host synchronization."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
    prefill_requests,
    write_prefill,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.types import PrefillInputs, Selection
from sglang.srt.layers.engram import EngramHasher
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4IndexerPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils.common import async_d2h
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Indexer:
    index_topk = 3

    def queries(self, q_lora, freqs):
        return q_lora[:, None, :]

    def head_weights(self, x):
        return x[:, :1]

    def scores(self, q, k, weights):
        return (q[:, 0] @ k.float().T).relu() * weights


def _fixture(*, ratio, rows, seq_lens, padding=0, device="cpu", cpu_slots=True):
    generator = torch.Generator().manual_seed(17)
    slots = [5, 2, 7]
    positions, req_rows = [], []
    for slot, count, seq_len in zip(slots, rows, seq_lens):
        positions.extend(range(seq_len - count, seq_len))
        req_rows.extend([slot] * count)
    positions.extend([0] * padding)
    req_rows.extend([0] * padding)
    n = len(positions)
    pool = DeepSeekV4IndexerPool(
        size=256,
        page_size=16,
        dtype=torch.bfloat16,
        index_head_dim=128,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
        use_fp4_indexer=True,
    )
    buf = pool.index_k_with_scale_buffer[0]
    packed = torch.randint(0, 256, buf.shape, dtype=torch.uint8, generator=generator)
    packed[:, 16 * 64 :] = 127  # unit ue8m0 scales
    buf.copy_(packed.to(device))
    inputs = PrefillInputs(
        indexer=_Indexer(),
        layer_id=0,
        compress_ratio=ratio,
        freqs_cis=torch.zeros((max(seq_lens) + 1, 1), device=device),
        x=torch.rand((n, 1), generator=generator).to(device),
        q_lora=torch.randn((n, 128), generator=generator).to(device),
        positions=torch.tensor(positions, device=device),
        req_rows=torch.tensor(req_rows, device=device),
        req_pool_indices=torch.tensor(slots, device=device),
        req_pool_indices_cpu=slots if cpu_slots else None,
        kv_page_table=torch.empty((3, 0), device=device),
        seq_lens_cpu=seq_lens,
        rows_per_request=rows,
        rows_per_request_device=None,
    )
    req_to_token = (
        torch.arange(9, device=device)[:, None] * 32
        + torch.arange(32, device=device)[None, :]
    )
    out = Selection(
        torch.empty((n, 3), dtype=torch.int32, device=device),
        torch.empty((n, 3), dtype=torch.int32, device=device),
    )
    wrapper = SimpleNamespace(get_low_ratio_index_k_dequant=pool.get_index_k_dequant)
    return inputs, out, wrapper, req_to_token


def _select(inputs, out, pool, req_to_token):
    for request, chunks in prefill_requests(
        inputs=inputs, out=out, token_to_kv_pool=pool, req_to_token=req_to_token
    ):
        for chunk in chunks:
            idx = chunk.scores.topk(request.k, dim=-1).indices
            write_prefill(out, request, chunk, idx)
    return out


def _reference(inputs, pool, req_to_token):
    """Independent per-query oracle; padding has no request owner."""
    pages = torch.full((inputs.x.shape[0], 3), -1, dtype=torch.int32)
    raw = torch.full_like(pages, -1)
    offset = 0
    for request_slot, count, seq_len in zip(
        inputs.req_pool_indices.tolist(), inputs.rows_per_request, inputs.seq_lens_cpu
    ):
        capacity = seq_len // inputs.compress_ratio
        if capacity:
            columns = torch.arange(capacity)
            slots = req_to_token[request_slot, columns * inputs.compress_ratio]
            slots = slots.long() // inputs.compress_ratio
            k = pool.get_low_ratio_index_k_dequant(0, slots).float()
            width = min(3, capacity)
            for row in range(offset, offset + count):
                visible = (int(inputs.positions[row]) + 1) // inputs.compress_ratio
                scores = (k @ inputs.q_lora[row]).relu() * inputs.x[row, 0]
                scores[visible:] = -torch.inf
                picks = scores.topk(width).indices.sort().values
                pages[row, :width] = torch.where(
                    picks < visible, slots[picks], -1
                ).int()
                raw[row, :width] = torch.where(picks < visible, picks, -1).int()
        offset += count
    return pages, raw


def _hasher(device):
    hasher = EngramHasher.__new__(EngramHasher)
    torch.nn.Module.__init__(hasher)
    hasher.max_ngram_size, hasher.pad_id, hasher.image_token_id = 3, 0, None
    for name, value in (
        ("token_map", torch.arange(512)),
        ("multipliers", torch.tensor([[3, 5, 7]])),
        ("primes", torch.tensor([[[37], [41]]])),
        ("offsets", torch.tensor([[0, 37]])),
    ):
        hasher.register_buffer(name, value.to(device))
    hasher.init_history(8, device)
    return hasher


def _engram_batch(device):
    return SimpleNamespace(
        forward_mode=ForwardMode.SPLIT_PREFILL,
        req_pool_indices=torch.tensor([5, 2, 7], device=device),
        extend_seq_lens=torch.tensor([2, 0, 3], device=device),
        extend_seq_lens_cpu=[2, 0, 3],
        extend_start_loc=torch.tensor([0, 2, 2], device=device),
        positions=torch.tensor([0, 1, 0, 1, 2, 0, 0], device=device),
        engram_history=None,
        out_cache_loc=torch.tensor([1, 2, 3, 4, 5, 0, 0], device=device),
    )


class TestDSV41PrefillSubmission(unittest.TestCase):
    def test_request_scoring_matches_per_query_oracle(self):
        for ratio in (1, 2):
            for rows, seq_lens in (([1, 0, 4], [1, 3, 19]), ([2, 1, 3], [9, 4, 15])):
                for cpu_slots in (True, False):
                    with self.subTest(ratio=ratio, rows=rows, cpu_slots=cpu_slots):
                        args = _fixture(
                            ratio=ratio,
                            rows=rows,
                            seq_lens=seq_lens,
                            padding=5,
                            cpu_slots=cpu_slots,
                        )
                        inputs, out, pool, req_to_token = args
                        expected_pages, expected_raw = _reference(
                            inputs, pool, req_to_token
                        )
                        _select(*args)
                        torch.testing.assert_close(out.page_indices, expected_pages)
                        torch.testing.assert_close(out.raw_indices, expected_raw)

    def test_engram_padding_does_not_commit_zero_length_request(self):
        hasher = _hasher("cpu")
        hasher.history[2] = torch.tensor([51, 52])
        result = hasher(torch.arange(1, 8), _engram_batch("cpu"))
        self.assertEqual(result.shape, (7, 1, 2))
        torch.testing.assert_close(
            hasher.history[2], torch.tensor([51, 52], dtype=torch.int32)
        )
        torch.testing.assert_close(
            hasher.history[5], torch.tensor([1, 2], dtype=torch.int32)
        )
        torch.testing.assert_close(
            hasher.history[7], torch.tensor([4, 5], dtype=torch.int32)
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.cuda, "requires CUDA"
    )
    def test_cuda_forward_submission_has_no_host_sync(self):
        cpu = _fixture(ratio=2, rows=[2, 0, 3], seq_lens=[9, 4, 15], padding=5)
        expected_pages, expected_raw = _reference(cpu[0], cpu[2], cpu[3])
        expected_hash = _hasher("cpu")(torch.arange(1, 8), _engram_batch("cpu"))
        for cpu_slots in (True, False):
            args = _fixture(
                ratio=2,
                rows=[2, 0, 3],
                seq_lens=[9, 4, 15],
                padding=5,
                device="cuda",
                cpu_slots=cpu_slots,
            )
            hasher, batch = _hasher("cuda"), _engram_batch("cuda")
            ids = torch.arange(1, 8, device="cuda")
            new_seq_lens = torch.tensor([9, 15], device="cuda")
            _select(*args)
            hasher(ids, batch)  # Warm Triton compilation outside the check.
            hasher.history.zero_()
            torch.cuda.synchronize()
            previous = torch.cuda.get_sync_debug_mode()
            try:
                torch.cuda.set_sync_debug_mode("error")
                _select(*args)  # Includes the real FP4 pool dequant readback.
                hashes = hasher(ids, batch)
                staged_seq_lens = async_d2h(new_seq_lens)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
            torch.testing.assert_close(args[1].page_indices.cpu(), expected_pages)
            torch.testing.assert_close(args[1].raw_indices.cpu(), expected_raw)
            torch.testing.assert_close(hashes.cpu(), expected_hash)
            torch.cuda.synchronize()
            torch.testing.assert_close(staged_seq_lens, torch.tensor([9, 15]))


if __name__ == "__main__":
    unittest.main()
