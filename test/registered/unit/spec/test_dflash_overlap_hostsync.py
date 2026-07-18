"""Unit tests for the DFlash spec-v2 host-sync removal: compact-rebuild
kernel bit-exactness, vocab-parallel draft sampler select, host seq-lens
upper bound, hybrid needs_cpu_seq_lens delegation, filter_batch host
keep-list."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

_HAS_CUDA = torch.cuda.is_available()


def _compact_lens_exact(seq_lens, window, page):
    fake_self = SimpleNamespace(
        device=seq_lens.device, draft_window_size=window, page_size=page
    )
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    return DFlashWorkerV2._compute_compact_draft_seq_lens(fake_self, seq_lens)


def _compact_lens_host(seq_lens, window, page):
    fake_self = SimpleNamespace(draft_window_size=window, page_size=page)
    out = torch.empty(seq_lens.numel(), dtype=torch.int32)
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    DFlashWorkerV2._compute_compact_draft_seq_lens_host(fake_self, seq_lens, out)
    return out


class TestCompactSeqLensHostBound(unittest.TestCase):
    def test_upper_bound_of_exact(self):
        g = torch.Generator().manual_seed(0)
        for window, page in [(4096, 64), (4096, 1), (128, 32), (64, 1)]:
            seq = torch.randint(1, 3 * window, (512,), generator=g)
            exact = _compact_lens_exact(seq, window, page).to(torch.int64)
            bound = _compact_lens_host(seq, window, page).to(torch.int64)
            self.assertTrue(
                bool((bound >= exact).all()),
                f"host bound under-shoots exact at window={window} page={page}",
            )

    def test_sawtooth_counterexample(self):
        # exact(4160) = 4096 < exact(4100) = 4100 at window=4096 page=64:
        # a host mirror of the exact math fed the reserved over-estimate
        # (4160 >= true 4100) would under-shoot; the envelope must not.
        window, page = 4096, 64
        true_len = torch.tensor([4100])
        reserved = torch.tensor([4160])
        exact_true = _compact_lens_exact(true_len, window, page).to(torch.int64)
        exact_reserved = _compact_lens_exact(reserved, window, page).to(torch.int64)
        self.assertLess(int(exact_reserved), int(exact_true))
        bound = _compact_lens_host(reserved, window, page).to(torch.int64)
        self.assertGreaterEqual(int(bound), int(exact_true))


class _FakeTpGroup:
    """Single-process stand-in for the TP GroupCoordinator: replays the
    concatenation of all ranks' recorded all-gather inputs."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.recording = True
        self.recorded = {}  # (rank, call_idx) -> tensor
        self.rank = 0
        self.call_idx = 0

    def all_gather_into_tensor(self, output, input_):
        if self.recording:
            self.recorded[(self.rank, self.call_idx)] = input_.clone()
        else:
            output.copy_(
                torch.cat(
                    [self.recorded[(r, self.call_idx)] for r in range(self.world_size)]
                )
            )
        self.call_idx += 1


class TestDflashDraftSamplerVocabParallel(unittest.TestCase):
    def _run(self, vocab, hidden, bs, block_size, world, dtype, weight=None):
        from sglang.srt.speculative.dflash_worker_v2 import _DflashDraftSampler

        device = torch.device("cuda" if _HAS_CUDA else "cpu")
        g = torch.Generator(device=device).manual_seed(0)
        if weight is None:
            weight = torch.randn(vocab, hidden, generator=g, device=device, dtype=dtype)
        hs = torch.randn(
            bs * block_size, hidden, generator=g, device=device, dtype=dtype
        )
        shard = vocab // world
        group = _FakeTpGroup(world)
        samplers = [
            _DflashDraftSampler(
                weight=weight[r * shard : (r + 1) * shard].contiguous(),
                block_size=block_size,
                num_org=shard,
                org_vocab_start=r * shard,
                max_bs=bs,
                tp_group=group,
            )
            for r in range(world)
        ]
        for phase_recording in (True, False):
            group.recording = phase_recording
            for r, s in enumerate(samplers):
                group.rank, group.call_idx = r, 0
                s(hs)

        n = bs * (block_size - 1)
        ref_hs = hs.view(bs, block_size, -1)[:, 1:, :].reshape(-1, hidden)
        ref = torch.argmax(torch.matmul(ref_hs.to(weight.dtype), weight.T), dim=-1).to(
            torch.long
        )
        for r, s in enumerate(samplers):
            torch.testing.assert_close(
                s.out[:n], ref, rtol=0, atol=0, msg=f"rank {r} mismatch"
            )

    def test_matches_full_vocab_argmax(self):
        self._run(
            vocab=512, hidden=64, bs=3, block_size=8, world=4, dtype=torch.float32
        )

    def test_shard_boundary_tie_resolves_to_first_global_index(self):
        # Duplicate row 10 (shard 0) at row 200 (shard 1): identical logits, so
        # a correct fold must pick 10 (torch.argmax first-max semantics).
        vocab, hidden = 256, 32
        device = torch.device("cuda" if _HAS_CUDA else "cpu")
        weight = torch.zeros(vocab, hidden, device=device)
        weight[10] = 1.0
        weight[200] = 1.0
        self._run(
            vocab=vocab,
            hidden=hidden,
            bs=1,
            block_size=4,
            world=2,
            dtype=torch.float32,
            weight=weight,
        )


@unittest.skipUnless(_HAS_CUDA, "triton kernel requires CUDA")
class TestRebuildCompactDraftReqToToken(unittest.TestCase):
    def _legacy(self, draft, target, req_idx, start, lens, verify_2d, bs, block):
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        lens64 = lens.to(torch.int64)
        max_len = int(lens64.max().item())
        offs = torch.arange(max_len, device=lens.device).unsqueeze(0)
        pos2d = start.to(torch.int64).unsqueeze(1) + offs
        mask = offs < lens64.unsqueeze(1)
        packed = target[req_idx.to(torch.int64)[:, None], pos2d.masked_fill(~mask, 0)][
            mask
        ].to(torch.int64)
        assign_req_to_token_pool_func(
            req_idx, draft, torch.zeros_like(lens), lens, packed, bs
        )
        assign_req_to_token_pool_func(
            req_idx, draft, lens, lens + block, verify_2d.reshape(-1), bs
        )

    def test_bitexact_vs_legacy(self):
        from sglang.kernels.ops.speculative.cache_locs import (
            rebuild_compact_draft_req_to_token_func,
        )

        device = torch.device("cuda")
        for bs, window, page, block, seed in [
            (1, 64, 1, 8, 0),
            (16, 64, 32, 8, 1),
            (13, 128, 64, 8, 2),
            (7, 512, 64, 16, 3),
        ]:
            g = torch.Generator(device=device).manual_seed(seed)
            pool_rows, width = 4 * bs, 4 * window
            seq = torch.randint(
                1, width - block - 1, (bs,), generator=g, device=device
            ).to(torch.int64)
            lens = _compact_lens_exact(seq, window, page).to(device)
            start = seq - lens.to(torch.int64)
            req_idx = torch.randperm(pool_rows, generator=g, device=device)[:bs]
            target = torch.randint(
                0, 2**30, (pool_rows, width), generator=g, device=device
            ).to(torch.int32)
            verify_2d = torch.randint(
                0, 2**30, (bs, block), generator=g, device=device
            ).to(torch.int64)
            draft_width = window + page + block + 8
            draft_a = torch.full(
                (pool_rows, draft_width), -1, dtype=torch.int32, device=device
            )
            draft_b = draft_a.clone()

            self._legacy(draft_a, target, req_idx, start, lens, verify_2d, bs, block)
            rebuild_compact_draft_req_to_token_func(
                draft_req_to_token=draft_b,
                target_req_to_token=target,
                req_pool_indices=req_idx,
                suffix_start=start,
                draft_prefix_lens=lens,
                verify_out_cache_loc_2d=verify_2d,
                batch_size=bs,
                block_size=block,
            )
            torch.testing.assert_close(draft_b, draft_a, rtol=0, atol=0)
            for i in range(bs):
                total = int(lens[i].item()) + block
                self.assertTrue(
                    bool((draft_b[req_idx[i], total:] == -1).all()),
                    "kernel wrote past the verify block",
                )


class TestHybridNeedsCpuSeqLens(unittest.TestCase):
    def _make(self, prefill_flag, decode_flag):
        from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend

        def backend(flag):
            return SimpleNamespace(needs_cpu_seq_lens=flag)

        runner = SimpleNamespace(
            server_args=SimpleNamespace(speculative_attention_mode="decode"),
            kv_cache_dtype=torch.bfloat16,
            token_to_kv_pool=None,
            req_to_token_pool=None,
        )
        return HybridAttnBackend(runner, backend(prefill_flag), backend(decode_flag))

    def test_delegation(self):
        self.assertFalse(self._make(False, False).needs_cpu_seq_lens)
        self.assertTrue(self._make(True, False).needs_cpu_seq_lens)
        self.assertTrue(self._make(False, True).needs_cpu_seq_lens)


class TestFilterBatchHostIndices(unittest.TestCase):
    def test_host_keep_list_matches_gpu_indices(self):
        from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

        def make():
            info = DFlashDraftInputV2.create_idle_input(device=torch.device("cpu"))
            info.reserved_seq_lens_cpu = torch.tensor(
                [10, 20, 30, 40], dtype=torch.int32
            )
            info.reserved_seq_lens_sum = 100
            info.future_indices = torch.tensor([5, 6, 7, 8])
            return info

        keep = [0, 2]
        a, b = make(), make()
        a.filter_batch(new_indices=torch.tensor(keep), has_been_filtered=False)
        b.filter_batch(
            new_indices=torch.tensor(keep),
            has_been_filtered=False,
            new_indices_cpu=keep,
        )
        torch.testing.assert_close(a.reserved_seq_lens_cpu, b.reserved_seq_lens_cpu)
        self.assertEqual(a.reserved_seq_lens_sum, b.reserved_seq_lens_sum)
        torch.testing.assert_close(a.future_indices, b.future_indices)


if __name__ == "__main__":
    unittest.main()
