import os
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_TRITON_OK = torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@unittest.skipUnless(_TRITON_OK, "needs CUDA or TRITON_INTERPRET=1")
class TestUniformVerifyWindowExtras(unittest.TestCase):
    """The one-launch verify window matches the separate positions add,
    cache-loc gather and anchor copy bit for bit."""

    def _run(self, *, bs, width, page_size, extras):
        from sglang.kernels.ops.speculative.cache_locs import (
            assign_extend_cache_locs_uniform,
        )

        dev = _device()
        g = torch.Generator().manual_seed(bs * 100 + width)
        pool_len, num_pages = 256, 64
        req_to_token = torch.randint(
            0, num_pages * page_size, (bs + 3, pool_len), generator=g
        ).to(dev)
        req_pool_indices = torch.randperm(bs + 3, generator=g)[:bs].to(dev)
        seq_lens = torch.randint(0, pool_len - width, (bs,), generator=g).to(dev)
        full_v2p = torch.randperm(num_pages, generator=g).to(dev)
        swa_v2p = torch.randperm(num_pages, generator=g).to(dev)
        out = torch.empty(bs * width, dtype=torch.int64, device=dev)
        full_out = torch.empty_like(out)
        swa_out = torch.empty_like(out)
        positions = torch.full((bs, width), -7, dtype=torch.int64, device=dev)
        anchor = torch.full((bs, width + 2), -9, dtype=torch.int64, device=dev)
        bonus = torch.randint(0, 1000, (bs,), generator=g).to(dev)
        kw = (
            dict(
                positions_out=positions,
                anchor_src=bonus,
                anchor_out=anchor,
                anchor_stride=anchor.stride(0),
            )
            if extras
            else {}
        )
        assign_extend_cache_locs_uniform[(bs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            out,
            pool_len,
            width,
            full_v2p=full_v2p,
            swa_v2p=swa_v2p,
            full_out=full_out,
            swa_out=swa_out,
            PAGE_SIZE=page_size,
            num_v_pages=num_pages,
            num_swa_pages=num_pages,
            **kw,
        )
        return dict(
            out=out,
            full_out=full_out,
            swa_out=swa_out,
            positions=positions,
            anchor=anchor,
            bonus=bonus,
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
        )

    def test_matches_separate_ops(self):
        for bs, width in ((1, 6), (5, 7), (64, 9)):
            for page_size in (1, 64):
                with self.subTest(bs=bs, width=width, page_size=page_size):
                    base = self._run(
                        bs=bs, width=width, page_size=page_size, extras=False
                    )
                    got = self._run(
                        bs=bs, width=width, page_size=page_size, extras=True
                    )
                    for k in ("out", "full_out", "swa_out"):
                        self.assertTrue(torch.equal(base[k], got[k]), k)
                    cols = torch.arange(width, device=got["seq_lens"].device)
                    ref_pos = got["seq_lens"].to(torch.int64).unsqueeze(1) + cols
                    self.assertTrue(torch.equal(got["positions"], ref_pos))
                    ref_loc = got["req_to_token"][
                        got["req_pool_indices"].unsqueeze(1), ref_pos
                    ].reshape(-1)
                    self.assertTrue(torch.equal(got["out"], ref_loc))
                    self.assertTrue(torch.equal(got["anchor"][:, 0], got["bonus"]))
                    self.assertTrue(bool((got["anchor"][:, 1:] == -9).all()))


class TestDraftSamplerVerifyIds(unittest.TestCase):
    """The folded draft tail writes [anchor | draft tokens], as the eager concat."""

    def test_tail_writes_verify_ids(self):
        from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
            DsparkDraftSampler,
        )

        bs, gamma, max_bs, hidden = 3, 4, 5, 8
        s = DsparkDraftSampler.__new__(DsparkDraftSampler)
        s.gamma, s.max_bs = gamma, max_bs
        s.sample_from_anchor, s.query_token_num = True, gamma
        s.out = torch.zeros(max_bs * gamma, dtype=torch.int64)
        s.verify_ids = torch.full((max_bs, gamma + 1), -1, dtype=torch.int64)
        s.confidence_out = None
        draft = torch.arange(bs * gamma, dtype=torch.int64).view(bs, gamma) + 100

        def step(model_hidden, anchor):
            s.out[: bs * gamma].copy_(draft.reshape(-1))
            return draft, None

        s._megakernel_steps = lambda b: step
        input_ids = torch.arange(bs * gamma, dtype=torch.int64) * 3
        s(torch.zeros(bs * gamma, hidden), input_ids)
        anchor = input_ids.view(bs, gamma)[:, :1]
        self.assertTrue(
            torch.equal(s.verify_ids[:bs], torch.cat([anchor, draft], dim=1))
        )
        self.assertTrue(bool((s.verify_ids[bs:] == -1).all()))


class TestInGraphBlockAcceptLens(unittest.TestCase):
    """The in-graph accept's block_accept_lens equals the worker's eager
    commit_lens + cap_trim_lens."""

    def test_matches_eager_sum(self):
        from sglang.srt.speculative.dspark_components.dspark_verify import (
            AcceptOuts,
            build_accept_outs,
        )
        from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite

        bs, gamma = 4, 5
        tp_sync = type("S", (), {"sync": lambda self, site, t: None})()
        kw = dict(
            correct_len=torch.tensor([0, 2, 5, 3]),
            bonus=torch.tensor([7, 8, 9, 10]),
            cap_trim_lens=torch.tensor([0, 1, 0, 2], dtype=torch.int32),
            prefix_lens=torch.tensor([10, 20, 30, 40]),
            draft_tokens=torch.arange(bs * gamma).view(bs, gamma),
            gamma=gamma,
            verify_num_draft_tokens=gamma + 1,
            tp_sync=tp_sync,
            sync_site=list(SpecTpSyncSite)[0],
        )
        eager = build_accept_outs(**kw)
        out = AcceptOuts(
            correct_len=torch.zeros(bs, dtype=torch.int64),
            bonus=torch.zeros(bs, dtype=torch.int64),
            cap_trim_lens=torch.zeros(bs, dtype=torch.int32),
            commit_lens=torch.zeros(bs, dtype=torch.int32),
            new_seq_lens=torch.zeros(bs, dtype=torch.int64),
            out_tokens=torch.zeros(bs, gamma + 1, dtype=torch.int64),
        )
        block = torch.full((bs,), -1, dtype=torch.int32)
        build_accept_outs(**kw, out=out, block_accept_lens_out=block)
        expected = eager.commit_lens + eager.cap_trim_lens
        self.assertTrue(torch.equal(block, expected.to(torch.int32)))


class TestPostFoldRelay(unittest.TestCase):
    """SGLANG_OPT_DSV41_DSPARK_POST_FOLD: the static verify graph scatters new seq
    lens and bonus into slot-indexed buffers the FutureMap adopts; the host skips
    its scatter only for the exact tensors the worker marks as relayed."""

    def tearDown(self):
        from sglang.srt.managers import overlap_utils

        overlap_utils._offered_relay = None

    def test_graph_scatter_feeds_the_adopted_future_map(self):
        from types import SimpleNamespace

        from sglang.srt.managers import overlap_utils
        from sglang.srt.speculative.dspark_components.dspark_verify import (
            DsparkVerifyEpilogue,
        )

        pool_size = 10
        epi = DsparkVerifyEpilogue(
            max_bs=4,
            verify_num_draft_tokens=6,
            device="cpu",
            tp_sync=type("S", (), {"sync": lambda self, site, t: None})(),
            commit_ctx=SimpleNamespace(
                resolve_req_to_token=lambda: torch.zeros(pool_size, 8)
            ),
            post_fold=True,
        )
        # A warmup scatter builds and offers the relay; the FutureMap adopts it
        # (which resets it to -1 in CI) before any live step scatters.
        epi._scatter_future_relay(torch.tensor([0, 0, 0, 0]), 4)
        relay = epi.future_relay
        self.assertIs(overlap_utils._offered_relay, relay)

        fm = overlap_utils.FutureMap.__new__(overlap_utils.FutureMap)
        fm.needs_confidence_relay = False
        fm.spec_algo = SimpleNamespace(is_some=lambda: False, is_ngram=lambda: False)
        fm._adopt_relay(relay)
        self.assertTrue(relay.adopted)

        epi.new_seq_lens_buf.copy_(torch.tensor([50, 60, 7, 7]))
        epi.bonus_buf.copy_(torch.tensor([500, 600, 9, 9]))
        # Two live rows; the padded rows index the dummy slot 0.
        epi._scatter_future_relay(torch.tensor([3, 7, 0, 0]), 4)
        self.assertEqual(fm.new_seq_lens_buf[[3, 7]].tolist(), [50, 60])
        self.assertEqual(fm.output_tokens_buf[[3, 7]].tolist(), [500, 600])

        # The worker-marked tensor skips the host scatter, once.
        marked = torch.tensor([-1, -1])
        relay.published_seq_lens = marked
        fm.publish(torch.tensor([3, 7]), marked)
        self.assertEqual(fm.new_seq_lens_buf[[3, 7]].tolist(), [50, 60])
        self.assertIsNone(relay.published_seq_lens)
        # Any other tensor (an eager accept) still scatters.
        fm.publish(torch.tensor([3, 7]), torch.tensor([51, 61]))
        self.assertEqual(fm.new_seq_lens_buf[[3, 7]].tolist(), [51, 61])

        fm._maybe_init_forward_bufs = lambda payload: None
        fm._maybe_init_dsa_topk_indices_buf = lambda payload: None
        fm.need_topk = fm.need_hidden_states = False
        fm.draft_probs_buf = fm.dsa_topk_indices_buf = None
        marked_bonus = torch.tensor([-1, -1])
        relay.published_bonus = marked_bonus
        fm.stash(
            torch.tensor([3, 7]), overlap_utils.RelayPayload(bonus_tokens=marked_bonus)
        )
        self.assertEqual(fm.output_tokens_buf[[3, 7]].tolist(), [500, 600])
        fm.stash(
            torch.tensor([3, 7]),
            overlap_utils.RelayPayload(bonus_tokens=torch.tensor([501, 601])),
        )
        self.assertEqual(fm.output_tokens_buf[[3, 7]].tolist(), [501, 601])


if __name__ == "__main__":
    unittest.main()
