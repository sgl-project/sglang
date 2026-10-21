"""GR prefill preserves the initial top-K order, including exact ties."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.beam_search.beam_group import BeamGroup, materialize_tokens
from sglang.srt.beam_search.coordinator import BeamCoordinator, _rows_topk_logprobs
from sglang.srt.beam_search.joint_select import joint_select, select_final_topk
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestGRPrefillSelection(CustomTestCase):
    def make_coordinator(self):
        return BeamCoordinator(
            model_config=None,
            spec_algorithm=None,
            dllm_enabled=False,
            max_req_len=8192,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            future_map=None,
        )

    def prefill(self, logits, width, depth, *, enable_gr=True, stop_ids=()):
        group = BeamGroup(
            beam_width=width,
            num_candidates=width if enable_gr else 2 * width,
            max_new_tokens=depth,
            stop_token_ids=stop_ids,
            device=logits.device,
        )
        req = SimpleNamespace(
            beam_group=group, kv=SimpleNamespace(req_pool_idx=0), output_ids=[]
        )
        group.leader = req
        coordinator = self.make_coordinator()
        output = SimpleNamespace(beam=SimpleNamespace(leader_logits=logits))
        with (
            patch(
                "sglang.srt.beam_search.coordinator.get_serving",
                return_value=SimpleNamespace(enable_gr=enable_gr),
            ),
            patch.object(BeamCoordinator, "_spawn_member_rows") as spawn,
            patch.object(BeamCoordinator, "_stash_next_tokens") as stash,
        ):
            coordinator.select_leader_prefill(req, 0, output)
        selected = group._pending_steps[0][1]
        self.assertEqual(spawn.call_count, int(depth > 1))
        relayed = stash.call_args.args[1]
        chosen = selected.tokens if depth == 1 else selected.next_tokens
        torch.testing.assert_close(
            relayed, chosen[:1] if depth == 1 else chosen, rtol=0, atol=0
        )
        group.commit_pending()
        return coordinator, group, selected

    def test_gr_prefill_preserves_sid_topk_order(self):
        # Equal logits and tied score blocks exercise ordering within the chosen
        # set and at its cutoff. Expected order comes from SID's one top-K rule.
        for logits in (
            torch.zeros(1, 256),
            (torch.arange(256) % 4).float().reshape(1, -1),
        ):
            for width in (1, 4, 16, 128, 256):
                expected_scores, expected_tokens = torch.log_softmax(logits, -1).topk(
                    width, dim=-1
                )
                for depth in (1, 3):
                    with self.subTest(
                        width=width, depth=depth, tied=bool(logits.max())
                    ):
                        _, group, selected = self.prefill(logits, width, depth)
                        actual_tokens = (
                            selected.tokens if depth == 1 else selected.next_tokens
                        )
                        actual_scores = (
                            selected.cum_logprobs
                            if depth == 1
                            else selected.new_cum_logprobs
                        )
                        self.assertEqual(
                            actual_tokens.tolist(), expected_tokens[0].tolist()
                        )
                        self.assertEqual(selected.parent_idx.tolist(), [0] * width)
                        torch.testing.assert_close(
                            actual_scores, expected_scores[0], rtol=1e-6, atol=1e-6
                        )
                        paths = (
                            [r.tokens for r in group.finalize()]
                            if depth == 1
                            else [materialize_tokens(leaf) for leaf in group.leaves]
                        )
                        self.assertEqual(
                            paths, [[t] for t in expected_tokens[0].tolist()]
                        )
                        if depth > 1:
                            self.assertEqual(int(selected.num_survivors), width)
                            self.assertEqual(int(selected.num_finished), 0)

    def test_ordinary_prefill_routes_eos(self):
        logits = torch.arange(16, dtype=torch.float32).reshape(1, -1)
        for depth in (1, 3):
            with self.subTest(depth=depth):
                _, group, selected = self.prefill(
                    logits, 4, depth, enable_gr=False, stop_ids=(15,)
                )
                if depth == 1:
                    self.assertEqual(
                        [r.tokens for r in group.finalize()], [[15], [14], [13], [12]]
                    )
                else:
                    self.assertEqual(
                        [materialize_tokens(leaf) for leaf in group.leaves],
                        [[14], [13], [12], [11]],
                    )
                    self.assertEqual(int(selected.num_finished), 1)
                    self.assertEqual(selected.fin_tokens[0].item(), 15)

    def test_gr_decode_still_selects_across_parents(self):
        logits = torch.arange(16, dtype=torch.float32).reshape(1, -1) / 10
        coordinator, group, _ = self.prefill(logits, 4, 2)
        prefixes = [materialize_tokens(leaf) for leaf in group.leaves]
        generator = torch.Generator().manual_seed(20261014)
        decode_logits = torch.randn(4, 16, generator=generator)
        expected_scores, expected_indices = (
            (
                torch.log_softmax(decode_logits, -1)
                + group.frontier_cum_logprobs[:, None]
            )
            .flatten()
            .topk(4)
        )
        self.assertGreater(len(set((expected_indices // 16).tolist())), 1)
        scores, tokens = _rows_topk_logprobs(
            [decode_logits[:1], decode_logits[1:]], group.num_candidates
        )
        with patch(
            "sglang.srt.beam_search.coordinator.get_serving",
            return_value=SimpleNamespace(enable_gr=True),
        ):
            coordinator._select_group(group, scores, tokens, tick=1)
        group.commit_pending()
        results = group.finalize()
        self.assertEqual(
            [r.tokens for r in results],
            [prefixes[i // 16] + [i % 16] for i in expected_indices.tolist()],
        )
        torch.testing.assert_close(
            torch.tensor([r.cum_logprob for r in results]),
            expected_scores,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_prefill_option_falls_back_for_other_frontiers(self):
        # Preserve normal selection for multiple parents, 2K candidates, or EOS.
        for rows, candidates, stops in (
            (4, 4, ()),
            (1, 8, ()),
            (1, 4, (0,)),
        ):
            cumulative = -torch.arange(rows, dtype=torch.float32)
            scores = torch.zeros(rows, candidates)
            tokens = torch.arange(rows * candidates).reshape(rows, candidates)
            stop_ids = torch.tensor(stops, dtype=torch.int64)
            for final in (False, True):
                if final and stops:
                    continue
                with self.subTest(
                    rows=rows, candidates=candidates, stops=stops, final=final
                ):
                    if final:
                        args = (cumulative, scores, tokens, 4)
                        select = select_final_topk
                    else:
                        args = (cumulative, scores, tokens, stop_ids, 4)
                        select = joint_select
                    expected = select(*args)
                    actual = select(*args, preserve_prefill_order=True)
                    for field in expected.__struct_fields__:
                        torch.testing.assert_close(
                            getattr(actual, field),
                            getattr(expected, field),
                            rtol=0,
                            atol=0,
                        )


if __name__ == "__main__":
    unittest.main()
