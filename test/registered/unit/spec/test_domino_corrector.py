import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.speculative.domino_corrector import (
    CorrectorContext,
    CorrectorDraftSampler,
    CorrectorSampling,
    TransformerCorrector,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _rotate(x, positions):
    # build complex pairs from the two halves used for NeoX rotary embeddings
    left, right = x.chunk(2, dim=-1)
    dims = torch.arange(left.shape[-1], dtype=x.dtype)
    angles = positions[:, None] / 10000 ** (dims / left.shape[-1])
    rotated = torch.complex(left, right) * torch.polar(torch.ones_like(angles), angles)
    return torch.cat((rotated.real, rotated.imag), -1)


def _attend(q, k, v, mask):
    scores = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    weights = scores.masked_fill(~mask, -torch.inf).softmax(-1).nan_to_num()
    return (weights @ v).transpose(-3, -2).flatten(-2)


def _teacher_forced_states(model, hidden, tokens, context, valid):
    """Evaluate every slot together with a causal mask, as in training."""
    slots = torch.arange(1, hidden.shape[1] + 1)
    past = torch.arange(-context.shape[1], 0)
    causal = slots[:, None] >= slots
    token_stream = model.input_token(tokens)
    streams = torch.stack((token_stream, token_stream + model.input_hidden(hidden)))

    def heads(x):
        return x.unflatten(-1, (model.blocks[0].heads, -1)).transpose(-3, -2)

    for block in model.blocks:
        query, key, value = block.self_qkv(block.self_norm(streams)).chunk(3, -1)
        update = _attend(
            _rotate(heads(query), slots),
            _rotate(heads(key[0]), slots),
            heads(value[0]),
            causal,
        )
        streams = streams + block.self_out(update)
        key, value = block.cross_kv(context).chunk(2, -1)
        query = block.cross_q(block.cross_norm(streams))
        update = _attend(
            _rotate(heads(query), slots),
            _rotate(heads(key), past),
            heads(value),
            valid[:, None, None, :],
        )
        streams = streams + block.cross_out(update)
        gate, up = block.mlp_gate_up(block.mlp_norm(streams)).chunk(2, -1)
        streams = streams + block.mlp_down(torch.nn.functional.silu(gate) * up)
    return model.output_norm(streams[1])


def _corrector_case():
    torch.manual_seed(42)
    config = {
        "d_model": 8,
        "heads": 2,
        "layers": 2,
        "context_window": 3,
        "context_attention": True,
    }
    model = TransformerCorrector(6, config, eps=1e-5).double()
    embed = nn.Embedding(13, 6).double()
    head = nn.Sequential(
        nn.Linear(14, 5, bias=False), nn.SiLU(), nn.Linear(5, 13, bias=False)
    ).double()
    hidden = torch.randn(3, 4, 6, dtype=torch.float64)
    context = torch.randn(3, 3, 6, dtype=torch.float64)
    valid = torch.tensor([[1, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=torch.bool)
    anchors = torch.tensor([1, 3, 5])
    unary = torch.randn(3, 4, 13, dtype=torch.float64) * 0.1
    return model, embed, head, hidden, context, valid, anchors, unary


class TestDominoCorrector(unittest.TestCase):
    def test_incremental_states_and_picks_match_teacher_forcing(self):
        model, embed, head, hidden, context, valid, anchors, unary = _corrector_case()
        corrected_inputs = []
        head[1].register_forward_hook(
            lambda module, args, output: corrected_inputs.append(args[0])
        )

        for prefix in (0, 1):
            corrected_inputs.clear()
            picks, _ = model.propose(
                hidden=hidden,
                anchors=anchors,
                context_kv=model.project_context(context),
                context_mask=valid,
                token_inputs=model.token_table(embed, 13),
                embed_proj=head,
                unary_picks=unary[:, :prefix].argmax(-1),
                unary_scores=unary[:, prefix:],
            )
            prior_picks = torch.cat((anchors[:, None], picks[:, :-1]), dim=1)
            states = _teacher_forced_states(
                model, hidden, embed(prior_picks), context, valid
            )
            expected = torch.cat((hidden[:, prefix:], states[:, prefix:]), dim=-1)
            torch.testing.assert_close(
                torch.stack(corrected_inputs, dim=1), head[0](expected)
            )
            torch.testing.assert_close(picks[:, :prefix], unary[:, :prefix].argmax(-1))
            torch.testing.assert_close(
                picks[:, prefix:], (unary[:, prefix:] + head(expected)).argmax(-1)
            )

    def test_candidate_pool_scores_only_its_tokens(self):
        model, embed, head, hidden, context, valid, anchors, unary = _corrector_case()

        def propose(candidate_ids):
            scores = unary
            if candidate_ids is not None:
                scores = unary.gather(2, candidate_ids[:, None, :].expand(-1, 4, -1))
            return model.propose(
                hidden=hidden,
                anchors=anchors,
                context_kv=model.project_context(context),
                context_mask=valid,
                token_inputs=model.token_table(embed, 13),
                embed_proj=head,
                unary_picks=unary[:, :0].argmax(-1),
                unary_scores=scores,
                candidate_ids=candidate_ids,
            )[0]

        full = propose(None)
        # A pool holding the whole vocabulary, in any order, changes nothing.
        everything = torch.stack([torch.randperm(13) for _ in range(3)])
        torch.testing.assert_close(propose(everything), full)

        pool = everything[:, :5]
        picks = propose(pool)
        prior_picks = torch.cat((anchors[:, None], picks[:, :-1]), dim=1)
        states = _teacher_forced_states(
            model, hidden, embed(prior_picks), context, valid
        )
        logits = unary + head(torch.cat((hidden, states), dim=-1))
        outside = torch.ones(3, 13, dtype=torch.bool).scatter_(1, pool, False)
        expected = logits.masked_fill(outside[:, None, :], -torch.inf).argmax(-1)
        torch.testing.assert_close(picks, expected)

    def test_sampled_slots_return_the_q_they_were_drawn_from(self):
        model, embed, head, hidden, context, valid, anchors, unary = _corrector_case()
        pool = torch.stack([torch.randperm(13)[:5] for _ in range(3)])
        slot_pool = pool[:, None, :].expand(-1, 3, -1)
        temperatures = torch.tensor([0.7, 1.0, 1.3])
        picks, q_rows = model.propose(
            hidden=hidden,
            anchors=anchors,
            context_kv=model.project_context(context),
            context_mask=valid,
            token_inputs=model.token_table(embed, 13),
            embed_proj=head,
            unary_picks=unary[:, :1].argmax(-1),
            unary_scores=unary[:, 1:].gather(2, slot_pool),
            candidate_ids=pool,
            sampling=CorrectorSampling(
                temperatures=temperatures,
                greedy_mask=torch.tensor([False, True, False]),
                noise=torch.empty(3, 5),
            ),
        )
        # q is conditioned on the tokens the walk drew, not on the greedy ones.
        prior_picks = torch.cat((anchors[:, None], picks[:, :-1]), dim=1)
        states = _teacher_forced_states(
            model, hidden, embed(prior_picks), context, valid
        )
        logits = unary + head(torch.cat((hidden, states), dim=-1))
        logits = logits[:, 1:].gather(2, slot_pool).float()
        q = (logits / temperatures[:, None, None]).softmax(-1)
        torch.testing.assert_close(q_rows[[0, 2]], q[[0, 2]])
        choices = (slot_pool == picks[:, 1:, None]).float().argmax(-1, keepdim=True)
        self.assertTrue((q_rows.gather(2, choices) > 0).all())
        # A greedy row of a sampled batch keeps the argmax as a point mass.
        self.assertEqual(choices[1, :, 0].tolist(), logits[1].argmax(-1).tolist())
        self.assertEqual(q_rows[1].gather(1, choices[1]).flatten().tolist(), [1.0] * 3)
        self.assertEqual(q_rows[1].sum(-1).tolist(), [1.0] * 3)

    def test_sampled_slot_follows_its_q(self):
        model, embed, head, hidden, context, valid, anchors, unary = _corrector_case()
        rows = 20000
        pool = torch.tensor([[2, 7, 11, 4]]).expand(rows, -1)
        torch.manual_seed(0)
        picks, q_rows = model.propose(
            hidden=hidden[:1, :1].expand(rows, -1, -1),
            anchors=anchors[:1].expand(rows),
            context_kv=model.project_context(context[:1]).expand(rows, -1, -1, -1),
            context_mask=valid[:1].expand(rows, -1),
            token_inputs=model.token_table(embed, 13),
            embed_proj=head,
            unary_picks=torch.empty((rows, 0), dtype=torch.int64),
            unary_scores=unary[:1, :1, [2, 7, 11, 4]].expand(rows, -1, -1) * 20,
            candidate_ids=pool,
            sampling=CorrectorSampling(
                temperatures=torch.full((rows,), 2.0),
                greedy_mask=torch.zeros(rows, dtype=torch.bool),
                noise=torch.empty(rows, 4),
            ),
        )
        drawn = (pool == picks).float().mean(0)
        self.assertGreater(q_rows[0, 0].min().item(), 0.05)
        torch.testing.assert_close(drawn, q_rows[0, 0], atol=0.015, rtol=0)


class TestCorrectorContext(unittest.TestCase):
    def test_window_across_extend_verify_and_slot_reuse(self):
        ctx = CorrectorContext(
            TransformerCorrector(
                hidden_size=2,
                config={"context_window": 3, "layers": 1, "heads": 1, "d_model": 2},
                eps=1e-5,
            ),
            pool=SimpleNamespace(maybe_get_custom_mem_pool=lambda: None),
            req_pool_rows=2,
        )
        generations = torch.zeros(2, dtype=torch.long)

        def write(locs, values):
            features = torch.tensor(values, dtype=torch.float32)[:, None]
            ctx.write(
                locs,
                torch.stack((features, -features, features, -features), dim=-1),
            )

        def extend(slots, starts, lengths, values):
            positions = torch.cat(
                [
                    torch.arange(start, start + length)
                    for start, length in zip(starts, lengths)
                ]
            )
            batch = SimpleNamespace(
                reqs=[
                    SimpleNamespace(kv=SimpleNamespace(req_pool_idx=i)) for i in slots
                ],
                prefix_lens=starts,
                req_pool_indices=torch.tensor(slots),
                seq_lens=torch.tensor(starts) + torch.tensor(lengths),
            )
            locs = ctx.extend_locs(batch, positions, torch.tensor(lengths), generations)
            write(locs, values)

        def check(slots, ends, values, valid):
            kv, mask = ctx.gather(torch.tensor(slots), torch.tensor(ends))
            self.assertEqual(kv[:, :, 0, 0].tolist(), values)
            self.assertEqual(
                kv[:, :, 0, 1].tolist(), [[-v for v in row] for row in values]
            )
            self.assertEqual(mask.tolist(), valid)

        # A long chunk wraps. A warm prefix starts with only its new rows.
        extend([1, 0], [0, 9], [5, 1], [1, 2, 3, 4, 5, 10])
        check([0, 1], [10, 5], [[0, 0, 10], [3, 4, 5]], [[0, 0, 1], [1, 1, 1]])
        extend([0], [10], [1], [11])
        check([0], [11], [[0, 10, 11]], [[0, 1, 1]])

        locs = ctx.verify_locs(
            torch.tensor([0, 1]),
            torch.tensor([[11, 12, 13, 14], [5, 6, 7, 8]]),
            torch.tensor([1, 4]),
        )
        write(locs, [12, 90, 91, 92, 6, 7, 8, 9])
        check([1, 0], [9, 12], [[7, 8, 9], [10, 11, 12]], [[1, 1, 1], [1, 1, 1]])

        generations[0] += 1
        extend([0], [11], [1], [42])
        check([0], [12], [[0, 0, 42]], [[0, 0, 1]])

        # PD decode: a transferred request starts with no window, whatever its slot
        # held, and sees what its own verifies append.
        generations[1] += 1
        transferred = SimpleNamespace(
            kv=SimpleNamespace(req_pool_idx=1),
            origin_input_ids=[0] * 9,
            output_ids=[7],
        )
        ctx.start_windows([transferred], generations)
        check([1], [9], [[0, 0, 0]], [[0, 0, 0]])
        locs = ctx.verify_locs(
            torch.tensor([1]), torch.tensor([[9, 10, 11, 12]]), torch.tensor([2])
        )
        write(locs, [20, 21, 93, 94])
        transferred.output_ids = [7, 8, 9]
        ctx.start_windows([transferred], generations)
        check([1], [11], [[0, 20, 21]], [[0, 1, 1]])


class TestCorrectorDraftSampler(unittest.TestCase):
    def test_captured_rollout_reads_the_graph_inputs(self):
        torch.manual_seed(7)
        config = {
            "d_model": 4,
            "heads": 2,
            "layers": 1,
            "context_window": 3,
            "context_attention": True,
        }
        corrector = TransformerCorrector(6, config, eps=1e-5)
        context = CorrectorContext(
            corrector,
            pool=SimpleNamespace(maybe_get_custom_mem_pool=lambda: None),
            req_pool_rows=3,
        )
        context.kv.normal_()
        context.ring.min_pos[2] = 6
        embed = nn.Embedding(11, 6)
        lm_head = nn.Linear(6, 11, bias=False)
        embed_proj = nn.Sequential(
            nn.Linear(10, 5, bias=False), nn.SiLU(), nn.Linear(5, 11, bias=False)
        )
        block_size, slots = 4, torch.tensor([2, 0])
        hidden_states = torch.randn(2 * block_size, 6)
        input_ids = torch.randint(0, 11, (2 * block_size,))
        positions = (torch.tensor([[8], [5]]) + torch.arange(block_size)).flatten()

        def sampler(num_unary_slots, candidate_pool_size, sampling_enabled=False):
            return CorrectorDraftSampler(
                corrector=corrector,
                context=context,
                embedding=embed,
                lm_head=lm_head,
                embed_proj=embed_proj,
                tp_group=SimpleNamespace(world_size=1, all_gather=lambda x, dim: x),
                vocab_size=11,
                block_size=block_size,
                num_unary_slots=num_unary_slots,
                candidate_pool_size=candidate_pool_size,
                sampling_enabled=sampling_enabled,
                max_bs=2,
            )

        def capture(head):
            head.capture_hook(
                None,
                SimpleNamespace(hidden_states=hidden_states),
                SimpleNamespace(
                    input_ids=input_ids, req_pool_indices=slots, positions=positions
                ),
                hidden_states.shape[0],
            )

        hidden = hidden_states.view(2, block_size, -1)[:, 1:]
        context_kv, context_mask = context.gather(slots, torch.tensor([8, 5]))
        self.assertEqual(context_mask.tolist(), [[0, 1, 1], [1, 1, 1]])
        unary = lm_head(hidden)
        for prefix in (0, 1):
            expected, _ = corrector.propose(
                hidden=hidden,
                anchors=input_ids.view(2, block_size)[:, 0],
                context_kv=context_kv,
                context_mask=context_mask,
                token_inputs=corrector.token_table(embed, 11),
                embed_proj=embed_proj,
                unary_picks=unary[:, :prefix].argmax(-1),
                unary_scores=unary[:, prefix:],
            )
            # A pool as large as the vocabulary scores the whole vocabulary.
            for pool_size in (0, 11):
                head = sampler(prefix, pool_size)
                capture(head)
                torch.testing.assert_close(head.out.view(2, -1), expected)
                eager, _, _ = head.propose(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    req_pool_indices=slots,
                    positions=positions,
                    in_cuda_graph=False,
                )
                torch.testing.assert_close(eager, expected)

        # Sampled: request 0 draws, request 1 is greedy, slot 1 is uncorrected.
        sampling_info = SimpleNamespace(
            temperatures=torch.tensor([[0.8], [0.0]]), top_ks=torch.tensor([50, 1])
        )
        head = sampler(1, 5, sampling_enabled=True)
        head.stage_sampling_params(bs=2, sampling_info=sampling_info)
        torch.manual_seed(3)
        capture(head)
        tokens = head.out.view(2, -1)
        torch.manual_seed(3)
        eager = head.propose(
            hidden_states=hidden_states,
            input_ids=input_ids,
            req_pool_indices=slots,
            positions=positions,
            in_cuda_graph=False,
            sampling=head.eager_sampling(bs=2, sampling_info=sampling_info),
        )
        for captured, ran in zip((tokens, head.candidate_out, head.q_out), eager):
            torch.testing.assert_close(captured, ran)
        # What the accept scatters is each slot's proposal distribution.
        dense = torch.zeros(2, block_size - 1, 11).scatter_(
            -1, head.candidate_out, head.q_out
        )
        torch.testing.assert_close(dense.sum(-1), torch.ones(2, block_size - 1))
        self.assertTrue((dense.gather(-1, tokens[..., None]) > 0).all())
        point_mass = torch.zeros(2, block_size - 1, 11).scatter_(
            -1, tokens[..., None], 1.0
        )
        torch.testing.assert_close(dense[:, 0], point_mass[:, 0])
        torch.testing.assert_close(dense[1], point_mass[1])
        self.assertLess(dense[0, 1:].max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
