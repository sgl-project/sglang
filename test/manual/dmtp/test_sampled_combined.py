"""The combined path must preserve top-k q and avoid duplicate embedding work."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.models.dflash import CandidateSelector
from sglang.srt.speculative.dmtp_head_conditioner_graph import (
    SampledHeadConditionerGraphs,
    conditioned_embeddings,
)
from sglang.srt.speculative.dmtp_selector_graph import native_proposal
from sglang.srt.speculative.mtp_first_controller import MTPFirstController
from sglang.srt.speculative.mtp_prefix_contract import MTPPrefixConditioner


def cpu_topk(hidden, head, k):
    vals, ids = torch.nn.functional.linear(hidden, head.weight).float().topk(k, dim=-1)
    return ids, vals, None


class CombinedTest(unittest.TestCase):
    def test_signature_invalidates_embedding_residual_and_runtime_changes(self):
        def fixture():
            model = torch.nn.Module()
            model.candidate_selector = None
            model.draft_config = SimpleNamespace(
                output_multiplier=1.0, final_logit_softcapping=None
            )
            head = torch.nn.Linear(8, 32, bias=False)
            head.shard_indices = SimpleNamespace(
                num_org_elements=32,
                num_org_elements_padded=32,
                num_added_elements=0,
                org_vocab_start_index=0,
                added_vocab_start_index=32,
            )
            embedding = torch.nn.Embedding(32, 8)
            condition = MTPPrefixConditioner(8, 4, 1e-6)
            model.mtp_prefix_conditioner = condition
            worker = SimpleNamespace(
                target_worker=SimpleNamespace(
                    model_runner=SimpleNamespace(
                        model=SimpleNamespace(
                            lm_head=head, get_input_embeddings=lambda: embedding
                        )
                    )
                ),
                draft_model=model,
                _full_embed_gpu=None,
                _noise_embed_scale=1.0,
            )
            runner = SampledHeadConditionerGraphs.__new__(SampledHeadConditionerGraphs)
            runner.worker, runner.model, runner.top_k = worker, model, 16
            runner.controller = SimpleNamespace(
                worker=worker,
                conditioner_graphs=SimpleNamespace(original=condition.forward),
            )
            return runner, head, embedding, condition

        changes = (
            lambda r, h, e, c: e.weight.add_(1),
            lambda r, h, e, c: c.down.weight.add_(1),
            lambda r, h, e, c: setattr(c, "eps", 1e-5),
            lambda r, h, e, c: setattr(e, "training", False),
            lambda r, h, e, c: setattr(e, "tp_size", 2),
            lambda r, h, e, c: setattr(r.worker, "_noise_embed_scale", 0.5),
            lambda r, h, e, c: setattr(r.worker, "_full_embed_gpu", torch.randn(32, 8)),
            lambda r, h, e, c: setattr(c, "forward", lambda a, b: a),
        )
        group = SimpleNamespace(world_size=1)
        with (
            patch(
                "sglang.srt.speculative.dmtp_selector_graph.get_tp_group",
                return_value=group,
            ),
            patch(
                "sglang.srt.speculative.dmtp_head_conditioner_graph.get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False),
            ),
            torch.no_grad(),
        ):
            for change in changes:
                runner, head, embedding, condition = fixture()
                before = runner.signature()
                change(runner, head, embedding, condition)
                self.assertNotEqual(before, runner.signature())

    def test_operators_preserve_tokens_support_q_and_all_eight_embeddings(self):
        torch.manual_seed(7)
        for dtype in (torch.float32, torch.bfloat16):
            model = torch.nn.Module()
            model.candidate_selector = CandidateSelector(
                hidden_size=8, vocab_size=32, state_rank=4, top_k=2
            ).to(dtype)
            model.lm_head = torch.nn.Linear(8, 32, bias=False).to(dtype)
            embedding = torch.nn.Embedding(32, 8).to(dtype)
            condition = MTPPrefixConditioner(8, 4, 1e-6).to(dtype)
            runner = SampledHeadConditionerGraphs.__new__(SampledHeadConditionerGraphs)
            runner.model, runner.embedding, runner.condition = (
                model,
                embedding,
                condition,
            )
            runner.top_k = 2
            runner.worker = SimpleNamespace(
                _full_embed_gpu=None, _noise_embed_scale=1.5
            )
            for bs in (1, 2, 7):
                original = torch.randint(0, 32, (bs, 8))
                runner.block_ids = original.clone()
                runner.hidden = torch.randn(bs, 1, 8, dtype=dtype)
                runner.anchors = original[:, 0]
                runner.temperatures = torch.linspace(0.7, 1.3, bs)
                runner.greedy_mask = torch.arange(bs) % 2 == 0
                runner.coins = torch.full((bs, 1), 0.9)
                with patch(
                    "sglang.srt.speculative.dmtp_selector_graph.candidate_topk",
                    side_effect=cpu_topk,
                ):
                    actual = runner.forward(bs)
                    tokens, ids, q = native_proposal(
                        model,
                        runner.hidden,
                        runner.anchors,
                        runner.temperatures,
                        runner.greedy_mask,
                        runner.coins,
                        t1=True,
                    )
                ref_ids = original.clone()
                ref_ids[:, 1] = tokens[:, 0]
                blocks = embedding(ref_ids) * 1.5
                expected = condition(blocks, runner.hidden[:, 0])
                for a, b in zip(actual, (tokens, ids, q, expected)):
                    self.assertTrue(
                        torch.equal(
                            a.contiguous().view(torch.uint8),
                            b.contiguous().view(torch.uint8),
                        )
                    )
                self.assertTrue(torch.equal(actual[3][:, 0], blocks[:, 0]))
                self.assertTrue(torch.equal(actual[3][:, 2:], blocks[:, 2:]))
                torch.testing.assert_close(actual[2].sum(-1), torch.ones(bs, 1))

    def test_full_embedding_and_scale_use_the_original_order(self):
        worker = SimpleNamespace(
            _full_embed_gpu=torch.randn(32, 8), _noise_embed_scale=0.5
        )
        ids, hidden = torch.randint(0, 32, (3, 8)), torch.randn(3, 8)
        module = MTPPrefixConditioner(8, 4, 1e-6)
        embedding = Mock(
            side_effect=AssertionError(
                "full table path should not call embedding module"
            )
        )
        actual = conditioned_embeddings(worker, embedding, module, ids, hidden)
        expected = module(
            torch.nn.functional.embedding(ids, worker._full_embed_gpu) * 0.5, hidden
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_guard_failure_falls_back_without_drawing_random_numbers(self):
        runner = SampledHeadConditionerGraphs.__new__(SampledHeadConditionerGraphs)
        runner.graphs = {1: object()}
        runner.block_ids = torch.zeros(1, 8, dtype=torch.long)
        runner.hidden = torch.zeros(1, 1, 8)
        runner.initial_signature = ("old",)
        runner.signature = lambda: ("changed",)
        runner.fallbacks = 0
        with patch(
            "torch.rand", side_effect=AssertionError("fallback consumed proposal RNG")
        ):
            result = runner.run(
                torch.zeros(1, 8),
                torch.zeros(1, 8, dtype=torch.long),
                SimpleNamespace(is_all_greedy=False),
            )
        self.assertIsNone(result)
        self.assertEqual(runner.fallbacks, 1)

    def test_controller_retains_t1_q_and_uses_prepared_embeddings(self):
        c = MTPFirstController.__new__(MTPFirstController)
        c.hidden = torch.zeros(2, 8)
        c.graph_used = {}
        c.selector_sampling = object()
        tokens = torch.tensor([[7], [9]])
        support = torch.tensor([[[7, 2]], [[3, 9]]])
        q = torch.tensor([[[0.4, 0.6]], [[0.2, 0.8]]])
        embeddings = torch.randn(2, 8, 8)
        c.t1_combined = SimpleNamespace(
            run=lambda *args: (tokens, support, q, embeddings), last_used=True
        )
        c.t1_selector = SimpleNamespace(
            run=Mock(side_effect=AssertionError("separate head replayed"))
        )
        block = torch.zeros(2, 8, dtype=torch.long)
        c.prepare_t1(block)
        self.assertEqual(block[:, 1].tolist(), [7, 9])
        self.assertIs(c.t1_sparse[0], support)
        self.assertIs(c.t1_sparse[1], q)
        self.assertIs(c.prepared_embeds, embeddings)
        self.assertTrue(c.graph_used["t1_combined"])

    def test_worker_prepared_branch_skips_embedding_and_conditioner(self):
        path = (
            Path(__file__).resolve().parents[3]
            / "python/sglang/srt/speculative/dflash_worker_v2.py"
        )
        cls = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "DFlashWorkerV2"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward_batch_generation"
        )
        index = next(
            i
            for i, n in enumerate(method.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "prepared" for t in n.targets)
        )
        code = ast.Module(body=method.body[index : index + 2], type_ignores=[])
        embeddings = torch.randn(2, 8, 8)
        namespace = dict(
            torch=torch,
            dmtp=SimpleNamespace(prepared_embeds=embeddings),
            bs=2,
            draft_slots=8,
            draft_ids=torch.zeros(2, 8, dtype=torch.long),
            self=SimpleNamespace(),
            embed_module=Mock(side_effect=AssertionError("duplicate embedding")),
        )
        exec(compile(code, str(path), "exec"), namespace)
        torch.testing.assert_close(
            namespace["input_embeds"], embeddings.reshape(16, 8), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
