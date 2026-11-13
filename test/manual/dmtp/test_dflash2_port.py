"""CPU regressions for checkpoint loading, proposal geometry and SWA lifecycle."""

import ast
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.dflash import CandidateSelector, DFlash2DraftModel
from sglang.srt.models.dflash_mtp_prefix import DFlash2T1ResidualDraftModel
from sglang.srt.speculative.dflash2_anchor_window import (
    install_anchor_window,
    requires_anchor_window,
    validate_anchor_capture,
)
from sglang.srt.speculative.dflash2_contract import checked_weights, validate_dflash2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.dmtp_runtime import validate_mtp_target
from sglang.srt.speculative.dmtp_selector_graph import (
    DMTPSelectorGraphs,
    _tensor_signature,
    native_proposal,
)
from sglang.srt.speculative.mtp_first_controller import MTPFirstController
from sglang.srt.speculative.mtp_prefix_contract import (
    normalize_residual_config,
    validate_config,
)
from sglang.srt.speculative.native_mtp_bridge import validate_bridge_proposals


def residual_config():
    return SimpleNamespace(
        hidden_size=5120,
        vocab_size=248320,
        num_hidden_layers=5,
        num_target_layers=64,
        head_dim=256,
        rms_norm_eps=1e-6,
        block_size=8,
        sliding_window=None,
        rope_theta=10000000.0,
        partial_rotary_factor=0.25,
        layer_types=["full_attention"] * 5,
        dflash_config=dict(
            block_size=8,
            known_prefix_tokens=2,
            mtp_prefix=True,
            mtp_injection_mode="t1_residual",
            mtp_t1_residual=True,
            mtp_hidden_only=True,
            mtp_prefix_injection="t1_residual",
            mtp_prefix_concat_order=[],
            mtp_prefix_rank=256,
            mtp_prefix_eps=1e-6,
            mtp_proposal_token_count=1,
            dflash_proposal_token_count=6,
            draft_prediction_offsets=list(range(2, 8)),
            context_fc_input_size=25600,
            conv_kernel_size=2,
            conv_group_size=16,
            selector_rank=256,
            selector_top_k=16,
            conv_base_kernel_identity_folded=True,
            target_layer_ids=[0, 15, 30, 45, 60],
            target_kv_layer_ids=[],
            legacy_double_mlp_norm=False,
            swa_window_size=4096,
            draft_layer_context_windows=[4096] * 5,
            block_attention="bidirectional",
            context_window_boundary="anchor_relative_not_query_relative",
            context_kv_boundary="strictly_before_anchor",
        ),
    )


def ordinary():
    model = type("DFlash2DraftModel", (), {})()
    model.config = residual_config()
    cfg = model.config.dflash_config
    cfg["mtp_prefix"] = False
    return model


class CheckpointTests(unittest.TestCase):
    def test_explicit_20000_export_alias_is_normalized_without_mutation(self):
        config = residual_config()
        cfg = config.dflash_config
        for name in (
            "mtp_injection_mode",
            "mtp_prefix_concat_order",
            "draft_prediction_offsets",
            "context_fc_input_size",
            "context_window_boundary",
        ):
            del cfg[name]
        cfg.update(
            mtp_prefix_injection="add_to_t1_input_embedding",
            mtp_drop_anchor=False,
            mtp_t1_fusion=False,
            mtp_t1_input_slot=1,
            input_layout=["A", "T1"] + ["MASK"] * 6,
            mtp_hidden_index_relative_to_T1=-2,
            target_context_end_offset=0,
            inference_requires_t1_residual_adapter=True,
        )
        config.is_causal = False
        before = deepcopy(cfg)
        normalized = normalize_residual_config(config)
        validate_config(normalized)
        self.assertEqual(cfg, before)
        for name, value in (
            ("mtp_drop_anchor", True),
            ("mtp_t1_input_slot", 0),
            ("target_context_end_offset", -1),
            ("context_fc_input_size", 5120),
            ("context_window_boundary", "query_relative"),
        ):
            bad = deepcopy(config)
            bad.dflash_config[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                normalize_residual_config(bad)

    def test_qsa_core_is_present_but_private_mtp_state_is_not_supported(self):
        validate_mtp_target(SimpleNamespace())
        qsa = SimpleNamespace(
            indexer_n_heads=8,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=4096,
            indexer_compress_ratio=2,
        )
        for config in (qsa, SimpleNamespace(text_config=qsa)):
            with self.assertRaisesRegex(ValueError, "compressed/indexer/ring"):
                validate_mtp_target(config)

    def test_only_double_anchor_t1_residual(self):
        config = residual_config()
        validate_config(config)
        validate_dflash2(config)
        for name, value in (
            ("mtp_drop_anchor", True),
            ("mtp_t1_fusion", True),
            ("known_prefix_tokens", 1),
            ("mtp_injection_mode", "hidden_only"),
        ):
            changed = deepcopy(config)
            changed.dflash_config[name] = value
            with self.assertRaises(ValueError):
                validate_config(changed)

    def test_contract_rejects_wrong_semantics(self):
        for key, value in (
            ("conv_base_kernel_identity_folded", False),
            ("conv_group_size", 17),
            ("selector_rank", 0),
            ("draft_layer_context_windows", [4096, 8192, 4096, 4096, 4096]),
            ("target_layer_ids", [0, 15, 30, 45, 64]),
            ("block_attention", "causal"),
        ):
            config = residual_config()
            config.dflash_config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(config)
                validate_dflash2(config)

    def test_native_components_not_legacy_copies(self):
        self.assertIs(
            DFlash2T1ResidualDraftModel.compute_candidates,
            DFlash2DraftModel.compute_candidates,
        )
        self.assertIs(
            DFlash2T1ResidualDraftModel._transform_unary_logits,
            DFlash2DraftModel._transform_unary_logits,
        )

    def test_loader_rejects_missing_duplicate_wrong_shape_and_nan(self):
        model = torch.nn.Module()
        model.candidate_selector = torch.nn.Linear(2, 2, bias=False)
        weights = [("candidate_selector.weight", torch.ones(2, 2))]
        self.assertEqual(len(list(checked_weights(model, weights))), 1)
        for bad in (
            [],
            weights * 2,
            [("unknown", torch.ones(2, 2))],
            [("candidate_selector.weight", torch.ones(1, 2))],
            [("candidate_selector.weight", torch.full((2, 2), float("nan")))],
        ):
            with self.assertRaises(ValueError):
                list(checked_weights(model, bad))


class WindowTests(unittest.TestCase):
    def test_only_explicit_ordinary_df2(self):
        self.assertTrue(requires_anchor_window(ordinary()))
        for name in ("DFlashDraftModel", "DFlash2T1ResidualDraftModel"):
            self.assertFalse(requires_anchor_window(type(name, (), {})()))
        model = ordinary()
        model.config.dflash_config.update(
            swa_window_size=None, draft_layer_context_windows=None
        )
        self.assertFalse(requires_anchor_window(model))

    def test_bad_window_or_query_relative_backend_rejected(self):
        for key, value in (
            ("block_attention", "causal"),
            ("draft_layer_context_windows", [8192] * 5),
        ):
            model = ordinary()
            model.config.dflash_config[key] = value
            with self.assertRaises(ValueError):
                requires_anchor_window(model)
        model = ordinary()
        model.config.sliding_window = 4096
        with self.assertRaises(ValueError):
            requires_anchor_window(model)

    def test_install_then_capture_assertion_and_eager(self):
        class Adapter:
            capture_calls = 0

            def __init__(self, backend):
                self.backend = backend

        backend = SimpleNamespace(fa_impl_ver=3)
        runner = SimpleNamespace(
            attn_backend=backend,
            runner_ids=SimpleNamespace(is_translating=False),
            decode_cuda_graph_runner=object(),
        )
        worker = SimpleNamespace(
            draft_model=ordinary(),
            draft_model_runner=runner,
            use_compact_draft_cache=False,
            block_size=8,
        )
        with (
            patch(
                "sglang.srt.runtime_context.get_schedule",
                return_value=SimpleNamespace(max_running_requests=32),
            ),
            patch("sglang.srt.speculative.mtp_anchor_fa3.AnchorFA3Forward", Adapter),
        ):
            install_anchor_window(worker)
            self.assertIs(backend.forward_extend, backend.dflash2_anchor_window)
            with self.assertRaisesRegex(RuntimeError, "not captured"):
                validate_anchor_capture(worker)
            backend.dflash2_anchor_window.capture_calls = 1
            validate_anchor_capture(worker)
            runner.decode_cuda_graph_runner = None
            backend.dflash2_anchor_window.capture_calls = 0
            validate_anchor_capture(worker)
            with self.assertRaisesRegex(RuntimeError, "twice"):
                install_anchor_window(worker)

    def test_split_worker_lifecycle_calls_real_helpers(self):
        root = Path(__file__).resolve().parents[3]
        tree = ast.parse(
            (root / "python/sglang/srt/speculative/dflash_worker_v2.py").read_text()
        )
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DFlashWorkerV2"
        )
        for method_name, first, second in (
            (
                "init_attention_backends",
                "init_attention_backends",
                "install_anchor_window",
            ),
            ("init_cuda_graphs", "init_cuda_graphs", "validate_anchor_capture"),
        ):
            method = next(
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == method_name
            )
            calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)]

            def line(name):
                return next(
                    n.lineno
                    for n in calls
                    if getattr(n.func, "id", None) == name
                    or getattr(n.func, "attr", None) == name
                )

            self.assertLess(line(first), line(second))


class ProposalTests(unittest.TestCase):
    def test_native_verifier_receives_t1_distribution_and_clears_all_rows(self):
        ids = torch.tensor([[[2, 7]] * 7])
        q = torch.tensor([[[0.25, 0.75]] * 7])
        for block in (False, True):
            worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
            worker._draft_probs_buf = None
            worker._block_verification = block

            def verify(**kwargs):
                self.assertEqual(kwargs["block_verification"], block)
                dense = kwargs["draft_probs"]
                self.assertEqual(dense.shape, (1, 7, 16))
                self.assertEqual(dense[0, 0, 2].item(), 0.25)
                self.assertEqual(dense[0, 0, 7].item(), 0.75)
                torch.testing.assert_close(dense.sum(-1), torch.ones(1, 7))
                return torch.tensor([1]), torch.tensor([9]), None

            with patch(
                "sglang.srt.speculative.dflash_worker_v2.accept_sampling",
                side_effect=verify,
            ):
                worker._selector_sampling_accept(
                    candidates=torch.zeros(1, 8, dtype=torch.long),
                    next_token_logits=torch.zeros(8, 16),
                    candidate_ids=ids,
                    q_rows=q,
                    sampling_info=object(),
                    draft_input=object(),
                )
            self.assertEqual(torch.count_nonzero(worker._draft_probs_buf).item(), 0)

    def test_selector_topk_change_invalidates_graph_signature(self):
        runner = DMTPSelectorGraphs.__new__(DMTPSelectorGraphs)
        runner.top_k = 2
        runner.model = torch.nn.Module()
        runner.model.candidate_selector = CandidateSelector(
            hidden_size=4, vocab_size=16, state_rank=2, top_k=2
        )
        runner.model.draft_config = SimpleNamespace(
            output_multiplier=1.0, final_logit_softcapping=None
        )
        head = torch.nn.Linear(4, 16, bias=False)
        head.shard_indices = SimpleNamespace(
            num_org_elements=16,
            num_org_elements_padded=16,
            num_added_elements=0,
            org_vocab_start_index=0,
            added_vocab_start_index=16,
        )
        runner.controller = SimpleNamespace(
            worker=SimpleNamespace(
                target_worker=SimpleNamespace(
                    model_runner=SimpleNamespace(model=SimpleNamespace(lm_head=head))
                )
            )
        )
        with patch(
            "sglang.srt.speculative.dmtp_selector_graph.get_tp_group",
            return_value=SimpleNamespace(world_size=1),
        ):
            before = runner.signature()
            runner.model.candidate_selector.top_k = 4
            self.assertNotEqual(before, runner.signature())

    def test_non_selector_tail_keeps_six_independent_topk_q_rows(self):
        model = SimpleNamespace(
            candidate_selector=None,
            lm_head=object(),
            _transform_unary_logits=lambda x: x.float(),
        )
        ids = torch.tensor([[3, 7]]).repeat(6, 1)
        unary = torch.tensor([[1.0, 0.0]]).repeat(6, 1)
        with patch(
            "sglang.srt.speculative.dmtp_selector_graph.candidate_topk",
            return_value=(ids, unary, None),
        ):
            tokens, support, q = native_proposal(
                model,
                torch.zeros(1, 6, 4),
                torch.tensor([9]),
                torch.ones(1),
                torch.tensor([False]),
                torch.full((1, 6), 0.9),
                t1=False,
                top_k=2,
            )
        self.assertEqual(tokens.tolist(), [[7] * 6])
        self.assertEqual(support.shape, (1, 6, 2))
        torch.testing.assert_close(q, unary.softmax(-1).view(1, 6, 2))

    def test_private_bridge_accepts_only_supported_proposal_owners(self):
        for target_only, selector in ((False, True),):
            c = SimpleNamespace(
                target_only=target_only,
                native_selector=selector,
                trace_dir=None,
                batch_trace=None,
            )
            validate_bridge_proposals(c)
            for name in ("trace_dir", "batch_trace"):
                other = deepcopy(c)
                setattr(other, name, "/trace")
                with self.assertRaises(ValueError):
                    validate_bridge_proposals(other)
        with self.assertRaises(ValueError):
            validate_bridge_proposals(
                SimpleNamespace(target_only=False, trace_dir=None, batch_trace=None)
            )

    def test_t1_uses_native_walk_without_learned_edges(self):
        selector = CandidateSelector(
            hidden_size=4, vocab_size=16, state_rank=2, top_k=2
        )
        model = SimpleNamespace(candidate_selector=selector, lm_head=object())
        hidden = torch.zeros(2, 1, 4)
        ids = torch.tensor([[3, 7], [4, 8]])
        unary = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        with (
            patch(
                "sglang.srt.speculative.dmtp_selector_graph.candidate_topk",
                return_value=(ids, unary, None),
            ),
            patch.object(
                selector,
                "build_lattice",
                side_effect=AssertionError("T1 is not a DF edge"),
            ),
        ):
            tokens, support, q = native_proposal(
                model,
                hidden,
                torch.zeros(2, dtype=torch.long),
                torch.ones(2),
                torch.tensor([True, False]),
                torch.tensor([[0.9], [0.9]]),
                t1=True,
            )
        self.assertEqual(tokens.tolist(), [[3], [8]])
        self.assertEqual(support.shape, (2, 1, 2))
        torch.testing.assert_close(q[0], torch.tensor([[1.0, 0.0]]), rtol=0, atol=0)
        torch.testing.assert_close(q[1, 0], unary[1].softmax(-1))

    def test_seven_q_rows_include_actual_t1_and_six_tail_rows(self):
        for slots in (8,):
            c = MTPFirstController.__new__(MTPFirstController)
            c.native_selector, c.draft_slots = True, slots
            c.graph_used = {}
            c.selector_sampling = SimpleNamespace(is_all_greedy=False)
            c.worker = SimpleNamespace(_selector_sample=None)
            c.t1_sparse = (torch.full((2, 1, 2), 3), torch.full((2, 1, 2), 0.5))

            def run(hidden, anchors, info):
                self.assertEqual(hidden.shape, (2, 6, 4))
                self.assertEqual(anchors.tolist(), [9, 10])
                return (
                    torch.ones(2, 6, dtype=torch.long),
                    torch.full((2, 6, 2), 4),
                    torch.full((2, 6, 2), 0.5),
                )

            c.tail_selector = SimpleNamespace(run=run, last_used=False)
            blocks = torch.zeros(2, 8, dtype=torch.long)
            blocks[:, 1] = torch.tensor([9, 10])
            result = c.proposal(torch.zeros(2, slots, 4), blocks, None)
            self.assertEqual(result.shape, (2, 7))
            self.assertEqual(result[:, 0].tolist(), [9, 10])
            self.assertEqual(c.worker._selector_sample[0].shape, (2, 7, 2))
            self.assertEqual(c.worker._selector_sample[0][:, 0, 0].tolist(), [3, 3])

    def test_weight_version_and_replacement_change_signature(self):
        x = torch.ones(4)
        before = _tensor_signature(x)
        x.add_(1)
        self.assertNotEqual(before, _tensor_signature(x))
        self.assertNotEqual(_tensor_signature(x), _tensor_signature(x.clone()))


if __name__ == "__main__":
    unittest.main()
