"""CPU regressions for sharing decisions made before draft weight loading."""

import ast
import contextlib
import logging
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.model_loader import draft_shared_weights
from sglang.srt.model_loader.draft_shared_weights import (
    DraftSharingContext,
    DraftWeightLoading,
    apply_draft_weight_sharing,
    draft_shares_embedding,
    is_shared_draft_module,
    plan_draft_weight_sharing,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Unquantized:
    pass


class _Quantized:
    pass


class _Vocab(nn.Module):
    def __init__(self, *, device="cpu", dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(8, 4, device=device, dtype=dtype))
        self.weight.weight_loader = self.load
        self.quant_method = _Unquantized()
        self.tp_size = 1
        self.shard_indices = (0, 8)
        self.embedding_dim = 4
        self.org_vocab_size = 8
        if device != "meta":
            with torch.no_grad():
                self.weight.copy_(torch.arange(32, dtype=dtype).reshape(8, 4))

    @staticmethod
    def load(parameter, value):
        with torch.no_grad():
            parameter.copy_(value)


class _Model(nn.Module):
    def __init__(self, *, device="cpu"):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = _Vocab(device=device)
        self.lm_head = _Vocab(device=device)
        self.load_lm_head_from_target = False
        self.keep_embedding = False
        self.keep_head = False

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed(self, embed):
        if not self.keep_embedding and embed is not None:
            self.model.embed_tokens.weight = embed

    def set_embed_and_head(self, embed, head):
        self.set_embed(embed)
        if not self.keep_head:
            self.lm_head.weight = head


class _CheckpointModel(_Model):
    def prepare_draft_weight_loading(self, checkpoint_names):
        self.keep_embedding = "draft.embedding.weight" in checkpoint_names
        self.keep_head = "draft.head.weight" in checkpoint_names


class _Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.language_model = model

    def set_embed_and_head(self, embed, head):
        self.language_model.set_embed_and_head(embed, head)


def _loading(target, draft, *, eagle3=False, token_map=False):
    loading = DraftWeightLoading(
        DraftSharingContext(target, eagle3, token_map, *target.get_embed_and_head()),
        torch.device("cpu"),
    )
    loading.deferred_modules.update(
        module
        for module in draft.modules()
        if isinstance(module, _Vocab)
        and isinstance(module.quant_method, _Unquantized)
        and module.weight.is_meta
    )
    return loading


def _actual_model_policy(filename, class_name):
    """Exercise real model methods without importing GPU-only model backends."""
    root = Path(__file__).resolve().parents[4]
    source = root / "python/sglang/srt/models" / filename
    tree = ast.parse(source.read_text())
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("prepare_draft_weight_loading", "set_embed_and_head")
    ]
    namespace = {
        "_Model": _Model,
        "torch": torch,
        "logger": logging.getLogger(__name__),
    }
    test_class = ast.ClassDef(
        name=class_name,
        bases=[ast.Name(id="_Model", ctx=ast.Load())],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[class_name]


def _actual_method(filename, class_name, method_name, namespace):
    root = Path(__file__).resolve().parents[4]
    source = root / "python/sglang/srt" / filename
    tree = ast.parse(source.read_text())
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    method.decorator_list = []
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                method,
            ],
            type_ignores=[],
        )
    )
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[method_name]


class TestDraftWeightSharing(unittest.TestCase):
    def test_shared_weights_stay_meta_until_loading_has_finished(self):
        target, draft = _Model(), _Model(device="meta")
        originals = [p.detach().clone() for p in target.parameters()]
        loading = _loading(target, draft)
        loading.prepare(draft)

        self.assertEqual(len(loading.shared_modules), 2)
        for module in (draft.model.embed_tokens, draft.lm_head):
            self.assertTrue(module.weight.is_meta)
            module.weight.weight_loader(module.weight, torch.full((8, 4), -100.0))
            self.assertTrue(module.weight.is_meta)

        loading.finish(draft)
        self.assertIs(draft.model.embed_tokens.weight, target.model.embed_tokens.weight)
        self.assertIs(draft.lm_head.weight, target.lm_head.weight)
        for parameter, original in zip(target.parameters(), originals):
            torch.testing.assert_close(parameter, original)
        tokens = torch.tensor([1, 3, 5])
        logits = draft.model.embed_tokens.weight[tokens] @ draft.lm_head.weight.T
        expected = originals[0][tokens] @ originals[1].T
        torch.testing.assert_close(logits, expected)

    def test_plan_uses_overridden_setters_and_leaves_models_unchanged(self):
        target, draft = _Model(), _Model(device="meta")
        draft.keep_embedding = True
        before = dict(draft.named_parameters())
        plan = plan_draft_weight_sharing(draft, *target.get_embed_and_head())
        self.assertEqual(set(plan), {"lm_head.weight"})
        self.assertIs(plan["lm_head.weight"], target.lm_head.weight)
        for name, parameter in draft.named_parameters():
            self.assertIs(parameter, before[name])
        self.assertTrue(draft.keep_embedding)
        self.assertFalse(
            draft_shares_embedding(draft, before["model.embed_tokens.weight"])
        )

    def test_wrapper_delegation_requires_no_module_path_declaration(self):
        target, draft = _Model(), _Wrapper(_Model(device="meta"))
        loading = _loading(target, draft)
        loading.prepare(draft)
        loading.finish(draft)
        self.assertIs(
            draft.language_model.model.embed_tokens.weight,
            target.model.embed_tokens.weight,
        )
        self.assertIs(draft.language_model.lm_head.weight, target.lm_head.weight)

    def test_eagle3_reuses_the_workers_choice_of_setter(self):
        for share_head in (False, True):
            with self.subTest(share_head=share_head):
                target, draft = _Model(), _Model(device="meta")
                draft.load_lm_head_from_target = share_head
                loading = _loading(target, draft, eagle3=True)
                loading.prepare(draft)
                self.assertTrue(draft.model.embed_tokens.weight.is_meta)
                self.assertEqual(draft.lm_head.weight.is_meta, share_head)
                loading.finish(draft)
                self.assertIs(
                    draft.model.embed_tokens.weight, target.model.embed_tokens.weight
                )
                self.assertEqual(
                    draft.lm_head.weight is target.lm_head.weight, share_head
                )

    def test_checkpoint_owned_parameters_are_loaded_and_preserved(self):
        for owned in (
            {"draft.embedding.weight"},
            {"draft.head.weight"},
            {"draft.embedding.weight", "draft.head.weight"},
        ):
            with self.subTest(owned=owned):
                target, draft = _Model(), _CheckpointModel(device="meta")
                loading = _loading(target, draft)
                loading.prepare(draft, owned)
                pairs = [
                    ("draft.embedding.weight", draft.model.embed_tokens),
                    ("draft.head.weight", draft.lm_head),
                ]
                expected = torch.full((8, 4), 42.0)
                for name, module in pairs:
                    self.assertEqual(module.weight.is_meta, name not in owned)
                    module.weight.weight_loader(module.weight, expected)
                loading.finish(draft)
                for name, module in pairs:
                    if name in owned:
                        torch.testing.assert_close(module.weight, expected)

    def test_missing_checkpoint_metadata_uses_normal_allocation(self):
        target, draft = _Model(), _CheckpointModel(device="meta")
        loading = _loading(target, draft)
        loading.prepare(draft)
        self.assertFalse(loading.bindings)
        self.assertFalse(any(p.is_meta for p in draft.parameters()))
        loading.finish(draft)

    def test_changed_checkpoint_ownership_fails_before_binding(self):
        target, draft = _Model(), _CheckpointModel(device="meta")
        loading = _loading(target, draft)
        loading.prepare(draft, ())
        draft.prepare_draft_weight_loading({"draft.embedding.weight"})
        with self.assertRaisesRegex(RuntimeError, "Draft sharing changed"):
            loading.finish(draft)
        self.assertTrue(draft.model.embed_tokens.weight.is_meta)
        self.assertTrue(draft.lm_head.weight.is_meta)

    def test_pp_local_head_can_share_without_a_target_embedding(self):
        target, draft = _Model(), _Model(device="meta")
        target.model.embed_tokens.weight = None
        loading = _loading(target, draft)
        loading.prepare(draft)
        self.assertFalse(draft.model.embed_tokens.weight.is_meta)
        self.assertTrue(draft.lm_head.weight.is_meta)
        local_embed = torch.full((8, 4), 7.0)
        draft.model.embed_tokens.weight.weight_loader(
            draft.model.embed_tokens.weight, local_embed
        )
        loading.finish(draft)
        torch.testing.assert_close(draft.model.embed_tokens.weight, local_embed)
        self.assertIs(draft.lm_head.weight, target.lm_head.weight)

    def test_token_mapped_head_keeps_storage_even_with_a_tied_target(self):
        target, draft = _Model(), _Model(device="meta")
        target.lm_head.weight = target.model.embed_tokens.weight
        loading = _loading(target, draft, token_map=True)
        loading.prepare(draft)
        self.assertTrue(draft.model.embed_tokens.weight.is_meta)
        self.assertFalse(draft.lm_head.weight.is_meta)
        loading.finish(draft)
        self.assertIs(draft.model.embed_tokens.weight, target.model.embed_tokens.weight)
        self.assertIsNot(draft.lm_head.weight, target.lm_head.weight)

    def test_incompatible_weights_use_normal_allocation(self):
        mutations = {
            "dtype": lambda m: setattr(
                m, "weight", nn.Parameter(torch.empty(8, 4, dtype=torch.float64))
            ),
            "shape": lambda m: setattr(m, "weight", nn.Parameter(torch.empty(9, 4))),
            "tp": lambda m: setattr(m, "tp_size", 2),
            "shard": lambda m: setattr(m, "shard_indices", (8, 16)),
            "quantization": lambda m: setattr(m, "quant_method", _Quantized()),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                target, draft = _Model(), _Model(device="meta")
                mutate(target.model.embed_tokens)
                loading = _loading(target, draft)
                loading.prepare(draft)
                self.assertFalse(draft.model.embed_tokens.weight.is_meta)
                self.assertTrue(draft.lm_head.weight.is_meta)
                loading.finish(draft)
                self.assertIsNot(
                    draft.model.embed_tokens.weight, target.model.embed_tokens.weight
                )

    def test_registered_parameter_aliases_are_preserved(self):
        target, draft = _Model(), _Model(device="meta")
        draft.embedding_alias = draft.model.embed_tokens.weight
        loading = _loading(target, draft)
        loading.prepare(draft)
        loading.finish(draft)
        self.assertIs(draft.embedding_alias, draft.model.embed_tokens.weight)
        self.assertIs(draft.embedding_alias, target.model.embed_tokens.weight)

    def test_conflicting_bindings_for_one_parameter_use_normal_allocation(self):
        target, draft = _Model(), _Model(device="meta")
        draft.lm_head.weight = draft.model.embed_tokens.weight
        loading = _loading(target, draft)
        loading.prepare(draft)
        self.assertFalse(loading.bindings)
        self.assertFalse(draft.model.embed_tokens.weight.is_meta)
        self.assertIs(draft.model.embed_tokens.weight, draft.lm_head.weight)
        loading.finish(draft)

    def test_nested_checkpoint_preparation_is_reused_by_wrapper_setter(self):
        target, draft = _Model(), _Wrapper(_CheckpointModel(device="meta"))
        loading = _loading(target, draft)
        loading.prepare(draft, {"draft.embedding.weight"})
        self.assertFalse(draft.language_model.model.embed_tokens.weight.is_meta)
        self.assertTrue(draft.language_model.lm_head.weight.is_meta)
        loading.finish(draft)
        self.assertIs(draft.language_model.lm_head.weight, target.lm_head.weight)
        self.assertIsNot(
            draft.language_model.model.embed_tokens.weight,
            target.model.embed_tokens.weight,
        )

    def test_each_draft_has_an_independent_loading_plan(self):
        target = _Model()
        first, second = _Model(device="meta"), _CheckpointModel(device="meta")
        first_loading, second_loading = (
            _loading(target, first),
            _loading(target, second),
        )
        first_loading.prepare(first)
        second_loading.prepare(second, {"draft.head.weight"})
        first_loading.finish(first)
        second_loading.finish(second)
        self.assertIs(first.lm_head.weight, target.lm_head.weight)
        self.assertIsNot(second.lm_head.weight, target.lm_head.weight)
        self.assertIs(first.model.embed_tokens.weight, second.model.embed_tokens.weight)

    def test_real_postprocessor_never_stages_shared_parameters(self):
        target, draft = _Model(), _CheckpointModel(device="meta")
        loading = _loading(target, draft)
        loading.prepare(draft, {"draft.head.weight"})
        processed = []
        staged = []

        def stage(module, device):
            self.assertFalse(module.weight.is_meta)
            staged.append(module)
            return contextlib.nullcontext()

        for module in (draft.model.embed_tokens, draft.lm_head):
            module.quant_method.process_weights_after_loading = processed.append
        postprocess = _actual_method(
            "model_loader/loader.py",
            "DefaultModelLoader",
            "postprocess_weights",
            {
                "_modules_with_quant_method": lambda model: (
                    (m, m.quant_method)
                    for m in model.modules()
                    if isinstance(m, _Vocab)
                ),
                "is_shared_draft_module": is_shared_draft_module,
                "device_loading_context": stage,
            },
        )
        token = draft_shared_weights._draft_loading.set(loading)
        try:
            postprocess(draft, torch.device("cpu"))
        finally:
            draft_shared_weights._draft_loading.reset(token)
        self.assertEqual(staged, [draft.lm_head])
        self.assertEqual(processed, [draft.lm_head])
        loading.finish(draft)

    def test_direct_vocab_weight_loader_also_skips_shared_parameter(self):
        target, draft = _Model(), _Model(device="meta")
        loading = _loading(target, draft)
        loading.prepare(draft)
        load = _actual_method(
            "layers/vocab_parallel_embedding.py",
            "VocabParallelEmbedding",
            "weight_loader",
            {"torch": torch},
        )
        load(
            draft.model.embed_tokens,
            draft.model.embed_tokens.weight,
            torch.full((8, 4), -1.0),
        )
        self.assertTrue(draft.model.embed_tokens.weight.is_meta)
        loading.finish(draft)

    def test_finish_rejects_an_unbound_meta_parameter(self):
        target, draft = _Model(), _Model(device="meta")
        draft.unexpected = nn.Parameter(torch.empty(1, device="meta"))
        loading = _loading(target, draft)
        loading.prepare(draft)
        with self.assertRaisesRegex(RuntimeError, "Unbound draft parameters"):
            loading.finish(draft)

    def test_quantized_consumer_of_tied_parameter_requires_real_storage(self):
        target, draft = _Model(), _Model(device="meta")
        target.lm_head.weight = target.model.embed_tokens.weight
        draft.lm_head.weight = draft.model.embed_tokens.weight
        draft.lm_head.quant_method = _Quantized()
        loading = _loading(target, draft)
        loading.prepare(draft)
        self.assertFalse(loading.bindings)
        self.assertFalse(draft.model.embed_tokens.weight.is_meta)
        self.assertIs(draft.model.embed_tokens.weight, draft.lm_head.weight)
        loading.finish(draft)

    def test_module_reassignment_is_planned_without_mutating_the_original(self):
        class TiedModel(_Model):
            def set_embed_and_head(self, embed, head):
                self.set_embed(embed)
                self.lm_head = self.model.embed_tokens

        target, draft = _Model(), TiedModel(device="meta")
        old_head = draft.lm_head
        plan = plan_draft_weight_sharing(draft, *target.get_embed_and_head())
        self.assertEqual(set(plan), {"model.embed_tokens.weight", "lm_head.weight"})
        self.assertIs(plan["lm_head.weight"], target.model.embed_tokens.weight)
        self.assertIs(draft.lm_head, old_head)
        self.assertIsNot(draft.lm_head, draft.model.embed_tokens)

    def test_setter_failure_cannot_leave_the_draft_partially_rebound(self):
        class FailingModel(_Model):
            def set_embed_and_head(self, embed, head):
                self.model.embed_tokens.weight = embed
                raise RuntimeError("setter failed")

        target, draft = _Model(), FailingModel(device="meta")
        original = draft.model.embed_tokens.weight
        with self.assertRaisesRegex(RuntimeError, "setter failed"):
            plan_draft_weight_sharing(draft, *target.get_embed_and_head())
        self.assertIs(draft.model.embed_tokens.weight, original)

    def test_real_dots3_checkpoint_rule_drives_both_planning_and_sharing(self):
        model_type = _actual_model_policy(
            "dots3_common/nextn.py", "Dots3NoteForCausalLMNextN"
        )
        for owns_embedding in (False, True):
            with (
                self.subTest(owns_embedding=owns_embedding),
                patch("torch.cuda.empty_cache"),
                patch("torch.cuda.synchronize"),
            ):
                target, draft = _Model(), model_type(device="meta")
                keys = {"model.mtp.embed_tokens.weight"} if owns_embedding else set()
                loading = _loading(target, draft)
                loading.prepare(draft, keys)
                self.assertEqual(
                    draft.model.embed_tokens.weight.is_meta, not owns_embedding
                )
                draft.prepare_draft_weight_loading(keys)
                loading.finish(draft)
                original = draft.model.embed_tokens.weight
                apply_draft_weight_sharing(draft, *target.get_embed_and_head())
                self.assertIs(draft.model.embed_tokens.weight, original)
                self.assertIs(draft.lm_head.weight, target.lm_head.weight)

    def test_real_nemotron_checkpoint_rule_keeps_standalone_head(self):
        model_type = _actual_model_policy(
            "nemotron_h_mtp.py", "NemotronHForCausalLMMTP"
        )
        for standalone in (False, True):
            with (
                self.subTest(standalone=standalone),
                patch("torch.cuda.empty_cache"),
                patch("torch.cuda.synchronize"),
            ):
                target, draft = _Model(), model_type(device="meta")
                keys = {"language_model.mtp.layers.0.weight", "lm_head.weight"}
                if not standalone:
                    keys.add("backbone.layers.0.weight")
                loading = _loading(target, draft)
                loading.prepare(draft, keys)
                self.assertEqual(draft.lm_head.weight.is_meta, not standalone)
                draft.prepare_draft_weight_loading(keys)
                loading.finish(draft)
                original = draft.lm_head.weight
                apply_draft_weight_sharing(draft, *target.get_embed_and_head())
                self.assertIs(draft.lm_head.weight, original)
                self.assertIs(
                    draft.model.embed_tokens.weight, target.model.embed_tokens.weight
                )


if __name__ == "__main__":
    unittest.main()
