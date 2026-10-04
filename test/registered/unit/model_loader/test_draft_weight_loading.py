"""Regressions for sharing through the normal model-loading lifecycle."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.model_loader import loader as loader_module
from sglang.srt.model_loader.draft_weight_loading import (
    draft_weight_sharing,
    get_draft_weight_sharing,
    unloaded_shared_weight,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDraftWeightLoading(unittest.TestCase):
    def test_shared_and_owned_weights(self):
        for shares_embedding in (False, True):
            with self.subTest(shares_embedding=shares_embedding):
                model = nn.Sequential(nn.Linear(2, 2, bias=False))
                target = nn.Parameter(torch.full((2, 2), 7.0))
                original = model[0].weight
                model[0].quant_method = Mock()
                phases = []

                def share(draft, *, before_load=False):
                    phases.append(before_load)
                    if shares_embedding:
                        draft[0].weight = (
                            unloaded_shared_weight(target) if before_load else target
                        )

                def load_weights(weights):
                    self.assertEqual(model[0].weight.is_meta, shares_embedding)
                    for _, value in weights:
                        weight = model[0].weight
                        if hasattr(weight, "weight_loader"):
                            weight.weight_loader(weight, value)
                        else:
                            weight.data.copy_(value)

                model.load_weights = load_weights
                self._load(model, share)
                self.assertEqual(phases, [True, False])
                torch.testing.assert_close(target, torch.full((2, 2), 7.0))
                postprocess = model[0].quant_method.process_weights_after_loading
                if shares_embedding:
                    self.assertIs(model[0].weight, target)
                    postprocess.assert_not_called()
                else:
                    self.assertIs(model[0].weight, original)
                    torch.testing.assert_close(original, torch.ones(2, 2))
                    postprocess.assert_called_once()
                self.assertIsNone(get_draft_weight_sharing())

    def test_checkpoint_condition_before_loading_and_metadata_fallback(self):
        for names, phases in (({"draft.embed.weight"}, [True, False]), (None, [False])):
            with self.subTest(names=names):
                model = nn.Linear(2, 2, bias=False)
                model.prepare_draft_weight_loading = Mock()
                model.load_weights = Mock()
                seen = []

                def share(draft, *, before_load=False):
                    seen.append(before_load)
                    if before_load:
                        draft.prepare_draft_weight_loading.assert_called_once_with(
                            names
                        )

                with patch.object(
                    loader_module.DefaultModelLoader,
                    "_draft_checkpoint_names",
                    return_value=names,
                ):
                    self._load(model, share)
                self.assertEqual(seen, phases)
                model.load_weights.assert_called_once()

    def test_unbound_placeholder_fails(self):
        model = nn.Linear(2, 2, bias=False)
        model.load_weights = Mock()

        def share(draft, *, before_load=False):
            if before_load:
                draft.weight = unloaded_shared_weight(draft.weight)

        with self.assertRaisesRegex(RuntimeError, "not bound"):
            self._load(model, share)
        self.assertIsNone(get_draft_weight_sharing())

    @staticmethod
    def _load(model, share):
        loader = object.__new__(loader_module.DefaultModelLoader)
        loader.load_config = SimpleNamespace()
        with (
            draft_weight_sharing(share),
            patch.object(loader_module, "_initialize_model", return_value=model),
            patch.object(loader_module, "_get_quantization_config", return_value=None),
            patch.object(loader_module, "is_cuda_alike", return_value=False),
            patch.object(
                loader, "_get_all_weights", return_value=[("weight", torch.ones(2, 2))]
            ),
        ):
            return loader.load_model(
                model_config=SimpleNamespace(dtype=torch.float32),
                device_config=SimpleNamespace(device="cpu"),
            )


if __name__ == "__main__":
    unittest.main()
