# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.models.grok import Grok1ForCausalLM
from sglang.srt.models.locate_anything import LocateAnythingForConditionalGeneration
from sglang.srt.models.unlimited_ocr import UnlimitedOCRForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAdditionalPartialWeightLoad(unittest.TestCase):
    def test_grok_composes_legacy_and_partial_diagnostic_controls(self):
        model = object.__new__(Grok1ForCausalLM)
        model.config = SimpleNamespace(num_local_experts=0)
        model.loaded_param_names = set()
        model.named_parameters = lambda: iter([])

        with self.assertRaisesRegex(ValueError, "did not hit any names"):
            Grok1ForCausalLM.load_weights(
                model, [], check_hit_names=True, is_full_load=True
            )

        for check_hit_names, is_full_load in (
            (False, True),
            (True, False),
            (False, False),
        ):
            with self.subTest(
                check_hit_names=check_hit_names, is_full_load=is_full_load
            ):
                self.assertEqual(
                    Grok1ForCausalLM.load_weights(
                        model,
                        [],
                        check_hit_names=check_hit_names,
                        is_full_load=is_full_load,
                    ),
                    set(),
                )

    def test_unlimited_ocr_only_rejects_missing_weights_on_full_load(self):
        model = object.__new__(UnlimitedOCRForCausalLM)
        model.named_parameters = lambda: iter([("missing.weight", torch.empty(1))])
        model.post_load_weights = Mock()

        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            UnlimitedOCRForCausalLM.load_weights(model, [], is_full_load=True)

        UnlimitedOCRForCausalLM.load_weights(model, [], is_full_load=False)
        model.post_load_weights.assert_called_once_with()

    def test_locate_anything_only_warns_about_missing_weights_on_full_load(self):
        model = object.__new__(LocateAnythingForConditionalGeneration)
        model.config = SimpleNamespace(
            text_config=SimpleNamespace(tie_word_embeddings=False)
        )
        model.named_parameters = lambda: iter([("missing.weight", torch.empty(1))])

        with self.assertLogs("sglang.srt.models.locate_anything", level="WARNING"):
            LocateAnythingForConditionalGeneration.load_weights(
                model, [], is_full_load=True
            )

        with self.assertNoLogs("sglang.srt.models.locate_anything", level="WARNING"):
            LocateAnythingForConditionalGeneration.load_weights(
                model, [], is_full_load=False
            )


if __name__ == "__main__":
    unittest.main()
