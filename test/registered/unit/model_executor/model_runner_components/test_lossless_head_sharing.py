"""Embedded MTP releases its duplicate head before memory-pool sizing."""

import os
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.nemotron_h_mtp import NemotronHForCausalLMMTP
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class TestLosslessHeadSharing(CustomTestCase):
    def test_share_full_module_before_pool_and_preserve_standalone_head(self):
        target_head = torch.nn.Linear(3, 4, bias=False)
        target_head._lossless_bf16_source = weakref.ref(target_head.weight)
        target_head.register_buffer("_lossless_bf16_payload", torch.ones(8))
        embed = torch.nn.Embedding(4, 3)
        target = SimpleNamespace(
            lm_head=target_head,
            get_embed_and_head=lambda: (embed.weight, target_head.weight),
        )
        draft = NemotronHForCausalLMMTP.__new__(NemotronHForCausalLMMTP)
        torch.nn.Module.__init__(draft)
        draft.config = SimpleNamespace(tie_word_embeddings=False)
        draft._owns_lm_head = True
        draft.model = SimpleNamespace(embed_tokens=torch.nn.Embedding(4, 3))
        draft.lm_head = torch.nn.Linear(3, 4, bias=False)
        old_head = weakref.ref(draft.lm_head)
        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        worker.draft_runner = SimpleNamespace(model=draft)
        worker.target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(model=target)
        )
        worker.speculative_algorithm = SimpleNamespace(is_eagle3=lambda: False)
        module = "sglang.srt.speculative.eagle_worker_v2"
        with (
            patch.dict(os.environ, {"SGLANG_SM120_LOSSLESS_BF16_HEAD": "1"}),
            patch(
                module + ".get_spec",
                return_value=SimpleNamespace(speculative_token_map=None),
            ),
            patch(
                module + ".get_model",
                return_value=SimpleNamespace(is_startup_weight_load_overlap=False),
            ),
            patch("torch.cuda.empty_cache"),
            patch("torch.cuda.synchronize"),
        ):
            self.assertFalse(worker._init_lossless_lm_head_early())
            self.assertIs(draft.lm_head, old_head())
            draft._owns_lm_head = False
            self.assertTrue(worker._init_lossless_lm_head_early())
            self.assertIsNone(old_head())
            self.assertIs(draft.lm_head, target_head)
            self.assertIs(draft.model.embed_tokens.weight, embed.weight)
            self.assertIs(
                draft.lm_head._lossless_bf16_payload, target_head._lossless_bf16_payload
            )
            self.assertIs(draft.lm_head._lossless_bf16_source(), target_head.weight)


if __name__ == "__main__":
    unittest.main()
