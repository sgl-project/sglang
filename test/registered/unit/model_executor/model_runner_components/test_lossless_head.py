"""Lossless BF16 head loading, sharing, and update lifecycle."""

import os
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.gemm.sm120_lossless_bf16_head import lossless_bf16_head_enabled
from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.models.nemotron_h_mtp import NemotronHForCausalLMMTP
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class TestLosslessHeadLoading(CustomTestCase):
    def test_modelopt_unquantized_head_prepares_after_loading(self):
        config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "packed_modules_mapping": {},
                "quantized_layers": {
                    "model.layers.0.experts.up_proj": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    }
                },
            }
        )
        head = ParallelLMHead.__new__(ParallelLMHead)
        torch.nn.Module.__init__(head)
        method = config.get_quant_method(head, prefix="lm_head")
        self.assertIsInstance(method, UnquantizedLinearMethod)
        with (
            patch.dict(os.environ, {"SGLANG_SM120_LOSSLESS_BF16_HEAD": "1"}),
            patch("sglang.srt.layers.quantization.unquant._is_cpu", False),
            patch(
                "sglang.kernels.ops.gemm.sm120_lossless_bf16_head.prepare_lossless_bf16_head"
            ) as prepare,
        ):
            method.process_weights_after_loading(head)
            prepare.assert_called_once_with(head)


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


class TestLosslessHeadUpdateGuard(CustomTestCase):
    def test_all_public_updates_reject_before_mutation(self):
        fake = SimpleNamespace(_assert_weight_cache_inactive=Mock(), get_model=Mock())
        calls = (
            (WeightUpdater.update_weights_from_disk, ("unused", "auto")),
            (WeightUpdater.update_weights_from_distributed, ([], [], [], "unused")),
            (WeightUpdater.update_weights_from_tensor, ([],)),
            (WeightUpdater.update_weights_from_ipc, (None,)),
        )
        with patch.dict(os.environ, {"SGLANG_SM120_LOSSLESS_BF16_HEAD": "1"}):
            lossless_bf16_head_enabled.cache_clear()
            try:
                for method, args in calls:
                    with self.subTest(method=method.__name__):
                        success, error = method(fake, *args)
                        self.assertFalse(success)
                        self.assertIn("SGLANG_SM120_LOSSLESS_BF16_HEAD", error)
                fake.get_model.assert_not_called()
            finally:
                lossless_bf16_head_enabled.cache_clear()


if __name__ == "__main__":
    unittest.main()
