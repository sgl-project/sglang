"""Prompt-only KV eligibility, model exits, and request-state lifetime."""

import unittest
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.dllm.algorithm.low_confidence import LowConfidence
from sglang.srt.dllm.prefill import (
    can_use_kv_only_prefill,
    supports_kv_only_prefill,
)
from sglang.srt.environ import envs
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.models.llada2 import (
    LLaDA2MoeBlock,
    LLaDA2MoeModel,
    LLaDA2MoeModelLM,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_runner():
    return SimpleNamespace(
        model=SimpleNamespace(supports_dllm_kv_only_prefill=True, quant_config=None),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        device="cuda",
        is_multimodal=False,
        is_draft_worker=False,
        server_args=SimpleNamespace(
            dllm_algorithm="LowConfidence",
            dllm_fdfo=True,
            tp_size=1,
            pp_size=1,
            dp_size=1,
            attn_cp_size=1,
            attention_backend="flashinfer",
            moe_runner_backend="triton",
            enable_lora=False,
            enable_torch_compile=False,
            enable_two_batch_overlap=False,
            enable_return_hidden_states=False,
            return_hidden_states_mode=None,
            enable_return_aux_hidden_states=False,
            enable_return_routed_experts=False,
            enable_return_indexer_topk=False,
            speculative_algorithm=None,
        ),
    )


def make_batch():
    return ForwardBatch(
        forward_mode=ForwardMode.DLLM_EXTEND,
        batch_size=1,
        input_ids=torch.arange(4),
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([4]),
        out_cache_loc=torch.arange(4),
        seq_lens_sum=4,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


class TestKvOnlyPrefillEligibility(CustomTestCase):
    def setUp(self):
        self.enterContext(envs.SGLANG_ENABLE_DLLM_KV_ONLY_PREFILL.override(True))
        self.enterContext(patch.object(torch.version, "hip", None))
        self.runner = make_runner()
        self.forward_batch = make_batch()
        self.batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(return_routed_experts=False, return_indexer_topk=False)
            ]
        )

    def test_opt_in_and_model_capability(self):
        self.assertTrue(supports_kv_only_prefill(self.runner))
        with envs.SGLANG_ENABLE_DLLM_KV_ONLY_PREFILL.override(False):
            self.assertFalse(supports_kv_only_prefill(self.runner))
        self.runner.model.supports_dllm_kv_only_prefill = False
        self.assertFalse(supports_kv_only_prefill(self.runner))

    def test_launch_configurations_fall_back(self):
        changes = {
            "dllm_algorithm": "JointThreshold",
            "dllm_fdfo": False,
            "tp_size": 2,
            "pp_size": 2,
            "dp_size": 2,
            "attn_cp_size": 2,
            "attention_backend": "triton",
            "moe_runner_backend": "flashinfer_trtllm",
            "enable_lora": True,
            "enable_torch_compile": True,
            "enable_two_batch_overlap": True,
            "enable_return_hidden_states": True,
            "return_hidden_states_mode": "last",
            "enable_return_aux_hidden_states": True,
            "enable_return_routed_experts": True,
            "enable_return_indexer_topk": True,
            "speculative_algorithm": "EAGLE",
        }
        for field, value in changes.items():
            with (
                self.subTest(field=field),
                patch.object(self.runner.server_args, field, value),
            ):
                self.assertFalse(supports_kv_only_prefill(self.runner))
        for field, value in (
            ("device", "cpu"),
            ("is_multimodal", True),
            ("is_draft_worker", True),
        ):
            with self.subTest(field=field), patch.object(self.runner, field, value):
                self.assertFalse(supports_kv_only_prefill(self.runner))
        with patch.object(self.runner.model_config, "dtype", torch.float16):
            self.assertFalse(supports_kv_only_prefill(self.runner))
        with patch.object(self.runner.model, "quant_config", object()):
            self.assertFalse(supports_kv_only_prefill(self.runner))
        with patch.object(torch.version, "hip", "7.0"):
            self.assertFalse(supports_kv_only_prefill(self.runner))

    def test_output_consumers_fall_back(self):
        self.assertTrue(
            can_use_kv_only_prefill(self.runner, self.forward_batch, self.batch)
        )
        for field, value in (
            ("return_logprob", True),
            ("capture_hidden_mode", CaptureHiddenMode.LAST),
            ("capture_hidden_mode", CaptureHiddenMode.FULL),
            ("return_hidden_states_before_norm", True),
            ("return_pooled_hidden_states", True),
            ("is_prefill_only", True),
            ("beam", object()),
            ("forward_mode", ForwardMode.EXTEND),
        ):
            with (
                self.subTest(field=field),
                patch.object(self.forward_batch, field, value),
            ):
                self.assertFalse(
                    can_use_kv_only_prefill(self.runner, self.forward_batch, self.batch)
                )
        for field in ("return_routed_experts", "return_indexer_topk"):
            with (
                self.subTest(field=field),
                patch.object(self.batch.reqs[0], field, True),
            ):
                self.assertFalse(
                    can_use_kv_only_prefill(self.runner, self.forward_batch, self.batch)
                )

    def test_flag_survives_batch_copy_and_selects_its_graph(self):
        self.forward_batch.dllm_kv_only_prefill = True
        copied = replace(self.forward_batch)
        runner = SimpleNamespace(record_nolora_graph=False)
        self.assertTrue(copied.dllm_kv_only_prefill)
        self.assertEqual(
            DecodeCudaGraphRunner._resolve_lora_variant(runner, copied), "dllm_kv_only"
        )
        copied.dllm_kv_only_prefill = False
        self.assertIsNone(DecodeCudaGraphRunner._resolve_lora_variant(runner, copied))

    def test_worker_clears_flag_even_if_forward_fails(self):
        algorithm = LowConfidence(
            SimpleNamespace(
                block_size=4,
                mask_id=99,
                first_done_first_out_mode=True,
                algorithm_config={},
            )
        )
        algorithm.prefill_tokens = Mock(return_value=[[1, 2, 3, 4]])
        worker = SimpleNamespace(dllm_algorithm=algorithm, model_runner=self.runner)

        def fail(batch, **kwargs):
            self.assertTrue(batch.dllm_kv_only_prefill)
            raise RuntimeError("forward failed")

        self.runner.forward = fail
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            TpModelWorker._forward_batch_generation_dllm(
                worker, self.forward_batch, self.batch
            )
        self.assertFalse(self.forward_batch.dllm_kv_only_prefill)


class TestKvOnlyModelExits(CustomTestCase):
    def test_only_last_block_skips_ffn_after_attention(self):
        hidden = torch.ones(4, 8)
        for is_last in (False, True):
            for enabled in (False, True):
                with self.subTest(is_last=is_last, enabled=enabled):
                    block = LLaDA2MoeBlock.__new__(LLaDA2MoeBlock)
                    nn.Module.__init__(block)
                    block.is_last_layer = is_last
                    block.attn_boundary = SimpleNamespace(
                        prepare=Mock(return_value=hidden),
                        finish=Mock(return_value=hidden),
                    )
                    block.attention = Mock(return_value=hidden)
                    block.ffn_boundary = SimpleNamespace(
                        prepare=Mock(return_value=hidden),
                        finish=Mock(return_value=hidden),
                    )
                    block.mlp = Mock(return_value=hidden)
                    fb = SimpleNamespace(dllm_kv_only_prefill=enabled)
                    out = block(torch.arange(4), hidden, fb)
                    self.assertIs(out, hidden)
                    block.attention.assert_called_once()
                    block.attn_boundary.finish.assert_called_once()
                    self.assertEqual(
                        block.mlp.call_count, int(not (is_last and enabled))
                    )
                    self.assertEqual(
                        block.ffn_boundary.prepare.call_count,
                        int(not (is_last and enabled)),
                    )

    def test_backbone_releases_residual_without_final_norm(self):
        model = LLaDA2MoeModel.__new__(LLaDA2MoeModel)
        nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        model.word_embeddings = nn.Embedding(8, 4)
        model.start_layer, model.end_layer = 0, 1
        fb = SimpleNamespace(dllm_kv_only_prefill=True, residual_stream=None)

        def layer(positions, hidden, batch):
            residual_batch.stream_of(batch).residual = hidden
            return hidden

        model.layers = [Mock(side_effect=layer)]
        model.norm = Mock(side_effect=AssertionError("unexpected output norm"))
        recorder = SimpleNamespace(with_current_layer=lambda _: nullcontext())
        with patch(
            "sglang.srt.models.llada2.get_global_expert_distribution_recorder",
            return_value=recorder,
        ):
            model(torch.arange(4), torch.arange(4), fb)
        model.layers[0].assert_called_once()
        model.norm.assert_not_called()
        self.assertIsNone(fb.residual_stream)

    def test_language_head_runs_for_denoising(self):
        model = LLaDA2MoeModelLM.__new__(LLaDA2MoeModelLM)
        nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(is_last_rank=True)
        hidden = torch.ones(4, 8)
        model.model = Mock(return_value=hidden)
        model.lm_head = nn.Linear(8, 16)
        model.logits_processor = Mock(return_value="ordinary logits")
        fb = SimpleNamespace(dllm_kv_only_prefill=True)
        result = model(torch.arange(4), torch.arange(4), fb)
        self.assertIsNone(result.full_logits)
        model.logits_processor.assert_not_called()
        fb.dllm_kv_only_prefill = False
        self.assertEqual(model(torch.arange(4), torch.arange(4), fb), "ordinary logits")
        model.logits_processor.assert_called_once()


if __name__ == "__main__":
    unittest.main()
