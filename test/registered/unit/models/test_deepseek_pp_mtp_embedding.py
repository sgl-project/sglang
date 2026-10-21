"""Pipeline-stage embedding ownership for DeepSeek/GLM NextN."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.layer_boundary import PLAIN_ADD, SumGroup
from sglang.srt.layers.layer_boundary.output import DeferredFinalize
from sglang.srt.layers.layer_boundary.residual import stream as residual_stream
from sglang.srt.layers.layer_boundary.residual.stream import OwedOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v2
from sglang.srt.models.deepseek_v2 import DeepseekV2Model, pp_stage_needs_embedding
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _pp_group(*, rank: int, size: int) -> SimpleNamespace:
    return SimpleNamespace(is_first_rank=rank == 0, is_last_rank=rank == size - 1)


class TestDeepseekPPStageEmbedding(CustomTestCase):
    def test_without_speculative_decoding_only_the_first_stage_embeds(self):
        for rank in range(4):
            with self.subTest(rank=rank):
                self.assertEqual(
                    pp_stage_needs_embedding(_pp_group(rank=rank, size=4), None),
                    rank == 0,
                )

    def test_speculative_decoding_adds_the_last_stage(self):
        wants = [
            pp_stage_needs_embedding(_pp_group(rank=rank, size=4), "EAGLE")
            for rank in range(4)
        ]
        self.assertEqual(wants, [True, False, False, True])

    def test_middle_stages_never_embed(self):
        self.assertFalse(pp_stage_needs_embedding(_pp_group(rank=2, size=4), "EAGLE"))

    def test_without_pipeline_parallelism_the_single_stage_embeds_either_way(self):
        for spec in (None, "EAGLE"):
            with self.subTest(spec=spec):
                self.assertTrue(
                    pp_stage_needs_embedding(_pp_group(rank=0, size=1), spec)
                )


class TestDeepseekDSAPPHandoff(CustomTestCase):
    def test_skip_topk_pp_handoff_uses_exported_hidden_states(self):
        for rows in (2, 0):
            for deferred_finalize in (False, True):
                with self.subTest(rows=rows, deferred_finalize=deferred_finalize):
                    self._run_forward(rows, deferred_finalize)

    def _run_forward(self, rows, deferred_finalize):
        class Group:
            def __init__(self):
                self.all_reduce_calls = 0

            def all_reduce(self, value):
                self.all_reduce_calls += 1
                return value

        forward_batch = SimpleNamespace(
            can_run_tbo=False,
            forward_mode=ForwardMode.DECODE,
            residual_stream=None,
            reuse_dsa_topk_indices=False,
            spec_info=None,
        )
        expected_hidden = torch.ones(rows, 4)
        expected_topk = torch.ones(rows, 3, dtype=torch.int32) if rows else None
        group = Group()
        finalize = Mock(return_value=expected_hidden)

        class Layer:
            def __call__(self, *args, **kwargs):
                stream = deepseek_v2.residual_batch.stream_of(forward_batch)
                if deferred_finalize:
                    hidden = stream.record(DeferredFinalize(), PLAIN_ADD)
                else:
                    hidden = stream.record(
                        expected_hidden,
                        PLAIN_ADD,
                        declared_sum=SumGroup.TP,
                    )
                assert isinstance(hidden, OwedOutput)
                assert group.all_reduce_calls == 0
                return hidden, expected_topk

        model = DeepseekV2Model.__new__(DeepseekV2Model)
        torch.nn.Module.__init__(model)
        model.use_dsa = True
        model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=False)
        model.config = SimpleNamespace(num_hidden_layers=2)
        model.start_layer = 0
        model.end_layer = 1
        model.layers = [Layer()]
        model.embed_tokens = torch.nn.Identity()
        model.layers_to_capture = []
        model.next_full_attention_layer_id = {}
        model.llama_4_scaling_config = None

        finalizer_patch = (
            patch.object(DeferredFinalize, "complete", finalize)
            if deferred_finalize
            else patch.object(DeferredFinalize, "complete")
        )
        with (
            patch.object(model, "_dsa_forward_uses_topk", return_value=True),
            patch.object(deepseek_v2, "dsa_layer_skips_topk", return_value=True),
            patch.object(deepseek_v2, "get_dsa_index_topk", return_value=3),
            patch.object(deepseek_v2, "BumpAllocator", return_value=None),
            patch.object(deepseek_v2, "check_cuda_graph_backend", return_value=True),
            patch.object(residual_stream, "_sum_group", return_value=group),
            finalizer_patch,
        ):
            proxy = model.forward(
                input_ids=torch.zeros(rows, dtype=torch.long),
                positions=torch.zeros(rows, dtype=torch.long),
                forward_batch=forward_batch,
            )

        if deferred_finalize:
            finalize.assert_called_once_with()
            self.assertEqual(group.all_reduce_calls, 0)
        else:
            finalize.assert_not_called()
            self.assertEqual(group.all_reduce_calls, 1)
        self.assertIsNone(forward_batch.residual_stream)
        self.assertEqual(proxy["hidden_states"].shape, (rows, 4))
        torch.testing.assert_close(proxy["hidden_states"], expected_hidden)
        if rows:
            self.assertIs(proxy["topk_indices"], expected_topk)
            torch.testing.assert_close(proxy["topk_indices"], expected_topk)
        else:
            self.assertEqual(proxy["topk_indices"].shape, (0, 3))
            self.assertEqual(proxy["topk_indices"].dtype, torch.int32)
            self.assertEqual(
                proxy["topk_indices"].device, proxy["hidden_states"].device
            )


if __name__ == "__main__":
    unittest.main()
