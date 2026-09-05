"""Regression tests for Bailing modality metadata and per-token routing bias."""

import unittest

import torch

from sglang.srt.layers.moe.topk import biased_grouped_topk_impl
from sglang.srt.layers.multi_gate import create_multi_gate_mm_indices
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.model_executor.forward_batch_info import (
    _build_forward_token_modalities,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestBailingModalityMetadata(CustomTestCase):
    def test_offsets_survive_hash_padding(self):
        """Hash-derived token replacement must not erase image/audio identity."""
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=[(1, 2)],
                feature=torch.ones(1),
            ),
            MultimodalDataItem(
                modality=Modality.AUDIO,
                offsets=[(4, 5)],
                feature=torch.ones(1),
            ),
        ]
        output = MultimodalProcessorOutput(
            mm_items=items,
            input_ids=[100, 11, 11, 101, 12, 12, 102],
        )

        inputs = MultimodalInputs.from_processor_output(output)

        self.assertEqual(
            inputs.token_modalities,
            [
                0,
                Modality.IMAGE.value,
                Modality.IMAGE.value,
                0,
                Modality.AUDIO.value,
                Modality.AUDIO.value,
                0,
            ],
        )
        self.assertNotEqual(items[0].pad_value, 11)
        self.assertNotEqual(items[1].pad_value, 12)

    def test_chunked_metadata_is_identical_on_every_pp_stage(self):
        """Each PP stage must independently receive the same active token map."""
        mm_inputs = [
            MultimodalInputs(
                mm_items=[],
                token_modalities=[0, Modality.IMAGE.value, Modality.IMAGE.value, 0],
            ),
            MultimodalInputs(
                mm_items=[],
                token_modalities=[Modality.AUDIO.value, Modality.AUDIO.value, 0],
            ),
        ]
        expected = torch.tensor(
            [
                Modality.IMAGE.value,
                Modality.IMAGE.value,
                0,
                Modality.AUDIO.value,
                Modality.AUDIO.value,
            ],
            dtype=torch.int8,
        )

        stage_maps = [
            _build_forward_token_modalities(
                mm_inputs,
                extend_prefix_lens=[1, 0],
                extend_seq_lens=[3, 2],
                num_tokens=5,
                device=torch.device("cpu"),
            )
            for _ in range(2)
        ]

        for stage_map in stage_maps:
            torch.testing.assert_close(stage_map, expected)

    def test_mixed_modalities_select_reference_experts(self):
        """Per-token bias must select image/audio experts after modality grouping."""
        modalities = torch.tensor(
            [
                Modality.IMAGE.value,
                Modality.IMAGE.value,
                0,
                Modality.AUDIO.value,
                Modality.AUDIO.value,
            ],
            dtype=torch.int8,
        )
        token_indices, modality_ids = create_multi_gate_mm_indices(modalities)
        self.assertEqual(modality_ids.tolist(), [0, 1, 2])
        self.assertEqual(token_indices[:1].tolist(), [2])
        self.assertEqual(token_indices[64:66].tolist(), [0, 1])
        self.assertEqual(token_indices[128:130].tolist(), [3, 4])

        router_logits = torch.zeros(5, 8)
        dynamic_bias = torch.zeros_like(router_logits)
        expected_experts = torch.tensor([1, 1, 0, 6, 6], dtype=torch.int32)
        dynamic_bias.scatter_(1, expected_experts.long().unsqueeze(1), 10.0)
        _, expert_ids = biased_grouped_topk_impl(
            hidden_states=torch.zeros(5, 4),
            gating_output=router_logits,
            correction_bias=dynamic_bias,
            topk=1,
            renormalize=True,
            num_expert_group=2,
            topk_group=1,
        )

        torch.testing.assert_close(expert_ids.squeeze(1), expected_experts)


if __name__ == "__main__":
    unittest.main()
