import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4Model,
    _dsv41_multimodal_enabled,
    _should_build_dsv41_vision,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Layer:
    engram = None

    def __init__(self):
        self.calls = []

    def forward_hc_pre_from_prev(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["hidden_states"] + 1, kwargs["prev_pre"] + 1


class TestDeepSeekV41PP(unittest.TestCase):
    def test_decode_candidate_mask_survives_pipeline_boundary(self):
        from sglang.srt.layers.attention.dsv4.candidate_indexer import CandidateMasks

        mask = torch.tensor([[True, False], [False, True]])
        core = SimpleNamespace(
            page_table=torch.tensor([[3], [4]]),
            page_size=256,
            seq_lens_casual=torch.tensor([1, 1]),
        )
        sender_metadata = SimpleNamespace(
            core_metadata=core,
            candidate_metadata=CandidateMasks(mask=mask),
        )
        receiver_metadata = SimpleNamespace(core_metadata=core, candidate_metadata=None)
        model = SimpleNamespace(
            end_layer=1,
            config=SimpleNamespace(compress_ratios=[0, 0]),
            _pp_attention_metadata=lambda tail: sender_metadata,
        )
        tensors = {}
        DeepseekV4Model._export_pp_state(model, tensors, None, False)
        model._pp_attention_metadata = lambda tail: receiver_metadata
        DeepseekV4Model._install_pp_state(model, PPProxyTensors(tensors), None, False)
        self.assertIsInstance(receiver_metadata.candidate_metadata, CandidateMasks)
        self.assertTrue(torch.equal(receiver_metadata.candidate_metadata.mask, mask))

    def test_sparse_slots_follow_receiver_page_allocation(self):
        from sglang.srt.layers.attention.dsv4.pp import remap_sparse_slots

        # One shared sender page can map to different pages on the receiver.
        source = torch.tensor([[3, 1, 0], [3, 2, 0]], dtype=torch.int32)
        target = torch.tensor([[7, 5, 0], [9, 6, 0]], dtype=torch.int32)
        slots = torch.tensor([[12, 7, -1], [13, 8, -1]], dtype=torch.int32)
        result = remap_sparse_slots(
            slots, source, target, 4, torch.tensor([2, 2], dtype=torch.int32)
        )
        self.assertTrue(torch.equal(result, torch.tensor([[28, 23, -1], [37, 24, -1]])))

    def test_vision_tower_is_owned_by_first_pp_stage(self):
        config = SimpleNamespace(
            model_type="deepseek_v41",
            vision_n_layers=8,
            language_only=False,
            language_model_only=False,
        )

        self.assertTrue(_dsv41_multimodal_enabled(config))
        self.assertTrue(
            _should_build_dsv41_vision(config, SimpleNamespace(is_first_rank=True))
        )
        self.assertFalse(
            _should_build_dsv41_vision(config, SimpleNamespace(is_first_rank=False))
        )

    def test_language_model_only_disables_vision_tower(self):
        config = SimpleNamespace(
            model_type="deepseek_v41",
            vision_n_layers=8,
            language_only=False,
            language_model_only=True,
        )

        self.assertFalse(_dsv41_multimodal_enabled(config))

    def test_tail_continuation_uses_proxy_state_without_reslicing_activation(self):
        tail = LateLayerTail(
            token_indices=torch.tensor([2, 3]),
            positions=torch.tensor([12, 13]),
            extend_seq_lens=torch.tensor([2], dtype=torch.int32),
            extend_seq_lens_cpu=[2],
            swa_out_cache_loc=torch.tensor([6, 7], dtype=torch.int32),
            contiguous_start=2,
        )
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail),
            enter_late_layer_tail=MagicMock(return_value=("full", None, 0)),
            exit_late_layer_tail=MagicMock(),
        )
        layer = _Layer()
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="deepseek_v41"),
            pp_group=SimpleNamespace(world_size=4),
            engram_hasher=None,
            late_layer_start=21,
            start_layer=30,
            end_layer=31,
            layers={30: layer},
            _check_late_layer_tail_readers=MagicMock(),
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend=lambda: True,
                is_extend_without_speculative=lambda: True,
                is_decode=lambda: False,
                is_target_verify=lambda: False,
            )
        )
        hidden_states = torch.arange(8).reshape(2, 1, 4)
        prev_pre = torch.arange(8).reshape(2, 1, 4)
        input_ids = torch.tensor([10, 11, 12, 13])
        positions = torch.tensor([10, 11, 12, 13])

        with (
            patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
            patch.object(deepseek_v4, "is_cp_active", return_value=False),
            patch.object(
                deepseek_v4,
                "check_cuda_graph_backend",
                return_value=True,
            ),
            patch.object(deepseek_v4, "nullcontext", return_value=nullcontext()),
        ):
            output, output_pre, output_tail = (
                DeepseekV4Model._forward_layers_hc_pre_from_prev(
                    model,
                    positions,
                    hidden_states,
                    forward_batch,
                    input_ids,
                    input_ids,
                    False,
                    [],
                    prev_pre,
                    True,
                )
            )

        backend.enter_late_layer_tail.assert_called_once_with(
            forward_batch,
            inherit_full_state=False,
        )
        backend.exit_late_layer_tail.assert_called_once()
        self.assertTrue(torch.equal(layer.calls[0]["hidden_states"], hidden_states))
        self.assertTrue(
            torch.equal(layer.calls[0]["input_ids"], torch.tensor([12, 13]))
        )
        self.assertTrue(torch.equal(output, hidden_states + 1))
        self.assertTrue(torch.equal(output_pre, prev_pre + 1))
        self.assertIs(output_tail, tail)


if __name__ == "__main__":
    unittest.main()
