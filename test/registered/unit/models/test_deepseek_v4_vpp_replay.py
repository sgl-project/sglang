import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.layers.attention.deepseek_v4_backend as dsv4_backend
import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    LateLayerTail,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4Model
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _tail():
    return LateLayerTail(
        token_indices=torch.tensor([2, 3]),
        positions=torch.tensor([12, 13]),
        extend_seq_lens=torch.tensor([2], dtype=torch.int32),
        extend_seq_lens_cpu=[2],
        swa_out_cache_loc=torch.tensor([6, 7], dtype=torch.int32),
        contiguous_start=2,
    )


class _Layer:
    engram = None

    def __init__(self):
        self.calls = []

    def forward_hc_pre_from_prev(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["hidden_states"] + 1, kwargs["prev_pre"] + 1


class TestDeepSeekV41VPPReplay(unittest.TestCase):
    def test_tail_continuation_does_not_inherit_full_metadata(self):
        tail = _tail()
        full_candidate = object()
        tail_candidate = object()
        full_metadata = SimpleNamespace(candidate_metadata=full_candidate)
        tail_metadata = SimpleNamespace(
            late_layer_tail=tail,
            candidate_metadata=tail_candidate,
            core_attn_metadata=SimpleNamespace(),
        )
        backend = SimpleNamespace(
            forward_metadata=full_metadata,
            tail_forward_metadata=tail_metadata,
            token_to_kv_pool=SimpleNamespace(request_window=None),
        )
        forward_batch = SimpleNamespace(attn_cp_metadata=None)

        with (
            patch.object(dsv4_backend, "get_local_dp_buffer_len", return_value=7),
            patch.object(dsv4_backend, "set_local_dp_buffer_len"),
        ):
            saved = DeepseekV4AttnBackend.enter_late_layer_tail(
                backend,
                forward_batch,
                inherit_full_state=False,
            )

        self.assertIs(backend.forward_metadata, tail_metadata)
        self.assertIs(tail_metadata.candidate_metadata, tail_candidate)
        self.assertIs(saved[0], full_metadata)

    def test_replay_boundary_switches_from_full_rows_to_tail_rows(self):
        tail = _tail()
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail),
            enter_late_layer_tail=MagicMock(return_value=("full", None, 0)),
            exit_late_layer_tail=MagicMock(),
        )
        layer20 = _Layer()
        layer21 = _Layer()
        model = SimpleNamespace(
            pp_group=SimpleNamespace(world_size=4),
            pipeline_layout=SimpleNamespace(is_interleaved=True),
            config=SimpleNamespace(model_type="deepseek_v41", vision_n_layers=0),
            engram_hasher=None,
            engram_prefetch_stream=None,
            late_layer_start=21,
            end_layer=22,
            layers={20: layer20, 21: layer21},
            _check_late_layer_tail_readers=MagicMock(),
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend=lambda: True,
                is_extend_without_speculative=lambda: True,
                is_decode=lambda: False,
            )
        )
        hidden_states = torch.arange(16).reshape(4, 1, 4)
        prev_pre = torch.arange(16).reshape(4, 1, 4)
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
            output, _, output_tail = DeepseekV4Model._forward_layers_hc_pre_from_prev(
                model,
                positions,
                hidden_states,
                forward_batch,
                input_ids,
                input_ids,
                False,
                [],
                (20, 21),
                prev_pre,
            )

        backend.enter_late_layer_tail.assert_called_once_with(forward_batch)
        backend.exit_late_layer_tail.assert_called_once()
        self.assertEqual(layer20.calls[0]["hidden_states"].shape[0], 4)
        self.assertEqual(layer21.calls[0]["hidden_states"].shape[0], 2)
        self.assertTrue(
            torch.equal(layer21.calls[0]["input_ids"], torch.tensor([12, 13]))
        )
        self.assertEqual(output.shape[0], 2)
        self.assertIs(output_tail, tail)

    def test_vpp_continuation_uses_tail_inputs_without_reslicing_activation(self):
        tail = _tail()
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail),
            enter_late_layer_tail=MagicMock(return_value=("full", None, 0)),
            exit_late_layer_tail=MagicMock(),
        )
        layer = _Layer()
        model = SimpleNamespace(
            pp_group=SimpleNamespace(world_size=4),
            pipeline_layout=SimpleNamespace(is_interleaved=True),
            config=SimpleNamespace(model_type="deepseek_v41", vision_n_layers=0),
            engram_hasher=None,
            engram_prefetch_stream=None,
            late_layer_start=21,
            end_layer=26,
            layers={25: layer},
            _check_late_layer_tail_readers=MagicMock(),
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend=lambda: True,
                is_extend_without_speculative=lambda: True,
                is_decode=lambda: False,
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
                    (25,),
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
        self.assertTrue(torch.equal(layer.calls[0]["positions"], tail.positions))
        self.assertTrue(torch.equal(output, hidden_states + 1))
        self.assertTrue(torch.equal(output_pre, prev_pre + 1))
        self.assertIs(output_tail, tail)

    def test_vpp_state_selects_tail_metadata_after_replay_boundary(self):
        full_metadata = object()
        tail_metadata = object()
        backend = SimpleNamespace(
            forward_metadata=full_metadata,
            tail_forward_metadata=tail_metadata,
        )
        with patch.object(deepseek_v4, "get_attn_backend", return_value=backend):
            self.assertIs(
                DeepseekV4Model._vpp_attention_metadata(
                    SimpleNamespace(),
                    False,
                ),
                full_metadata,
            )
            self.assertIs(
                DeepseekV4Model._vpp_attention_metadata(
                    SimpleNamespace(),
                    True,
                ),
                tail_metadata,
            )


if __name__ == "__main__":
    unittest.main()
