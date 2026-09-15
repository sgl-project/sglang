import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.aux_hidden_states import (
    RUNTIME_AUX_CAPTURE_LOGITS_ATTR,
    RUNTIME_AUX_HIDDEN_STATES_ATTR,
    resolve_runtime_aux_hidden_states,
)
from sglang.srt.model_executor.dspark_aux_hidden_state import (
    _RUNTIME_CAPTURE_INSTALLED_ATTR,
    attach_runtime_dspark_aux_hidden_state_capture,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    configure_aux_hidden_state_capture,
)
from sglang.srt.models.sarvam_moe import (
    AttnForwardMethod,
    get_attn_forward_method,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Communicator:
    def prepare_attn(self, hidden_states, residual, _forward_batch):
        residual = hidden_states if residual is None else hidden_states + residual
        return hidden_states, residual

    def capture_last_layer_output(
        self, residual, _forward_batch, captured_last_layer_outputs
    ):
        captured_last_layer_outputs.append(residual.clone())


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_communicator = _Communicator()

    def forward(self, positions, hidden_states, forward_batch, residual):
        del positions
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        return hidden_states + 1, residual


class _Norm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states + residual, residual


class _LayerModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.start_layer = 0
        self.end_layer = num_layers
        self.layers = nn.ModuleList([_Layer() for _ in range(num_layers)])
        self.norm = _Norm()

    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        del input_ids, pp_proxy_tensors
        hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class _Target(nn.Module):
    def __init__(self, *, num_layers=4, pp_size=1, is_last_rank=True):
        super().__init__()
        self.pp_group = SimpleNamespace(world_size=pp_size, is_last_rank=is_last_rank)
        self.config = SimpleNamespace(num_hidden_layers=num_layers)
        self.model = _LayerModel(num_layers)
        self.logits_processor = SimpleNamespace()


class _NativeDSparkTarget(_Target):
    def __init__(self):
        super().__init__()
        self.capture_aux_hidden_states = False
        self.configured_layer_ids = None

    def set_dspark_layers_to_capture(self, layer_ids):
        self.capture_aux_hidden_states = True
        self.configured_layer_ids = layer_ids


class _NativeDFlashTarget(_Target):
    def __init__(self):
        super().__init__()
        self.capture_aux_hidden_states = False
        self.configured_layer_ids = None

    def set_dflash_layers_to_capture(self, layer_ids):
        self.capture_aux_hidden_states = True
        self.configured_layer_ids = layer_ids


class TestSarvamDSparkRuntimeCapture(CustomTestCase):
    def test_runtime_capture_preserves_target_output(self):
        input_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        positions = torch.arange(2)

        baseline = _Target()
        expected = baseline.model(
            input_ids=None,
            positions=positions,
            forward_batch=SimpleNamespace(),
            input_embeds=input_embeds,
        )

        target = _Target()
        attach_runtime_dspark_aux_hidden_state_capture(target, [0, 2, 3])
        actual, _ = target.model(
            input_ids=None,
            positions=positions,
            forward_batch=SimpleNamespace(),
            input_embeds=input_embeds,
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_runtime_capture_uses_raw_post_layer_ids(self):
        target = _Target()
        original_signature = inspect.signature(target.model.forward)

        attach_runtime_dspark_aux_hidden_state_capture(target, [0, 2, 3])

        forward_batch = SimpleNamespace()
        final_hidden, captured = target.model(
            input_ids=None,
            positions=torch.arange(2),
            forward_batch=forward_batch,
            input_embeds=torch.zeros(2, 4),
        )

        torch.testing.assert_close(final_hidden, torch.full((2, 4), 10.0))
        self.assertEqual(len(captured), 3)
        for value, expected in zip(captured, (1.0, 6.0, 10.0)):
            torch.testing.assert_close(value, torch.full((2, 4), expected))
        self.assertEqual(inspect.signature(target.model.forward), original_signature)
        self.assertTrue(target.capture_aux_hidden_states)
        self.assertTrue(
            getattr(target.logits_processor, RUNTIME_AUX_CAPTURE_LOGITS_ATTR)
        )

    def test_runtime_capture_resets_between_forwards(self):
        target = _Target()
        attach_runtime_dspark_aux_hidden_state_capture(target, [0, 2])
        forward_batch = SimpleNamespace()

        for fill in (0.0, 2.0):
            _, captured = target.model(
                None,
                torch.arange(2),
                forward_batch,
                input_embeds=torch.full((2, 4), fill),
            )
            self.assertEqual(len(captured), 2)

    def test_configure_uses_runtime_fallback(self):
        target = _Target()

        configure_aux_hidden_state_capture(
            model=target,
            eagle_use_aux_hidden_state=False,
            eagle_aux_hidden_state_layer_ids=None,
            dflash_use_aux_hidden_state=True,
            dflash_target_layer_ids=[0, 2],
            is_dspark=True,
        )

        self.assertTrue(target.capture_aux_hidden_states)

    def test_configure_is_noop_when_aux_capture_is_disabled(self):
        target = _Target()

        configure_aux_hidden_state_capture(
            model=target,
            eagle_use_aux_hidden_state=False,
            eagle_aux_hidden_state_layer_ids=None,
            dflash_use_aux_hidden_state=False,
            dflash_target_layer_ids=None,
            is_dspark=True,
        )

        self.assertFalse(hasattr(target, "capture_aux_hidden_states"))
        self.assertFalse(hasattr(target, _RUNTIME_CAPTURE_INSTALLED_ATTR))

    def test_configure_prefers_model_managed_capture_hooks(self):
        for target in (_NativeDSparkTarget(), _NativeDFlashTarget()):
            with self.subTest(target=type(target).__name__):
                configure_aux_hidden_state_capture(
                    model=target,
                    eagle_use_aux_hidden_state=False,
                    eagle_aux_hidden_state_layer_ids=None,
                    dflash_use_aux_hidden_state=True,
                    dflash_target_layer_ids=[0, 2],
                    is_dspark=True,
                )

                self.assertEqual(target.configured_layer_ids, [0, 2])
                self.assertTrue(target.capture_aux_hidden_states)
                self.assertFalse(hasattr(target, _RUNTIME_CAPTURE_INSTALLED_ATTR))

    def test_runtime_capture_rejects_invalid_contracts(self):
        for layer_ids in (None, [], [1, 1], [2, 1], [-1, 1], [1, 4]):
            with self.subTest(layer_ids=layer_ids), self.assertRaises(ValueError):
                attach_runtime_dspark_aux_hidden_state_capture(_Target(), layer_ids)

        with self.assertRaises(NotImplementedError):
            attach_runtime_dspark_aux_hidden_state_capture(_Target(pp_size=2), [0, 2])

        target = _Target()
        target.capture_aux_hidden_states = False
        with self.assertRaises(TypeError):
            attach_runtime_dspark_aux_hidden_state_capture(target, [0, 2])

    def test_logits_resolution_accepts_body_or_batch_aux_states(self):
        processor = SimpleNamespace()
        setattr(processor, RUNTIME_AUX_CAPTURE_LOGITS_ATTR, True)
        hidden_states = torch.zeros(2, 4)
        aux_hidden_states = [torch.ones(2, 4)]
        forward_batch = SimpleNamespace()
        setattr(
            forward_batch,
            RUNTIME_AUX_HIDDEN_STATES_ATTR,
            aux_hidden_states,
        )

        resolved_hidden, resolved_aux = resolve_runtime_aux_hidden_states(
            processor, hidden_states, forward_batch, None
        )
        self.assertIs(resolved_hidden, hidden_states)
        self.assertIs(resolved_aux, aux_hidden_states)

        resolved_hidden, resolved_aux = resolve_runtime_aux_hidden_states(
            processor,
            (hidden_states, aux_hidden_states),
            SimpleNamespace(),
            None,
        )
        self.assertIs(resolved_hidden, hidden_states)
        self.assertIs(resolved_aux, aux_hidden_states)

    def test_logits_resolution_is_noop_without_runtime_marker(self):
        hidden_states = (torch.zeros(2, 4), [torch.ones(2, 4)])

        resolved_hidden, resolved_aux = resolve_runtime_aux_hidden_states(
            SimpleNamespace(), hidden_states, SimpleNamespace(), None
        )

        self.assertIs(resolved_hidden, hidden_states)
        self.assertIsNone(resolved_aux)

    def test_sarvam_target_verify_uses_speculative_attention_mode(self):
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)

        with patch(
            "sglang.srt.models.sarvam_moe.attention_backends",
            return_value=("triton", "trtllm_mla"),
        ):
            with patch(
                "sglang.srt.models.sarvam_moe.get_spec",
                return_value=SimpleNamespace(speculative_attention_mode="decode"),
            ):
                self.assertEqual(
                    get_attn_forward_method(forward_batch),
                    AttnForwardMethod.MLA_SEPARATE_ROPE,
                )

            with patch(
                "sglang.srt.models.sarvam_moe.get_spec",
                return_value=SimpleNamespace(speculative_attention_mode="prefill"),
            ):
                self.assertEqual(
                    get_attn_forward_method(forward_batch),
                    AttnForwardMethod.MLA_CONCAT_ROPE,
                )


if __name__ == "__main__":
    unittest.main()
