"""CPU state-transition checks for BCG plus eager decoder-tail fallback."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch

from sglang.srt.arg_groups import deepseek_v4_hook as config_module
from sglang.srt.layers.attention import deepseek_v4_backend as backend_module
from sglang.srt.models import deepseek_v4 as model_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

Backend = backend_module.DeepseekV4AttnBackend


class TestDecoderTailGraphConfig(unittest.TestCase):
    def check_config(self, backend, dp=False):
        cfg = SimpleNamespace(
            enable_encoder_swa_bounded_replay=False,
            enable_decoder_swa_bounded_replay=True,
            speculative_algorithm="DSPARK",
            enable_hisparse=False,
            dsv4_attn_backend="flashmla",
            enable_two_batch_overlap=False,
            pp_size=1,
            disaggregation_mode="null",
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=backend, max_seq_len=1048576)
            ),
        )
        with (
            patch.object(config_module, "resolving_view", return_value=cfg),
            patch.object(
                config_module,
                "model_config_of",
                return_value=SimpleNamespace(
                    hf_config=SimpleNamespace(model_type="deepseek_v41")
                ),
            ),
            patch.object(config_module, "attn_dp_enabled_of", return_value=dp),
            patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                return_value=False,
            ),
        ):
            config_module.validate_deepseek_v41_features(object())

    def test_only_breakable_and_disabled_prefill_are_allowed(self):
        for backend in ("disabled", "breakable"):
            with self.subTest(backend=backend):
                self.check_config(backend)
        for backend in ("full", "tc_piecewise"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "prefill CUDA graph"):
                    self.check_config(backend)

    def test_dp_attention_stays_rejected(self):
        for backend in ("disabled", "breakable"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "DP attention"):
                    self.check_config(backend, dp=True)


class TestDecoderTailGraphTransitions(unittest.TestCase):
    def backend(self):
        backend = Backend.__new__(Backend)
        backend.tail_forward_metadata = object()
        backend._build_forward_metadata = Mock(return_value=object())
        return backend

    def test_capture_drops_stale_eager_tail(self):
        backend = self.backend()
        batch = SimpleNamespace(max_seq_len_override=131072)
        with patch.object(
            Backend,
            "low_ratio_prefill_graph",
            new_callable=PropertyMock,
            return_value=False,
        ):
            metadata = backend.init_forward_metadata_for_breakable_cuda_graph_capture(
                batch
            )
        self.assertIsNone(backend.tail_forward_metadata)
        self.assertIs(metadata, backend.forward_metadata)
        backend._build_forward_metadata.assert_called_once_with(
            batch, max_seq_len_override=131072, use_prefill_cuda_graph=True
        )

    def test_replay_drops_eager_tail_and_eager_rebuilds_it(self):
        backend = self.backend()
        live = SimpleNamespace(
            encoder_swa_replay=None,
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
        )
        static = SimpleNamespace(max_seq_len_override=131072)
        captured = Mock(spec=backend_module.DSV4Metadata)
        backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
            captured, live, static_forward_batch=static
        )
        self.assertIsNone(backend.tail_forward_metadata)
        self.assertIs(backend.forward_metadata, captured)
        backend._build_forward_metadata.assert_called_once_with(
            static, max_seq_len_override=131072, use_prefill_cuda_graph=True
        )
        captured.refresh_for_breakable_cuda_graph_replay_.assert_called_once_with(
            backend._build_forward_metadata.return_value
        )

        # Graph -> eager must restore the current batch's tail rather than
        # leave bounded replay silently disabled after the first graph.
        backend.mtp_enabled = False
        backend.enable_decoder_swa_bounded_replay = True
        backend.init_forward_metadata_in_graph = Mock()
        current_tail = object()
        backend._build_late_layer_tail_metadata = Mock(return_value=current_tail)
        backend.token_to_kv_pool = SimpleNamespace(request_window=None)
        with patch.object(backend_module, "_get_logical_forward_mode"):
            backend.init_forward_metadata(live)
        self.assertIs(backend.tail_forward_metadata, current_tail)
        backend._build_late_layer_tail_metadata.assert_called_once_with(live)

    def test_outer_logits_after_graph_keeps_full_rows(self):
        self.check_logits(tail=None)

    def test_outer_logits_after_eager_uses_tail_rows(self):
        tail = SimpleNamespace(
            rows=lambda x: x[-128:],
            extend_seq_lens=torch.tensor([128]),
            extend_seq_lens_cpu=[128],
            token_indices=torch.arange(91, 219),
        )
        self.check_logits(tail=tail)

    def check_logits(self, tail):
        ids = torch.arange(219)
        hidden = torch.zeros(219 if tail is None else 128, 4)
        aux = object()
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True)
        )
        logits = Mock(return_value=SimpleNamespace())
        model = SimpleNamespace(
            vision=None,
            model=SimpleNamespace(
                late_layer_start=20, forward=Mock(return_value=((hidden, None), aux))
            ),
            pp_group=SimpleNamespace(is_last_rank=True),
            capture_aux_hidden_states=True,
            logits_processor=logits,
            lm_head=object(),
        )
        backend = SimpleNamespace(
            tail_forward_metadata=None
            if tail is None
            else SimpleNamespace(late_layer_tail=tail)
        )
        context = SimpleNamespace(maybe_input_scattered=lambda _: nullcontext())
        metadata = SimpleNamespace()
        with (
            patch.object(model_module, "get_attn_backend", return_value=backend),
            patch.object(model_module, "get_attn_tp_context", return_value=context),
            patch.object(
                model_module.LogitsMetadata, "from_forward_batch", return_value=metadata
            ),
        ):
            output = model_module.DeepseekV4ForCausalLM.forward(model, ids, ids, batch)
        passed_ids, passed_hidden, _, passed_metadata, passed_aux = (
            logits.call_args.args
        )
        self.assertIs(passed_hidden, hidden)
        self.assertIs(passed_aux, aux)
        if tail is None:
            self.assertIs(passed_ids, ids)
            self.assertIs(passed_metadata, batch)
            self.assertFalse(hasattr(output, "hidden_states_token_indices"))
        else:
            torch.testing.assert_close(passed_ids, ids[-128:], rtol=0, atol=0)
            self.assertIs(passed_metadata, metadata)
            self.assertEqual(metadata.extend_seq_lens_cpu, [128])
            self.assertIs(output.hidden_states_token_indices, tail.token_indices)


if __name__ == "__main__":
    unittest.main()
