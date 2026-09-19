"""DSpark's opt-in window must not rebase positions or alter target attention."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.models.dflash import DFlashAttention
from sglang.srt.models.dspark import DSparkDraftModel
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _draft(windows=(-1, -1), attn_type=AttentionType.DECODER):
    # No weights or distributed initialization; use the real attention modules.
    model = DSparkDraftModel.__new__(DSparkDraftModel)
    nn.Module.__init__(model)
    model._draft_window_size = None
    model.config = PretrainedConfig(
        layer_types=[
            "full_attention" if window == -1 else "sliding_attention"
            for window in windows
        ],
        sliding_window=next((w + 1 for w in windows if w >= 0), None),
    )
    model.layers = nn.ModuleList()
    for i, window in enumerate(windows):
        attention = DFlashAttention.__new__(DFlashAttention)
        nn.Module.__init__(attention)
        attention.sliding_window_size = window
        attention.attn = RadixAttention(
            num_heads=1,
            head_dim=4,
            scaling=0.5,
            num_kv_heads=1,
            layer_id=i,
            sliding_window_size=window,
            attn_type=attn_type,
        )
        layer = nn.Module()
        layer.self_attn = attention
        model.layers.append(layer)
    return model


def _worker(model):
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker.draft_model = model
    worker.draft_model_runner = SimpleNamespace(
        model=model,
        sliding_window_size=model.get_attention_sliding_window_size(),
        model_config=SimpleNamespace(context_len=8192, hf_config=model.config),
    )
    return worker


class TestDSparkDraftWindow(CustomTestCase):
    def test_unset_preserves_native_windows_and_other_backends(self):
        with get_context().override_server_args(speculative_draft_window_size=None):
            for backend in ("trtllm_mha", "flashinfer", "triton", "dsv4"):
                for windows in ((), (-1, -1), (31, 31), (-1, 31)):
                    for attn_type in AttentionType:
                        with self.subTest(
                            backend=backend, windows=windows, type=attn_type
                        ):
                            model = _draft(windows, attn_type)
                            worker = _worker(model)
                            worker._configure_draft_window(backend)
                            self.assertEqual(
                                [
                                    l.self_attn.attn.sliding_window_size
                                    for l in model.layers
                                ],
                                list(windows),
                            )
                            self.assertEqual(
                                worker.draft_model_runner.sliding_window_size,
                                model.get_attention_sliding_window_size(),
                            )

    def test_window_includes_current_token_and_only_changes_draft(self):
        for window in (1, 2, 2048):
            with (
                self.subTest(window=window),
                get_context().override_server_args(
                    speculative_draft_window_size=window
                ),
            ):
                target = _draft()
                draft = _draft()
                config_before = draft.config.to_dict()
                worker = _worker(draft)
                worker._target_worker = SimpleNamespace(
                    model_runner=_worker(target).draft_model_runner
                )
                worker._configure_draft_window("trtllm_mha")
                self.assertEqual(
                    [l.self_attn.attn.sliding_window_size for l in draft.layers],
                    [window - 1, window - 1],
                )
                self.assertEqual(draft.get_attention_sliding_window_size(), window - 1)
                self.assertEqual(
                    worker.draft_model_runner.sliding_window_size, window - 1
                )
                self.assertEqual(
                    worker.draft_model_runner.model_config.context_len, 8192
                )
                self.assertEqual(draft.config.to_dict(), config_before)
                self.assertEqual(
                    [l.self_attn.attn.sliding_window_size for l in target.layers],
                    [-1, -1],
                )
                self.assertIsNone(target.get_attention_sliding_window_size())

    def test_constructor_configures_window_after_loading_draft(self):
        draft = _draft()
        runner = _worker(draft).draft_model_runner
        bundle = SimpleNamespace(
            draft_worker=SimpleNamespace(model_runner=runner),
            draft_model_runner=runner,
            draft_model=draft,
            resolved_attention_backend="trtllm_mha",
        )
        target_config = mock.Mock()
        # Stop before the rest of worker setup needs model weights or devices.
        type(target_config).hf_text_config = mock.PropertyMock(
            side_effect=RuntimeError("stop after draft configuration")
        )
        target = SimpleNamespace(
            device="cpu", model_runner=SimpleNamespace(model_config=target_config)
        )
        module = "sglang.srt.speculative.dspark_components.dspark_worker_v2"
        with (
            get_context().override_server_args(
                speculative_draft_window_size=2,
                cuda_graph_config=CudaGraphConfig(
                    decode=PhaseConfig(), prefill=PhaseConfig()
                ),
            ) as args,
            mock.patch(f"{module}.draft_is_deepseek_v4", return_value=False),
            mock.patch(f"{module}.build_draft_tp_worker", return_value=bundle),
            self.assertRaisesRegex(RuntimeError, "stop after draft configuration"),
        ):
            DSparkWorkerV2(
                server_args=args,
                gpu_id=0,
                ps=ParallelState.trivial(),
                nccl_port=0,
                target_worker=target,
            )
        self.assertEqual(draft.layers[0].self_attn.attn.sliding_window_size, 1)
        self.assertEqual(runner.sliding_window_size, 1)

    def test_explicit_window_rejects_other_backends(self):
        with get_context().override_server_args(speculative_draft_window_size=2):
            for backend in ("flashinfer", "triton", "fa3", "dsv4"):
                with self.subTest(backend=backend):
                    model = _draft()
                    with self.assertRaisesRegex(ValueError, "trtllm_mha"):
                        _worker(model)._configure_draft_window(backend)
                    self.assertIsNone(model.get_attention_sliding_window_size())

    def test_empty_or_noncausal_layers_fail_without_partial_update(self):
        with get_context().override_server_args(speculative_draft_window_size=2):
            models = [_draft(())]
            for attn_type in (
                AttentionType.ENCODER_ONLY,
                AttentionType.DECODER_BIDIRECTIONAL,
            ):
                model = _draft()
                model.layers[-1].self_attn.attn.attn_type = attn_type
                models.append(model)
            model = _draft()
            model.layers[-1].self_attn.attn = nn.Identity()
            models.append(model)
            for model in models:
                with self.subTest(model=model):
                    with self.assertRaisesRegex(ValueError, "nonempty causal"):
                        _worker(model)._configure_draft_window("trtllm_mha")
                    self.assertEqual(
                        [l.self_attn.sliding_window_size for l in model.layers],
                        [-1] * len(model.layers),
                    )
                    self.assertIsNone(model.get_attention_sliding_window_size())

    def test_checkpoint_window_layout_is_not_silently_overridden(self):
        with get_context().override_server_args(speculative_draft_window_size=2):
            for windows in ((-1, 1), (31, 31), (1, 31)):
                with self.subTest(windows=windows):
                    model = _draft(windows)
                    with self.assertRaisesRegex(ValueError, "checkpoint"):
                        _worker(model)._configure_draft_window("trtllm_mha")
                    self.assertEqual(
                        [l.self_attn.sliding_window_size for l in model.layers],
                        list(windows),
                    )
            for window in (1, 2, 2048):
                # A matching uniform checkpoint is compatible, including window_left=0.
                with get_context().override_server_args(
                    speculative_draft_window_size=window
                ):
                    model = _draft((window - 1, window - 1))
                    _worker(model)._configure_draft_window("trtllm_mha")
                    self.assertEqual(
                        model.get_attention_sliding_window_size(), window - 1
                    )
                    self.assertEqual(
                        model.layers[0].self_attn.attn.sliding_window_size, window - 1
                    )

    def test_window_does_not_rebase_rope_positions(self):
        model = _draft()
        model.set_attention_window(2)
        attention = model.layers[0].self_attn
        attention.q_size = attention.kv_size = attention.head_dim = 4
        attention.use_table_qk_norm_rope = False
        attention.v_scale = attention.attention_sink_bias = None
        attention.q_norm = RMSNorm(4)
        attention.k_norm = RMSNorm(4)
        hidden = torch.ones(3, 4)
        attention.qkv_proj = mock.Mock(return_value=(hidden.repeat(1, 3), None))
        attention.o_proj = mock.Mock(side_effect=lambda out: (out, None))
        seen_positions = []

        def rope(positions, q, k):
            seen_positions.append(positions.clone())
            return q, k

        attention.rotary_emb = rope
        positions = torch.tensor([8190, 8191, 8192])
        with mock.patch.object(
            attention.attn, "forward", side_effect=lambda q, k, v, batch: q
        ):
            attention(positions, hidden, SimpleNamespace())
        torch.testing.assert_close(seen_positions[0], torch.tensor([8190, 8191, 8192]))
        torch.testing.assert_close(positions, seen_positions[0])

    def test_nonpositive_windows_fail_in_central_validation(self):
        for window in (0, -1, -2048):
            with self.subTest(window=window):
                args = ServerArgs(
                    model_path="dummy",
                    speculative_algorithm="DSPARK",
                    speculative_draft_window_size=window,
                )
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    handle_speculative_decoding(args)

    def test_dspark_window_is_recognized_by_central_validation(self):
        args = ServerArgs(
            model_path="dummy",
            speculative_algorithm="DSPARK",
            speculative_draft_window_size=2048,
        )
        # Isolate the common validation from checkpoint loading in the algorithm hook.
        with (
            mock.patch("sglang.srt.arg_groups.speculative_hook._handle_dspark"),
            mock.patch(
                "sglang.srt.arg_groups.speculative_hook.logger.warning"
            ) as warning,
        ):
            handle_speculative_decoding(args)
        self.assertEqual(resolution_result(args, "speculative_draft_window_size"), 2048)
        self.assertFalse(
            any("has no effect" in str(call) for call in warning.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
