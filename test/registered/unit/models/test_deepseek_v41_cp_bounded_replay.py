"""CP bounded replay preserves compact global rows through logits and DSpark."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.arg_groups import deepseek_v4_hook
from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import eager_runner
from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_model(capture=True, bounded=True):
    model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
    torch.nn.Module.__init__(model)
    model.capture_aux_hidden_states = capture
    model.model = SimpleNamespace(late_layer_start=3 if bounded else None)
    model.lm_head = object()
    model.logits_processor = mock.Mock(return_value=SimpleNamespace())
    return model


def make_batch():
    return SimpleNamespace(
        input_ids=torch.arange(512),
        forward_mode=ForwardMode.EXTEND,
        attn_cp_metadata=object(),
        return_logprob=False,
        extend_seq_lens_cpu=[302, 210],
        extend_logprob_start_lens_cpu=[302, 210],
    )


def make_tail(*, cp=True, graph=False):
    indices = torch.tensor([300, 301, 510, 511])
    return LateLayerTail(
        token_indices=torch.tensor([75, 127]) if cp else indices,
        positions=torch.arange(2 if cp else 4),
        extend_seq_lens=torch.tensor([2, 2]),
        extend_seq_lens_cpu=[2, 2],
        swa_out_cache_loc=torch.arange(4),
        cp_metadata=object() if cp else None,
        global_token_indices=indices if cp else None,
        real_rows_count=3 if graph else None,
    )


class TestCpBoundedReplayOutput(unittest.TestCase):
    def test_compact_global_rows_reach_logits_and_dspark(self):
        model, batch, tail = make_model(), make_batch(), make_tail()
        original_metadata = batch.attn_cp_metadata
        local = (torch.zeros(2, 3), torch.zeros(2, 6))
        aux = torch.zeros(2, 9)
        hidden, pre, global_aux = (torch.arange(4 * n).view(4, n) for n in (3, 6, 9))
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail)
        )
        seen_metadata = []

        def gather(value, fb, stream):
            seen_metadata.append(fb.attn_cp_metadata)
            return (hidden, pre) if isinstance(value, tuple) else global_aux

        with (
            mock.patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
            mock.patch.object(
                deepseek_v4, "cp_gather_after_forward", side_effect=gather
            ),
            mock.patch.object(torch.cuda, "current_stream", return_value=None),
            mock.patch.object(
                deepseek_v4.LogitsMetadata,
                "from_forward_batch",
                return_value=SimpleNamespace(),
            ),
        ):
            output = model.postprocess_cp_output((local, aux), batch)

        args, kwargs = model.logits_processor.call_args
        torch.testing.assert_close(args[0], torch.tensor([300, 301, 510, 511]))
        self.assertIs(args[1], hidden)
        self.assertIs(args[4], global_aux)
        self.assertEqual(args[3].extend_seq_lens_cpu, [2, 2])
        self.assertEqual(args[3].extend_logprob_start_lens_cpu, [2, 2])
        self.assertIsNone(kwargs["hidden_states_before_norm"])
        torch.testing.assert_close(
            output.hidden_states_token_indices, tail.global_token_indices
        )
        self.assertEqual(seen_metadata, [tail.cp_metadata] * 2)
        self.assertIs(batch.attn_cp_metadata, original_metadata)

    def test_collective_failure_restores_full_metadata(self):
        model, batch, tail = make_model(), make_batch(), make_tail()
        original_metadata = batch.attn_cp_metadata
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail)
        )
        with (
            mock.patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
            mock.patch.object(
                deepseek_v4,
                "cp_gather_after_forward",
                side_effect=RuntimeError("collective failed"),
            ),
            mock.patch.object(torch.cuda, "current_stream", return_value=None),
            self.assertRaisesRegex(RuntimeError, "collective failed"),
        ):
            model.postprocess_cp_output(((torch.zeros(2, 3), None), None), batch)
        self.assertIs(batch.attn_cp_metadata, original_metadata)
        model.logits_processor.assert_not_called()

    def test_missing_cp_layout_fails_before_collective(self):
        model, batch, tail = make_model(), make_batch(), make_tail(cp=False)
        original_metadata = batch.attn_cp_metadata
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail)
        )
        with (
            mock.patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
            mock.patch.object(deepseek_v4, "cp_gather_after_forward") as gather,
            self.assertRaisesRegex(RuntimeError, "missing its tail layout"),
        ):
            model.postprocess_cp_output(((torch.zeros(2, 3), None), None), batch)
        gather.assert_not_called()
        self.assertIs(batch.attn_cp_metadata, original_metadata)

    def test_full_cp_output_and_non_dspark_keep_original_metadata(self):
        for capture in (False, True):
            with self.subTest(capture=capture):
                model, batch = make_model(capture=capture, bounded=False), make_batch()
                metadata = batch.attn_cp_metadata
                hidden, pre, aux = (torch.ones(512, n) for n in (3, 6, 9))
                raw = ((hidden, pre), aux) if capture else (hidden, pre)
                with (
                    mock.patch.object(
                        deepseek_v4,
                        "cp_gather_after_forward",
                        side_effect=lambda x, *_: x,
                    ),
                    mock.patch.object(torch.cuda, "current_stream", return_value=None),
                ):
                    output = model.postprocess_cp_output(raw, batch)
                args, kwargs = model.logits_processor.call_args
                self.assertIs(args[0], batch.input_ids)
                self.assertIs(args[3], batch)
                self.assertIs(args[4], aux if capture else None)
                self.assertIs(
                    kwargs["hidden_states_before_norm"], None if capture else pre
                )
                self.assertFalse(hasattr(output, "hidden_states_token_indices"))
                self.assertIs(batch.attn_cp_metadata, metadata)

    def test_non_cp_graph_tail_still_trims_padding(self):
        model, batch, tail = make_model(), make_batch(), make_tail(cp=False, graph=True)
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail)
        )
        raw = ((torch.ones(4, 3), torch.ones(4, 6)), torch.ones(4, 9))
        with (
            mock.patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
            mock.patch.object(
                deepseek_v4.LogitsMetadata,
                "from_forward_batch",
                return_value=SimpleNamespace(),
            ),
        ):
            output = model._postprocess_output(raw, batch.input_ids, batch)
        args, _ = model.logits_processor.call_args
        torch.testing.assert_close(args[0], torch.tensor([300, 301, 510]))
        self.assertEqual(args[1].shape[0], 3)
        self.assertEqual(args[4].shape[0], 3)
        torch.testing.assert_close(
            output.hidden_states_token_indices, tail.token_indices[:3]
        )

    def test_prompt_logprob_offsets_use_compact_lengths(self):
        model, batch, tail = make_model(), make_batch(), make_tail(cp=False)
        batch.return_logprob = True
        batch.extend_logprob_start_lens_cpu = [0, 210]
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail)
        )
        with (
            mock.patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
            mock.patch.object(
                deepseek_v4.LogitsMetadata,
                "from_forward_batch",
                return_value=SimpleNamespace(),
            ),
        ):
            model._postprocess_output(
                ((torch.ones(4, 3), torch.ones(4, 6)), torch.ones(4, 9)),
                batch.input_ids,
                batch,
            )
        metadata = model.logits_processor.call_args.args[3]
        self.assertEqual(metadata.extend_logprob_start_lens_cpu, [0, 2])


class TestEagerCpOutputHook(unittest.TestCase):
    def test_model_hook_receives_raw_output_before_generic_gather(self):
        raw, result = object(), object()
        model = SimpleNamespace(
            prepare_cp_inputs=lambda *_args, **_kwargs: (None, None, {}),
            model=mock.Mock(return_value=raw),
            pp_group=SimpleNamespace(is_last_rank=True),
            postprocess_cp_output=mock.Mock(return_value=result),
        )
        runner = SimpleNamespace(model_runner=SimpleNamespace(model=model))
        batch = make_batch()

        @contextlib.contextmanager
        def shard(*_args):
            yield None, None, None

        with (
            mock.patch.object(eager_runner, "cp_shard_model_inputs", shard),
            mock.patch.object(eager_runner, "cp_gather_after_forward") as generic,
        ):
            actual = eager_runner.EagerRunner._execute_extend_cp(runner, batch, {})
        self.assertIs(actual, result)
        model.postprocess_cp_output.assert_called_once_with(raw, batch)
        generic.assert_not_called()


class TestCpBoundedReplayConfiguration(unittest.TestCase):
    def test_eager_cp_opt_in_and_graph_rejection(self):
        for backend in (Backend.DISABLED, Backend.BREAKABLE):
            cfg = SimpleNamespace(
                enable_encoder_swa_bounded_replay=False,
                dsv4_attn_backend="auto",
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(backend=backend)
                ),
                enable_hisparse=False,
                enable_two_batch_overlap=False,
                pp_size=1,
                speculative_algorithm=None,
                disaggregation_mode="null",
                enable_decoder_swa_bounded_replay=True,
                enable_prefill_cp=True,
                attn_cp_size=4,
            )
            with (
                mock.patch.object(deepseek_v4_hook, "resolving_view", return_value=cfg),
                mock.patch.object(
                    deepseek_v4_hook,
                    "model_config_of",
                    return_value=SimpleNamespace(
                        hf_config=SimpleNamespace(model_type="deepseek_v41")
                    ),
                ),
                mock.patch(
                    "sglang.srt.layers.attention.dsv4.unified_kv_gate.is_unified_kv_triton",
                    return_value=False,
                ),
                mock.patch.object(
                    deepseek_v4_hook, "attn_dp_enabled_of", return_value=False
                ),
            ):
                if backend == Backend.DISABLED:
                    deepseek_v4_hook.validate_deepseek_v41_features(object())
                else:
                    with self.assertRaisesRegex(
                        ValueError, "prefill CP with the prefill CUDA graph"
                    ):
                        deepseek_v4_hook.validate_deepseek_v41_features(object())


if __name__ == "__main__":
    unittest.main()
