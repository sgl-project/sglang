import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeLayer:
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.calls = []
        self.hc_post_calls = 0

    def __call__(
        self,
        *,
        hidden_states,
        prev_residual,
        prev_post,
        prev_comb,
        **kwargs,
    ):
        self.calls.append((prev_residual, prev_post, prev_comb))
        value = self.layer_id + 1
        return (
            hidden_states + value,
            torch.tensor(value),
            torch.tensor(value + 10),
            torch.tensor(value + 20),
        )

    def hc_post(self, hidden_states, residual, post, comb):
        self.hc_post_calls += 1
        return hidden_states + residual + post + comb


class TestDeepseekV4SplitPrefill(unittest.TestCase):
    def _make_model(self):
        layers = [_FakeLayer(0), _FakeLayer(1)]
        model = SimpleNamespace(
            embed_tokens=lambda input_ids: input_ids.float().unsqueeze(-1),
            hc_mult=2,
            layers=layers,
            end_layer=len(layers),
            use_fused_mhc_post_pre=True,
            hc_head=lambda hidden, *args: hidden.sum(dim=1),
            hc_head_fn=None,
            hc_head_scale=None,
            hc_head_base=None,
            norm=lambda hidden: hidden,
        )
        return model, layers

    def _run_split(self, model, forward_batch, split_interval, input_embeds=None):
        with (
            patch(
                "sglang.srt.models.deepseek_v4.get_parallel",
                return_value=SimpleNamespace(attn_dp_size=1),
            ),
            patch(
                "sglang.srt.models.deepseek_v4.dsa_use_prefill_cp",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.deepseek_v4.check_cuda_graph_backend",
                return_value=True,
            ),
        ):
            return DeepseekV4Model.forward_split_prefill(
                model,
                torch.tensor([1, 2]),
                torch.tensor([0, 1]),
                forward_batch,
                split_interval,
                input_embeds,
            )

    def test_split_execution_preserves_cross_layer_mhc_state(self):
        model, layers = self._make_model()
        forward_batch = SimpleNamespace(
            hidden_states=None,
            model_specific_states=None,
            freqs_cis_c4=object(),
            freqs_cis_c128=object(),
        )

        self.assertIsNone(self._run_split(model, forward_batch, (0, 1)))
        self.assertFalse(hasattr(forward_batch, "freqs_cis_c4"))
        self.assertFalse(hasattr(forward_batch, "freqs_cis_c128"))
        self.assertEqual(layers[0].hc_post_calls, 0)

        result = self._run_split(model, forward_batch, (1, 2))

        self.assertIsNotNone(result)
        for actual, expected in zip(layers[1].calls[0], (1, 11, 21)):
            self.assertEqual(actual.item(), expected)
        self.assertEqual(layers[0].hc_post_calls, 0)
        self.assertEqual(layers[1].hc_post_calls, 1)

    def test_split_execution_matches_one_shot_execution(self):
        split_model, _ = self._make_model()
        split_batch = SimpleNamespace(hidden_states=None, model_specific_states=None)
        self._run_split(split_model, split_batch, (0, 1))
        split_result = self._run_split(split_model, split_batch, (1, 2))

        one_shot_model, _ = self._make_model()
        one_shot_batch = SimpleNamespace(hidden_states=None, model_specific_states=None)
        one_shot_result = self._run_split(one_shot_model, one_shot_batch, (0, 2))

        torch.testing.assert_close(split_result[0], one_shot_result[0])
        torch.testing.assert_close(split_result[1], one_shot_result[1])

    def test_split_execution_uses_supplied_embeddings_and_global_input_ids(self):
        model, _ = self._make_model()
        model.embed_tokens = Mock(side_effect=AssertionError("unexpected embedding"))
        global_ids = torch.tensor([3, 4])
        batch = SimpleNamespace(input_ids_global=global_ids)
        self._run_split(model, batch, (0, 1), torch.tensor([[5.0], [6.0]]))
        torch.testing.assert_close(
            batch.hidden_states[:, 0, 0], torch.tensor([6.0, 7.0])
        )
        self.assertIs(batch.model_specific_states["input_ids_global"], global_ids)

    def test_split_prefill_does_not_globally_enable_context_parallelism(self):
        self.assertFalse(ForwardMode.SPLIT_PREFILL.is_context_parallel_extend())
        self.assertFalse(ForwardMode.DECODE.is_context_parallel_extend())

    def test_runner_restores_cp_buffer_sizing_before_each_interval(self):
        operations = []
        batch = SimpleNamespace(split_index=0, forward_mode=ForwardMode.SPLIT_PREFILL)
        runner = SimpleNamespace(
            attn_backend=SimpleNamespace(
                init_forward_metadata=Mock(
                    side_effect=lambda batch: operations.append("attention")
                )
            ),
            model_config=SimpleNamespace(num_hidden_layers=2),
            device_timer=None,
            model=SimpleNamespace(
                supports_split_prefill_cp=True,
                forward_split_prefill=Mock(return_value=None),
            ),
        )
        batch.input_ids = torch.tensor([1, 2])
        batch.positions = torch.tensor([0, 1])
        with (
            patch(
                "sglang.srt.model_executor.model_runner.get_cp_strategy",
                return_value=object(),
            ),
            patch(
                "sglang.srt.model_executor.model_runner.is_cp_active", return_value=True
            ),
            patch(
                "sglang.srt.model_executor.model_runner.prepare_cp_forward",
                side_effect=lambda batch: operations.append("cp"),
            ),
            patch(
                "sglang.srt.model_executor.model_runner.device_timer_ctx",
                return_value=nullcontext(),
            ),
        ):
            ModelRunner.forward_split_prefill(runner, batch)
            ModelRunner.forward_split_prefill(runner, batch, reinit_attn_backend=True)
        self.assertEqual(operations, ["cp", "attention", "cp", "attention"])
        self.assertEqual(batch.split_index, 2)
        self.assertEqual(batch.forward_mode, ForwardMode.SPLIT_PREFILL)

    def test_runner_only_enables_cp_for_adapted_split_models(self):
        for supports_cp in (False, True):
            for fails in (False, True):
                with self.subTest(supports_cp=supports_cp, fails=fails):
                    batch = SimpleNamespace(
                        split_index=0,
                        forward_mode=ForwardMode.SPLIT_PREFILL,
                        input_ids=torch.tensor([1, 2]),
                        positions=torch.tensor([0, 1]),
                    )
                    expected_mode = (
                        ForwardMode.EXTEND if supports_cp else ForwardMode.SPLIT_PREFILL
                    )

                    def model_forward(*args):
                        self.assertEqual(batch.forward_mode, expected_mode)
                        if fails:
                            raise RuntimeError("injected model failure")

                    runner = SimpleNamespace(
                        attn_backend=Mock(),
                        model_config=SimpleNamespace(num_hidden_layers=2),
                        device_timer=None,
                        model=SimpleNamespace(
                            supports_split_prefill_cp=supports_cp,
                            forward_split_prefill=model_forward,
                        ),
                    )
                    with (
                        patch(
                            "sglang.srt.model_executor.model_runner.get_cp_strategy",
                            return_value=object(),
                        ),
                        patch(
                            "sglang.srt.model_executor.model_runner.is_cp_active",
                            side_effect=lambda b: (
                                b.forward_mode.is_context_parallel_extend()
                            ),
                        ),
                        patch(
                            "sglang.srt.model_executor.model_runner.prepare_cp_forward"
                        ) as prepare,
                        patch(
                            "sglang.srt.model_executor.model_runner.device_timer_ctx",
                            return_value=nullcontext(),
                        ),
                    ):
                        if fails:
                            with self.assertRaisesRegex(RuntimeError, "injected"):
                                ModelRunner.forward_split_prefill(runner, batch)
                        else:
                            ModelRunner.forward_split_prefill(runner, batch)
                        self.assertEqual(prepare.call_count, int(supports_cp))
                        self.assertEqual(batch.split_index, 0 if fails else 1)
                    self.assertEqual(batch.forward_mode, ForwardMode.SPLIT_PREFILL)

    def test_causal_lm_processes_only_final_logits(self):
        model_forward = Mock(
            side_effect=[None, (torch.tensor([1.0]), torch.tensor([2.0]))]
        )
        logits_processor = Mock(return_value="logits")
        model = SimpleNamespace(
            model=SimpleNamespace(forward_split_prefill=model_forward),
            logits_processor=logits_processor,
            lm_head=object(),
        )
        attn_context = SimpleNamespace(
            maybe_input_scattered=lambda forward_batch: nullcontext()
        )
        args = (
            torch.tensor([1]),
            torch.tensor([0]),
            SimpleNamespace(),
        )

        with (
            patch(
                "sglang.srt.models.deepseek_v4.get_attn_tp_context",
                return_value=attn_context,
            ),
            patch("sglang.srt.models.deepseek_v4.is_cp_active", return_value=False),
        ):
            self.assertIsNone(
                DeepseekV4ForCausalLM.forward_split_prefill(
                    model, *args, split_interval=(0, 1)
                )
            )
            result = DeepseekV4ForCausalLM.forward_split_prefill(
                model, *args, split_interval=(1, 2)
            )

        self.assertEqual(result, "logits")
        logits_processor.assert_called_once()

    def test_causal_lm_shards_first_interval_and_gathers_final_outputs(self):
        batch = SimpleNamespace()
        input_ids = torch.tensor([1, 2])
        positions = torch.tensor([0, 1])
        embeds = torch.tensor([[1.0], [2.0]])
        shard = (embeds[:1], positions[:1], input_ids[:1])
        local_result = (torch.tensor([3.0]), torch.tensor([4.0]))
        full_result = (torch.tensor([3.0, 5.0]), torch.tensor([4.0, 6.0]))
        calls = []

        @contextmanager
        def shard_inputs(*args):
            calls.append(args)
            batch.input_ids_global = input_ids
            try:
                yield shard
            finally:
                del batch.input_ids_global

        body = Mock(side_effect=[None, local_result])
        embedding = Mock(return_value=embeds)
        model = SimpleNamespace(
            model=SimpleNamespace(
                get_input_embeddings=lambda: embedding, forward_split_prefill=body
            ),
            logits_processor=Mock(return_value="logits"),
            lm_head=object(),
        )
        with (
            patch("sglang.srt.models.deepseek_v4.is_cp_active", return_value=True),
            patch("sglang.srt.models.deepseek_v4.cp_shard_model_inputs", shard_inputs),
            patch(
                "sglang.srt.models.deepseek_v4.cp_gather_after_forward",
                return_value=full_result,
            ) as gather,
            patch(
                "sglang.srt.models.deepseek_v4.torch.cuda.current_stream",
                return_value="stream",
            ),
            patch(
                "sglang.srt.models.deepseek_v4.get_attn_tp_context",
                return_value=SimpleNamespace(
                    maybe_input_scattered=lambda batch: nullcontext()
                ),
            ),
        ):
            self.assertIsNone(
                DeepseekV4ForCausalLM.forward_split_prefill(
                    model, input_ids, positions, batch, (0, 1)
                )
            )
            gather.assert_not_called()
            self.assertEqual(
                DeepseekV4ForCausalLM.forward_split_prefill(
                    model, input_ids, positions, batch, (1, 2)
                ),
                "logits",
            )
        self.assertEqual(len(calls), 1)
        embedding.assert_called_once_with(input_ids)
        self.assertFalse(hasattr(batch, "input_ids_global"))
        self.assertIs(body.call_args_list[0].args[0], shard[2])
        gather.assert_called_once_with(local_result, batch, "stream")
        model.logits_processor.assert_called_once_with(
            input_ids,
            full_result[0],
            model.lm_head,
            batch,
            hidden_states_before_norm=full_result[1],
        )


if __name__ == "__main__":
    unittest.main()
