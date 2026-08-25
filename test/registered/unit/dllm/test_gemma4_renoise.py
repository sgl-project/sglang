import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.overrides import _dllm_attention_backend
from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.algorithm.gemma4_renoise import Gemma4Renoise
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config(*, block_size=3, fdfo=False, **algorithm_config):
    return DllmConfig(
        algorithm="Gemma4Renoise",
        algorithm_config=algorithm_config,
        block_size=block_size,
        mask_id=-1,
        max_running_requests=8,
        first_done_first_out_mode=fdfo,
        requires_separate_context_encoding=True,
    )


def _batch(rids, block_size, *, sampling_seeds=None, encoder=False, empty=False):
    sampling_info = (
        None
        if sampling_seeds is None
        else SimpleNamespace(sampling_seed=torch.tensor(sampling_seeds))
    )
    num_tokens = 0 if empty else len(rids) * block_size
    return SimpleNamespace(
        batch_size=len(rids),
        input_ids=torch.full((num_tokens,), -1, dtype=torch.long),
        rids=list(rids),
        sampling_info=sampling_info,
        forward_mode=(ForwardMode.EXTEND if encoder else ForwardMode.DLLM_EXTEND),
        input_embeds=None,
    )


class _ScaledEmbedding(torch.nn.Embedding):
    def __init__(self, weight, scale):
        super().__init__(*weight.shape)
        with torch.no_grad():
            self.weight.copy_(weight)
        self.embed_scale = scale


class _FakeRunner:
    def __init__(self, vocab_size=4, hidden_size=3):
        weight = torch.arange(vocab_size * hidden_size, dtype=torch.float32).view(
            vocab_size, hidden_size
        )
        self.embedding = _ScaledEmbedding(weight / 10, 1.5)
        self.prepared_signals = []
        self.model = SimpleNamespace(
            get_input_embeddings=lambda: self.embedding,
            prepare_dllm_input_embeds=self.prepare_dllm_input_embeds,
        )
        self.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                text_config=SimpleNamespace(vocab_size=vocab_size)
            )
        )
        self.records = []

    def prepare_dllm_input_embeds(self, input_ids, signal):
        self.prepared_signals.append(None if signal is None else signal.clone())
        input_embeds = self.embedding(input_ids)
        if signal is not None:
            input_embeds = input_embeds + signal
        return input_embeds

    def forward(self, forward_batch, pp_proxy_tensors=None):
        self.records.append(
            {
                "input_ids": forward_batch.input_ids.clone(),
                "input_embeds": forward_batch.input_embeds.clone(),
                "self_conditioning": self.prepared_signals[-1],
            }
        )
        vocab_size = self.model_config.hf_config.text_config.vocab_size
        logits = torch.zeros(forward_batch.input_ids.numel(), vocab_size)
        logits[:, 0] = 3
        return SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=logits), can_run_graph=False
        )


class TestGemma4Renoise(unittest.TestCase):
    def _initialize(self, algorithm, batch, vocab_size=4, hidden_size=3):
        weight = torch.arange(vocab_size * hidden_size, dtype=torch.float32).view(
            vocab_size, hidden_size
        )
        algorithm.vocab_size = vocab_size
        algorithm.embed_tokens = _ScaledEmbedding(weight / 10, 1.5)
        states = algorithm.init_step_state(batch)
        runner = _FakeRunner(vocab_size=vocab_size, hidden_size=hidden_size)
        runner.embedding = algorithm.embed_tokens
        runner.model.get_input_embeddings = lambda: runner.embedding
        algorithm.prepare_inputs(runner, batch, states)
        return states

    def test_defaults_and_validation(self):
        algorithm = Gemma4Renoise(_config())
        self.assertEqual(algorithm.max_denoising_steps, 48)
        self.assertEqual(algorithm.entropy_bound, 0.1)
        self.assertEqual((algorithm.t_min, algorithm.t_max), (0.4, 0.8))
        self.assertEqual(algorithm.confidence_threshold, 0.005)
        self.assertEqual(algorithm.stability_threshold, 1)
        self.assertEqual(algorithm.max_steps(256), 48)

        invalid = [
            ({"max_denoising_steps": 0}, "max_denoising_steps"),
            ({"sampler_config": {"entropy_bound": 0}}, "entropy_bound"),
            (
                {"temperature_schedule": {"t_min": -0.1, "t_max": 0.8}},
                "temperature_schedule",
            ),
            (
                {"temperature_schedule": {"t_min": 0.8, "t_max": 0.8}},
                "temperature_schedule",
            ),
            (
                {"stopping_config": {"stability_threshold": -1}},
                "stopping_config",
            ),
            (
                {"stopping_config": {"confidence_threshold": 0}},
                "stopping_config",
            ),
            ({"sampler_config": "invalid"}, "sampler_config"),
            ({"seed": "invalid"}, "seed"),
        ]
        for values, message in invalid:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    Gemma4Renoise(_config(**values))

        flat = Gemma4Renoise(
            _config(
                t_min=0.2,
                t_max=0.6,
                confidence_threshold=0.01,
                stability_threshold=2,
            )
        )
        self.assertEqual((flat.t_min, flat.t_max), (0.2, 0.6))
        self.assertEqual(flat.confidence_threshold, 0.01)
        self.assertEqual(flat.stability_threshold, 2)

        with self.assertRaisesRegex(ValueError, "max_denoising_steps"):
            get_algorithm(_config(max_denoising_steps=0))

    def test_launch_constraints_are_owned_by_algorithm(self):
        hf_config = SimpleNamespace(
            tie_word_embeddings=True,
            text_config=SimpleNamespace(tie_word_embeddings=True),
        )
        server_args = SimpleNamespace(
            device="cuda",
            get_model_config=lambda: SimpleNamespace(
                hf_config=hf_config, quantization=None
            ),
            pp_size=1,
            dcp_size=1,
            attn_cp_size=1,
            attention_backend="flashinfer",
            prefill_attention_backend="fa3",
            decode_attention_backend=None,
            disable_radix_cache=False,
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend=Backend.FULL),
                prefill=SimpleNamespace(backend=Backend.FULL),
            ),
            chunked_prefill_size=128,
        )

        Gemma4Renoise.configure_server_args(server_args)

        self.assertEqual(server_args.attention_backend, "flashinfer")
        self.assertEqual(server_args.prefill_attention_backend, "fa3")
        self.assertIsNone(server_args.decode_attention_backend)
        self.assertTrue(server_args.disable_radix_cache)
        self.assertEqual(server_args.cuda_graph_config.decode.backend, Backend.DISABLED)
        self.assertEqual(
            server_args.cuda_graph_config.prefill.backend, Backend.DISABLED
        )
        self.assertEqual(server_args.chunked_prefill_size, -1)

        with self.assertRaisesRegex(ValueError, "GPU execution"):
            Gemma4Renoise.configure_server_args(SimpleNamespace(device="cpu"))

    def test_required_attention_backend_uses_generic_override_pass(self):
        view = SimpleNamespace(
            dllm_algorithm="Gemma4Renoise",
            attention_backend="flashinfer",
            prefill_attention_backend="fa3",
            decode_attention_backend=None,
        )
        self.assertEqual(
            _dllm_attention_backend(view),
            {
                "attention_backend": "triton",
                "prefill_attention_backend": "triton",
                "decode_attention_backend": "triton",
            },
        )

    def test_model_compatibility_populates_algorithm_capabilities(self):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["DiffusionGemmaForBlockDiffusion"],
                canvas_length=192,
            )
        )
        server_args = SimpleNamespace(
            dllm_algorithm="Gemma4Renoise",
            model_path="model",
            revision=None,
            max_running_requests=None,
            dllm_algorithm_config=None,
            dllm_fdfo=True,
        )

        with patch(
            "sglang.srt.dllm.config.ModelConfig.from_server_args",
            return_value=model_config,
        ):
            config = DllmConfig.from_server_args(server_args)
            self.assertEqual(config.block_size, 192)
            self.assertTrue(config.requires_separate_context_encoding)

            server_args.dllm_algorithm = "LowConfidence"
            with self.assertRaisesRegex(ValueError, "requires the Gemma4Renoise"):
                DllmConfig.from_server_args(server_args)

    def test_first_canvas_initialization_is_deterministic_and_written(self):
        algorithm = Gemma4Renoise(_config(block_size=4, seed=17))
        batch = _batch(["request-a", "request-b"], 4)
        algorithm.vocab_size = 7
        states = algorithm.init_step_state(batch)

        self.assertTrue(torch.all(batch.input_ids == -1))
        for state in states:
            self.assertEqual(state["step"], 48)
            self.assertEqual(state["current"].shape, (4,))
            self.assertTrue(torch.all((state["current"] >= 0) & (state["current"] < 7)))
            self.assertIsNone(state["self_conditioning"])
            self.assertFalse(state["finished"])

        expected = torch.stack([state["current"] for state in states]).reshape(-1)
        runner = _FakeRunner(vocab_size=7)
        algorithm.embed_tokens = runner.embedding
        algorithm.prepare_inputs(runner, batch, states)
        torch.testing.assert_close(batch.input_ids, expected)
        self.assertIsNotNone(batch.input_embeds)
        self.assertIsNone(runner.prepared_signals[-1])

        repeat = algorithm.init_step_state(_batch(["other-a", "other-b"], 4))
        for first, second in zip(states, repeat):
            torch.testing.assert_close(first["current"], second["current"])
            torch.testing.assert_close(first["rng_state"], second["rng_state"])

    def test_entropy_bound_and_state_progression(self):
        algorithm = Gemma4Renoise(
            _config(
                max_denoising_steps=3,
                sampler_config={"entropy_bound": 0.1},
                stopping_config={
                    "confidence_threshold": 1e-12,
                    "stability_threshold": 10,
                },
                seed=9,
            )
        )
        batch = _batch(["request"], 3)
        state = self._initialize(algorithm, batch, vocab_size=3)[0]
        logits = torch.tensor(
            [[100.0, -100.0, -100.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )

        processed = logits / algorithm._temperature(state["step"])
        log_probabilities = torch.log_softmax(processed, dim=-1)
        probabilities = log_probabilities.exp()
        generator = torch.Generator(device="cpu")
        generator.set_state(state["rng_state"])
        sampled = torch.multinomial(
            probabilities, num_samples=1, generator=generator
        ).squeeze(-1)
        entropy = -(probabilities * log_probabilities).sum(dim=-1)
        sorted_entropy, indices = torch.sort(entropy)
        cumulative = torch.cumsum(sorted_entropy, dim=-1)
        selected = cumulative - sorted_entropy <= algorithm.entropy_bound
        selected = torch.zeros_like(selected).scatter(-1, indices, selected)
        self.assertEqual(selected.tolist(), [True, True, False])
        random_canvas = torch.randint(3, (3,), generator=generator, dtype=torch.long)
        expected = torch.where(selected, sampled, random_canvas)

        self.assertEqual(algorithm.step(batch, logits, [state]), [False])
        self.assertEqual(state["step"], 2)
        self.assertEqual(len(state["history"]), 1)
        self.assertFalse(state["finished"])
        torch.testing.assert_close(state["current"], expected)
        torch.testing.assert_close(batch.input_ids, expected)
        self.assertEqual(state["self_conditioning"].shape, (3, 3))

    def test_soft_embeddings_apply_probabilities_and_embedding_scale(self):
        algorithm = Gemma4Renoise(_config())
        weight = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, -1.0]])
        algorithm.embed_tokens = _ScaledEmbedding(weight, 2.5)
        probabilities = torch.tensor([[0.2, 0.7, 0.1], [0.6, 0.1, 0.3]])

        actual = algorithm._soft_embeddings(probabilities)
        expected = probabilities @ weight * 2.5

        self.assertEqual(actual.shape, (2, 2))
        self.assertEqual(actual.dtype, weight.dtype)
        torch.testing.assert_close(actual, expected)

    def test_request_rng_is_invariant_to_batch_order(self):
        values = dict(
            max_denoising_steps=4,
            stopping_config={
                "confidence_threshold": 1e-12,
                "stability_threshold": 10,
            },
        )
        first = Gemma4Renoise(_config(block_size=2, **values))
        second = Gemma4Renoise(_config(block_size=2, **values))
        first_batch = _batch(["alpha", "beta"], 2)
        second_batch = _batch(["beta", "alpha"], 2)
        first_states = self._initialize(first, first_batch, vocab_size=5)
        second_states = self._initialize(second, second_batch, vocab_size=5)

        logits_by_rid = {
            "alpha": torch.tensor([[3.0, 1.0, 0.0, -1.0, -2.0]] * 2),
            "beta": torch.tensor([[-1.0, 0.0, 1.0, 2.0, 3.0]] * 2),
        }
        first_logits = torch.cat([logits_by_rid[rid] for rid in first_batch.rids])
        second_logits = torch.cat([logits_by_rid[rid] for rid in second_batch.rids])
        first.step(first_batch, first_logits, first_states)
        second.step(second_batch, second_logits, second_states)

        first_by_rid = dict(zip(first_batch.rids, first_states))
        second_by_rid = dict(zip(second_batch.rids, second_states))
        for rid in first_by_rid:
            left, right = first_by_rid[rid], second_by_rid[rid]
            self.assertEqual(left["step"], right["step"])
            for key in ("current", "argmax", "self_conditioning", "rng_state"):
                torch.testing.assert_close(left[key], right[key])

    def test_max_step_completion_is_idempotent(self):
        algorithm = Gemma4Renoise(
            _config(
                block_size=2,
                max_denoising_steps=2,
                stopping_config={
                    "confidence_threshold": 1e-12,
                    "stability_threshold": 10,
                },
                seed=5,
            )
        )
        batch = _batch(["request"], 2)
        state = self._initialize(algorithm, batch, vocab_size=4)[0]
        logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]] * 2)

        self.assertEqual(algorithm.step(batch, logits, [state]), [False])
        self.assertEqual(algorithm.step(batch, logits, [state]), [True])
        self.assertEqual(state["step"], 0)
        self.assertTrue(state["finished"])
        self.assertIsNone(state["self_conditioning"])
        torch.testing.assert_close(state["current"], state["argmax"])
        final = state["current"].clone()

        self.assertEqual(algorithm.step(batch, logits, [state]), [True])
        self.assertEqual(state["step"], 0)
        torch.testing.assert_close(state["current"], final)

    def test_encoder_returns_five_tuple_and_skips_denoising(self):
        marker = object()
        runner = SimpleNamespace(
            calls=0,
            forward=lambda forward_batch, pp_proxy_tensors=None: SimpleNamespace(
                logits_output=marker, can_run_graph=True
            ),
        )
        algorithm = Gemma4Renoise(_config())
        batch = _batch(["request"], 3, encoder=True)

        output = algorithm.run(runner, batch)
        self.assertEqual(output, (marker, [], None, None, True))

        empty = _batch(["request"], 3, encoder=True, empty=True)
        self.assertEqual(algorithm.run(runner, empty), (None, [], None, None, False))

    def test_fdfo_mixes_fresh_and_carried_state(self):
        algorithm = Gemma4Renoise(
            _config(
                block_size=2,
                fdfo=True,
                max_denoising_steps=3,
                stopping_config={
                    "confidence_threshold": 1e-12,
                    "stability_threshold": 10,
                },
                seed=23,
            )
        )
        runner = _FakeRunner()
        first_batch = _batch(["carried"], 2)
        first_output = algorithm.run(runner, first_batch)
        self.assertEqual(first_output[2], [0])
        carried = first_output[3][0]
        self.assertEqual(carried["step"], 2)
        carried_canvas = carried["current"].clone()
        carried_signal = carried["self_conditioning"].clone()

        reference = Gemma4Renoise(
            _config(block_size=2, fdfo=True, max_denoising_steps=3, seed=23)
        )
        reference.vocab_size = 4
        fresh_canvas = reference.init_step_state(_batch(["fresh"], 2))[0]["current"]

        mixed_batch = _batch(["carried", "fresh"], 2)
        mixed_output = algorithm.run(runner, mixed_batch, [carried, None])
        record = runner.records[-1]
        observed_canvas = record["input_ids"].view(2, 2)
        torch.testing.assert_close(observed_canvas[0], carried_canvas)
        torch.testing.assert_close(observed_canvas[1], fresh_canvas)

        observed_signal = record["self_conditioning"].view(2, 2, 3)
        torch.testing.assert_close(observed_signal[0], carried_signal)
        torch.testing.assert_close(observed_signal[1], torch.zeros_like(carried_signal))
        self.assertEqual(mixed_output[2], [0, 0])
        self.assertEqual([state["step"] for state in mixed_output[3]], [1, 2])


if __name__ == "__main__":
    unittest.main()
