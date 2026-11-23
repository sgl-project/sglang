"""CPU semantic checks for Clover's pure block preview."""

import unittest

import torch

from sglang.srt.layers.attention.linear.clover_pure_memory import bd_gdn_local_memory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCloverPure(unittest.TestCase):
    def test_experimental_policy_requires_valid_explicit_options(self):
        from sglang.srt.dllm.clover_sampling import validate_decode_option

        for key, value in (
            ("max_denoise_steps", 0),
            ("max_denoise_steps", 33),
            ("refine_uncertain", "yes"),
        ):
            with self.assertRaises(ValueError):
                validate_decode_option(key, value, 32)
        validate_decode_option("max_denoise_steps", None, 32)
        validate_decode_option("max_denoise_steps", 3, 32)

    def test_default_budget_and_checkpoint_stop_fallback(self):
        from types import SimpleNamespace

        from sglang.srt.dllm.clover_sampling import (
            prepare_sampling_params,
            resolve_stop_ids,
        )

        gen = SimpleNamespace(
            max_new_tokens=24576,
            eos_token_id=None,
            pad_token_id=None,
            stop_token_ids=[3],
        )
        cfg = SimpleNamespace(block_size=32, eos_token_id=[1, 2], pad_token_id=None)
        model = SimpleNamespace(hf_generation_config=gen, hf_config=cfg, vocab_size=5)
        self.assertEqual(resolve_stop_ids(gen, cfg), ({1, 2, 3}, 1))
        params = {"stop_token_ids": [4]}
        prepare_sampling_params(params, model, 27, 2048)
        self.assertEqual(params["max_new_tokens"], 2016)
        self.assertEqual(params["stop_token_ids"], [1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "context limit"):
            prepare_sampling_params({"max_new_tokens": 2017}, model, 27, 2048)

    def test_history_bucket_decay_only_affects_history(self):
        q = torch.ones(1, 2, 1, 1)
        state = torch.full((1, 1, 1, 1), 4.0)
        decay = torch.tensor([0.25, 0.5]).view(1, 2, 1)
        result = bd_gdn_local_memory(
            q,
            q,
            q * 3,
            decay.log(),
            clean_ssm_states=state,
            cache_indices=torch.tensor([0]),
            active_valid_mask=torch.tensor([True, True]),
            scale=1,
            softmax_logit_scale=torch.zeros(1),
            softmax_history_bias=torch.zeros(1),
            softmax_history_c=torch.zeros(1),
        )
        expected = (6 + 4 * decay * torch.rsqrt(torch.tensor(1.000001))) / 3
        torch.testing.assert_close(result.squeeze(-1), expected)
        torch.testing.assert_close(state, torch.full_like(state, 4))

    def test_preview_finishes_before_replay_and_clips_tail(self):
        from types import SimpleNamespace as NS

        from sglang.srt.dllm.algorithm.clover_pure import CloverPureBlock
        from sglang.srt.dllm.clover_sampling import PENALTY_DEFAULTS

        batch = NS(
            input_ids=torch.zeros(3, dtype=torch.long),
            forward_mode=NS(is_dllm_extend=lambda: True),
            mark_forward_metadata_ready=lambda **kwargs: None,
        )
        sampling = NS(
            max_new_tokens=5,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            stop_token_ids=set(),
            custom_params=None,
        )
        req = NS(output_ids=[1, 2, 3], sampling_params=sampling)
        gen = NS(**PENALTY_DEFAULTS, eos_token_id=4, pad_token_id=4)
        cfg = NS(
            hf_generation_config=gen,
            vocab_size=5,
            hf_config=NS(denoising_config={"threshold": 1.0}),
        )
        calls = []

        def forward(batch, pp_proxy_tensors):
            calls.append(batch.input_ids.clone().tolist())
            scores = torch.zeros(3, 5)
            scores[:, 0] = 999  # mask is never an output
            scores[0, 1], scores[1, 2], scores[2, 3] = 1.0, 2.0, 3.0
            return NS(logits_output=NS(full_logits=scores), can_run_graph=False)

        runner = NS(model_config=cfg, forward=forward)
        algo = CloverPureBlock(
            NS(
                block_size=3,
                mask_id=0,
                first_done_first_out_mode=False,
                algorithm_config={},
            )
        )
        _, tokens, *_ = algo.run(runner, batch, [req])
        assert calls == [[0, 0, 0], [0, 0, 3], [0, 2, 3]]
        assert tokens[0].tolist() == [1, 2]
        calls.clear()
        sampling.custom_params = {"threshold": 0.0}
        _, tokens, *_ = algo.run(runner, batch, [req])
        assert calls == [[0, 0, 0]]
        assert tokens[0].tolist() == [1, 2]

        gen.eos_token_id = None
        gen.pad_token_id = None
        gen.stop_token_ids = [1]
        _, tokens, *_ = algo.run(runner, batch, [req])
        assert tokens[0].tolist() == [1]


if __name__ == "__main__":
    unittest.main()
