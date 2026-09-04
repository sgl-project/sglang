"""Device-portable sampling contracts for linear UNO."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.hardware_backend.npu.graph_runner.uno_npu_graph_runner import (
    UnoNPUGraphRunner,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.speculative.uno_utils import (
    _dense_rejection_torch,
    _top_k,
    run_uno_sampling,
    sample_uno_candidates,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestUnoDeviceSampling(CustomTestCase):
    def test_uno_npu_graph_runner_keeps_npu_capture_and_replay(self):
        self.assertTrue(issubclass(UnoNPUGraphRunner, NPUGraphRunner))
        self.assertIs(UnoNPUGraphRunner.execute, NPUGraphRunner.execute)

    def test_npu_runner_forwards_nolora_graph_capture_option(self):
        from sglang.srt.compilation import torch_compile_decoration

        # Stop before shared initialization needs an accelerator or model.
        with (
            patch.object(torch_compile_decoration, "patch_model"),
            patch.object(
                NPUGraphRunner.__bases__[0],
                "__init__",
                side_effect=RuntimeError("shared runner reached"),
            ) as init,
            self.assertRaisesRegex(RuntimeError, "shared runner reached"),
        ):
            NPUGraphRunner(object(), record_nolora_graph=True)
        self.assertTrue(init.call_args.kwargs["record_nolora_graph"])

    def test_npu_replay_uses_uno_variant_and_final_host_lengths(self):
        module = "sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner"
        for variant in ("lora", "nolora"):
            with self.subTest(variant=variant):
                runner = UnoNPUGraphRunner.__new__(UnoNPUGraphRunner)
                runner.bs, runner.raw_bs, runner.raw_num_token = 4, 2, 8
                runner.is_dllm = False
                runner.load_batch = Mock()
                runner.model_runner = SimpleNamespace(
                    spec_algorithm=SimpleNamespace(is_uno=lambda: True),
                    model_config=SimpleNamespace(hf_config=object()),
                )
                runner._resolve_lora_variant = Mock(return_value=variant)
                runner._resolve_attention_variant = Mock(return_value=None)
                runner._make_graph_key = Mock(return_value="uno_graph")
                runner._get_update_attr_name = Mock(
                    return_value="actual_seq_lengths_kv"
                )
                runner._get_update_attr_type = Mock(return_value=[])
                runner.backend = SimpleNamespace(
                    replay_with_input_update=Mock(
                        return_value=LogitsProcessorOutput(
                            next_token_logits=torch.zeros(16, 3)
                        )
                    )
                )
                batch = SimpleNamespace(
                    needs_forward_metadata_init=lambda: True,
                    forward_mode=SimpleNamespace(is_target_verify=lambda: True),
                    seq_lens=torch.tensor([124, 252]),
                    seq_lens_cpu=torch.tensor([128, 256], dtype=torch.int32),
                )
                with (
                    patch(module + ".is_deepseek_dsa", return_value=False),
                    patch(module + ".is_deepseek_v4", return_value=False),
                ):
                    output = runner.execute(batch)
                runner._make_graph_key.assert_called_once_with(
                    4, variant_label=variant, attention_variant=None
                )
                self.assertEqual(
                    runner.backend.replay_with_input_update.call_args.kwargs[
                        "seq_lens"
                    ],
                    [128, 256, 0, 0],
                )
                self.assertEqual(output.next_token_logits.shape, (8, 3))

    def test_torch_topk_fallback_matches_reference(self):
        logits = torch.tensor([[0.1, 0.9, -0.2, 0.4]], dtype=torch.float32)
        expected_values, expected_indices = torch.topk(logits, k=3, dim=-1)

        with patch("sglang.srt.speculative.uno_utils._flashinfer_top_k", None):
            values, indices = _top_k(logits, 3)

        torch.testing.assert_close(values, expected_values)
        torch.testing.assert_close(indices, expected_indices)

    def test_npu_sampling_pipeline_avoids_shared_triton_sampler(self):
        module = "sglang.srt.speculative.uno_utils"
        # Use deterministic one-hot support so each output is independent of
        # RNG implementation, for both compact and dense draft distributions.
        for top_k in (2, 129):
            for width in (1, 4):
                with self.subTest(top_k=top_k, width=width):
                    logits = torch.full((2, width, 130), float("-inf"))
                    logits[:, :, 7] = 0
                    sampling = SimpleNamespace(
                        temperatures=torch.ones(2, 1),
                        top_ks=torch.full((2,), top_k, dtype=torch.int32),
                        top_ps=torch.ones(2),
                        need_top_k_sampling=True,
                        need_top_p_sampling=False,
                    )
                    frontiers = torch.tensor([124, 252], dtype=torch.int32)
                    with (
                        patch(module + "._is_npu", True),
                        patch(
                            module + ".fast_sample",
                            side_effect=AssertionError("shared Triton sampler reached"),
                        ),
                    ):
                        candidates, draft = sample_uno_candidates(
                            draft_logits=logits,
                            sampling_info=sampling,
                            max_top_k=top_k,
                            uniform_top_k_value=top_k,
                        )
                        result = run_uno_sampling(
                            candidates=candidates,
                            next_token_logits=logits.flatten(0, 1),
                            sampling_info=sampling,
                            committed_frontiers=frontiers,
                            draft_distribution=draft,
                            max_top_k=top_k,
                            uniform_top_k_value=top_k,
                        )
                    self.assertEqual(candidates.tolist(), [[7] * width] * 2)
                    self.assertEqual(
                        result.output_ids.tolist(), [[7] * (width + 1)] * 2
                    )
                    self.assertEqual(result.accept_lens.tolist(), [width + 1] * 2)
                    torch.testing.assert_close(
                        result.new_seq_lens, frontiers + width + 1
                    )

    def test_dense_rejection_all_accepted_and_single_token_width(self):
        for width in (1, 4):
            with self.subTest(width=width):
                target = torch.tensor([[[0.25, 0.75]]]).expand(1, width, 2)
                accepted, bonus = _dense_rejection_torch(
                    candidates=torch.ones((1, width), dtype=torch.int64),
                    target_probs=target,
                    draft_probs=target[:, : width - 1],
                    accept_uniforms=torch.full((1, width), 0.5),
                    final_uniforms=torch.tensor([0.25]),
                )
                self.assertEqual(accepted.tolist(), [width - 1])
                self.assertEqual(bonus.tolist(), [1])

    def test_dense_rejection_fallback_accepts_prefix_and_samples_residual(self):
        candidates = torch.tensor([[7, 1, 2]], dtype=torch.int64)
        target_probs = torch.tensor(
            [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
        )
        draft_probs = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])

        accepted, bonus = _dense_rejection_torch(
            candidates=candidates,
            target_probs=target_probs,
            draft_probs=draft_probs,
            accept_uniforms=torch.tensor([[0.5, 0.5, 0.5]]),
            final_uniforms=torch.tensor([0.0]),
        )

        self.assertEqual(accepted.tolist(), [1])
        self.assertEqual(bonus.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
