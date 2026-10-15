import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt import runtime_context as rc
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import prepare_server_args
from sglang.srt.speculative import eagle_utils
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=60, suite="full-1-npu-a3", nightly=True)


class TestNpuEagleSampleUsesCliTargetOnlyThresholds(CustomTestCase):
    def setUp(self):
        super().setUp()
        rc.reset_context()

    def tearDown(self):
        try:
            rc.reset_context()
        finally:
            super().tearDown()

    def _assert_sample_uses_thresholds(
        self,
        threshold_single,
        threshold_acc,
        expected_predict,
        expected_accept_len,
        expected_accept_index,
    ):
        """Exercise parsed EAGLE3 CLI config through eagle_sample and the NPU kernel."""
        self.assertTrue(
            eagle_utils._is_npu,
            "This integration test must execute on an NPU.",
        )

        # ServerArgs resolution inspects target/draft model configs even though
        # this fixture calls only the sampler. A tiny local config keeps EAGLE3
        # resolution real while requiring no checkpoint, tokenizer, or network.
        with tempfile.TemporaryDirectory() as model_dir:
            config_path = Path(model_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "architectures": ["LlamaForCausalLM"],
                        "model_type": "llama",
                        "hidden_size": 8,
                        "intermediate_size": 16,
                        "max_position_embeddings": 32,
                        "num_attention_heads": 2,
                        "num_hidden_layers": 1,
                        "num_key_value_heads": 2,
                        "rms_norm_eps": 1e-5,
                        "vocab_size": 3,
                    }
                )
            )
            args = prepare_server_args(
                [
                    "--model-path",
                    model_dir,
                    "--speculative-draft-model-path",
                    model_dir,
                    "--device",
                    "npu",
                    "--attention-backend",
                    "ascend",
                    "--speculative-algorithm",
                    "EAGLE3",
                    "--speculative-num-steps",
                    "1",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "2",
                    "--speculative-accept-threshold-single",
                    str(threshold_single),
                    "--speculative-accept-threshold-acc",
                    str(threshold_acc),
                ]
            )
            rc.publish(args, role="test")
            spec = rc.get_spec()
            self.assertEqual(
                spec.speculative_accept_threshold_single,
                threshold_single,
            )
            self.assertEqual(spec.speculative_accept_threshold_acc, threshold_acc)
            self.assertFalse(spec.speculative_use_rejection_sampling)

            candidates = torch.tensor([[0, 1]], dtype=torch.int64, device="npu")
            verify_input = EagleVerifyInput(
                draft_token=candidates.reshape(-1),
                custom_mask=torch.ones((1, 2), dtype=torch.bool, device="npu"),
                positions=torch.tensor([0, 1], dtype=torch.int64, device="npu"),
                retrieve_index=torch.tensor([[0, 1]], dtype=torch.int64, device="npu"),
                retrieve_next_token=torch.tensor(
                    [[1, -1]], dtype=torch.int64, device="npu"
                ),
                retrieve_next_sibling=torch.tensor(
                    [[-1, -1]], dtype=torch.int64, device="npu"
                ),
                retrieve_cum_len=None,
                spec_steps=1,
                topk=1,
                draft_token_num=2,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                seq_lens_sum=2,
                seq_lens_cpu=torch.tensor([2], dtype=torch.int64),
            )
            batch = SimpleNamespace(
                device=torch.device("npu"),
                seq_lens=torch.tensor([2], dtype=torch.int64, device="npu"),
                sampling_info=SimpleNamespace(
                    is_all_greedy=False,
                    acc_additive_penalties=None,
                    acc_scaling_penalties=None,
                    logit_bias=None,
                    temperatures=torch.ones((1, 1), dtype=torch.float32, device="npu"),
                    need_top_k_sampling=False,
                    need_top_p_sampling=False,
                    sampling_seed=None,
                ),
                forward_mode=SimpleNamespace(is_idle=lambda: False),
            )
            target_probs = torch.tensor(
                [[[0.1, 0.6, 0.3], [0.8, 0.1, 0.1]]],
                dtype=torch.float32,
                device="npu",
            )
            logits_output = SimpleNamespace(
                next_token_logits=target_probs.reshape(2, 3).log()
            )

            # Keep the same acceptance and final coins across all three cases;
            # only the parsed threshold values vary.
            coins = torch.tensor([[0.7, 0.5]], dtype=torch.float32, device="npu")
            final_coin = torch.tensor([0.4], dtype=torch.float32, device="npu")
            sampler_calls = []
            from sgl_kernel_npu.sample import tree_speculative_sampling_target_only

            def record_and_forward(**kwargs):
                sampler_calls.append(
                    (
                        kwargs["threshold_single"],
                        kwargs["threshold_acc"],
                        kwargs["deterministic"],
                    )
                )
                return tree_speculative_sampling_target_only(**kwargs)

            with (
                patch(
                    "sglang.srt.speculative.spec_utils.SIMULATE_ACC_LEN",
                    0,
                ),
                patch.object(
                    eagle_utils,
                    "_verify_coins",
                    return_value=(coins, final_coin),
                ),
                patch(
                    "sgl_kernel_npu.sample.tree_speculative_sampling_target_only",
                    side_effect=record_and_forward,
                ),
                patch(
                    "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                    return_value=False,
                ),
                patch(
                    "sglang.srt.runtime_context.get_parallel",
                    return_value=SimpleNamespace(
                        tp_group=SimpleNamespace(world_size=1),
                        attn_tp_group=SimpleNamespace(world_size=1),
                    ),
                ),
            ):
                predict, accept_len, accept_index = eagle_utils.eagle_sample(
                    verify_input,
                    batch,
                    logits_output,
                )

            self.assertEqual(
                sampler_calls,
                [(threshold_single, threshold_acc, True)],
            )
            self.assertEqual(predict.cpu().tolist(), expected_predict)
            self.assertEqual(accept_len.cpu().tolist(), expected_accept_len)
            self.assertEqual(accept_index.cpu().tolist(), expected_accept_index)

    def test_single_threshold_changes_acceptance(self):
        self._assert_sample_uses_thresholds(0.5, 1.0, [1, 0], [2], [[0, 1]])

    def test_acc_threshold_changes_acceptance(self):
        self._assert_sample_uses_thresholds(1.0, 0.8, [1, 0], [2], [[0, 1]])

    def test_default_thresholds_reject(self):
        self._assert_sample_uses_thresholds(1.0, 1.0, [2, 0], [1], [[0, -1]])


if __name__ == "__main__":
    unittest.main()
