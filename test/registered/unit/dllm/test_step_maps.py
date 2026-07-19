import unittest
from types import SimpleNamespace

import torch

from sglang.srt.dllm.algorithm.joint_threshold import JointThreshold
from sglang.srt.dllm.algorithm.low_confidence import LowConfidence
from sglang.srt.dllm.algorithm.sampling import sample_block_tokens
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-b-test-cpu")


def _config(*, fdfo=False, algorithm_config=None):
    return SimpleNamespace(
        block_size=4,
        mask_id=0,
        first_done_first_out_mode=fdfo,
        algorithm_config=algorithm_config or {"threshold": 1.1},
    )


def _sampling_info(*, temperature=1.0, top_p=1.0, top_k=TOP_K_ALL):
    return SamplingBatchInfo(
        temperatures=torch.tensor([[temperature]], dtype=torch.float32),
        top_ps=torch.tensor([top_p], dtype=torch.float32),
        top_ks=torch.tensor([top_k], dtype=torch.int32),
        min_ps=torch.tensor([0.0], dtype=torch.float32),
        is_all_greedy=top_k <= 1,
        is_any_greedy=top_k <= 1,
        need_top_p_sampling=top_p != 1.0,
        need_top_k_sampling=top_k != TOP_K_ALL,
        need_min_p_sampling=False,
        vocab_size=5,
        device="cpu",
    )


class _FakeRunner:
    def __init__(self, batch_size=1):
        logits = torch.full((batch_size, 4, 5), -10.0)
        for batch_id in range(batch_size):
            for position in range(4):
                logits[batch_id, position, 1 + position % 3] = 6.0 - position
        self.logits = logits.view(batch_size * 4, 5)

    def forward(self, forward_batch, pp_proxy_tensors=None):
        return SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=self.logits.clone()),
            can_run_graph=False,
        )


def _forward_batch(input_ids):
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    return SimpleNamespace(
        batch_size=input_ids.numel() // 4,
        input_ids=input_ids,
        positions=torch.arange(input_ids.numel(), dtype=torch.long),
        sampling_info=None,
    )


class TestDllmStepMaps(unittest.TestCase):
    def test_request_flag_normalizes_for_single_and_mixed_batch(self):
        single = GenerateReqInput(input_ids=[1], return_step_maps=True)
        single.normalize_batch_and_arguments()
        self.assertTrue(single.return_step_maps)

        batch = GenerateReqInput(input_ids=[[1], [2]], return_step_maps=[True, False])
        batch.normalize_batch_and_arguments()
        self.assertEqual(batch.return_step_maps, [True, False])
        self.assertTrue(batch[0].return_step_maps)
        self.assertFalse(batch[1].return_step_maps)

    def test_sync_maps_are_one_based_token_aligned_and_opt_in(self):
        algorithm = LowConfidence(_config())
        forward_batch = _forward_batch([4, 0, 0, 0, 0, 0, 0, 0])
        result = algorithm.run(
            _FakeRunner(batch_size=2),
            forward_batch,
            return_step_maps=[True, False],
        )

        output_ids = result[1]
        step_maps = result[4]
        self.assertEqual(len(output_ids[0]), 3)
        self.assertEqual(step_maps[0], [1, 2, 3])
        self.assertEqual(len(step_maps[0]), len(output_ids[0]))
        self.assertIsNone(step_maps[1])

    def test_fdfo_carries_trace_until_the_block_is_emitted(self):
        algorithm = LowConfidence(_config(fdfo=True))
        forward_batch = _forward_batch([0, 0, 0, 0])
        algo_states = None
        step_map_states = None

        for _ in range(6):
            result = algorithm.run(
                _FakeRunner(),
                forward_batch,
                algo_states,
                [True],
                step_map_states,
            )
            algo_states = result[3]
            step_map_states = result[5]
            if result[2][0] == 4:
                break
        else:
            self.fail("FDFO block did not resolve")

        self.assertEqual(result[4], [[1, 2, 3, 4]])
        self.assertIsNone(step_map_states[0])

    def test_non_low_confidence_algorithm_fails_fast(self):
        algorithm = JointThreshold(_config())
        with self.assertRaisesRegex(ValueError, "LowConfidence"):
            algorithm.run(
                _FakeRunner(),
                _forward_batch([0, 0, 0, 0]),
                return_step_maps=[True],
            )

    def test_scheduler_exports_validated_maps_via_public_metadata_key(self):
        req = SimpleNamespace(
            rid="request-0",
            return_step_maps=True,
            dllm_step_maps=[],
            output_ids=[11, 12],
            customized_info=None,
        )
        SchedulerDllmMixin._append_dllm_step_maps(req, [1, 2], 2)
        self.assertEqual(req.customized_info, {"step_maps": [1, 2]})

        with self.assertRaisesRegex(RuntimeError, "positive one-based"):
            SchedulerDllmMixin._append_dllm_step_maps(req, [0], 1)


class TestDllmSampling(unittest.TestCase):
    def test_greedy_preserves_argmax(self):
        logits = torch.tensor([[0.0, 2.0, 1.0, -1.0, -2.0]])
        token_ids, token_probs = sample_block_tokens(
            logits, _sampling_info(temperature=0.7, top_p=0.5, top_k=1), 0
        )
        expected = torch.softmax(logits, dim=-1)
        self.assertEqual(token_ids.tolist(), [1])
        self.assertTrue(
            torch.allclose(
                token_probs,
                expected.gather(1, token_ids.unsqueeze(-1)).squeeze(-1),
            )
        )

    def test_top_k_limits_sampled_tokens(self):
        torch.manual_seed(0)
        logits = torch.tensor([[0.0, 10.0, 9.0, 8.0, 7.0]]).repeat(128, 1)
        token_ids, _ = sample_block_tokens(logits, _sampling_info(top_k=2), 0)
        self.assertTrue(set(token_ids.tolist()).issubset({1, 2}))


if __name__ == "__main__":
    unittest.main()
