import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.algorithm.dream import Dream, sample_tokens
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.dllm.mixin.scheduler import DllmManager, SchedulerDllmMixin
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _config(**kwargs):
    values = dict(
        algorithm="Dream",
        algorithm_config={"steps": 1},
        block_size=None,
        mask_id=99,
        max_running_requests=4,
        needs_full_prefill=True,
    )
    values.update(kwargs)
    return DllmConfig(**values)


class TestDreamRequestCanvas(CustomTestCase):
    def _make_req(self, config, *, max_new_tokens=3):
        return Req(
            rid="req",
            origin_input_text="prompt",
            origin_input_ids=array("q", [10, 11]),
            sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
            dllm_config=config,
        )

    def test_dream_initializes_one_full_generation_canvas(self):
        req = self._make_req(_config(mask_id=151666))

        self.assertEqual(req.dllm_phase, DllmReqPhase.INCOMING_DECODE)
        self.assertEqual(req.dllm_algo_state, {"prompt_len": 2})

        req.init_next_round_input()

        self.assertEqual(
            list(req.full_untruncated_fill_ids), [10, 11, 151666, 151666, 151666]
        )
        self.assertEqual(req.dllm_phase, DllmReqPhase.STAGING_DECODE)
        self.assertEqual(req.dllm_block_offset, 0)

        req.output_ids = array("q", [20])
        req.init_next_round_input()
        self.assertEqual(
            list(req.full_untruncated_fill_ids), [10, 11, 20, 151666, 151666]
        )

    def test_block_dllm_still_uses_fixed_block_canvas(self):
        config = DllmConfig(
            algorithm="LLaDA",
            algorithm_config={},
            block_size=4,
            mask_id=156895,
            max_running_requests=1,
        )
        req = self._make_req(config)
        req.init_next_round_input()

        self.assertEqual(
            list(req.full_untruncated_fill_ids),
            [10, 11, 156895, 156895, 156895, 156895],
        )
        self.assertEqual(req.dllm_phase, DllmReqPhase.STAGING_DECODE)


class TestDreamAlgorithm(CustomTestCase):
    def _forward_batch(self, input_ids, lengths, *, top_k=None):
        batch_size = len(lengths)
        return SimpleNamespace(
            input_ids=input_ids,
            extend_seq_lens_cpu=lengths,
            batch_size=batch_size,
            sampling_info=SimpleNamespace(
                original_temperatures=torch.ones(batch_size),
                original_top_ks=torch.tensor(
                    [-1 if top_k is None else top_k] * batch_size, dtype=torch.int32
                ),
                top_ps=torch.ones(batch_size),
            ),
        )

    def test_sample_tokens_applies_greedy_top_k(self):
        logits = torch.tensor([[0.0, 3.0, 1.0], [4.0, 1.0, 2.0]])
        confidence, tokens = sample_tokens(logits, temperature=0.0, top_k=1)

        self.assertEqual(tokens.tolist(), [1, 0])
        self.assertTrue(torch.all(confidence > 0))

    def test_dream_batches_requests_with_different_sequence_lengths(self):
        dream = Dream(_config())
        input_ids = torch.tensor([10, 11, 99, 99, 20, 99, 99])
        forward_batch = self._forward_batch(input_ids, [4, 3], top_k=1)
        logits = torch.zeros((7, 5))
        logits[:, 2] = 5.0
        states = [
            {"prompt_len": 2, "step": 0},
            {"prompt_len": 1, "step": 0},
        ]

        done = dream.step(forward_batch, logits, states)

        self.assertEqual(done, [True, True])
        self.assertEqual(forward_batch.input_ids.tolist(), [10, 11, 2, 2, 20, 2, 2])
        self.assertEqual([state["step"] for state in states], [1, 1])

    def test_dream_confidence_algorithms_resolve_final_canvas(self):
        for alg in ("maskgit_plus", "topk_margin", "entropy"):
            with self.subTest(alg=alg):
                dream = Dream(_config(algorithm_config={"steps": 1, "alg": alg}))
                forward_batch = self._forward_batch(
                    torch.tensor([10, 99, 99]), [3], top_k=1
                )
                logits = torch.zeros((3, 5))
                logits[:, 4] = 5.0
                states = [{"prompt_len": 1, "step": 0}]

                done = dream.step(forward_batch, logits, states)

                self.assertEqual(done, [True])
                self.assertEqual(forward_batch.input_ids.tolist(), [10, 4, 4])

    def test_dream_requires_cpu_sequence_lengths(self):
        dream = Dream(_config())
        forward_batch = self._forward_batch(torch.tensor([10, 99]), [2])
        forward_batch.extend_seq_lens_cpu = None

        with self.assertRaisesRegex(RuntimeError, "sequence lengths"):
            dream.step(
                forward_batch, torch.zeros((2, 5)), [{"prompt_len": 1, "step": 0}]
            )


class _OneStepAlgorithm(DllmAlgorithm):
    def __init__(self):
        super().__init__(
            DllmConfig(
                algorithm="stub",
                algorithm_config={},
                block_size=4,
                mask_id=99,
                max_running_requests=2,
            )
        )
        self.seen_states = None

    def init_step_state(self, forward_batch):
        return [{"prompt_len": 1} for _ in range(forward_batch.batch_size)]

    def max_steps(self, block_size):
        return 1

    def step(self, forward_batch, full_logits, states):
        self.seen_states = states
        return [True] * forward_batch.batch_size


class TestDllmAlgorithmSyncRun(CustomTestCase):
    def test_sync_run_splits_variable_length_requests(self):
        algorithm = _OneStepAlgorithm()
        forward_batch = SimpleNamespace(
            batch_size=2,
            input_ids=torch.tensor([10, 99, 99, 20, 99]),
            extend_seq_lens_cpu=[3, 2],
        )
        logits_output = SimpleNamespace(full_logits=torch.zeros((5, 4)))
        model_runner = MagicMock()
        model_runner.forward.return_value = SimpleNamespace(
            logits_output=logits_output,
            can_run_graph=False,
        )

        result = algorithm.run(
            model_runner,
            forward_batch,
            algo_states=[{"prompt_len": 1, "round": 7}, None],
        )

        self.assertEqual([tokens.tolist() for tokens in result[1]], [[99, 99], [99]])
        self.assertEqual(algorithm.seen_states[0]["round"], 7)
        model_runner.forward.assert_called_once()

    def test_sync_run_requires_sequence_lengths(self):
        algorithm = _OneStepAlgorithm()
        forward_batch = SimpleNamespace(
            batch_size=1,
            input_ids=torch.tensor([10, 99, 99, 99]),
            extend_seq_lens_cpu=None,
        )

        with self.assertRaisesRegex(RuntimeError, "sequence lengths"):
            algorithm.run(MagicMock(), forward_batch)


class TestDreamConfig(CustomTestCase):
    def _server_args(self, *, max_running_requests=None, fdfo=True):
        return SimpleNamespace(
            dllm_algorithm="Dream",
            dllm_algorithm_config=None,
            dllm_fdfo=fdfo,
            max_running_requests=max_running_requests,
            model_path="dream",
            revision=None,
        )

    @patch("sglang.srt.dllm.config.ModelConfig.from_server_args")
    def test_dream_uses_full_prefill_and_scheduler_capacity(self, from_server_args):
        from_server_args.return_value = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DreamModel"])
        )

        config = DllmConfig.from_server_args(
            self._server_args(max_running_requests=4, fdfo=True)
        )

        self.assertTrue(config.needs_full_prefill)
        self.assertIsNone(config.block_size)
        self.assertEqual(config.mask_id, 151666)
        self.assertEqual(config.max_running_requests, 4)
        self.assertFalse(config.first_done_first_out_mode)

    @patch("sglang.srt.dllm.config.ModelConfig.from_server_args")
    def test_block_dllm_defaults_remain_unchanged(self, from_server_args):
        from_server_args.return_value = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LLaDA2MoeModelLM"])
        )

        config = DllmConfig.from_server_args(self._server_args(fdfo=True))

        self.assertFalse(config.needs_full_prefill)
        self.assertEqual(config.block_size, 32)
        self.assertEqual(config.mask_id, 156895)
        self.assertEqual(config.max_running_requests, 1)
        self.assertTrue(config.first_done_first_out_mode)

    @patch("sglang.srt.arg_groups.overrides.run_post_process_pass")
    def test_server_guard_disables_unsupported_dream_features(
        self, run_post_process_pass
    ):
        args = ServerArgs.__new__(ServerArgs)
        args.dllm_algorithm = "Dream"
        args.dllm_fdfo = True
        args.tp_size = 1
        args.pp_size = 1
        args.disable_radix_cache = False
        args.disable_cuda_graph = False
        args.cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(backend=object()),
            prefill=SimpleNamespace(backend=object()),
        )
        args.enable_hierarchical_cache = False
        args.enable_lmcache = False
        args.enable_flexkv = False
        args.enable_lora = False
        args.disaggregation_mode = "null"
        args.enable_mixed_chunk = False
        args.get_model_config = MagicMock(
            return_value=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["DreamModel"])
            )
        )

        with patch("sglang.srt.server_args.is_hip", return_value=False):
            args._handle_dllm_inference()

        self.assertTrue(args.disable_radix_cache)
        self.assertTrue(args.disable_cuda_graph)
        self.assertFalse(args.dllm_fdfo)
        self.assertEqual(run_post_process_pass.call_count, 3)


class TestDreamSchedulerAdmission(CustomTestCase):
    def _scheduler(self, config, max_running_requests=4):
        scheduler = SimpleNamespace(
            dllm_config=config,
            dllm_manager=DllmManager(config),
            max_running_requests=max_running_requests,
            waiting_queue=[],
        )
        scheduler._fetch_waiting_reqs = SchedulerDllmMixin._fetch_waiting_reqs.__get__(
            scheduler
        )
        return scheduler

    def test_dream_fetch_uses_resolved_scheduler_capacity(self):
        scheduler = self._scheduler(_config(max_running_requests=1))
        requests = [SimpleNamespace(rid=f"req-{i}") for i in range(4)]
        scheduler.waiting_queue = requests

        scheduler._fetch_waiting_reqs()

        self.assertEqual(scheduler.dllm_manager.waiting_queue, requests)
        self.assertEqual(scheduler.waiting_queue, [])

    def test_block_dllm_keeps_configured_dllm_capacity(self):
        config = DllmConfig(
            algorithm="LLaDA",
            algorithm_config={},
            block_size=4,
            mask_id=156895,
            max_running_requests=1,
        )
        scheduler = self._scheduler(config)
        scheduler.waiting_queue = [SimpleNamespace(rid=f"req-{i}") for i in range(4)]

        scheduler._fetch_waiting_reqs()

        self.assertEqual(len(scheduler.dllm_manager.waiting_queue), 1)
        self.assertEqual(len(scheduler.waiting_queue), 3)


if __name__ == "__main__":
    unittest.main()
