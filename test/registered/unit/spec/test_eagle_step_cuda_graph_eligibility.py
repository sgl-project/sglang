# SPDX-License-Identifier: Apache-2.0
"""Eligibility of the EAGLE step CUDA graph (``--enable-speculative-step-cuda-graph``).

CPU only:
  * server args: supported setups keep the flag; each unsupported one has a
    reason, and the launch resolves the flag to False instead of failing;
  * per step: the reason a batch runs the regular step (including a device
    tensor on the batch that the runner does not know), and the capture key;
  * fallbacks are counted in the stats a server reports;
  * a weight update that asks for a CUDA graph recapture is refused while the
    flag is on, before any runner loads a weight.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.managers.scheduler_components import (
    weight_updater as weight_updater_module,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import eagle_step_cuda_graph_runner as runner_module
from sglang.srt.speculative.eagle_step_cuda_graph_runner import (
    EagleStepCudaGraphRunner,
    step_cuda_graph_unsupported_reason,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _server_args(**overrides):
    args = ServerArgs(
        model_path="dummy",
        device="cuda",
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        enable_speculative_step_cuda_graph=True,
    )
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@contextlib.contextmanager
def _cuda_platform():
    """Pretend to run on CUDA with cuda-python and the native allocator."""
    with (
        mock.patch.object(runner_module, "is_cuda", return_value=True),
        mock.patch.object(runner_module, "_cuda_python_available", return_value=True),
        mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"),
    ):
        yield


class TestServerArgs(CustomTestCase):
    def test_supported_setup_keeps_the_flag(self):
        with _cuda_platform():
            for algorithm in ("EAGLE", "EAGLE3", "NEXTN"):
                with self.subTest(algorithm=algorithm):
                    args = _server_args(speculative_algorithm=algorithm)
                    self.assertIsNone(step_cuda_graph_unsupported_reason(args))
                    handle_speculative_decoding(args)
                    self.assertTrue(
                        resolution_result(args, "enable_speculative_step_cuda_graph")
                    )

    def test_unsupported_setups_turn_the_flag_off(self):
        cases = {
            "topk": dict(speculative_eagle_topk=2, speculative_num_draft_tokens=8),
            "multi-layer": dict(enable_multi_layer_eagle=True),
            "rejection sampling": dict(speculative_use_rejection_sampling=True),
            "adaptive": dict(speculative_adaptive=True),
            "pipeline": dict(pp_size=2),
            "two-batch overlap": dict(enable_two_batch_overlap=True),
            "lora": dict(enable_lora=True),
            "memory saver": dict(enable_memory_saver=True),
            "disaggregation": dict(disaggregation_mode="decode"),
        }
        with _cuda_platform():
            for name, overrides in cases.items():
                with self.subTest(case=name):
                    args = _server_args(**overrides)
                    self.assertIsNotNone(step_cuda_graph_unsupported_reason(args))

    def test_launch_resolves_the_flag_off(self):
        self.assertFalse(
            ServerArgs(model_path="dummy").enable_speculative_step_cuda_graph
        )
        with mock.patch.object(runner_module, "is_cuda", return_value=False):
            args = _server_args()
            self.assertIn("CUDA", step_cuda_graph_unsupported_reason(args))
            handle_speculative_decoding(args)
            self.assertFalse(
                resolution_result(args, "enable_speculative_step_cuda_graph")
            )


def _runner():
    runner = EagleStepCudaGraphRunner.__new__(EagleStepCudaGraphRunner)
    runner.worker = SimpleNamespace(speculative_num_steps=3)
    runner._init_state()
    return runner


def _batch(bs=1, **overrides):
    spec_info = SimpleNamespace(
        topk_p=torch.zeros(bs, 1),
        topk_index=torch.zeros(bs, 1, dtype=torch.int64),
        capture_hidden_mode="LAST",
        future_dsa_topk_indices_available=True,
        cuda_graph_compatible=True,
        is_verify_input=lambda: False,
    )
    sampling_info = SimpleNamespace(
        temperatures=torch.ones(bs, 1),
        is_all_greedy=False,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        has_custom_logit_processor=False,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        has_watermark_candidates=False,
        sampling_mask_batch_indices=None,
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        seq_lens=torch.full((bs,), 100, dtype=torch.int64),
        req_pool_indices=torch.arange(bs),
        spec_info=spec_info,
        sampling_info=sampling_info,
        return_logprob=False,
        return_hidden_states=False,
        has_grammar=False,
        seq_lens_cpu=None,
        seq_lens_sum=None,
        hisparse_coordinator=None,
        mamba_track_indices=None,
        multimodal_inputs=None,
        reqs=[SimpleNamespace(grammar=None) for _ in range(bs)],
    )
    for key, value in overrides.items():
        setattr(batch, key, value)
    return batch


class TestPerStepEligibility(CustomTestCase):
    def test_eligible_batches(self):
        runner = _runner()
        self.assertIsNone(runner._ineligible_reason(_batch(), None))
        # Every request constrained by a grammar: still eligible.
        batch = _batch(has_grammar=True, reqs=[SimpleNamespace(grammar=object())])
        self.assertIsNone(runner._ineligible_reason(batch, None))

    def test_reasons(self):
        runner = _runner()
        cases = {
            "idle batch": dict(forward_mode=ForwardMode.IDLE),
            "batch size above 8": dict(seq_lens=torch.zeros(9, dtype=torch.int64)),
            "logprobs requested": dict(return_logprob=True),
            "hidden states requested": dict(return_hidden_states=True),
            "constrained and unconstrained requests mixed": dict(
                has_grammar=True,
                reqs=[SimpleNamespace(grammar=object()), SimpleNamespace(grammar=None)],
                seq_lens=torch.zeros(2, dtype=torch.int64),
            ),
            "host sequence lengths": dict(seq_lens_cpu=torch.zeros(1)),
            "no draft input": dict(spec_info=None),
        }
        for reason, overrides in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(
                    runner._ineligible_reason(_batch(**overrides), None), reason
                )
        self.assertEqual(
            runner._ineligible_reason(_batch(), object()), "pipeline proxy tensors"
        )
        batch = _batch()
        batch.sampling_info.logit_bias = torch.zeros(1, 8)
        self.assertEqual(runner._ineligible_reason(batch, None), "logit bias")

    def test_unknown_device_tensor_on_the_batch_is_refused(self):
        runner = _runner()
        table = torch.zeros(4)
        batch = _batch(extra_table=table)
        with mock.patch.object(
            runner_module, "_is_device_tensor", side_effect=lambda v: v is table
        ):
            reason = runner._ineligible_reason(batch, None)
        self.assertEqual(reason, "batch tensor extra_table")

    def test_key_tracks_sampling_and_layout(self):
        runner = _runner()
        key = runner._key(_batch(), [])
        self.assertEqual(key, runner._key(_batch(), []))
        greedy = _batch()
        greedy.sampling_info.is_all_greedy = True
        self.assertNotEqual(key, runner._key(greedy, []))
        self.assertNotEqual(key, runner._key(_batch(bs=2), []))
        layout = [("spec_info", "topk_p", torch.zeros(1, 1))]
        self.assertNotEqual(key, runner._key(_batch(), layout))

    def test_fallbacks_are_counted(self):
        runner = _runner()
        runner._fallback("logprobs requested")
        runner._fallback("logprobs requested")
        stats = runner.stats()
        self.assertEqual(stats["fallbacks"], {"logprobs requested": 2})
        self.assertEqual((stats["replays"], stats["keys"]), (0, 0))
        self.assertIsNone(stats["disabled_reason"])


def _weight_updater(target, draft):
    """The scheduler's weight-update manager over two fake runners."""
    parallel = SimpleNamespace(tp_group=SimpleNamespace(cpu_group=None))
    with mock.patch.object(
        weight_updater_module, "get_parallel", return_value=parallel
    ):
        return weight_updater_module.SchedulerWeightUpdaterManager(
            tp_worker=SimpleNamespace(
                weight_update_runners=lambda: [("target", target)]
            ),
            draft_worker=SimpleNamespace(
                weight_update_runners=lambda: [("draft", draft)]
            ),
            memory_saver_adapter=None,
            flush_cache=mock.Mock(return_value=True),
            is_fully_idle=lambda **kwargs: True,
            scheduler=mock.Mock(),
        )


class TestWeightUpdateRecapture(CustomTestCase):
    def _update(self, *, step_graph, recapture):
        target, draft = mock.Mock(), mock.Mock()
        for runner in (target, draft):
            runner.weight_updater.update_weights_from_disk.return_value = (True, "ok")
        manager = _weight_updater(target, draft)
        request = SimpleNamespace(
            model_path="m",
            load_format=None,
            recapture_cuda_graph=recapture,
            weight_version="v2",
            flush_cache=True,
            torch_empty_cache=False,
        )
        spec = SimpleNamespace(enable_speculative_step_cuda_graph=step_graph)
        with mock.patch.object(weight_updater_module, "get_spec", return_value=spec):
            output = manager.update_weights_from_disk(request)
        return output, manager, (target, draft)

    def test_recapture_is_refused_with_the_step_graph(self):
        output, manager, runners = self._update(step_graph=True, recapture=True)
        self.assertFalse(output.success)
        self.assertIn("--enable-speculative-step-cuda-graph", output.message)
        for runner in runners:
            runner.weight_updater.update_weights_from_disk.assert_not_called()
        manager.flush_cache.assert_not_called()
        manager.scheduler.record_weight_version_change.assert_not_called()

    def test_updates_without_recapture_or_without_the_flag_are_unchanged(self):
        for step_graph, recapture in ((True, False), (False, True), (False, False)):
            with self.subTest(step_graph=step_graph, recapture=recapture):
                output, manager, runners = self._update(
                    step_graph=step_graph, recapture=recapture
                )
                self.assertTrue(output.success, output.message)
                for runner in runners:
                    runner.weight_updater.update_weights_from_disk.assert_called_once_with(
                        "m", None, recapture_cuda_graph=recapture
                    )
                manager.flush_cache.assert_called_once()


if __name__ == "__main__":
    unittest.main()
