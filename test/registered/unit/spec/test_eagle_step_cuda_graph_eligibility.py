# SPDX-License-Identifier: Apache-2.0
"""Eligibility of the EAGLE step CUDA graph (``--enable-speculative-step-cuda-graph``).

CPU only:
  * server args: supported setups keep the flag; each unsupported one has a
    reason, and the launch resolves the flag to False instead of failing;
  * per step: the reason a batch runs the regular step (including a device
    tensor on the batch that the runner does not know), and the capture key;
  * fallbacks are counted in the stats a server reports;
  * a capture refused for lack of free memory is counted and logged once per
    key with the free and needed memory; when PyTorch caches enough unused
    memory, the cache is released once per key (and at most a fixed number of
    times) and the need checked again;
  * the start-up warm-up runs requests with the sampling a chat request without
    sampling fields resolves to (same capture key flags), as one batch of k
    requests for each batch size k up to the smaller of the runner's largest
    batch size and the maximum number of running requests, so that the
    runner captures every one of those keys; a missing one gets one more
    batch, then a warning; nothing runs without an active step graph runner;
  * a weight update that asks for a CUDA graph recapture is refused while the
    flag is on, before any runner loads a weight.
"""

import asyncio
import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.entrypoints import warmup as warmup_module
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import (
    weight_updater as weight_updater_module,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams
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

    def test_post_verify_graphs_and_compact_verify_sampling_turn_the_flag_off(self):
        # The post-verify graphs run their own capture inside the verify, and the
        # compact verify sampling picks its branch on the host per batch.
        for env in (
            envs.SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH,
            envs.SGLANG_Q35MF_EDGE_SPARSE_VERIFY_SAMPLING,
        ):
            with self.subTest(env=env.name), _cuda_platform(), env.override(True):
                args = _server_args()
                self.assertIn(env.name, step_cuda_graph_unsupported_reason(args))
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


class TestMemoryRefusal(CustomTestCase):
    MiB = 1 << 20

    def _check(self, runner, batch, free):
        with (
            mock.patch.object(runner_module, "_tp_min", side_effect=lambda *v: v),
            mock.patch.object(torch.cuda, "mem_get_info", return_value=(free, 0)),
            mock.patch.object(
                torch.cuda, "memory_reserved", return_value=900 * self.MiB
            ),
            mock.patch.object(
                torch.cuda, "memory_allocated", return_value=700 * self.MiB
            ),
            mock.patch.object(runner_module.logger, "info") as info,
        ):
            result = runner._capture_and_replay(("key",), batch, [], None, None)
        return result, info

    def test_a_refusal_is_counted_and_logged_once_per_key(self):
        runner = _runner()
        batch = _batch()
        # First key: the estimate (256 MiB) plus the 1 GiB headroom.
        result, info = self._check(runner, batch, free=1000 * self.MiB)
        self.assertIsNone(result)
        stats = runner.stats()
        self.assertEqual(stats["memory_refusals"], 1)
        self.assertEqual(
            stats["last_memory_refusal"],
            {
                "batch_size": 1,
                "greedy": False,
                "constrained": False,
                "free_mib": 1000,
                "need_mib": 1280,
                "reserved_mib": 900,
                "allocated_mib": 700,
            },
        )
        self.assertEqual(stats["fallbacks"], {"not enough free memory to capture": 1})
        self.assertEqual(info.call_count, 2)  # the numbers + the fallback reason
        self.assertIn("free %d MiB, need %d MiB", info.call_args_list[0].args[0])
        self.assertEqual(info.call_args_list[0].args[4:6], (1000, 1280))

        # The retry of the same key is counted, not logged again.
        _, info = self._check(runner, batch, free=1000 * self.MiB)
        self.assertEqual(runner.stats()["memory_refusals"], 2)
        self.assertEqual(info.call_count, 0)

    def test_the_need_follows_the_largest_captured_key(self):
        runner = _runner()
        runner._steps[runner._key(_batch(), [])] = SimpleNamespace(
            memory_bytes=22 * self.MiB, batch_size=1, greedy=True, constrained=False
        )
        with (
            mock.patch.object(runner_module, "_tp_min", side_effect=lambda *v: v),
            mock.patch.object(
                torch.cuda, "mem_get_info", return_value=(1100 * self.MiB, 0)
            ),
        ):
            self.assertTrue(runner._memory_ok(("key",), _batch()))
        self.assertEqual(runner.stats()["memory_refusals"], 0)
        self.assertIsNone(runner.stats()["last_memory_refusal"])


MiB = 1 << 20


class _FakeDevice:
    """One rank's device memory as the runner reads it. A capture takes
    ``capture_bytes``; releasing the cache returns ``releasable`` of the unused
    reserved memory (unused blocks of live graph pools stay)."""

    def __init__(self, free, reserved=0, allocated=0, releasable=1.0):
        self.free, self.reserved, self.allocated = free, reserved, allocated
        self.releasable = releasable
        self.capture_bytes = 22 * MiB
        self.empty_cache_calls = 0

    def empty_cache(self):
        self.empty_cache_calls += 1
        released = int((self.reserved - self.allocated) * self.releasable)
        self.free += released
        self.reserved -= released

    def capture(self):
        self.free -= self.capture_bytes
        self.reserved += self.capture_bytes
        self.allocated += self.capture_bytes

    @contextlib.contextmanager
    def patched(self):
        cuda = torch.cuda
        with contextlib.ExitStack() as stack:
            for target, name, kwargs in (
                (cuda, "mem_get_info", dict(side_effect=lambda: (self.free, 0))),
                (cuda, "memory_reserved", dict(side_effect=lambda: self.reserved)),
                (cuda, "memory_allocated", dict(side_effect=lambda: self.allocated)),
                (cuda, "empty_cache", dict(side_effect=self.empty_cache)),
                (cuda, "synchronize", {}),
                (cuda, "is_current_stream_capturing", dict(return_value=False)),
                (cuda, "stream", dict(side_effect=lambda s: contextlib.nullcontext())),
                (
                    runner_module,
                    "get_tp_group",
                    dict(return_value=SimpleNamespace(barrier=lambda: None)),
                ),
            ):
                stack.enter_context(mock.patch.object(target, name, **kwargs))
            tp_min = stack.enter_context(
                mock.patch.object(runner_module, "_tp_min", side_effect=lambda *v: v)
            )
            yield tp_min


class TestCacheRelease(CustomTestCase):
    def test_a_refusal_releases_the_cache_once_per_key(self):
        runner = _runner()
        # 500 MiB free, ~34 GiB reserved but unused: the window of a long prefill.
        device = _FakeDevice(500 * MiB, reserved=36000 * MiB, allocated=1000 * MiB)
        with device.patched(), mock.patch.object(runner_module.logger, "info") as info:
            self.assertTrue(runner._memory_ok(("a",), _batch()))
            self.assertEqual(device.empty_cache_calls, 1)
            self.assertIn("released PyTorch's cached memory", info.call_args.args[0])
            self.assertEqual(info.call_args.args[3:6], (500, 35500, 1280))

            # The cache grew back: the same key does not release it again.
            device.free, device.reserved = 500 * MiB, 36000 * MiB
            self.assertFalse(runner._memory_ok(("a",), _batch()))
            self.assertEqual(device.empty_cache_calls, 1)
            # Another key may release it once.
            self.assertTrue(runner._memory_ok(("b",), _batch(bs=2)))
            self.assertEqual(device.empty_cache_calls, 2)
        stats = runner.stats()
        self.assertEqual(stats["cache_releases"], 2)
        self.assertEqual(stats["memory_refusals"], 1)

    def test_no_release_when_the_cache_cannot_cover_the_need(self):
        runner = _runner()
        device = _FakeDevice(500 * MiB, reserved=1000 * MiB, allocated=900 * MiB)
        with device.patched():
            self.assertFalse(runner._memory_ok(("a",), _batch()))
        self.assertEqual(device.empty_cache_calls, 0)
        self.assertEqual(runner.stats()["cache_releases"], 0)
        self.assertEqual(runner.stats()["last_memory_refusal"]["free_mib"], 500)

    def test_a_release_that_frees_too_little_is_refused_and_not_repeated(self):
        runner = _runner()
        # The unused memory sits in live graph pools: releasing frees nothing.
        device = _FakeDevice(
            500 * MiB, reserved=36000 * MiB, allocated=1000 * MiB, releasable=0.0
        )
        with device.patched():
            for _ in range(3):
                self.assertFalse(runner._memory_ok(("a",), _batch()))
        self.assertEqual(device.empty_cache_calls, 1)
        stats = runner.stats()
        self.assertEqual((stats["cache_releases"], stats["memory_refusals"]), (1, 3))

    def test_releases_are_capped(self):
        runner = _runner()
        runner._cache_released = {
            ("old", i) for i in range(runner_module._MAX_CACHE_RELEASES)
        }
        device = _FakeDevice(500 * MiB, reserved=36000 * MiB, allocated=1000 * MiB)
        with device.patched():
            self.assertFalse(runner._memory_ok(("new",), _batch()))
        self.assertEqual(device.empty_cache_calls, 0)

    def test_enough_free_memory_is_one_check_as_before(self):
        runner = _runner()
        device = _FakeDevice(2000 * MiB, reserved=36000 * MiB, allocated=1000 * MiB)
        with device.patched() as tp_min:
            self.assertTrue(runner._memory_ok(("a",), _batch()))
        self.assertEqual(tp_min.call_count, 1)
        self.assertEqual(device.empty_cache_calls, 0)


def _key_flags(sampling_params):
    """(is_all_greedy, need_top_p, need_top_k, need_min_p) of one request, as
    SamplingBatchInfo derives the capture key's sampling fields."""
    params = SamplingParams(**sampling_params)
    return (
        params.top_k <= 1,
        params.top_p != 1.0,
        params.top_k != TOP_K_ALL,
        params.min_p > 0,
    )


def _prompts(req):
    """The prompts of one GenerateReqInput (one or a batch of token id lists)."""
    return req.input_ids if isinstance(req.input_ids[0], list) else [req.input_ids]


def _capturing_runner(device):
    """A runner whose probe step and capture are stand-ins (a capture takes
    memory from ``device``); everything else is the runner's own code."""
    runner = _runner()

    def probe(key, batch, on_publish, grammar_barrier):
        runner._output_layouts[key] = ()
        return "regular step"

    def capture(batch, step_tensors, output_layout):
        device.capture()
        return runner_module._StepGraph(
            graph=SimpleNamespace(_segments=[None]),
            input_names=[],
            static_inputs=[],
            result=None,
            stage_graphs=[],
            ring_slot=None,
        )

    runner._probe_step = probe
    runner._capture = capture
    runner._replay = lambda step, batch, on_publish, grammar_barrier: "replayed"
    return runner


class _FakeServer:
    """The tokenizer manager of a server with one scheduler. The prompts of one
    request are prefilled together and decode as one batch through the
    runner's step path; every step gives each request the most tokens a step
    can (the fewest steps). ``split`` lists batch sizes whose first batch
    does not form, and ``never`` those that never form."""

    def __init__(self, runner, max_running_requests, split=(), never=()):
        self.runner = runner
        self.preferred_sampling_params = None
        self.max_running_requests = max_running_requests
        self.split, self.never = set(split), set(never)
        self.model_config = SimpleNamespace(
            get_default_sampling_params=lambda: {"top_p": 0.95}
        )
        self.requests = []
        self.state_reads = 0

    async def get_internal_state(self):
        self.state_reads += 1
        state = {"effective_max_running_requests_per_dp": self.max_running_requests}
        if self.runner is not None:
            state["spec_step_cuda_graph"] = self.runner.stats()
        return [state]

    async def generate_request(self, req, request):
        self.requests.append(req)
        bs = len(_prompts(req))
        if bs in self.split:
            self.split.discard(bs)
        elif bs not in self.never:
            tokens_per_step = self.runner.worker.speculative_num_steps + 1
            batch = _batch(bs)
            (
                batch.sampling_info.is_all_greedy,
                batch.sampling_info.need_top_p_sampling,
                batch.sampling_info.need_top_k_sampling,
                batch.sampling_info.need_min_p_sampling,
            ) = _key_flags(req.sampling_params)
            for _ in range(req.sampling_params["max_new_tokens"] // tokens_per_step):
                self.runner.try_replay(batch)
        yield {}


class _FakeTokenizerManager:
    """Fixed scheduler states; records the requests."""

    def __init__(self, states, generation_config):
        self.states = states
        self.preferred_sampling_params = None
        self.model_config = SimpleNamespace(
            get_default_sampling_params=lambda: dict(generation_config)
        )
        self.requests = []

    async def get_internal_state(self):
        return self.states

    async def generate_request(self, req, request):
        self.requests.append(req)
        yield {}


def _active_state(max_running_requests, captured=()):
    keys = [
        {
            "batch_size": bs,
            "greedy": False,
            "need_top_k_sampling": False,
            "need_top_p_sampling": True,
            "need_min_p_sampling": False,
            "constrained": False,
            "memory_mb": 22.0,
        }
        for bs in captured
    ]
    return {
        "effective_max_running_requests_per_dp": max_running_requests,
        "spec_step_cuda_graph": {
            "disabled_reason": None,
            "max_batch_size": 8,
            "captured_keys": keys,
        },
    }


def _warmup(manager, disaggregation_mode="null", num_dp_ranks=1):
    with mock.patch(
        "sglang.srt.runtime_context.get_parallel",
        return_value=SimpleNamespace(num_dp_ranks=num_dp_ranks),
    ):
        asyncio.run(
            warmup_module.speculative_step_cuda_graph_warmup(
                disaggregation_mode, manager
            )
        )
    return manager.requests


class TestStartupWarmup(CustomTestCase):
    GENERATION_CONFIGS = {
        "none or openai defaults": ({}, (False, False, False, False)),
        "temperature only": ({"temperature": 0.6}, (False, False, False, False)),
        "top_p": ({"temperature": 1.0, "top_p": 0.95}, (False, True, False, False)),
        "top_k": ({"top_k": 40}, (False, False, True, False)),
        "min_p": ({"min_p": 0.05}, (False, False, False, True)),
        "greedy": ({"temperature": 0.0}, (True, False, True, False)),
    }

    def test_a_different_sampling_key_does_not_hide_a_missing_warmup_key(self):
        for field, value in (
            ("greedy", True),
            ("constrained", True),
            ("need_top_k_sampling", True),
            ("need_top_p_sampling", False),
            ("need_min_p_sampling", True),
        ):
            with self.subTest(field=field):
                state = _active_state(1, captured=[1])
                state["spec_step_cuda_graph"]["captured_keys"][0][field] = value
                manager = _FakeTokenizerManager([state], {"top_p": 0.95})
                with mock.patch.object(warmup_module.logger, "warning") as warning:
                    requests = _warmup(manager)
                self.assertEqual(len(requests), 2)
                warning.assert_called_once()

    def test_warmup_sampling_matches_a_chat_request_without_sampling_fields(self):
        for name, (config, flags) in self.GENERATION_CONFIGS.items():
            with self.subTest(generation_config=name):
                model_config = SimpleNamespace(
                    get_default_sampling_params=lambda config=config: dict(config)
                )
                warmup = warmup_module.default_chat_sampling_params(model_config, 64)
                request = ChatCompletionRequest.model_validate(
                    {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
                )
                chat = request.to_sampling_params(
                    stop=[],
                    model_generation_config=model_config.get_default_sampling_params(),
                )
                self.assertEqual(_key_flags(warmup), _key_flags(chat))
                self.assertEqual(_key_flags(warmup), flags)
                self.assertEqual(warmup["max_new_tokens"], 64)
                self.assertTrue(warmup["ignore_eos"])

    def test_keys_up_to_the_max_running_requests_are_captured_at_startup(self):
        cases = {
            # (max running requests, runner's largest batch size): batch sizes
            (1, 8): [1],
            (4, 8): [1, 2, 3, 4],
            (8, 8): [1, 2, 3, 4, 5, 6, 7, 8],
            (16, 8): [1, 2, 3, 4, 5, 6, 7, 8],
            (8, 4): [1, 2, 3, 4],
        }
        for (max_running_requests, max_batch_size), sizes in cases.items():
            with self.subTest(mr=max_running_requests, max_bs=max_batch_size):
                device = _FakeDevice(40 << 30)
                runner = _capturing_runner(device)
                runner.max_batch_size = max_batch_size
                server = _FakeServer(runner, max_running_requests)
                with (
                    device.patched(),
                    mock.patch.object(warmup_module.logger, "info") as info,
                    mock.patch.object(warmup_module.logger, "warning") as warning,
                ):
                    requests = _warmup(server)
                # One request per batch size, in order, and no retry.
                self.assertEqual([len(_prompts(r)) for r in requests], sizes)
                stats = runner.stats()
                keys = stats["captured_keys"]
                self.assertEqual([key["batch_size"] for key in keys], sizes)
                self.assertFalse(any(key["greedy"] for key in keys))
                self.assertEqual(
                    {key["memory_mb"] for key in keys}, {device.capture_bytes / MiB}
                )
                self.assertEqual(stats["memory_refusals"], 0)
                warning.assert_not_called()
                (done,) = [
                    c for c in info.call_args_list if "warm-up done" in c.args[0]
                ]
                self.assertEqual(done.args[4:6], (len(sizes), 22.0 * len(sizes)))

    def test_largest_batch_size_follows_the_stage_graphs(self):
        def stage(max_bs):
            return SimpleNamespace(max_bs=max_bs)

        draft_extend_backend = SimpleNamespace(shared_read_ends=lambda mode: None)
        worker = SimpleNamespace(
            draft_worker=SimpleNamespace(
                draft_runner=SimpleNamespace(),
                cuda_graph_runner=stage(8),
                cuda_graph_runner_for_draft_extend=stage(4),
                draft_extend_attn_backend=draft_extend_backend,
            ),
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(decode_cuda_graph_runner=stage(6))
            ),
            device="cpu",
        )
        tp_group = SimpleNamespace(world_size=1, pynccl_comm=None)
        no_fused_copy = SimpleNamespace(
            SGLANG_SPEC_STEP_GRAPH_FUSED_INPUT_COPY=SimpleNamespace(get=lambda: False)
        )
        with (
            mock.patch.object(
                EagleStepCudaGraphRunner,
                "_construction_unsupported_reason",
                return_value=None,
            ),
            mock.patch.object(runner_module, "get_tp_group", return_value=tp_group),
            mock.patch.object(runner_module, "get_stream", return_value=None),
            mock.patch.object(runner_module, "envs", no_fused_copy),
        ):
            runner = EagleStepCudaGraphRunner(worker)
        self.assertEqual(runner.stats()["max_batch_size"], 4)

    def test_one_batch_reaches_every_size_within_the_new_tokens(self):
        # Two regular steps, then the capture, at the most tokens per step.
        for num_steps in range(1, 21):
            self.assertGreaterEqual(
                warmup_module._STEP_GRAPH_WARMUP_NEW_TOKENS // (num_steps + 1),
                runner_module._NUM_WARMUP_STEPS + 1,
            )

    def test_a_missing_size_gets_one_more_batch_then_a_warning(self):
        for split, never, retried_sizes, missing in (
            ({3}, (), [3], []),
            ((), {3}, [3], [3]),
        ):
            with self.subTest(split=split, never=never):
                device = _FakeDevice(40 << 30)
                runner = _capturing_runner(device)
                server = _FakeServer(runner, 4, split=split, never=never)
                with (
                    device.patched(),
                    mock.patch.object(warmup_module.logger, "warning") as warning,
                ):
                    requests = _warmup(server)
                self.assertEqual(
                    [len(_prompts(r)) for r in requests], [1, 2, 3, 4] + retried_sizes
                )
                captured = [k["batch_size"] for k in runner.stats()["captured_keys"]]
                self.assertEqual(
                    sorted(captured), [b for b in (1, 2, 3, 4) if b not in missing]
                )
                if missing:
                    warning.assert_called_once()
                    self.assertEqual(warning.call_args.args[1], missing)
                else:
                    warning.assert_not_called()

    def test_each_batch_is_one_request_of_distinct_prompts_per_dp_rank(self):
        manager = _FakeTokenizerManager(
            [_active_state(1, captured=[1])], {"top_p": 0.95}
        )
        (req,) = _warmup(manager)
        self.assertEqual(req.input_ids, [10, 11, 12])
        self.assertEqual(req.sampling_params["top_p"], 0.95)
        self.assertEqual(req.sampling_params["max_new_tokens"], 64)
        self.assertTrue(req.sampling_params["ignore_eos"])

        states = [_active_state(2, captured=[1, 2])] * 2
        manager = _FakeTokenizerManager(states, {"top_p": 0.95})
        first, second = _warmup(manager, num_dp_ranks=2)
        self.assertEqual(first.input_ids, [[10, 11, 12], [11, 11, 12]])
        self.assertEqual(
            second.input_ids,
            [[10, 11, 12], [11, 11, 12], [12, 11, 12], [13, 11, 12]],
        )

    def test_flag_off_or_no_active_runner_sends_nothing(self):
        cases = {
            "flag off (no runner)": [{"effective_max_running_requests_per_dp": 8}],
            "runner turned itself off": [
                {
                    "effective_max_running_requests_per_dp": 8,
                    "spec_step_cuda_graph": {"disabled_reason": "no draft runner"},
                }
            ],
        }
        for name, states in cases.items():
            with self.subTest(case=name):
                self.assertEqual(_warmup(_FakeTokenizerManager(states, {})), [])
        server = _FakeServer(None, 8)
        self.assertEqual(_warmup(server), [])
        self.assertEqual(server.state_reads, 1)
        active = [_active_state(8)]
        manager = _FakeTokenizerManager(active, {})
        self.assertEqual(_warmup(manager, disaggregation_mode="decode"), [])


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
