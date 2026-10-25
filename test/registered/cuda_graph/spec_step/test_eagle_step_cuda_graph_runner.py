# SPDX-License-Identifier: Apache-2.0
"""The EAGLE step CUDA graph runner's capture / replay loop on one GPU.

A fake worker has the shape of an EAGLE v2 decode step: three per-stage graphs
held by FullCudaGraphBackends with raw graphs kept, and eager work between
them (including a GEMM). Driving it through ``EagleStepCudaGraphRunner.
try_replay`` at batch sizes 1 and 2:
  * each batch size is captured once (after the warm-up steps) and every step,
    replayed through the 2-slot output ring, equals the eager step bitwise;
  * recapturing the stage graphs (a graph epoch change) drops every key
    together: one reset, and only the key in use is captured again;
  * recapture-then-replay: a weight update that asks for a recapture (which
    would build new stage backends the step graph does not track) is refused
    while the step graph is on, so later steps still replay and equal eager.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

D = 64
STEPS = 8


class _Stage:
    """A per-stage graph held by a FullCudaGraphBackend with raw graphs kept."""

    def __init__(self, fn):
        from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
            FullCudaGraphBackend,
        )

        self.backend = FullCudaGraphBackend.__new__(FullCudaGraphBackend)
        self.backend._graphs, self.backend._outputs = {}, {}
        self.backend._keep_raw_graph, self.backend.graph_epoch = True, 0
        self.fn, self.inp = fn, torch.zeros(8, D, device="cuda")
        self.capture()

    def capture(self):
        for _ in range(2):
            self.fn(self.inp)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            out = self.fn(self.inp) * 1.0
        graph.instantiate()
        self.backend._graphs, self.backend._outputs = {"k": graph}, {"k": out}
        self.backend.graph_epoch += 1

    def run(self, x):
        self.inp[: x.shape[0]].copy_(x)
        return self.backend.replay("k", None)[: x.shape[0]]


class _Worker:
    """Three stage graphs with eager work between them."""

    speculative_num_steps = 3

    def __init__(self, a, w, stages=None):
        self.stages = stages or [
            _Stage(lambda x: x.sin() + 1.0),
            _Stage(lambda x: torch.cumsum(x, 1) * 0.01),
            _Stage(lambda x: x.cos() * 0.5),
        ]
        self.a, self.w, self.runner = a, w, None

    def forward_batch_generation(
        self, batch, on_publish=None, grammar_barrier=None, pp_proxy_tensors=None
    ):
        if self.runner is not None:
            result = self.runner.try_replay(
                batch, on_publish, grammar_barrier, pp_proxy_tensors
            )
            if result is not None:
                return result
        from sglang.srt.managers.utils import GenerationBatchResult

        bs = batch.seq_lens.shape[0]
        gemm = (self.a @ self.w)[:bs, :D].float() * 1e-4
        h = self.stages[0].run(batch.spec_info.hidden_states + gemm)
        v = self.stages[1].run(h)
        tokens = ((v[:, :4].abs() * 1000).long() % 997).flatten().to(torch.int32)
        accept_lens = torch.ones(bs, dtype=torch.int32, device="cuda")
        e = self.stages[2].run(v)
        return GenerationBatchResult(
            next_token_ids=tokens,
            accept_lens=accept_lens,
            new_seq_lens=batch.seq_lens + accept_lens,
            next_draft_input=_draft_input(e * 1.0),
            extra_keep_alive_refs=[],
        )


def _draft_input(hidden):
    return SimpleNamespace(
        hidden_states=hidden,
        is_verify_input=lambda: False,
        capture_hidden_mode="LAST",
        future_dsa_topk_indices_available=False,
        cuda_graph_compatible=True,
    )


def _batch(bs, hidden, seq_lens=None):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    sampling = SimpleNamespace(
        temperatures=torch.ones(bs, 1, device="cuda"),
        is_all_greedy=True,
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
    return SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        seq_lens=(
            torch.full((bs,), 100, device="cuda") if seq_lens is None else seq_lens
        ),
        req_pool_indices=torch.arange(1, bs + 1, device="cuda"),
        spec_info=_draft_input(hidden),
        sampling_info=sampling,
        return_logprob=False,
        return_hidden_states=False,
        has_grammar=False,
        seq_lens_cpu=None,
        seq_lens_sum=None,
        hisparse_coordinator=None,
        mamba_track_indices=None,
        multimodal_inputs=None,
        reqs=[],
    )


class TestEagleStepCudaGraphRunner(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from cuda.bindings import runtime  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cuda-python not available")
        from sglang.srt.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from sglang.test.test_utils import publish_build_topology

        # The runner agrees on capture outcomes over the TP group (world size 1).
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29625")
        torch.cuda.set_device(0)
        publish_build_topology(tp_size=1)
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method="env://"
        )
        initialize_model_parallel()
        torch.manual_seed(0)
        cls.a = torch.randn(16, 6144, device="cuda", dtype=torch.bfloat16)
        cls.w = torch.randn(6144, 2048, device="cuda", dtype=torch.bfloat16)

    @staticmethod
    def _runner(worker):
        from sglang.srt.speculative.eagle_step_cuda_graph_runner import (
            EagleStepCudaGraphRunner,
        )

        runner = EagleStepCudaGraphRunner.__new__(EagleStepCudaGraphRunner)
        runner.worker = worker
        runner.target_model_runner = SimpleNamespace(forward_pass_id=0)
        runner.draft_model_runner = SimpleNamespace(shared_read_done_event=None)
        runner.device = "cuda"
        runner._init_state()
        runner._stream = torch.cuda.Stream()
        worker.runner = runner
        return runner

    @staticmethod
    def _drive(worker, bs, n=STEPS):
        hidden = torch.randn(
            bs, D, device="cuda", generator=torch.Generator("cuda").manual_seed(bs)
        )
        batch, outs = _batch(bs, hidden), []
        for _ in range(n):
            result = worker.forward_batch_generation(batch)
            hidden = result.next_draft_input.hidden_states
            outs.append((result.next_token_ids.tolist(), hidden.cpu()))
            batch = _batch(bs, hidden.clone(), result.new_seq_lens.clone())
        return outs

    def test_replays_match_eager_steps_and_an_epoch_change_resets_every_key(self):
        worker = _Worker(self.a, self.w)
        runner = self._runner(worker)
        got = self._drive(worker, 1) + self._drive(worker, 2)
        stats = runner.stats()
        self.assertEqual((stats["captures"], stats["keys"]), (2, 2), stats)
        self.assertGreater(stats["replays"], 0, stats)

        reference = _Worker(self.a, self.w, stages=worker.stages)
        expected = self._drive(reference, 1) + self._drive(reference, 2)
        for i, ((tokens, hidden), (ref_tokens, ref_hidden)) in enumerate(
            zip(got, expected)
        ):
            self.assertEqual(tokens, ref_tokens, f"step {i}")
            self.assertTrue(torch.equal(hidden, ref_hidden), f"step {i}")

        for stage in worker.stages:
            stage.capture()  # a new graph epoch: every key is released
        self._drive(worker, 1)
        stats = runner.stats()
        self.assertEqual((stats["resets"], stats["keys"]), (1, 1), stats)
        self.assertEqual(stats["capture_failures"], 0, stats)

    def test_a_recapture_request_is_refused_and_replays_go_on(self):
        from sglang.srt.managers.scheduler_components import (
            weight_updater as weight_updater_module,
        )

        worker = _Worker(self.a, self.w)
        runner = self._runner(worker)
        self._drive(worker, 1)
        before = runner.stats()
        stages = list(worker.stages)

        def update_weights_from_disk(model_path, load_format, recapture_cuda_graph):
            # What a recapture does to the stage graphs: new backends (and here
            # new weights), not an epoch change of the captured ones.
            if recapture_cuda_graph:
                worker.stages[:] = [
                    _Stage(lambda x, f=stage.fn: f(x) + 0.5) for stage in stages
                ]
            return True, "ok"

        target = SimpleNamespace(
            weight_updater=SimpleNamespace(
                update_weights_from_disk=mock.Mock(side_effect=update_weights_from_disk)
            )
        )
        parallel = SimpleNamespace(tp_group=SimpleNamespace(cpu_group=None))
        with mock.patch.object(
            weight_updater_module, "get_parallel", return_value=parallel
        ):
            manager = weight_updater_module.SchedulerWeightUpdaterManager(
                tp_worker=SimpleNamespace(
                    weight_update_runners=lambda: [("target", target)]
                ),
                draft_worker=None,
                memory_saver_adapter=None,
                flush_cache=lambda **kwargs: True,
                is_fully_idle=lambda **kwargs: True,
                scheduler=mock.Mock(),
            )
        request = SimpleNamespace(
            model_path="m",
            load_format=None,
            recapture_cuda_graph=True,
            weight_version=None,
            flush_cache=True,
            torch_empty_cache=False,
        )
        spec = SimpleNamespace(enable_speculative_step_cuda_graph=True)
        with mock.patch.object(weight_updater_module, "get_spec", return_value=spec):
            output = manager.update_weights_from_disk(request)
        self.assertFalse(output.success)
        target.weight_updater.update_weights_from_disk.assert_not_called()
        self.assertEqual(
            [id(stage) for stage in worker.stages], [id(stage) for stage in stages]
        )

        got = self._drive(worker, 1)
        stats = runner.stats()
        self.assertEqual(stats["captures"], before["captures"], stats)
        self.assertEqual(stats["resets"], 0, stats)
        self.assertEqual(stats["replays"], before["replays"] + STEPS, stats)
        reference = _Worker(self.a, self.w, stages=worker.stages)
        expected = self._drive(reference, 1)
        for i, ((tokens, hidden), (ref_tokens, ref_hidden)) in enumerate(
            zip(got, expected)
        ):
            self.assertEqual(tokens, ref_tokens, f"step {i}")
            self.assertTrue(torch.equal(hidden, ref_hidden), f"step {i}")


if __name__ == "__main__":
    unittest.main()
