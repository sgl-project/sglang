# SPDX-License-Identifier: Apache-2.0
"""Cache accounting must describe real projections and survive report transport."""

import dataclasses
import json
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
from sglang.multimodal_gen.runtime.disaggregation.orchestrator import (
    _deserialize_request_metrics,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import (
    GPUWorker,
    _ExpandedOutputParts,
)
from sglang.multimodal_gen.runtime.models.dits import (
    minimax_h3_adaln_cache as cache_module,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_adaln_cache import (
    MiniMaxH3AdalnCache,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
    _record_adaln_cache_stats,
)
from sglang.multimodal_gen.runtime.utils import perf_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import (
    PerformanceLogger,
    RequestMetrics,
    RequestPerfRecord,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Measured CPU: 0.027 s for the cases; 97.354 s including shared-filesystem imports.
register_cuda_ci(
    est_time=100, stage="base-b", runner_config="diffusion-unit-1-gpu-h100"
)

NAME = "minimax_h3_adaln"


class TestAdalnPerfStats(CustomTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        # CPU component tests only replace the distributed/environment boundary.
        self.stack.enter_context(
            patch.object(cache_module, "get_tp_world_size", return_value=1)
        )
        self.stack.enter_context(
            patch.object(cache_module, "world_group_is_initialized", return_value=False)
        )
        self.stack.enter_context(
            patch.object(cache_module, "_host_cache_budget_bytes", return_value=1 << 20)
        )
        self.arch = MiniMaxH3DiTArchConfig(
            num_layers=2, hidden_size=4, time_embed_dim=3
        )
        self.weights = {}
        block_width = (
            6 * cache_module.MINIMAX_H3_ADALN_MODALITY_NUM * self.arch.hidden_size
        )
        for prefix, width in [
            (f"blocks.{i}.adaln_proj.linear", block_width) for i in range(2)
        ] + [("final_layer.adaln_proj.linear", 8)]:
            self.weights[prefix + ".weight"] = (
                torch.arange(width * 3).reshape(width, 3) % 11 - 5
            ).float() / 16
            self.weights[prefix + ".bias"] = torch.arange(width).float() / 32
        self.path = self.root / "tiny.safetensors"
        save_file(self.weights, str(self.path))
        self.page_bytes = (2 * block_width + 8) * 2

    def cache(self, host_pages=0):
        cache = MiniMaxH3AdalnCache(
            self.arch,
            weight_files=[str(self.path)],
            max_plans=2,
            max_plan_width=2,
            host_cache_bytes=host_pages * self.page_bytes,
            precision="fp32",
        )
        cache.load(torch.device("cpu"))
        return cache

    @staticmethod
    def embed(t):
        return torch.stack((t, t + 0.25, t * 2), dim=-1)

    def collect(self, cache, plans, metrics=None, *, warmup=False, fail=False):
        metrics = metrics if metrics is not None else RequestMetrics("request")
        with _record_adaln_cache_stats(
            SimpleNamespace(adaln_cache=cache),
            SimpleNamespace(metrics=metrics, is_warmup=warmup),
        ):
            cache.build(plans, embed=self.embed)
            if fail:
                raise RuntimeError("after prepare")
        return metrics

    def check_projection(self, cache, plans):
        for plan in plans:
            slot = int(cache.lookup(plan))
            torch.testing.assert_close(cache.plan_timesteps[slot, : len(plan)], plan)
            self.assertEqual(int(cache.plan_lengths[slot]), len(plan))
            x = self.embed(plan)
            for layer in range(2):
                prefix = f"blocks.{layer}.adaln_proj.linear"
                # Independent projection oracle; no cache indexing in the expected math.
                expected = (
                    x @ self.weights[prefix + ".weight"].T
                    + self.weights[prefix + ".bias"]
                ).bfloat16()
                torch.testing.assert_close(
                    cache.block_params[slot, : len(plan), layer],
                    expected,
                    rtol=0,
                    atol=0,
                )
            prefix = "final_layer.adaln_proj.linear"
            expected = (
                x @ self.weights[prefix + ".weight"].T + self.weights[prefix + ".bias"]
            ).bfloat16()
            torch.testing.assert_close(
                cache.final_params[slot, : len(plan)], expected, rtol=0, atol=0
            )

    def counts(self, metrics, *, gpu=0, host=0, built=0, evicted=0, skipped=0):
        self.assertEqual(
            metrics.cache_stats[NAME]["request"],
            dict(
                gpu_hit_plans=gpu,
                host_hit_plans=host,
                built_plans=built,
                host_evicted_groups=evicted,
                host_pressure_skips=skipped,
            ),
        )

    def test_cold_duplicate_device_hit_and_snapshot_isolation(self):
        cache = self.cache()
        plans = [torch.tensor([0.0]), torch.tensor([0.5, 0.75]), torch.tensor([0.0])]
        first = self.collect(cache, plans)
        self.counts(first, built=2)
        self.check_projection(cache, plans)
        saved = first.to_dict()
        second = self.collect(cache, plans)
        self.counts(second, gpu=2)
        self.assertEqual(second.cache_stats[NAME]["cumulative"]["built_plans"], 2)
        self.assertEqual(first.to_dict(), saved)
        saved["cache_stats"][NAME]["request"]["built_plans"] = -1
        self.counts(first, built=2)
        snap = cache.stats.snapshot()
        snap["built_plans"] = -1
        self.assertEqual(cache.stats.built_plans, 2)
        self.collect(cache, plans, first)
        self.counts(first, gpu=2, built=2)
        self.counts(second, gpu=2)

    def test_host_restore_and_capacity_transitions(self):
        a = [torch.tensor([0.0]), torch.tensor([0.25])]
        b = [torch.tensor([0.5]), torch.tensor([0.75])]
        cache = self.cache(host_pages=4)
        self.collect(cache, a)
        original = cache.block_params.clone()
        self.collect(cache, b)
        restored = self.collect(cache, a)
        self.counts(restored, host=2)
        self.check_projection(cache, a)
        torch.testing.assert_close(cache.block_params, original, rtol=0, atol=0)
        self.assertEqual(restored.cache_stats[NAME]["cumulative"]["built_plans"], 4)
        cache = self.cache(host_pages=2)
        self.collect(cache, a)
        self.counts(self.collect(cache, b), built=2, evicted=1)
        self.assertFalse(
            cache._host_tier.has_all([cache_module._plan_key(p) for p in a])
        )
        self.assertTrue(
            cache._host_tier.has_all([cache_module._plan_key(p) for p in b])
        )
        wide = [torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])]
        self.counts(self.collect(cache, wide), built=2, skipped=1)
        self.check_projection(cache, wide)
        self.assertEqual(len(cache._host_tier._groups), 1)
        self.assertTrue(
            cache._host_tier.has_all([cache_module._plan_key(p) for p in b])
        )

    def test_warmup_failure_retry_and_invalidation(self):
        cache = self.cache()
        a, b = [torch.tensor([0.0])], [torch.tensor([0.5])]
        warmup = self.collect(cache, a, warmup=True)
        self.assertEqual(warmup.cache_stats, {})
        self.counts(self.collect(cache, a), gpu=1)
        failed = RequestMetrics("failed")
        with self.assertRaisesRegex(RuntimeError, "after prepare"):
            self.collect(cache, b, failed, fail=True)
        self.counts(failed, built=1)
        retried = self.collect(cache, b)
        self.counts(retried, gpu=1)
        self.assertEqual(retried.cache_stats[NAME]["cumulative"]["built_plans"], 2)
        cache.invalidate()
        rebuilt = self.collect(cache, a)
        self.counts(rebuilt, built=1)
        self.assertEqual(rebuilt.cache_stats[NAME]["cumulative"]["built_plans"], 3)
        # A real truncated checkpoint fails after slots have been reserved.
        broken = dict(self.weights)
        del broken["final_layer.adaln_proj.linear.bias"]
        save_file(broken, str(self.path))
        failed = RequestMetrics("broken")
        with self.assertRaises(KeyError):
            self.collect(cache, b, failed)
        self.counts(failed)
        self.assertEqual(len(cache._slots) + len(cache._free_slots), cache.max_plans)
        self.assertTrue(all(int(cache.plan_lengths[s]) == 0 for s in cache._free_slots))
        save_file(self.weights, str(self.path))
        self.counts(self.collect(cache, b), built=1)
        self.check_projection(cache, b)

    def test_suppression_disabled_and_sidecar(self):
        cache = self.cache()
        plans = [torch.tensor([0.0]), torch.tensor([0.5])]
        for mode in ("warmup", "suppressed", "no_metrics"):
            with self.subTest(mode=mode):
                cache.invalidate()
                metrics = RequestMetrics(mode)
                metrics.suppress_stage_breakdown = mode == "suppressed"
                batch = SimpleNamespace(
                    metrics=None if mode == "no_metrics" else metrics,
                    is_warmup=mode == "warmup",
                )
                with _record_adaln_cache_stats(
                    SimpleNamespace(adaln_cache=cache), batch
                ):
                    cache.build(plans, embed=self.embed)
                self.check_projection(cache, plans)
                self.assertEqual(metrics.cache_stats, {})
        path = self.root / "sidecar.safetensors"
        save_file(
            {
                name: getattr(cache, name)
                for name in (
                    "plan_timesteps",
                    "plan_lengths",
                    "block_params",
                    "final_params",
                )
            },
            str(path),
            metadata={"format_version": "2"},
        )
        sidecar = MiniMaxH3AdalnCache(self.arch, path=str(path))
        sidecar.load(torch.device("cpu"))
        for selected in (None, sidecar):
            metrics = RequestMetrics("non-online")
            with _record_adaln_cache_stats(
                SimpleNamespace(adaln_cache=selected),
                SimpleNamespace(metrics=metrics, is_warmup=False),
            ):
                if selected is not None:
                    self.assertEqual(selected.resolve_slots(plans).numel(), len(plans))
                    self.check_projection(selected, plans)
            self.assertEqual(metrics.cache_stats, {})
        self.assertEqual(RequestMetrics("other_model").to_dict()["cache_stats"], {})

    def test_real_loop_counts_prepare_before_first_step(self):
        cache = self.cache()
        packed = minimax_h3_packed_sequence(
            text_len=1,
            latent_t=1,
            latent_h=4,
            latent_w=4,
            audio_t=1,
            include_keyframe_cond=False,
        )
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=torch.zeros(1, 4),
            token_tags=packed["token_tags"],
            device=torch.device("cpu"),
        )
        prepared = []

        def prepare(plans):
            prepared.extend(plans)
            cache.build(plans, embed=self.embed)
            return cache.resolve_slots(plans)

        def forward(model, kwargs, step):
            self.assertEqual(cache.stats.built_plans, 2)
            self.check_projection(cache, prepared)
            return torch.zeros(int(branch.update_mask.sum()), 96), torch.zeros(
                branch.audio_pos.numel(), 32
            )

        metrics = RequestMetrics("loop")
        model = SimpleNamespace(adaln_cache=cache, prepare_adaln_plans=prepare)
        # Exercise the real stage entry point, so moving collection inside the
        # step profiler (and losing prepare) breaks this test.
        model = torch.nn.Module()
        model.adaln_cache = cache
        model.prepare_adaln_plans = prepare
        model._resolve_attention_backend_once = lambda: None
        model._resolved_attention_backend = None
        stage = MiniMaxH3DenoisingStage.__new__(MiniMaxH3DenoisingStage)
        stage.transformer = model
        stage._component_residency_manager = None
        stage._maybe_enable_cache_dit_and_torch_compile = lambda *args: None
        stage._finish_active_component_use = lambda: None
        stage._forward_dit = lambda model, kwargs, step, **unused: forward(
            model, kwargs, step
        )
        stage.step_profile = lambda: None
        stage.progress_bar = lambda **kwargs: nullcontext(
            SimpleNamespace(update=lambda: None)
        )
        stage._profile_denoising_step = lambda *args, **kwargs: nullcontext()
        batch = SimpleNamespace(
            metrics=metrics,
            is_warmup=False,
            sampling_params=SimpleNamespace(),
            extra={
                "minimax_h3_text_embeddings": {
                    "positive": {
                        "text_len": 1,
                        "hidden_states": torch.zeros(1, 4),
                        "text_token_tags": packed["token_tags"][
                            packed["text_pos"].view(-1)
                        ],
                    }
                },
                "minimax_h3_denoise_state": {
                    "latent_t": 1,
                    "latent_h": 4,
                    "latent_w": 4,
                    "audio_t": 1,
                    "initial_video_rows": torch.zeros(branch.img_pos.numel(), 96),
                    "initial_audio_rows": torch.zeros(branch.audio_pos.numel(), 32),
                },
                "minimax_h3_sigmas": {
                    "video": [1.0, 0.5, 0.0],
                    "audio": [1.0, 0.5, 0.0],
                },
            },
        )
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(uses_subblock_attention=lambda _: False),
            attention_backend="torch_sdpa",
            component_attention_backends={},
        )
        with patch.object(
            MiniMaxH3DenoisingStage,
            "current_use_nvtx",
            new_callable=PropertyMock,
            return_value=False,
        ):
            stage._run_full_loop(batch, server_args)
        self.assertEqual(batch.latents.ndim, 5)
        self.assertTrue(torch.isfinite(batch.latents).all())
        self.counts(metrics, built=2)

    def test_transport_logs_writer_and_old_records(self):
        metrics = self.collect(self.cache(), [torch.tensor([0.0])])
        expected = metrics.to_dict()["cache_stats"]
        detached = RequestMetrics("detached")
        request = dict(expected[NAME]["request"])
        cumulative = dict(expected[NAME]["cumulative"])
        detached.record_cache_stats(NAME, request, cumulative)
        request["built_plans"] = cumulative["built_plans"] = 99
        self.assertEqual(detached.cache_stats, expected)
        detached.suppress_stage_breakdown = True
        detached.record_cache_stats(NAME, request, cumulative)
        self.assertEqual(detached.cache_stats, expected)
        wire = json.loads(json.dumps(metrics.to_dict()))
        restored = _deserialize_request_metrics(wire)
        wire["cache_stats"][NAME]["request"]["built_plans"] = 99
        self.assertEqual(restored.cache_stats, expected)
        self.assertEqual(
            _deserialize_request_metrics({"request_id": "old"}).cache_stats, {}
        )
        old = dict(
            request_id="old",
            commit_hash="test",
            tag="test",
            stages=[],
            steps=[],
            total_duration_ms=0,
        )
        self.assertEqual(RequestPerfRecord(**old).cache_stats, {})
        record = RequestPerfRecord(**old, cache_stats=expected)
        expected[NAME]["request"]["built_plans"] = 77
        self.assertEqual(record.cache_stats, metrics.cache_stats)
        self.assertEqual(
            RequestPerfRecord(
                **json.loads(json.dumps(dataclasses.asdict(record)))
            ).cache_stats,
            metrics.cache_stats,
        )
        with (
            patch.object(perf_logger, "get_git_commit_hash", return_value="test"),
            patch.object(perf_logger, "get_is_main_process", return_value=True),
            patch.object(
                perf_logger, "get_diffusion_perf_log_dir", return_value=str(self.root)
            ),
        ):
            PerformanceLogger.log_request_summary(metrics)
            log = json.loads((self.root / "performance.log").read_text())
            self.assertEqual(log["cache_stats"], metrics.cache_stats)
            for output_rank in (False, True):
                worker = GPUWorker.__new__(GPUWorker)
                worker.is_output_rank = output_rank
                worker.server_args = SimpleNamespace(model_path="tiny")
                path = self.root / f"rank-{output_rank}.json"
                other = RequestMetrics("other")
                other.record_cache_stats(NAME, {"built_plans": 7}, {"built_plans": 9})
                grouped = OutputBatch()
                GPUWorker._finalize_expanded_parts(
                    grouped,
                    _ExpandedOutputParts(metrics_list=[metrics, other]),
                    audio_sample_rate=None,
                )
                transported = [
                    _deserialize_request_metrics(json.loads(json.dumps(item.to_dict())))
                    for item in grouped.metrics_list
                ]
                self.assertEqual(
                    [item.cache_stats for item in transported],
                    [metrics.cache_stats, other.cache_stats],
                )
                self.assertIs(grouped.metrics, metrics)
                worker._dump_perf_report(
                    SimpleNamespace(perf_dump_path=str(path), is_warmup=False), grouped
                )
                self.assertEqual(path.exists(), output_rank)
                if output_rank:
                    self.assertEqual(
                        json.loads(path.read_text())["cache_stats"], metrics.cache_stats
                    )
                    self.assertEqual(
                        other.cache_stats[NAME]["request"]["built_plans"], 7
                    )


if __name__ == "__main__":
    unittest.main()
