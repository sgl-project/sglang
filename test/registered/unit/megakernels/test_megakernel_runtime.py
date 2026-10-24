"""CPU checks of the megakernel install, dispatch and capture check."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
from msgspec.structs import replace

from sglang.srt.megakernels import runtime
from sglang.srt.megakernels import spec as registry
from sglang.srt.megakernels.binding import MegakernelBinding, PreparationView
from sglang.srt.runtime_context import publish, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fakes import (  # noqa: E402
    ANY_TP,
    ARCH,
    ISA,
    SPEC,
    Lm,
    batch,
    server_args,
)

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def install(
    *,
    archs=(ARCH,),
    gathered=None,
    tokens=2,
    hooks=None,
    is_draft=False,
    model=None,
):
    """install_megakernel on an Lm as rank 0 of the published server; gathered maps
    a gathered value to what the other ranks report, default the same as this rank."""
    lm, hooks = (Lm() if model is None else model), [] if hooks is None else hooks
    gpu = SimpleNamespace(multi_processor_count=160)
    with (
        patch.object(runtime, "device_isa", return_value=ISA),
        patch.object(runtime, "deadline_s", return_value=30.0),
        patch.object(
            runtime, "_gather_tp", side_effect=gathered or (lambda x, group=None: [x])
        ),
        patch.object(runtime, "_tokens_per_request", return_value=tokens),
        patch.object(runtime, "_decode_graphs_enabled", return_value=True),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.cuda, "get_device_properties", return_value=gpu),
        patch.object(torch.cuda, "mem_get_info", return_value=(0, 0)),
    ):
        runtime.install_megakernel(
            model=lm,
            hf_config=SimpleNamespace(architectures=list(archs)),
            capture_tail_hooks=hooks,
            is_draft=is_draft,
        )
    return lm


class TestMegakernelRuntime(CustomTestCase):
    def setUp(self):
        registry_patch = patch.dict(registry._REGISTRY, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        registry.register(architecture=ARCH, path="_fakes:SPEC")
        publish(server_args(enable_nvidia_megakernel=True), role="test")
        self.addCleanup(reset_context)
        # the running forward's view reads its attention backend, which tests lack
        view_patch = patch.object(
            MegakernelBinding, "forward_view", return_value="view"
        )
        view_patch.start()
        self.addCleanup(view_patch.stop)
        capture_view_patch = patch.object(MegakernelBinding, "seal_capture_view")
        capture_view_patch.start()
        self.addCleanup(capture_view_patch.stop)

    def _install_spec(self, spec, **kw):
        with patch.object(runtime, "find_spec", return_value=(spec, "")):
            return install(**kw)

    def _warmup(self, model_runner, events):
        from sglang.srt.model_executor.runner import base_runner

        runner = SimpleNamespace(
            model_runner=model_runner,
            _pre_initialize_flashinfer_allreduce_workspace=lambda: events.append(
                "allreduce"
            ),
            _pre_initialize_fi_a2a_workspace=lambda: events.append("a2a"),
            _pre_initialize_pcie_ipc_workspace=lambda: events.append("pcie"),
            _autotune_buffers=lambda: (object(), 1),
            _flashinfer_autotune=lambda **kw: events.append("autotune"),
        )
        with (
            patch.object(
                base_runner, "should_run_flashinfer_autotune", return_value=True
            ),
            patch.object(
                base_runner,
                "maybe_flashinfer_autotune_extend",
                side_effect=lambda *args, **kw: events.append("extend"),
            ),
            patch.object(
                base_runner.envs.SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP,
                "get",
                return_value=False,
            ),
        ):
            base_runner.BaseRunner._warmup(runner)

    def test_covered_steps_run_the_megakernel_and_others_the_forward(self):
        lm = install(tokens=1)
        built = runtime.built_megakernel(lm)
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            self.assertEqual(lm(batch(4).input_ids, None, batch(4)), "megakernel")
            self.assertEqual(lm(batch(6).input_ids, None, batch(6)), "standard")
        # only a covered step runs the megakernel, with the running forward's view
        self.assertEqual((built.steps, built.views), ([4], ["view"]))
        MegakernelBinding.seal_capture_view.assert_not_called()

    def test_dispatch_passes_and_seals_the_current_capture_view(self):
        lm = install(tokens=1)
        forward_batch = batch(4)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            lm(forward_batch.input_ids, None, forward_batch)
        MegakernelBinding.forward_view.assert_called_once_with(forward_batch)
        MegakernelBinding.seal_capture_view.assert_called_once_with(
            forward_batch, "view"
        )

    def test_preparation_runs_after_model_once_per_warmup_lifecycle(self):
        """All phase runners share preparation; an explicit warmup reset repeats it."""
        events = []
        prepared_view = PreparationView()

        def prepare(built, view):
            built.prepared = view
            events.append(("plugin", built, view))

        model = Lm()
        original = Mock(side_effect=lambda runner: events.append(("model", runner)))
        model.__dict__["prepare_before_cuda_graph_capture"] = original
        lm = self._install_spec(replace(SPEC, prepare=prepare), model=model)
        built = runtime.built_megakernel(lm)
        self.assertIs(lm.prepare_before_cuda_graph_capture, original)
        mr = SimpleNamespace(model=lm, device="cuda")
        expected = [
            "allreduce",
            "a2a",
            "pcie",
            ("model", mr),
            ("plugin", built, prepared_view),
            "autotune",
            "extend",
        ]
        # Preparation must share the binding passed to build and forward dispatch.
        with patch.object(
            lm.context.binding, "preparation_view", return_value=prepared_view
        ) as make_view:
            self._warmup(mr, events)
            self.assertTrue(mr._kernel_warmed_up)
            self.assertIs(built.prepared, prepared_view)
            self.assertEqual(events, expected)
            self._warmup(mr, events)
            self.assertEqual(events, expected)
            mr._kernel_warmed_up = False
            self._warmup(mr, events)
        self.assertEqual(events, expected * 2)
        self.assertEqual(make_view.call_args_list, [call(mr), call(mr)])

    def test_absent_preparation_is_a_noop_and_leaves_model_hooks_untouched(self):
        model = Lm()
        original = Mock()
        model.__dict__["prepare_before_cuda_graph_capture"] = original
        lm = install(model=model)
        self.assertIs(lm.__dict__["prepare_before_cuda_graph_capture"], original)

        hookless = install()
        self.assertNotIn("prepare_before_cuda_graph_capture", hookless.__dict__)
        uninstalled = Lm()
        with patch.object(
            MegakernelBinding,
            "preparation_view",
            side_effect=AssertionError("unrequested preparation"),
        ):
            for model in (lm, hookless, uninstalled):
                runtime.prepare_megakernel(SimpleNamespace(model=model))
        original.assert_not_called()
        self.assertIsNone(runtime.built_megakernel(uninstalled))

    def test_preparation_failures_propagate_in_order(self):
        stages = ["model", "view", "plugin"]
        for failed_stage in stages:
            with self.subTest(failed_stage=failed_stage):
                events = []

                def record(stage):
                    events.append(stage)
                    if stage == failed_stage:
                        raise RuntimeError(f"{stage} failed")

                def make_view(runner):
                    record("view")
                    return PreparationView()

                model = Lm()
                model.__dict__["prepare_before_cuda_graph_capture"] = lambda runner: (
                    record("model")
                )
                lm = self._install_spec(
                    replace(SPEC, prepare=lambda built, view: record("plugin")),
                    model=model,
                )
                with (
                    patch.object(
                        lm.context.binding, "preparation_view", side_effect=make_view
                    ),
                    self.assertRaisesRegex(RuntimeError, f"{failed_stage} failed"),
                ):
                    self._warmup(SimpleNamespace(model=lm, device="cuda"), events)
                self.assertEqual(
                    events,
                    ["allreduce", "a2a", "pcie"]
                    + stages[: stages.index(failed_stage) + 1],
                )

    def test_zero_width_build_prepares_without_a_model_hook(self):
        """Sampler-only builds still prepare before their captured sampler tail."""
        prepared_view = PreparationView()

        def prepare(built, view):
            built.prepared = view

        zero = SimpleNamespace(
            widths=(),
            draft_sampler=print,
            summary=lambda: "draft sampler",
        )
        spec = replace(
            SPEC, max_rows=0, build=lambda model, hf, ctx: zero, prepare=prepare
        )
        with patch.object(
            MegakernelBinding, "preparation_view", return_value=prepared_view
        ):
            lm = self._install_spec(spec, is_draft=True)
            self._warmup(SimpleNamespace(model=lm, device="cuda"), [])
        self.assertIs(zero.prepared, prepared_view)
        self.assertIs(runtime.built_megakernel(lm), zero)
        self.assertIs(runtime.megakernel_draft_sampler(lm), print)
        self.assertNotIn("forward", lm.__dict__)
        self.assertNotIn("prepare_before_cuda_graph_capture", lm.__dict__)

    def test_a_target_without_a_usable_spec_is_refused_and_a_draft_keeps_its_forward(
        self,
    ):
        # the flag must not silently run the model's own forward
        with self.assertRaisesRegex(ValueError, "in the model workers"):
            install(archs=["TargetWithoutSpec"])
        tp8 = replace(SPEC, requires=replace(SPEC.requires, tp_sizes=frozenset({8})))
        with self.assertRaisesRegex(ValueError, "tp_size 4"):
            self._install_spec(tp8)
        # an empty field accepts any value
        self.assertIsNotNone(runtime.built_megakernel(self._install_spec(ANY_TP)))
        with self.assertLogs(runtime.logger, "INFO") as logs:
            lm = install(archs=["DraftWithoutSpec"], is_draft=True)
        self.assertNotIn("forward", lm.__dict__)
        self.assertIn("keeps its own forward", "".join(logs.output))
        registry.register(architecture="Draft", path="_fakes:DRAFT")
        # a draft spec offers only its draft sampler
        lm = install(archs=["Draft"], is_draft=True)
        self.assertNotIn("forward", lm.__dict__)
        self.assertIs(runtime.megakernel_draft_sampler(lm), print)

    def test_ranks_that_disagree_are_refused_on_every_rank(self):
        """A rank that differs would wait in a different collective."""

        def no_spec_on_rank_1(value, group=None):
            return [value, (value[0], "rank-1 reason", None)]

        def fewer_widths_on_rank_1(value, group=None):
            widths = isinstance(value, tuple) and all(isinstance(w, int) for w in value)
            return [value, value[:1]] if widths else [value]

        with self.assertRaisesRegex(ValueError, "rank 1: rank-1 reason"):
            install(gathered=no_spec_on_rank_1)
        with self.assertRaisesRegex(RuntimeError, "widths fake built differ"):
            install(gathered=fewer_widths_on_rank_1)

    def test_capture_check_refuses_when_no_graph_runs_the_megakernel(self):
        def capture(sizes, capture_bs):
            hooks = []
            lm = install(tokens=1, hooks=hooks)
            runner = SimpleNamespace(capture_bs=capture_bs)
            with patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ):
                for bs in sizes:
                    lm(batch(bs).input_ids, None, batch(bs))
                    hooks[0](runner, None, batch(bs), bs)

        capture([2, 8], capture_bs=[2, 8, 16])  # not all captured yet: no verdict
        capture([2, 8], capture_bs=[2, 8])
        with self.assertRaisesRegex(ValueError, "none of the 2 captured decode graphs"):
            capture([8, 16], capture_bs=[8, 16])


if __name__ == "__main__":
    unittest.main()
