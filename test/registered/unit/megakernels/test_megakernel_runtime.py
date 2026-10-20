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
from sglang.srt.megakernels.binding import MegakernelBinding
from sglang.srt.runtime_context import publish, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fakes import (  # noqa: E402
    ANY_TP,
    ARCH,
    ISA,
    SPEC,
    FakeBuilt,
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

    def test_optional_preparation_runs_after_the_model_hook_and_repeats(self):
        events = []
        prepared_view = object()

        # Use the ordinary fake object, adding only the optional callback under test.
        def build(model, hf_config, context):
            model.context = context
            built = FakeBuilt(context.widths)

            def prepare(view):
                events.append(("plugin", view))

            built.prepare_before_cuda_graph_capture = prepare
            return built

        model = Lm()
        original = Mock(side_effect=lambda runner: events.append(("model", runner)))
        model.__dict__["prepare_before_cuda_graph_capture"] = original
        spec = replace(SPEC, build=build)
        with patch.object(
            MegakernelBinding, "preparation_view", return_value=prepared_view
        ) as make_view:
            lm = self._install_spec(spec, model=model)
            runner = object()
            lm.prepare_before_cuda_graph_capture(runner)
            lm.prepare_before_cuda_graph_capture(runner)
        self.assertEqual(
            events,
            [
                ("model", runner),
                ("plugin", prepared_view),
                ("model", runner),
                ("plugin", prepared_view),
            ],
        )
        self.assertEqual(original.call_args_list, [call(runner), call(runner)])
        self.assertEqual(make_view.call_args_list, [call(runner), call(runner)])

    def test_absent_preparation_callback_leaves_the_model_hook_untouched(self):
        model = Lm()
        original = Mock()
        model.__dict__["prepare_before_cuda_graph_capture"] = original
        lm = install(model=model)
        self.assertIs(lm.__dict__["prepare_before_cuda_graph_capture"], original)

        hookless = install()
        self.assertNotIn("prepare_before_cuda_graph_capture", hookless.__dict__)

    def test_preparation_failures_propagate_in_order(self):
        def prepared_spec(plugin):
            def build(model, hf_config, context):
                model.context = context
                built = FakeBuilt(context.widths)
                built.prepare_before_cuda_graph_capture = plugin
                return built

            return replace(SPEC, build=build)

        model = Lm()
        model.__dict__["prepare_before_cuda_graph_capture"] = Mock(
            side_effect=RuntimeError("model failed")
        )
        plugin = Mock()
        lm = self._install_spec(prepared_spec(plugin), model=model)
        with (
            patch.object(MegakernelBinding, "preparation_view") as make_view,
            self.assertRaisesRegex(RuntimeError, "model failed"),
        ):
            lm.prepare_before_cuda_graph_capture(object())
        make_view.assert_not_called()
        plugin.assert_not_called()

        model = Lm()
        plugin = Mock()
        with patch.object(
            MegakernelBinding,
            "preparation_view",
            side_effect=RuntimeError("view failed"),
        ):
            lm = self._install_spec(prepared_spec(plugin), model=model)
            with self.assertRaisesRegex(RuntimeError, "view failed"):
                lm.prepare_before_cuda_graph_capture(object())
        plugin.assert_not_called()

        model = Lm()
        plugin = Mock(side_effect=RuntimeError("plugin failed"))
        with patch.object(MegakernelBinding, "preparation_view", return_value=object()):
            lm = self._install_spec(prepared_spec(plugin), model=model)
            with self.assertRaisesRegex(RuntimeError, "plugin failed"):
                lm.prepare_before_cuda_graph_capture(object())

    def test_zero_width_build_still_gets_the_preparation_bridge(self):
        plugin = Mock()
        zero = SimpleNamespace(
            widths=(),
            draft_sampler=print,
            summary=lambda: "draft sampler",
            prepare_before_cuda_graph_capture=plugin,
        )
        spec = replace(SPEC, max_rows=0, build=lambda model, hf, ctx: zero)
        with patch.object(
            MegakernelBinding, "preparation_view", return_value="prepared"
        ):
            lm = self._install_spec(spec, is_draft=True)
            lm.prepare_before_cuda_graph_capture("runner")
        plugin.assert_called_once_with("prepared")

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
