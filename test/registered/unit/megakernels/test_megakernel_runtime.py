"""CPU contracts of the megakernel registry, install, dispatch and capture check."""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from msgspec.structs import replace

from sglang.srt.megakernels import runtime
from sglang.srt.megakernels import spec as registry
from sglang.srt.megakernels.guards import deadline_s
from sglang.srt.megakernels.spec import (
    MegakernelSpec,
    Numerics,
    NumericsKind,
    Requirements,
    Step,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    default_cuda_graph_config,
    with_phase,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

ARCH = "FakeMegaForCausalLM"
ISA = "sm_103"


class FakeBuilt:
    """Covers steps of 1-4 rows and records the steps it ran."""

    draft_sampler = None

    def __init__(self, widths):
        self.widths = tuple(w for w in widths if w <= 4)
        self.steps = []

    def summary(self):
        return "fake"

    def plan(self, input_ids, forward_batch, *args, **kwargs):
        rows = int(input_ids.shape[0])
        return Step(mode="decode", rows=rows) if rows in self.widths else None

    def forward(self, step, original, input_ids, positions, forward_batch, **kw):
        self.steps.append(step.rows)
        return "megakernel"


def _build(model, hf_config, context):
    model.context = context
    return FakeBuilt(context.widths)


SPEC = MegakernelSpec(
    name="fake",
    requires=Requirements(isas=frozenset({ISA}), tp_sizes=frozenset({4})),
    features=frozenset(),
    numerics=Numerics(kind=NumericsKind.BITWISE),
    max_rows=8,
    build=_build,
    api_version=registry.MEGAKERNEL_API_VERSION,
)
OLD_SPEC = replace(SPEC, api_version=registry.MEGAKERNEL_API_VERSION - 1)
# a draft spec: builds no width, offers only the draft sampler
DRAFT_SPEC = replace(
    SPEC,
    max_rows=0,
    build=lambda model, hf, ctx: SimpleNamespace(
        widths=(), summary=lambda: "draft sampler", draft_sampler=print
    ),
)


class Lm(torch.nn.Module):
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return "standard"


def batch(rows):
    return SimpleNamespace(batch_size=rows, input_ids=torch.zeros(rows))


def server_args(**kw):
    defaults = {
        "model_path": "dummy",
        "enable_nvidia_megakernel": True,
        "tp_size": 4,
        "speculative_algorithm": "DSPARK",
        "speculative_num_draft_tokens": 6,
        "cuda_graph_config": default_cuda_graph_config(),
    }
    return ServerArgs(**{**defaults, **kw})


def install(
    *, archs=(ARCH,), gathered=None, tokens=2, graphs=True, hooks=None, lm=None
):
    """install_megakernel on an Lm as rank 0 of the published server; gathered maps
    a gathered value to what the other ranks report, default the same as this rank."""
    lm, hooks = lm or Lm(), [] if hooks is None else hooks
    gather = gathered or (lambda x, group=None: [x])
    gpu = SimpleNamespace(multi_processor_count=160)
    with (
        patch.object(runtime, "device_isa", return_value=ISA),
        patch.object(runtime, "deadline_s", return_value=30.0),
        patch.object(runtime, "_gather_tp", side_effect=gather),
        patch.object(runtime, "_tokens_per_request", return_value=tokens),
        patch.object(runtime, "_decode_graphs_enabled", return_value=graphs),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.cuda, "get_device_properties", return_value=gpu),
        patch.object(torch.cuda, "mem_get_info", return_value=(0, 0)),
    ):
        runtime.install_megakernel(
            model=lm,
            hf_config=SimpleNamespace(architectures=list(archs)),
            capture_tail_hooks=hooks,
        )
    return lm


class TestMegakernelRuntime(CustomTestCase):
    def setUp(self):
        registry_patch = patch.dict(registry._REGISTRY, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        registry.register(architecture=ARCH, path=f"{__name__}:SPEC")
        publish(server_args(), role="test")
        self.addCleanup(reset_context)

    def _install_spec(self, spec, **kw):
        with patch.object(runtime, "find_spec", return_value=(spec, "")):
            return install(**kw)

    def test_the_runtime_reads_the_published_config(self):
        self.assertTrue(runtime.megakernel_enabled())
        self.assertEqual(runtime._tokens_per_request(), 6)
        self.assertTrue(runtime._decode_graphs_enabled())
        reset_context()
        eager = with_phase(
            default_cuda_graph_config(), Phase.DECODE, backend=Backend.DISABLED
        )
        publish(server_args(watchdog_timeout=40, cuda_graph_config=eager), role="test")
        self.assertFalse(runtime._decode_graphs_enabled())
        self.assertEqual(deadline_s(runtime.get_device().watchdog_timeout), 20.0)
        reset_context()
        publish(server_args(enable_nvidia_megakernel=False), role="test")
        self.assertFalse(runtime.megakernel_enabled())

    def test_a_spec_found_at_load_is_checked_on_what_the_runner_reads(self):
        # a draft's spec is never checked at resolution
        strict = replace(
            SPEC,
            requires=replace(
                SPEC.requires,
                tp_sizes=frozenset({8}),
                decode_graph_backends=frozenset({"breakable"}),
                hf_config={"model_type": "fake"},
            ),
        )
        with self.assertRaises(ValueError) as refused:
            self._install_spec(strict)
        for want in ("tp_size 4", "decode CUDA graphs 'full'", "config.model_type"):
            self.assertIn(want, str(refused.exception))

    def test_each_load_refusal_is_raised(self):
        def other_spec_on_rank_1(value, group=None):
            is_lookup = isinstance(value, tuple) and isinstance(value[-1], tuple)
            other = (value[0], "", ("other", 1)) if is_lookup else value
            return [value, other]

        with self.assertRaisesRegex(ValueError, "load different specs"):
            install(gathered=other_spec_on_rank_1)
        partitioned = replace(
            SPEC, requires=replace(SPEC.requires, resident_blocks=200)
        )
        with self.assertRaisesRegex(ValueError, "160 SMs"):
            self._install_spec(partitioned)
        with self.assertRaisesRegex(ValueError, "already replaced"):
            install(lm=install(tokens=1))
        # a target that covers no width must not quietly keep the standard forward
        empty = replace(SPEC, build=lambda model, hf, ctx: FakeBuilt(()))
        with self.assertRaisesRegex(ValueError, "built no width of"):
            self._install_spec(empty)

    def test_find_spec_refuses_with_a_reason(self):
        registry.register(architecture="Broken", path="no_such_module:SPEC")
        registry.register(architecture="Old", path=f"{__name__}:OLD_SPEC")
        self.assertIs(registry.find_spec(architectures=[ARCH], isa=ISA)[0], SPEC)
        version = registry.MEGAKERNEL_API_VERSION
        for archs, isa, why in (
            ([ARCH], "sm_90", "supports ['sm_103'], not sm_90"),
            (["Other"], ISA, "no megakernel spec is registered for ['Other']"),
            (["Broken"], ISA, "does not import: ModuleNotFoundError"),
            (["Old"], ISA, f"{version - 1}; this SGLang requires version {version}"),
        ):
            with self.subTest(archs=archs, isa=isa):
                spec, reason = registry.find_spec(architectures=archs, isa=isa)
                self.assertIsNone(spec)
                self.assertIn(why, reason)

    def test_an_architecture_has_one_spec(self):
        # Repeated registration must not silently replace another plugin's spec.
        registry.register(architecture=ARCH, path=f"{__name__}:SPEC")
        with self.assertRaises(ValueError):
            registry.register(architecture=ARCH, path="other:SPEC")

    def test_without_the_flag_nothing_is_read(self):
        model = SimpleNamespace()  # any attribute read would raise
        with patch.object(runtime, "megakernel_enabled", return_value=False):
            runtime.install_megakernel(
                model=model, hf_config=None, capture_tail_hooks=[]
            )
        self.assertEqual(vars(model), {})

    def test_a_model_without_a_spec_on_every_rank_keeps_its_forward(self):
        lm = install(archs=["DraftWithoutSpec"])
        self.assertNotIn("forward", lm.__dict__)
        self.assertIsNone(runtime.built_megakernel(lm))

    def test_every_rank_refuses_when_any_rank_cannot_serve(self):
        # rank 1 lacks the spec: rank 0 must raise too, or it waits in the build
        def no_spec_on_rank_1(value, group=None):
            return [value, (value[0], "rank-1 reason", None)]

        with self.assertRaisesRegex(ValueError, "rank 1: rank-1 reason"):
            install(gathered=no_spec_on_rank_1)

        def model_reason_on_rank_1(value, group=None):
            return [value, ["no mHC pre-mix"]] if isinstance(value, list) else [value]

        with self.assertRaisesRegex(ValueError, "rank 1: no mHC pre-mix"):
            install(gathered=model_reason_on_rank_1)

    def test_ranks_that_built_different_widths_are_refused(self):
        def fewer_on_rank_1(value, group=None):
            is_widths = isinstance(value, tuple) and all(
                isinstance(w, int) for w in value
            )
            return [value, value[:1]] if is_widths else [value]

        with self.assertRaisesRegex(RuntimeError, "widths fake built differ"):
            install(gathered=fewer_on_rank_1)

    def test_widths_are_whole_requests_up_to_max_rows(self):
        self.assertEqual(install(tokens=2).context.widths, (2, 4, 6, 8))
        self.assertEqual(install(tokens=3).context.widths, (3, 6))

    def test_covered_steps_run_the_megakernel_and_others_the_forward(self):
        lm = install(tokens=1)
        built = runtime.built_megakernel(lm)
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            self.assertEqual(lm(batch(4).input_ids, None, batch(4)), "megakernel")
            self.assertEqual(lm(batch(6).input_ids, None, batch(6)), "standard")
        self.assertEqual(built.steps, [4])
        # graph runners and supports_pp read the original forward's parameters
        self.assertEqual(inspect.signature(lm.forward), inspect.signature(Lm().forward))
        self.assertIn("--enable-nvidia-megakernel", lm._derived_weight_cache_error)

    def test_a_draft_spec_installs_no_forward_and_offers_its_sampler(self):
        registry.register(architecture="Draft", path=f"{__name__}:DRAFT_SPEC")
        hooks = []
        lm = install(archs=["Draft"], hooks=hooks)
        self.assertNotIn("forward", lm.__dict__)
        self.assertEqual(hooks, [])
        self.assertIs(runtime.megakernel_draft_sampler(lm), print)
        self.assertIsNone(runtime.megakernel_draft_sampler(Lm()))
        nothing = replace(
            DRAFT_SPEC,
            build=lambda model, hf, ctx: SimpleNamespace(
                widths=(), summary=lambda: "", draft_sampler=None
            ),
        )
        with self.assertLogs(runtime.logger, "INFO") as logs:
            self._install_spec(nothing)
        self.assertIn("no step and no draft sampler", "".join(logs.output))

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
        hooks = []
        install(graphs=False, hooks=hooks)
        self.assertEqual(hooks, [])  # eager steps are routed by plan alone


if __name__ == "__main__":
    unittest.main()
