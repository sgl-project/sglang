"""CPU contracts of --enable-nvidia-megakernel's startup checks and wait deadlines."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from msgspec.structs import replace

from sglang.srt.environ import envs
from sglang.srt.megakernels import guards
from sglang.srt.megakernels import spec as registry
from sglang.srt.megakernels.guards import (
    MAX_DEADLINE_S,
    deadline_s,
    features_in_use,
    handle_nvidia_megakernel,
    megakernel_disable_reasons,
)
from sglang.srt.megakernels.spec import (
    MEGAKERNEL_API_VERSION,
    Feature,
    MegakernelSpec,
    Numerics,
    NumericsKind,
    Requirements,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    default_cuda_graph_config,
    with_phase,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, stage="base-a", runner_config="cpu")

ARCH = "FakeMegaForCausalLM"

SPEC = MegakernelSpec(
    name="fake",
    requires=Requirements(
        isas=frozenset({"sm_103"}),
        tp_sizes=frozenset({4}),
        dtypes=frozenset({"bfloat16"}),
        server_args={"kv_cache_dtype": frozenset({"fp8_e4m3"})},
        hf_config={"model_type": "fake"},
        env={"SGLANG_DSPARK_EMBED_IN_GRAPH": True},
    ),
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    numerics=Numerics(kind=NumericsKind.BITWISE),
    max_rows=48,
    build=lambda model, hf_config, ctx: None,
    api_version=MEGAKERNEL_API_VERSION,
    config_disable_reasons=lambda cfg, hf: (
        ["context_length too long"] if cfg.context_length == 7 else []
    ),
)


def server_args(**kw):
    """Arguments as the resolution step sees them, with the graph config resolved."""
    defaults = {
        "model_path": "dummy",
        "tp_size": 4,
        "kv_cache_dtype": "fp8_e4m3",
        "cuda_graph_config": default_cuda_graph_config(),
    }
    return ServerArgs(**{**defaults, **kw})


def entry_point(name):
    ep = MagicMock()
    ep.name = name
    return ep


def reasons_for(cfg, *, arch=ARCH, isa="sm_103", model_type="fake"):
    hf_config = SimpleNamespace(architectures=[arch], model_type=model_type)
    return megakernel_disable_reasons(cfg=cfg, hf_config=hf_config, isa=isa)


class TestMegakernelGuards(CustomTestCase):
    def setUp(self):
        registry_patch = patch.dict(registry._REGISTRY, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        registry.register(architecture=ARCH, path=f"{__name__}:SPEC")

    def test_every_feature_has_a_probe_and_none_is_on_by_default(self):
        # a feature without a probe is never refused; a probe stuck on refuses every server
        post_load = {Feature.PARTITIONED_GPU}  # the runtime checks it on the device
        self.assertEqual(set(guards.FEATURE_PROBES), set(Feature) - post_load)
        self.assertEqual(features_in_use(ServerArgs(model_path="dummy")), set())

    def test_features_in_use_are_refused_unless_accepted(self):
        _, reasons = reasons_for(server_args(pp_size=2, speculative_algorithm="DSPARK"))
        self.assertEqual(len(reasons), 1, reasons)  # SPEC accepts speculative decoding
        self.assertIn(Feature.PIPELINE_PARALLEL.value, reasons[0])

    def test_every_refusal_reason_is_listed(self):
        with envs.SGLANG_DSPARK_EMBED_IN_GRAPH.override(False):
            spec, reasons = reasons_for(
                server_args(
                    tp_size=8,
                    kv_cache_dtype="auto",
                    enable_eplb=True,
                    context_length=7,
                    watchdog_timeout=6,
                ),
                model_type="other",
            )
        self.assertIs(spec, SPEC)
        for want in (
            "tp_size 8",
            "kv_cache_dtype 'auto'",
            "config.model_type 'other'",
            "SGLANG_DSPARK_EMBED_IN_GRAPH False",
            Feature.EPLB.value,
            "context_length",
            "--watchdog-timeout",
        ):
            self.assertTrue(any(want in r for r in reasons), (want, reasons))
        self.assertEqual(reasons_for(server_args())[1], [])
        self.assertEqual(reasons_for(server_args(), isa=None)[1], ["no GPU is visible"])
        for arch, isa, why in (
            ("Other", "sm_103", "Other"),
            (ARCH, "sm_100", "sm_100"),
        ):
            spec, reasons = reasons_for(server_args(), arch=arch, isa=isa)
            self.assertIsNone(spec)
            self.assertIn(why, reasons[0])

    def test_a_plugin_that_registered_nothing_is_named(self):
        """A plugin that fails to load is only logged, so an empty registry must not
        read as an unsupported model."""
        registry._REGISTRY.clear()
        spec, reasons = reasons_for(server_args())
        self.assertIsNone(spec)
        self.assertIn("installed but registered no spec", reasons[0])

    def test_the_decode_graph_backend_is_checked(self):
        full_only = replace(
            SPEC, requires=replace(SPEC.requires, decode_graph_backends={"full"})
        )
        graphs = with_phase(
            default_cuda_graph_config(), Phase.DECODE, backend=Backend.BREAKABLE
        )
        with patch.object(guards, "find_spec", return_value=(full_only, "")):
            _, reasons = reasons_for(server_args(cuda_graph_config=graphs))
        self.assertTrue(any("decode CUDA graphs" in r for r in reasons), reasons)

    def test_the_flag_needs_its_plugin_to_load(self):
        """Without its plugin, the flag must fail at startup, not serve the standard path."""
        cases = (
            ([], "", "install sgl-nvidia-mega"),
            ([entry_point("nvidia_megakernel")], "other", "SGLANG_PLUGINS='other'"),
        )
        for eps, allowed, want in cases:
            with (
                self.subTest(want=want),
                patch.object(guards, "entry_points", return_value=eps),
                patch.object(guards.envs.SGLANG_PLUGINS, "get", return_value=allowed),
            ):
                self.assertTrue(any(want in r for r in guards.plugin_reasons()))
        eps = [entry_point("nvidia_megakernel")]
        for allowed in ("", "other, nvidia_megakernel"):
            with (
                patch.object(guards, "entry_points", return_value=eps),
                patch.object(guards.envs.SGLANG_PLUGINS, "get", return_value=allowed),
            ):
                self.assertEqual(guards.plugin_reasons(), [])

    def test_the_resolution_step_raises_every_reason_or_nothing(self):
        with patch.object(guards, "plugin_reasons") as plugin:
            handle_nvidia_megakernel(server_args())  # flag off: nothing is looked up
            plugin.assert_not_called()
        on = server_args(enable_nvidia_megakernel=True, tp_size=8)
        with (
            patch.object(guards, "plugin_reasons", return_value=[]),
            patch.object(guards, "device_isa", return_value="sm_103"),
            patch.object(guards, "megakernel_disable_reasons") as reasons,
            patch.object(guards, "model_config_of") as model_config,
        ):
            model_config.return_value = SimpleNamespace(hf_config=None, dtype="float16")
            reasons.return_value = (SPEC, ["tp_size 8 is not one of ['4']"])
            # the resolved dtype is checked here, and listed with the other reasons
            with self.assertRaisesRegex(
                ValueError, "(?s)tp_size 8.*\n  - dtype 'float16'"
            ):
                handle_nvidia_megakernel(on)
            model_config.return_value = SimpleNamespace(
                hf_config=None, dtype="bfloat16"
            )
            reasons.return_value = (SPEC, [])
            handle_nvidia_megakernel(on)
            # both flags replace the model's forward: refused before any weight loads,
            # listed with the spec's own reasons
            both = server_args(enable_nvidia_megakernel=True, monokernel_backend="x")
            reasons.return_value = (SPEC, ["tp_size 8 is not one of ['4']"])
            with self.assertRaisesRegex(
                ValueError, "(?s)tp_size 8.*\n  - --monokernel-backend x is also set"
            ):
                handle_nvidia_megakernel(both)

    def test_deadlines_stay_below_the_watchdog(self):
        self.assertEqual(deadline_s(None), MAX_DEADLINE_S)
        self.assertEqual(deadline_s(3600), MAX_DEADLINE_S)
        self.assertEqual(deadline_s(30), 15.0)


if __name__ == "__main__":
    unittest.main()
