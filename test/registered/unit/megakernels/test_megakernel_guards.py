"""CPU contracts of --enable-nvidia-megakernel's startup checks and wait deadlines."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.megakernels import guards
from sglang.srt.megakernels import spec as registry
from sglang.srt.megakernels.guards import (
    MAX_DEADLINE_S,
    deadline_s,
    handle_nvidia_megakernel,
    megakernel_disable_reasons,
)
from sglang.srt.megakernels.spec import Feature
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fakes import ARCH, STRICT, server_args  # noqa: E402

register_cpu_ci(est_time=10, stage="base-a", runner_config="cpu")


def reasons_for(cfg, *, arch=ARCH, isa="sm_103", model_type="fake"):
    hf_config = SimpleNamespace(architectures=[arch], model_type=model_type)
    return megakernel_disable_reasons(cfg=cfg, hf_config=hf_config, isa=isa)


class TestMegakernelGuards(CustomTestCase):
    def setUp(self):
        registry_patch = patch.dict(registry._REGISTRY, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        registry.register(architecture=ARCH, path="_fakes:STRICT")

    def test_every_refusal_reason_is_listed(self):
        with envs.SGLANG_DSPARK_EMBED_IN_GRAPH.override(False):
            spec, reasons = reasons_for(
                server_args(
                    tp_size=8,
                    pp_size=2,
                    kv_cache_dtype="auto",
                    enable_eplb=True,
                    context_length=7,
                    watchdog_timeout=6,
                ),
                model_type="other",
            )
        self.assertIs(spec, STRICT)
        for want in (
            "tp_size 8",
            "kv_cache_dtype 'auto'",
            "config.model_type 'other'",
            "SGLANG_DSPARK_EMBED_IN_GRAPH False",
            Feature.EPLB.value,
            Feature.PIPELINE_PARALLEL.value,
            "context_length",
            "--watchdog-timeout",
        ):
            self.assertTrue(any(want in r for r in reasons), (want, reasons))
        with envs.SGLANG_DSPARK_EMBED_IN_GRAPH.override(True):
            self.assertEqual(reasons_for(server_args())[1], [])
        self.assertEqual(reasons_for(server_args(), isa=None)[1], ["no GPU is visible"])
        for arch, isa, why in (
            ("Other", "sm_103", "Other"),
            (ARCH, "sm_100", "sm_100"),
        ):
            spec, reasons = reasons_for(server_args(), arch=arch, isa=isa)
            self.assertIsNone(spec)
            self.assertIn(why, reasons[0])
        # an empty field accepts any value
        registry._REGISTRY.clear()
        registry.register(architecture=ARCH, path="_fakes:ANY_TP")
        tp8 = reasons_for(server_args(tp_size=8))[1]
        self.assertFalse(any("tp_size" in r for r in tp8), tp8)
        # a plugin that fails to load is only logged: name it, not the model
        registry._REGISTRY.clear()
        self.assertIn("registered no spec", reasons_for(server_args())[1][0])
        with patch.object(guards, "entry_points", return_value=[]):
            self.assertIn("install sgl-nvidia-mega", guards.plugin_reasons()[0])

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
            reasons.return_value = (STRICT, ["tp_size 8 is not one of ['4']"])
            # the resolved dtype is checked here, and listed with the other reasons
            with self.assertRaisesRegex(
                ValueError, "(?s)tp_size 8.*\n  - dtype 'float16'"
            ):
                handle_nvidia_megakernel(on)
            model_config.return_value = SimpleNamespace(
                hf_config=None, dtype="bfloat16"
            )
            reasons.return_value = (STRICT, [])
            handle_nvidia_megakernel(on)
            # both flags replace the model's forward: refused before any weight loads
            both = server_args(enable_nvidia_megakernel=True, monokernel_backend="x")
            with self.assertRaisesRegex(ValueError, "--monokernel-backend x"):
                handle_nvidia_megakernel(both)
        # every wait's deadline stays below the watchdog
        self.assertEqual(deadline_s(None), MAX_DEADLINE_S)
        self.assertEqual(deadline_s(30), 15.0)


if __name__ == "__main__":
    unittest.main()
