"""CPU checks of --enable-nvidia-megakernel's startup resolution."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.kernels.plugin_kernels.nvidia import spec as registry
from sglang.srt.kernel_plugins.nvidia import binding
from sglang.srt.kernel_plugins.nvidia.binding import (
    megakernel_disable_reasons,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    default_cuda_graph_config,
    with_phase,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fakes import ARCH, ISA, server_args  # noqa: E402

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestMegakernelGuards(CustomTestCase):
    def setUp(self):
        registry_patch = patch.dict(registry._REGISTRY, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)

    def test_a_disabled_flag_looks_nothing_up_and_keeps_the_forward(self):
        with patch.object(binding, "plugin_reasons") as plugin:
            binding.handle_nvidia_megakernel(server_args())
        plugin.assert_not_called()
        model = SimpleNamespace()  # any attribute read would raise
        with patch.object(binding, "megakernel_enabled", return_value=False):
            binding.install_megakernel(
                model=model, hf_config=None, capture_tail_hooks=[], is_draft=False
            )
        self.assertEqual(vars(model), {})

    def test_an_unmet_tp_size_is_refused_and_an_empty_field_accepts_any(self):
        hf_config = SimpleNamespace(architectures=[ARCH], model_type="fake")

        def tp_reasons(path):
            registry._REGISTRY.clear()
            registry.register(architecture=ARCH, path=path)
            _, reasons = megakernel_disable_reasons(
                cfg=server_args(tp_size=8), hf_config=hf_config, isa=ISA
            )
            return [r for r in reasons if "tp_size" in r]

        self.assertTrue(tp_reasons("_fakes:SPEC"))
        self.assertEqual(tp_reasons("_fakes:ANY_TP"), [])

    def test_only_a_spec_that_covers_prefill_is_held_to_the_prefill_graphs(self):
        hf_config = SimpleNamespace(architectures=[ARCH], model_type="fake")

        def prefill_reasons(path, backend):
            registry._REGISTRY.clear()
            registry.register(architecture=ARCH, path=path)
            graphs = with_phase(
                default_cuda_graph_config(), Phase.PREFILL, backend=backend
            )
            _, reasons = megakernel_disable_reasons(
                cfg=server_args(cuda_graph_config=graphs), hf_config=hf_config, isa=ISA
            )
            return [r for r in reasons if "prefill CUDA graphs" in r]

        self.assertTrue(prefill_reasons("_fakes:PREFILL", Backend.DISABLED))
        self.assertEqual(prefill_reasons("_fakes:PREFILL", Backend.BREAKABLE), [])
        # a decode-only spec is not asked about prefill graphs at all
        self.assertEqual(prefill_reasons("_fakes:SPEC", Backend.DISABLED), [])


if __name__ == "__main__":
    unittest.main()
