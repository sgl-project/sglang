# SPDX-License-Identifier: Apache-2.0
"""`AdapterLoader` places each component by its own offload policy.

The loader serves three components, and the policy does not answer the same for
all of them: `connectors` follows `dit_cpu_offload`, while `duration_head` and
`diffusion_decoder` default to staying resident. Asking about a fixed name
would push the latter two to the CPU whenever `--dit-cpu-offload` is set.
"""

import unittest

from sglang.multimodal_gen.runtime.loader.component_loaders.adapter_loader import (
    AdapterLoader,
)
from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs


class TestAdapterLoaderOffloadTarget(unittest.TestCase):
    def _server_args(self, **overrides):
        server_args = ServerArgs(model_path="x")
        server_args.cpu_offload_components = None
        server_args.dit_cpu_offload = False
        server_args.vae_cpu_offload = False
        for key, value in overrides.items():
            setattr(server_args, key, value)
        return server_args

    def test_every_component_the_loader_serves_has_a_policy_answer(self):
        server_args = self._server_args()
        for component_name in AdapterLoader.component_names:
            # Must not raise: the loader asks the policy about each of these.
            self.assertIsInstance(
                server_args.should_cpu_offload_component(component_name), bool
            )

    def test_dit_offload_moves_connectors_but_not_the_other_two(self):
        server_args = self._server_args(dit_cpu_offload=True)
        self.assertTrue(server_args.should_cpu_offload_component("connectors"))
        self.assertFalse(server_args.should_cpu_offload_component("duration_head"))
        self.assertFalse(server_args.should_cpu_offload_component("diffusion_decoder"))

    def test_explicit_selection_reaches_the_diffusion_decoder(self):
        server_args = self._server_args(
            cpu_offload_components=["diffusion_decoder"],
        )
        self.assertTrue(server_args.should_cpu_offload_component("diffusion_decoder"))
        self.assertFalse(server_args.should_cpu_offload_component("connectors"))

    def test_loader_does_not_hardcode_a_component_name_for_placement(self):
        # The regression this guards: a literal "connectors" in the placement
        # call, which would make all three components follow dit_cpu_offload.
        import inspect

        source = inspect.getsource(AdapterLoader.load_customized)
        placement = [
            line
            for line in source.splitlines()
            if "should_cpu_offload_component" in line
        ]
        self.assertTrue(placement, "expected a placement call to inspect")
        for line in placement:
            self.assertNotIn(
                '"',
                line,
                f"placement must use the component_name parameter, got: {line.strip()}",
            )


if __name__ == "__main__":
    unittest.main()
