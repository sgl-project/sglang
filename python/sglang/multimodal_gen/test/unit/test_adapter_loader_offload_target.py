# SPDX-License-Identifier: Apache-2.0
"""Custom component loaders place each component by its own offload policy.

`connectors` follows `dit_cpu_offload`; the duration head and standalone
diffusion decoder stay resident by default.
"""

import unittest

from sglang.multimodal_gen.runtime.loader.component_loaders.adapter_loader import (
    AdapterLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.diffusion_decoder_loader import (
    DiffusionDecoderLoader,
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

    def test_dit_offload_moves_connectors_but_not_optional_modules(self):
        server_args = self._server_args(dit_cpu_offload=True)
        self.assertTrue(server_args.should_cpu_offload_component("connectors"))
        self.assertFalse(server_args.should_cpu_offload_component("duration_head"))
        self.assertFalse(server_args.should_cpu_offload_component("diffusion_decoder"))

    def test_diffusion_decoder_has_a_dedicated_loader(self):
        self.assertNotIn("diffusion_decoder", AdapterLoader.component_names)
        self.assertEqual(DiffusionDecoderLoader.component_names, ["diffusion_decoder"])

    def test_explicit_selection_reaches_the_diffusion_decoder(self):
        server_args = self._server_args(
            cpu_offload_components=["diffusion_decoder"],
        )
        self.assertTrue(server_args.should_cpu_offload_component("diffusion_decoder"))
        self.assertFalse(server_args.should_cpu_offload_component("connectors"))

    def test_vae_group_includes_the_diffusion_decoder(self):
        server_args = self._server_args(cpu_offload_components=["vae"])
        self.assertTrue(server_args.should_cpu_offload_component("diffusion_decoder"))

    def test_layerwise_diffusion_decoder_starts_on_cpu(self):
        server_args = self._server_args(
            component_residency={"diffusion_decoder": "layerwise-offload"}
        )

        self.assertFalse(server_args.should_cpu_offload_component("diffusion_decoder"))
        self.assertTrue(server_args.should_start_component_on_cpu("diffusion_decoder"))


if __name__ == "__main__":
    unittest.main()
