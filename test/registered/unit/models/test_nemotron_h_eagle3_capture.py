"""NemotronH exposes EAGLE3 aux-hidden-state capture.

The model runner calls ``set_eagle3_layers_to_capture`` on the target for
EAGLE3 drafts; NemotronH (and the Nano VL wrapper) only had the DFLASH
variant, so an EAGLE3 draft could not be served on it. Layer ``i``'s output is
captured at the input of layer ``i + 1``; with no ids the default is the input
of layers 2, L // 2 and L - 3, as for Llama.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models.nano_nemotron_vl import NemotronH_Nano_VL_V2  # noqa: E402
from sglang.srt.models.nemotron_h import NemotronHForCausalLM  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _model(last_rank: bool = True):
    model = object.__new__(NemotronHForCausalLM)
    model.__dict__.update(
        pp_group=SimpleNamespace(is_last_rank=last_rank),
        config=SimpleNamespace(num_hidden_layers=52),
        model=SimpleNamespace(layers_to_capture=set()),
        capture_aux_hidden_states=False,
    )
    return model


class TestNemotronHEagle3Capture(CustomTestCase):
    def test_explicit_layer_ids(self):
        model = _model()
        model.set_eagle3_layers_to_capture([1, 25, 48])
        self.assertTrue(model.capture_aux_hidden_states)
        self.assertEqual(model.model.layers_to_capture, {2, 26, 49})

    def test_default_layers(self):
        model = _model()
        model.set_eagle3_layers_to_capture()
        self.assertEqual(model.model.layers_to_capture, {2, 26, 49})

    def test_not_last_pp_rank(self):
        model = _model(last_rank=False)
        model.set_eagle3_layers_to_capture([1])
        self.assertFalse(model.capture_aux_hidden_states)
        self.assertEqual(model.model.layers_to_capture, set())

    def test_vl_wrapper_forwards(self):
        vl = object.__new__(NemotronH_Nano_VL_V2)
        vl.__dict__["language_model"] = _model()
        vl.set_eagle3_layers_to_capture([3])
        self.assertEqual(vl.language_model.model.layers_to_capture, {4})


if __name__ == "__main__":
    unittest.main()
