"""The quantization a Qwen3.5 MTP module is built with.

The MTP module is embedded in the target checkpoint, so it inherits the
target's `quantization_config` unless something says otherwise. Quark MXFP4
checkpoints ship it in bf16; `amd/Qwen3.5-397B-A17B-MXFP4` declares that by
listing every `mtp.*` layer under `exclude`, but its successor
`amd/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2` ships the same bf16 tensors and
declares none of them. Trusting the declaration alone allocates packed MXFP4
experts for bf16 weights, and the only symptom is a shape mismatch deep inside
`FusedMoE._load_w13` — so these cases pin what the checkpoint itself is asked.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.model_loader.weight_utils import probe_module_is_quantized
from sglang.srt.models.qwen3_5_mtp import _mtp_quant_config
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# One `mtp.*` and one target tensor per layout, enough for the name-only probe.
_UNQUANTIZED_MTP_KEYS = [
    "mtp.fc.weight",
    "mtp.layers.0.mlp.experts.0.gate_proj.weight",
    "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale",
]
_QUANTIZED_MTP_KEYS = _UNQUANTIZED_MTP_KEYS + [
    "mtp.layers.0.mlp.experts.0.gate_proj.weight_scale",
]


def _quark_config(exclude_layers=(), is_prequantized=True):
    return SimpleNamespace(
        get_name=lambda: "quark",
        exclude_layers=list(exclude_layers),
        is_prequantized=is_prequantized,
    )


class _MtpQuantConfigCase(CustomTestCase):
    def _checkpoint(self, keys):
        """A checkpoint directory carrying just a safetensors index."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        model_path = tmp.name
        with open(os.path.join(model_path, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {k: "model-00001.safetensors" for k in keys}}, f)
        # The probe caches per (path, prefix, revision); tempfile paths are
        # unique per case, so cases cannot bleed into one another.
        return model_path

    def _seed(self, model_path):
        override = get_context().override_server_args(model_path=model_path)
        override.install()
        self.addCleanup(override.restore)


class TestProbeModuleIsQuantized(_MtpQuantConfigCase):
    def test_a_bf16_submodule_reads_as_unquantized(self):
        path = self._checkpoint(_UNQUANTIZED_MTP_KEYS)
        self.assertIs(probe_module_is_quantized(path, "mtp."), False)

    def test_a_scaled_submodule_reads_as_quantized(self):
        path = self._checkpoint(_QUANTIZED_MTP_KEYS)
        self.assertIs(probe_module_is_quantized(path, "mtp."), True)
        # The target side of the same checkpoint is quantized either way.
        self.assertIs(probe_module_is_quantized(path, "model.language_model."), True)

    def test_an_absent_submodule_has_no_answer(self):
        path = self._checkpoint(_UNQUANTIZED_MTP_KEYS)
        self.assertIsNone(probe_module_is_quantized(path, "draft."))

    def test_an_uncached_remote_checkpoint_has_no_answer(self):
        self.assertIsNone(
            probe_module_is_quantized("sglang-test/definitely-not-a-repo", "mtp.")
        )


class TestQwen3_5MtpQuantConfig(_MtpQuantConfigCase):
    def test_an_exclude_list_naming_mtp_drops_quantization(self):
        # `amd/Qwen3.5-397B-A17B-MXFP4`: the declaration alone settles it, so
        # the checkpoint is never opened.
        self._seed(self._checkpoint(_QUANTIZED_MTP_KEYS))
        quant_config = _quark_config(exclude_layers=["mtp.mlp.experts"])
        self.assertIsNone(_mtp_quant_config(quant_config))

    def test_an_undeclared_bf16_mtp_module_drops_quantization(self):
        # `amd/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2`: nothing declared, bf16 on
        # disk. Inheriting MXFP4 here is the 2048-vs-4096 crash.
        self._seed(self._checkpoint(_UNQUANTIZED_MTP_KEYS))
        self.assertIsNone(_mtp_quant_config(_quark_config()))

    def test_a_quantized_mtp_module_keeps_quantization(self):
        self._seed(self._checkpoint(_QUANTIZED_MTP_KEYS))
        quant_config = _quark_config()
        self.assertIs(_mtp_quant_config(quant_config), quant_config)

    def test_an_online_requantizing_source_keeps_quantization(self):
        # Quark requantizing a bf16 / FP8 / NVFP4 source converts the MTP
        # weights on load, so bf16 `mtp.*` tensors are expected there.
        self._seed(self._checkpoint(_UNQUANTIZED_MTP_KEYS))
        quant_config = _quark_config(is_prequantized=False)
        self.assertIs(_mtp_quant_config(quant_config), quant_config)

    def test_a_checkpoint_without_an_mtp_module_keeps_quantization(self):
        self._seed(self._checkpoint(["model.language_model.layers.0.mlp.gate.weight"]))
        quant_config = _quark_config()
        self.assertIs(_mtp_quant_config(quant_config), quant_config)

    def test_an_unreadable_checkpoint_keeps_quantization(self):
        self._seed("sglang-test/definitely-not-a-repo")
        quant_config = _quark_config()
        self.assertIs(_mtp_quant_config(quant_config), quant_config)


if __name__ == "__main__":
    unittest.main()
