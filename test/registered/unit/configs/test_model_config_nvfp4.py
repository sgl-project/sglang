"""Unit tests for srt/configs/model_config.py NVFP4 hybrid-checkpoint helpers.

Covers `ModelConfig._load_hf_quant_config_dict` (raw JSON load) and
`ModelConfig._extract_nvfp4_moe_meta` (group_size + exclude_modules
extraction) — the two pieces that recognise a hybrid FP8(linear)+NVFP4(MoE)
checkpoint such as ``nvidia/DeepSeek-V4-Pro-NVFP4``.

Both methods only read ``self.model_path`` and ``self.revision``, so the
tests call them on a ``SimpleNamespace`` stub and avoid constructing a
real ``ModelConfig`` (which would require an HF model on disk).
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.configs.model_config import ModelConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _write_hf_quant_config(model_dir: Path, body: dict) -> None:
    (model_dir / "hf_quant_config.json").write_text(json.dumps(body))


def _nvfp4_quantized_layers(num_layers: int, num_experts: int) -> dict:
    """DSV4-Pro-NVFP4 shape: keys are ``layers.X.ffn.experts.Y.wN``,
    each entry has ``quant_algo=NVFP4`` + ``awq_block_size=16``."""
    out = {}
    for layer in range(num_layers):
        for expert in range(num_experts):
            for w in ("w1", "w2", "w3"):
                key = f"layers.{layer}.ffn.experts.{expert}.{w}"
                out[key] = {"quant_algo": "NVFP4", "awq_block_size": 16}
    return out


_DSV4_EXCLUDE_MODULES = [
    "*.attn.*",
    "*.attn_norm.*",
    "*.ffn.shared_experts.*",
    "*.ffn.gate.*",
    "*.ffn_norm.*",
    "embed.weight",
    "head.weight",
    "norm.weight",
]


def _dsv4_pro_nvfp4_body(num_layers: int = 2, num_experts: int = 2) -> dict:
    """Mirror of nvidia/DeepSeek-V4-Pro-NVFP4 hf_quant_config.json shape."""
    return {
        "producer": {"name": "modelopt", "version": "dsv4-nvfp4-experts"},
        "quantization": {
            "quant_algo": None,
            "kv_cache_quant_algo": None,
            "quantized_layers": _nvfp4_quantized_layers(num_layers, num_experts),
            "exclude_modules": _DSV4_EXCLUDE_MODULES,
        },
    }


class TestLoadHfQuantConfigDict(CustomTestCase):
    """`_load_hf_quant_config_dict` returns the raw JSON dict or None."""

    def _make_stub(self, model_path: str):
        return SimpleNamespace(model_path=model_path, revision=None)

    def test_loads_local_hf_quant_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body = _dsv4_pro_nvfp4_body()
            _write_hf_quant_config(Path(tmpdir), body)
            stub = self._make_stub(tmpdir)

            loaded = ModelConfig._load_hf_quant_config_dict(stub)

            self.assertEqual(loaded, body)
            self.assertEqual(loaded["producer"]["version"], "dsv4-nvfp4-experts")

    def test_returns_none_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # No hf_quant_config.json written.
            stub = self._make_stub(tmpdir)

            self.assertIsNone(ModelConfig._load_hf_quant_config_dict(stub))


class TestExtractNvfp4MoeMeta(CustomTestCase):
    """`_extract_nvfp4_moe_meta` returns {group_size, exclude_modules} or None."""

    def _make_stub(self, model_path: str):
        return SimpleNamespace(model_path=model_path, revision=None)

    def test_extracts_dsv4_pro_nvfp4_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_hf_quant_config(Path(tmpdir), _dsv4_pro_nvfp4_body())
            stub = self._make_stub(tmpdir)

            meta = ModelConfig._extract_nvfp4_moe_meta(stub)

            self.assertIsNotNone(meta)
            self.assertEqual(meta["group_size"], 16)
            self.assertEqual(meta["exclude_modules"], _DSV4_EXCLUDE_MODULES)

    def test_returns_none_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stub = self._make_stub(tmpdir)

            self.assertIsNone(ModelConfig._extract_nvfp4_moe_meta(stub))

    def test_returns_none_when_no_quantized_layers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body = {"quantization": {"quant_algo": None}}  # no quantized_layers
            _write_hf_quant_config(Path(tmpdir), body)
            stub = self._make_stub(tmpdir)

            self.assertIsNone(ModelConfig._extract_nvfp4_moe_meta(stub))

    def test_returns_none_when_no_nvfp4_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body = {
                "quantization": {
                    "quantized_layers": {
                        "layers.0.ffn.experts.0.w1": {
                            "quant_algo": "FP8",  # not NVFP4
                            "awq_block_size": 16,
                        },
                    },
                    "exclude_modules": [],
                },
            }
            _write_hf_quant_config(Path(tmpdir), body)
            stub = self._make_stub(tmpdir)

            self.assertIsNone(ModelConfig._extract_nvfp4_moe_meta(stub))

    def test_accepts_group_size_key_instead_of_awq_block_size(self):
        # `_extract_nvfp4_moe_meta` falls back to "group_size" if
        # "awq_block_size" is absent — keep that behaviour covered so a
        # future refactor doesn't silently break checkpoints that use the
        # alternate key.
        with tempfile.TemporaryDirectory() as tmpdir:
            body = {
                "quantization": {
                    "quantized_layers": {
                        "layers.0.ffn.experts.0.w1": {
                            "quant_algo": "NVFP4",
                            "group_size": 32,
                        },
                    },
                    "exclude_modules": ["a", "b"],
                },
            }
            _write_hf_quant_config(Path(tmpdir), body)
            stub = self._make_stub(tmpdir)

            meta = ModelConfig._extract_nvfp4_moe_meta(stub)

            self.assertIsNotNone(meta)
            self.assertEqual(meta["group_size"], 32)
            self.assertEqual(meta["exclude_modules"], ["a", "b"])

    def test_defaults_empty_exclude_modules_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            body = {
                "quantization": {
                    "quantized_layers": {
                        "layers.0.ffn.experts.0.w1": {
                            "quant_algo": "NVFP4",
                            "awq_block_size": 16,
                        },
                    },
                    # No exclude_modules key.
                },
            }
            _write_hf_quant_config(Path(tmpdir), body)
            stub = self._make_stub(tmpdir)

            meta = ModelConfig._extract_nvfp4_moe_meta(stub)

            self.assertIsNotNone(meta)
            self.assertEqual(meta["exclude_modules"], [])


if __name__ == "__main__":
    unittest.main()
