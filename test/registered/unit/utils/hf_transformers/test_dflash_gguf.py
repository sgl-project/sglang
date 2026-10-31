"""Dense BF16 GGUF regressions for Bolt E4B DFlash.

Checkpoint: nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash.
Fixtures cover indexing, loader dispatch, and exact BF16 decoding.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gguf
import numpy as np
import torch

from sglang.srt.utils.hf_transformers.config import get_config
from sglang.srt.utils.hf_transformers.dflash_gguf import (
    build_plain_dflash_gguf_config,
    plain_dflash_gguf_weights,
)
from sglang.srt.utils.hf_transformers.gguf_native import has_native_gguf_support
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def write_draft(
    path, *, missing=None, extra=None, targets=(2, 6), lilicorr=False, packed=False
):
    writer = gguf.GGUFWriter(path, "dflash")
    for key, value in {
        "block_count": 1,
        "embedding_length": 32,
        "feed_forward_length": 64,
        "attention.head_count": 4,
        "attention.head_count_kv": 2,
        "attention.key_length": 8,
        "attention.value_length": 8,
        "context_length": 128,
        "block_size": 8,
        "attention.sliding_window": 32,
    }.items():
        writer.add_uint32("dflash." + key, value)
    writer.add_float32("dflash.rope.freq_base", 10000.0)
    writer.add_float32("dflash.attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_array("dflash.target_layers", list(targets))
    writer.add_array("dflash.attention.sliding_window_pattern", [True])
    writer.add_bool("dflash.attention.causal", False)
    writer.add_uint32("tokenizer.ggml.mask_token_id", 3)
    writer.add_array("tokenizer.ggml.tokens", ["a", "b", "c", "mask"])
    if lilicorr:
        writer.add_uint32("lilicorr.num_layers", 1)
    tensors = {
        "fc.weight": np.arange(32 * 64, dtype=np.float32).reshape(32, 64) / 256,
        "enc.output_norm.weight": np.ones(32, dtype=np.float32),
        "output_norm.weight": np.ones(32, dtype=np.float32),
    }
    for name, shape in {
        "attn_norm": (32,),
        "ffn_norm": (32,),
        "attn_q": (32, 32),
        "attn_k": (16, 32),
        "attn_v": (16, 32),
        "attn_output": (32, 32),
        "attn_q_norm": (8,),
        "attn_k_norm": (8,),
        "ffn_gate": (64, 32),
        "ffn_up": (64, 32),
        "ffn_down": (32, 64),
    }.items():
        tensors[f"blk.0.{name}.weight"] = np.ones(shape, dtype=np.float32)
    if extra:
        tensors[extra] = np.ones(32, dtype=np.float32)
    for name, values in tensors.items():
        if name == missing:
            continue
        if values.ndim == 2:
            weight_type = (
                gguf.GGMLQuantizationType.Q8_0
                if packed
                else gguf.GGMLQuantizationType.BF16
            )
            values = gguf.quants.quantize(values, weight_type)
            writer.add_tensor(name, values, raw_dtype=weight_type)
        else:
            writer.add_tensor(name, values)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class TestDFlashGGUF(CustomTestCase):
    def test_config_and_weights_parse_the_gguf_once(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "draft.gguf"
            write_draft(path)
            with patch("gguf.GGUFReader", wraps=gguf.GGUFReader) as reader:
                self.assertTrue(has_native_gguf_support(str(path)))
                get_config(str(path), trust_remote_code=False)
                list(plain_dflash_gguf_weights(str(path)))
            self.assertEqual(reader.call_count, 1)

    def test_dense_config_preserves_capture_indices_and_window(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "draft.gguf"
            write_draft(path)
            self.assertTrue(has_native_gguf_support(str(path)))
            config = build_plain_dflash_gguf_config(str(path))
            self.assertEqual(config.dflash_config["target_layer_ids"], [1, 5])
            self.assertEqual(config.sliding_window, 32)
            self.assertEqual(config.layer_types, ["sliding_attention"])
            self.assertEqual(config.architectures, ["DFlashDraftModel"])
            self.assertFalse(hasattr(config, "quantization_config"))

    def test_bf16_bytes_load_as_dense_linear_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "draft.gguf"
            write_draft(path)
            weights = dict(plain_dflash_gguf_weights(str(path)))
            expected = (torch.arange(2048).reshape(32, 64) / 256).bfloat16()
            torch.testing.assert_close(weights["fc.weight"], expected, rtol=0, atol=0)
            self.assertEqual(
                weights["layers.0.self_attn.k_proj.weight"].shape, (16, 32)
            )
            self.assertEqual(weights["hidden_norm.weight"].dtype, torch.float32)
            self.assertFalse(any("qweight" in name for name in weights))

    def test_gguf_loader_uses_dense_parameters_for_the_draft(self):
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.configs.load_config import LoadConfig
        from sglang.srt.model_loader.loader import GGUFModelLoader

        class DenseDraft(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(64, 32, bias=False)

            def load_weights(self, weights):
                loaded = dict(weights)
                with torch.no_grad():
                    self.fc.weight.copy_(loaded["fc.weight"])

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "draft.gguf"
            write_draft(path)
            config = SimpleNamespace(
                model_path=str(path),
                hf_config=get_config(str(path), trust_remote_code=False),
                dtype=torch.bfloat16,
            )
            with patch(
                "sglang.srt.model_loader.loader._initialize_model",
                side_effect=lambda *_: DenseDraft(),
            ) as initialize:
                model = GGUFModelLoader(LoadConfig(load_format="gguf")).load_model(
                    model_config=config, device_config=DeviceConfig("cpu")
                )
            self.assertIsNone(initialize.call_args.args[2])
            expected = (torch.arange(2048).reshape(32, 64) / 256).bfloat16()
            torch.testing.assert_close(model.fc.weight, expected, rtol=0, atol=0)

    def test_incomplete_or_extra_weights_cannot_be_silently_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            for index, kwargs in enumerate(
                [
                    {"missing": "blk.0.attn_v.weight"},
                    {"extra": "output.weight"},
                ]
            ):
                with self.subTest(kwargs=kwargs):
                    path = Path(temp) / f"draft{index}.gguf"
                    write_draft(path, **kwargs)
                    with self.assertRaisesRegex(
                        ValueError, "Unsupported DFlash GGUF tensors"
                    ):
                        build_plain_dflash_gguf_config(str(path))
                    with self.assertRaisesRegex(
                        ValueError, "Unsupported DFlash GGUF tensors"
                    ):
                        list(plain_dflash_gguf_weights(str(path)))

    def test_lilicorr_is_not_claimed_by_plain_dflash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "draft.gguf"
            write_draft(path, lilicorr=True)
            self.assertFalse(has_native_gguf_support(str(path)))
            with self.assertRaisesRegex(ValueError, "without LiLiCorr"):
                build_plain_dflash_gguf_config(str(path))

    def test_packed_dflash_does_not_enter_the_dense_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "draft.gguf"
            write_draft(path, packed=True)
            self.assertFalse(has_native_gguf_support(str(path)))
            with self.assertRaisesRegex(
                ValueError, "requires F32, F16, or BF16 weights"
            ):
                list(plain_dflash_gguf_weights(str(path)))

    def test_invalid_capture_indices_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            for targets in [(0, 6), (2, 2)]:
                path = Path(temp) / f"draft-{targets[0]}.gguf"
                write_draft(path, targets=targets)
                with self.assertRaisesRegex(ValueError, "distinct positive"):
                    build_plain_dflash_gguf_config(str(path))


if __name__ == "__main__":
    unittest.main()
