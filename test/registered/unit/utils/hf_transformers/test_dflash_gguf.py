"""Dense BF16 GGUF regressions for Bolt E4B DFlash.

Checkpoint: nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash.
Fixtures cover indexing, loader dispatch, and exact BF16 decoding.
"""

import gc
import tempfile
import unittest
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gguf
import numpy as np
import torch

from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.models import dflash
from sglang.srt.utils.hf_transformers.config import get_config
from sglang.srt.utils.hf_transformers.dflash_gguf import (
    build_plain_dflash_gguf_config,
    plain_dflash_gguf_weights,
)
from sglang.srt.utils.hf_transformers.gguf_native import (
    has_native_gguf_support,
    read_gguf_architecture,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def write_draft(
    path,
    *,
    missing=None,
    extra=None,
    targets=(2, 6),
    lilicorr=False,
    packed=False,
    causal=False,
    tokens=("a", "b", "c", "mask"),
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
    writer.add_bool("dflash.attention.causal", causal)
    writer.add_uint32("tokenizer.ggml.mask_token_id", 3)
    if isinstance(tokens, str):
        writer.add_string("tokenizer.ggml.tokens", tokens)
    else:
        writer.add_array("tokenizer.ggml.tokens", list(tokens))
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


@contextmanager
def track_readers():
    reader_type = gguf.GGUFReader
    readers = []
    token_parts = []

    def create(*args, **kwargs):
        reader = reader_type(*args, **kwargs)
        readers.append(weakref.ref(reader))
        token_parts.extend(
            weakref.ref(part) for part in reader.fields["tokenizer.ggml.tokens"].parts
        )
        return reader

    with patch("gguf.GGUFReader", side_effect=create):
        yield readers, token_parts


class TestDFlashGGUF(CustomTestCase):
    def test_config_cache_releases_readers_and_keeps_configs_independent(self):
        """Cached config data must not retain tokenizer views or caller mutations."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "draft.gguf"
            write_draft(path)
            with track_readers() as (readers, token_parts):
                self.assertEqual(read_gguf_architecture(str(path)), "dflash")
                self.assertTrue(has_native_gguf_support(str(path)))
                config = get_config(str(path), trust_remote_code=False)
                config.dflash_config["target_layer_ids"].clear()
                again = build_plain_dflash_gguf_config(str(path))
                self.assertEqual(again.dflash_config["target_layer_ids"], [1, 5])
                self.assertEqual(len(readers), 1)
                self.assertTrue(token_parts)
                gc.collect()
                self.assertTrue(all(ref() is None for ref in readers + token_parts))
                list(plain_dflash_gguf_weights(str(path)))
                self.assertEqual(len(readers), 2)
            gc.collect()
            self.assertTrue(all(ref() is None for ref in readers + token_parts))

    def test_architecture_probe_releases_metadata_and_refreshes_replaced_files(self):
        """Ordinary GGUF probes must release arrays and observe file replacements."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "target.gguf"
            self.assertIsNone(read_gguf_architecture(str(path)))
            with (
                track_readers() as (readers, token_parts),
                patch(
                    "sglang.srt.utils.hf_transformers.gguf_native._extract_dflash_gguf_metadata",
                    side_effect=AssertionError(
                        "DFlash extraction reached another model"
                    ),
                ),
            ):
                for index, arch in enumerate(("llama", "qwen2"), 1):
                    replacement = Path(temp) / "replacement.gguf"
                    writer = gguf.GGUFWriter(replacement, arch)
                    writer.add_array("tokenizer.ggml.tokens", ["a", "b"])
                    writer.add_tensor(
                        "token_embd.weight", np.ones((2, 32), dtype=np.float32)
                    )
                    writer.write_header_to_file()
                    writer.write_kv_data_to_file()
                    writer.write_tensors_to_file()
                    writer.close()
                    replacement.replace(path)
                    self.assertEqual(read_gguf_architecture(str(path)), arch)
                    self.assertFalse(has_native_gguf_support(str(path)))
                    self.assertEqual(len(readers), index)
                    gc.collect()
                    self.assertTrue(token_parts)
                    self.assertTrue(all(ref() is None for ref in readers + token_parts))
            path.unlink()
            self.assertIsNone(read_gguf_architecture(str(path)))

    def test_config_preserves_vocabulary_capture_indices_and_attention(self):
        tokens = ("", "é", "long token", "mask", "z")
        with tempfile.TemporaryDirectory() as temp:
            for causal, expected in [
                (False, AttentionType.ENCODER_ONLY),
                (True, AttentionType.DECODER),
            ]:
                with self.subTest(causal=causal):
                    path = Path(temp) / f"draft-{causal}.gguf"
                    write_draft(path, causal=causal, tokens=tokens)
                    self.assertTrue(has_native_gguf_support(str(path)))
                    config = build_plain_dflash_gguf_config(str(path))
                    self.assertEqual(config.dflash_config["target_layer_ids"], [1, 5])
                    self.assertEqual(config.vocab_size, len(tokens))
                    self.assertEqual(config.sliding_window, 32)
                    self.assertEqual(config.layer_types, ["sliding_attention"])
                    self.assertEqual(config.architectures, ["DFlashDraftModel"])
                    self.assertEqual(
                        dflash._get_dflash_layer_attention_params(config, 0)[1],
                        expected,
                    )

    def test_invalid_token_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for index, tokens in enumerate(((0, 1, 2, 3), "mask")):
                with self.subTest(tokens=tokens):
                    path = Path(temp) / f"draft-{index}.gguf"
                    write_draft(path, tokens=tokens)
                    self.assertTrue(has_native_gguf_support(str(path)))
                    with self.assertRaisesRegex(
                        ValueError, "tokenizer.ggml.tokens must be an array of strings"
                    ):
                        build_plain_dflash_gguf_config(str(path))

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
                    self.assertTrue(has_native_gguf_support(str(path)))
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
            with patch(
                "sglang.srt.utils.hf_transformers.gguf_native._extract_dflash_gguf_metadata",
                side_effect=AssertionError("Plain DFlash extraction reached LiLiCorr"),
            ):
                self.assertEqual(read_gguf_architecture(str(path)), "dflash")
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
