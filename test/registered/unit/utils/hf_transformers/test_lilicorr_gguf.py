import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gguf
import numpy as np
import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_loader.loader import LiLiCorrGGUFModelLoader, get_model_loader
from sglang.srt.models.lilicorr import LiLiCorrHead
from sglang.srt.speculative.lilicorr_utils import LiLiCorrConfig
from sglang.srt.utils.hf_transformers.gguf_native import (
    build_gguf_config,
    has_native_gguf_support,
)
from sglang.srt.utils.hf_transformers.gguf_reader import get_gguf_reader
from sglang.srt.utils.hf_transformers.lilicorr_gguf import (
    build_lilicorr_gguf_config,
    lilicorr_gguf_weights,
    repack_nvfp4,
    slot_bias_from_gguf,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _write_checkpoint(path, blocks, scale, extra_tensor=None, prec_a4=False):
    writer = gguf.GGUFWriter(path, "dflash")
    for key, value in {
        "dflash.block_count": 2,
        "dflash.embedding_length": 16,
        "dflash.feed_forward_length": 16,
        "dflash.attention.head_count": 2,
        "dflash.attention.head_count_kv": 1,
        "dflash.attention.key_length": 8,
        "dflash.context_length": 128,
        "dflash.block_size": 4,
        "dflash.conv_kernel_size": 0,
        "dflash.conv_group_size": 0,
        "dflash.attention.sliding_window": 32,
        "tokenizer.ggml.mask_token_id": 3,
        "lilicorr.hidden_size": 8,
        "lilicorr.candidate_topk": 2,
        "lilicorr.num_heads": 2,
        "lilicorr.num_layers": 1,
        "lilicorr.factor_dim": 8,
        "lilicorr.mlp_hidden_size": 16,
    }.items():
        writer.add_uint32(key, value)
    writer.add_float32("dflash.rope.freq_base", 10000.0)
    writer.add_float32("dflash.attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_float32("lilicorr.vector_eps", 1e-4)
    writer.add_float32("lilicorr.logit_scale", 8.0)
    writer.add_bool("dflash.attention.causal", False)
    writer.add_array("general.tensor_extra.name", ["fc.weight"])
    writer.add_array("general.tensor_extra.prec_a4", [prec_a4])
    writer.add_array("dflash.target_layers", [2, 6])
    writer.add_array("dflash.attention.sliding_window_pattern", [True, True])
    writer.add_array("tokenizer.ggml.tokens", ["a", "b", "c", "mask"])
    writer.add_tensor("fc.weight", blocks, raw_dtype=gguf.GGMLQuantizationType.NVFP4)
    writer.add_tensor("fc.scale", np.array([scale], dtype=np.float32))
    writer.add_tensor("enc.output_norm.weight", np.ones(16, dtype=np.float32))
    for name, rows in (("q_proj", 16), ("k_proj", 8), ("v_proj", 8)):
        gguf_name = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v"}[name]
        writer.add_tensor(
            f"blk.0.{gguf_name}.weight", np.ones((rows, 16), dtype=np.float32)
        )
    if extra_tensor is not None:
        writer.add_tensor(*extra_tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class TestLiLiCorrGGUF(CustomTestCase):
    def test_model_config_and_weights_parse_checkpoint_once(self):
        """Repeated config probes must not reparse large tokenizer metadata."""
        get_gguf_reader.cache_clear()
        self.addCleanup(get_gguf_reader.cache_clear)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "draft.gguf"
            _write_checkpoint(path, np.zeros((2, 72), dtype=np.uint8), 0.25)
            with patch("gguf.GGUFReader", wraps=gguf.GGUFReader) as reader:
                config = ModelConfig(model_path=str(path), is_draft_model=True)
                weights = dict(lilicorr_gguf_weights(str(path), config.hf_config))
            self.assertEqual(weights["fc.weight_scale_2"].item(), 0.25)
            self.assertEqual(reader.call_count, 1)

    def test_native_detection_does_not_claim_plain_dflash(self):
        with tempfile.TemporaryDirectory() as temp:
            for index, (architecture, head_layers, supported) in enumerate(
                [
                    ("dflash", None, False),
                    ("dflash", 0, True),
                    ("muse-glimmer", None, True),
                    ("llama", 2, False),
                ]
            ):
                with self.subTest(architecture=architecture, head_layers=head_layers):
                    path = Path(temp) / f"draft-{index}.gguf"
                    writer = gguf.GGUFWriter(path, architecture)
                    if head_layers is not None:
                        writer.add_uint32("lilicorr.num_layers", head_layers)
                    writer.write_header_to_file()
                    writer.write_kv_data_to_file()
                    writer.write_tensors_to_file()
                    writer.close()
                    self.assertEqual(has_native_gguf_support(str(path)), supported)
                    if not supported:
                        with self.assertRaisesRegex(
                            ValueError, "Unsupported native GGUF"
                        ):
                            build_gguf_config(str(path))

    def test_repacked_nvfp4_matches_independent_gguf_decoder(self):
        rng = np.random.default_rng(21)
        blocks = rng.integers(0, 256, size=(3, 2, 36), dtype=np.uint8)
        blocks[:, :, :4] %= 128
        blocks[0, 0, 0] = 127
        weight, scales = repack_nvfp4(blocks, [128, 3])
        codes = torch.stack((weight & 15, weight >> 4), dim=-1).reshape(3, 128)
        values = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        actual = values[codes.long()] * scales.float().repeat_interleave(16, dim=-1)
        expected = gguf.dequantize(
            blocks.reshape(3, 72), gguf.GGMLQuantizationType.NVFP4
        )
        torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=0, atol=0)

    def test_real_gguf_retains_capture_layers_and_global_scale(self):
        blocks = np.zeros((2, 72), dtype=np.uint8)
        blocks.reshape(2, 2, 36)[:, :, :4] = 56
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "draft.bin"
            _write_checkpoint(path, blocks, 0.25)
            config = build_lilicorr_gguf_config(str(path))
            self.assertEqual(config.dflash_config["target_layer_ids"], [1, 5])
            self.assertEqual(config.sliding_window, 32)
            quant = ModelOptFp4Config.from_config(config.quantization_config)
            quant.packed_modules_mapping = {}
            qkv = QKVParallelLinear(
                hidden_size=16,
                head_size=8,
                parallel_group="replicated",
                total_num_heads=2,
                total_num_kv_heads=1,
                quant_config=quant,
                prefix="layers.0.self_attn.qkv_proj",
            )
            self.assertIsInstance(qkv.quant_method, UnquantizedLinearMethod)
            weights = dict(lilicorr_gguf_weights(str(path), config))
            self.assertEqual(weights["fc.weight_scale_2"].item(), 0.25)
            self.assertEqual(tuple(weights["fc.weight"].shape), (2, 64))
            self.assertEqual(weights["fc.weight_scale"].dtype, torch.float8_e4m3fn)
            self.assertIn("hidden_norm.weight", weights)
            model_config = ModelConfig(model_path=str(path), is_draft_model=True)
            self.assertIsInstance(
                get_model_loader(LoadConfig(), model_config), LiLiCorrGGUFModelLoader
            )

    def test_explicit_fp4_activations_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "draft.gguf"
            _write_checkpoint(
                path, np.zeros((2, 72), dtype=np.uint8), 1.0, prec_a4=True
            )
            with self.assertRaisesRegex(ValueError, "16-bit activations"):
                build_lilicorr_gguf_config(str(path))

    def test_unrecognized_embedding_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "draft.gguf"
            _write_checkpoint(
                path,
                np.zeros((2, 72), dtype=np.uint8),
                1.0,
                extra_tensor=("token_embd.weight", np.ones((4, 16), dtype=np.float32)),
            )
            with self.assertRaisesRegex(ValueError, "Unknown LiLiCorr GGUF tensor"):
                build_lilicorr_gguf_config(str(path))

    def test_materialized_bias_preserves_head_and_shorter_blocks(self):
        config = LiLiCorrConfig(
            candidate_topk=2,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            mlp_ratio=2,
            factor_dim=8,
            vector_eps=1e-4,
            logit_scale=8,
        )
        head = LiLiCorrHead(
            model_hidden_size=8, block_size=4, rms_norm_eps=1e-6, config=config
        )
        with torch.no_grad():
            head.relative_slot_bias.copy_(torch.arange(14).reshape(2, 7))
            head.same_slot_bias.copy_(torch.tensor([0.5, -0.5]))
        expected = head._build_attention_bias(
            device=torch.device("cpu"), dtype=torch.float32
        )
        relative, same = slot_bias_from_gguf(expected, 4, 2)
        with torch.no_grad():
            head.relative_slot_bias.copy_(relative)
            head.same_slot_bias.copy_(same)
        head.materialize_inference_buffers(torch.device("cpu"), torch.float32)
        torch.testing.assert_close(head._attn_bias, expected, rtol=0, atol=0)
        head.set_active_block_size(2)
        torch.testing.assert_close(head._attn_bias, expected[:, :2, :2], rtol=0, atol=0)

    def test_rank_dependent_bias_is_rejected(self):
        bias = torch.zeros(2, 6, 6)
        bias[0, 0, 1] = 1
        with self.assertRaisesRegex(ValueError, "not slot-relative"):
            slot_bias_from_gguf(bias, 4, 2)


if __name__ == "__main__":
    unittest.main()
