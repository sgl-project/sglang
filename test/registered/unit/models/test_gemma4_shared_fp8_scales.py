"""CPU loading regression for Gemma4 FP8 shared-cache scale consumers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from transformers import Gemma4TextConfig

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.models.gemma4_causal import (
    Gemma4Attention,
    Gemma4TextModel,
    _bind_shared_fp8_scales,
)
from sglang.srt.runtime_context import (
    get_context,
    get_parallel,
    restore_context,
    snapshot_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


def _layers(bits=8):
    config = Gemma4TextConfig.from_dict(
        dict(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=3,
            vocab_size=32,
            vocab_size_per_layer_input=32,
            hidden_size_per_layer_input=0,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            num_kv_shared_layers=2,
            layer_types=["full_attention"] * 3,
            max_position_embeddings=32,
            sliding_window=16,
        )
    )
    config.allow_global_per_layer_attribute_access = True
    quant = CompressedTensorsConfig.from_config(
        {
            "format": "dense",
            "quant_method": "compressed-tensors",
            "config_groups": {},
            "ignore": [],
            "kv_cache_scheme": {
                "type": "float",
                "num_bits": bits,
                "strategy": "tensor",
                "symmetric": True,
                "dynamic": False,
            },
        }
    )
    layers = torch.nn.ModuleList()
    get_parallel().override_permanently(
        tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0
    )
    for index in range(3):
        with torch.device("meta"):
            attention = Gemma4Attention(index, config, 64, 32)
        method = quant.get_quant_method(
            attention.attn, f"model.layers.{index}.self_attn.attn"
        )
        method.create_weights(attention.attn)
        attention.attn.quant_method = method
        layer = torch.nn.Module()
        layer.self_attn = attention
        layers.append(layer)
    return layers, quant


class TestGemma4SharedFP8Scales(CustomTestCase):
    def setUp(self):
        super().setUp()
        state = snapshot_context()
        state["__parallel__"] = {
            key: value.copy() if isinstance(value, dict) else value
            for key, value in state["__parallel__"].items()
        }
        self.addCleanup(restore_context, state)

    def test_text_model_constructor_binds_reader_scales(self):
        layers, quant = _layers()
        config = layers[0].self_attn.config
        get_context().set_server_args(
            ServerArgs(model_path="unused", device="cpu", boundary_reduction="ar")
        )
        get_parallel().override_permanently(
            pp_group=SimpleNamespace(
                world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True
            ),
            tp_group=None,
            pp_rank=0,
            pp_size=1,
            attn_dp_size=1,
            attn_cp_size=1,
            dp_size=1,
        )
        with torch.device("meta"):
            model = Gemma4TextModel(config, quant_config=quant)
        writer = model.layers[0].self_attn.attn
        self.assertEqual(
            writer.quant_method.readers,
            [layer.self_attn.attn for layer in model.layers[1:]],
        )
        self.assertTrue(
            all(layer.self_attn.attn.quant_method.reader for layer in model.layers[1:])
        )

    def test_writer_publishes_final_scales_in_either_order_and_on_reload(self):
        for reverse in (False, True):
            for fnuz in (False, True):
                with self.subTest(reverse=reverse, fnuz=fnuz):
                    layers, quant = _layers()
                    radix = [layer.self_attn.attn for layer in layers]
                    identities = [
                        (id(layer.k_scale), id(layer.v_scale)) for layer in radix
                    ]
                    before_names = list(layers.named_parameters())
                    _bind_shared_fp8_scales(layers, quant)
                    self.assertEqual(
                        [name for name, _ in before_names],
                        [name for name, _ in layers.named_parameters()],
                    )
                    for k, v in ((0.125, 0.25), (0.375, 0.5)):
                        radix[0].k_scale.data.fill_(k)
                        radix[0].v_scale.data.fill_(v)
                        # Serialized reader values cannot override the owner.
                        for reader in radix[1:]:
                            reader.k_scale.data.fill_(9)
                            reader.v_scale.data.fill_(11)
                        with patch(
                            "sglang.srt.layers.quantization.kv_cache.is_fp8_fnuz",
                            return_value=fnuz,
                        ):
                            for layer in reversed(radix) if reverse else radix:
                                layer.quant_method.process_weights_after_loading(layer)
                        factor = 2 if fnuz else 1
                        for layer in radix:
                            self.assertEqual(
                                (layer.k_scale_float, layer.v_scale_float),
                                (k * factor, v * factor),
                            )
                            self.assertEqual(
                                (layer.k_scale.item(), layer.v_scale.item()),
                                (k * factor, v * factor),
                            )
                    self.assertEqual(
                        identities,
                        [(id(layer.k_scale), id(layer.v_scale)) for layer in radix],
                    )
                    # Direct references live only in a non-Module quant method;
                    # they do not create parameter registrations or writer cycles.
                    restored = pickle.loads(pickle.dumps(layers))
                    self.assertEqual(
                        len(list(restored.named_parameters())), len(before_names)
                    )

    def test_missing_pipeline_writer_fails_closed(self):
        layers, quant = _layers()
        layers[0] = torch.nn.Module()
        with self.assertRaisesRegex(ValueError, "writer on the same pipeline rank"):
            _bind_shared_fp8_scales(layers, quant)

    def test_flashinfer_reader_consumer_gets_writer_scales_without_writing_cache(self):
        layers, quant = _layers()
        _bind_shared_fp8_scales(layers, quant)
        writer = layers[0].self_attn.attn
        writer.k_scale.data.fill_(0.125)
        writer.v_scale.data.fill_(0.25)
        for decoder in layers:
            layer = decoder.self_attn.attn
            layer.quant_method.process_weights_after_loading(layer)
        reader = layers[1].self_attn.attn
        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        backend.num_wrappers = 1
        backend.decode_uses_dequant_workspace = False
        backend.token_to_kv_pool = MagicMock()
        wrapper = MagicMock()
        q = torch.zeros(1, reader.tp_q_head_num, reader.head_dim)
        wrapper.forward.return_value = q
        backend.forward_metadata = SimpleNamespace(decode_wrappers=[wrapper])
        # Only the external kernel/pool boundary is replaced; execute the actual
        # backend consumer on CPU without claiming CUDA kernel correctness.
        backend.forward_decode(
            q, None, None, reader, SimpleNamespace(), save_kv_cache=False
        )
        self.assertEqual(wrapper.forward.call_args.kwargs["k_scale"], 0.125)
        self.assertEqual(wrapper.forward.call_args.kwargs["v_scale"], 0.25)
        backend.token_to_kv_pool.get_kv_buffer.assert_called_once_with(writer.layer_id)
        backend.token_to_kv_pool.set_kv_buffer.assert_not_called()

    def test_unquantized_and_nvfp4_bindings_remain_unchanged(self):
        for quant in (None, SimpleNamespace(kv_cache_quant_algo="NVFP4")):
            layers, _ = _layers()
            original = [layer.self_attn.attn.quant_method for layer in layers]
            _bind_shared_fp8_scales(layers, quant)
            self.assertEqual(
                original, [layer.self_attn.attn.quant_method for layer in layers]
            )


if __name__ == "__main__":
    unittest.main()
