"""CPU regression test: the Gemma-4 vision tower must load unquantized.

`google/gemma-4-31B-it-qat-w4a16-ct` lists every vision-tower linear in
`quantization_config.ignore`, but the checkpoint wraps each tower `nn.Linear`
in a clip module, so the entries carry a trailing `.linear`
(`...mlp.gate_proj.linear`). SGLang keeps that suffix for the unfused clippable
linears, but its fused `qkv_proj` / `gate_up_proj` do not have it, and the
compressed-tensors fused mapping expands them to `...q_proj` / `...gate_proj`,
which never match the `.linear` entries. Those two layers per block are
therefore quantized, and the fused `gate_up_proj` (2 * 4304 = 8608) is not
divisible by Marlin's min_thread_n = 64, so load crashes in
`gptq_marlin_repack`.

The model now drops a compressed-tensors config for the vision tower. These
tests build the real vision-tower modules on CPU against a two-layer replica of
the checkpoint's config -- the real vision geometry, and `ignore` entries in
the checkpoint's own spelling -- and pin that contract. No weights are loaded
and no kernels run.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import os
import socket
import unittest
from unittest import mock

from transformers import Gemma4Config

import sglang.srt.models.gemma4_mm as gemma4_mm
import sglang.srt.models.gemma4_vision as gemma4_vision
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.test.test_utils import CustomTestCase

# Real shapes from google/gemma-4-31B-it-qat-w4a16-ct's vision_config. The
# intermediate_size is what makes the fused gate_up_proj 8608 wide.
VISION_HIDDEN_SIZE = 1152
VISION_INTERMEDIATE_SIZE = 4304
VISION_LAYERS = 2

# The checkpoint's ignore entries, in the checkpoint's own spelling: every
# tower linear is listed with the trailing ".linear" of the clip wrapper.
TOWER_IGNORE = ["model.vision_tower.patch_embedder.input_proj"] + [
    f"model.vision_tower.encoder.layers.{i}.{sub}.linear"
    for i in range(VISION_LAYERS)
    for sub in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
]
IGNORE = TOWER_IGNORE + ["model.embed_vision.embedding_projection", "lm_head"]

# group_0 targets "Linear", i.e. everything that is not ignored is quantized.
W4A16_QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "ignore": IGNORE,
    "config_groups": {
        "group_0": {
            "format": "pack-quantized",
            "targets": ["Linear"],
            "input_activations": None,
            "output_activations": None,
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
                "dynamic": False,
            },
        }
    },
}


def _gemma4_config() -> Gemma4Config:
    """A Gemma-4 config with the real vision geometry and a stub text tower."""
    text_config = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 1024,
    }
    vision_config = {
        "hidden_size": VISION_HIDDEN_SIZE,
        "intermediate_size": VISION_INTERMEDIATE_SIZE,
        "num_hidden_layers": VISION_LAYERS,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 72,
        "patch_size": 16,
    }
    return Gemma4Config(text_config=text_config, vision_config=vision_config)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init_single_rank_cpu_parallelism() -> None:
    """TP=PP=1 on gloo: the linear layers need the parallel groups to exist."""
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if model_parallel_is_initialized():
        return
    # A free port, not a fixed one: CI may run several test files in parallel
    # on the same host.
    port = int(os.environ.get("SGLANG_TEST_MASTER_PORT") or _free_port())
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)


class _StubMMConfig:
    mm_attention_backend = "sdpa"
    mm_enable_dp_encoder = False


def _build_model(quant_config):
    """Build the real vision tower through Gemma4ForConditionalGeneration.

    The text model is stubbed out: it needs kernels that are not available on
    CPU and it is not what these tests are about. Everything on the vision path
    is the real thing.
    """
    with mock.patch.object(
        CompressedTensorsConfig, "_check_scheme_supported", return_value=True
    ), mock.patch.object(
        gemma4_vision, "get_mm", lambda: _StubMMConfig()
    ), mock.patch.object(
        gemma4_mm, "Gemma4TextModel"
    ), mock.patch.object(
        gemma4_mm, "LogitsProcessor"
    ), mock.patch.object(
        Gemma4ForConditionalGeneration, "post_init"
    ):
        model = Gemma4ForConditionalGeneration(
            config=_gemma4_config(), quant_config=quant_config, prefix=""
        )
        return model, gemma4_mm.Gemma4TextModel


def _build_vision_tower(quant_config):
    return _build_model(quant_config)[0].vision_tower


def _linear_methods(module):
    return {
        name: type(layer.quant_method).__name__
        for name, layer in module.named_modules()
        if isinstance(layer, LinearBase)
    }


class TestGemma4TowerQuantization(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _init_single_rank_cpu_parallelism()

    def test_vision_tower_is_unquantized_under_w4a16(self):
        quant_config = CompressedTensorsConfig.from_config(W4A16_QUANT_CONFIG)
        tower = _build_vision_tower(quant_config)

        methods = _linear_methods(tower)
        self.assertTrue(methods, "no linear layers found in the vision tower")
        quantized = {
            name: method
            for name, method in methods.items()
            if method != UnquantizedLinearMethod.__name__
        }
        self.assertEqual(
            quantized,
            {},
            "vision-tower linears must not be quantized",
        )

    def test_fused_gate_up_proj_is_the_marlin_unfriendly_shape(self):
        # Pins why this matters: the layer that used to reach gptq_marlin_repack
        # is 8608 wide, which is not divisible by Marlin's min_thread_n = 64.
        tower = _build_vision_tower(None)
        gate_up = tower.encoder.layers[0].mlp.gate_up.gate_up_proj
        self.assertEqual(gate_up.output_size, 2 * VISION_INTERMEDIATE_SIZE)
        self.assertEqual(gate_up.output_size % 64, 32)

    def test_other_quant_methods_are_passed_through(self):
        # Only compressed-tensors is dropped. Anything else keeps whatever
        # behaviour it had; we have no checkpoint evidence to de-quantize it.
        other = object()
        self.assertIs(
            Gemma4ForConditionalGeneration._vision_tower_quant_config(other), other
        )
        self.assertIsNone(
            Gemma4ForConditionalGeneration._vision_tower_quant_config(None)
        )

    def test_language_model_still_gets_the_quant_config(self):
        quant_config = CompressedTensorsConfig.from_config(W4A16_QUANT_CONFIG)
        _, text_model_cls = _build_model(quant_config)

        args, _ = text_model_cls.call_args
        self.assertIs(args[1], quant_config)

    def test_ignore_list_alone_does_not_save_the_tower(self):
        # Negative control for the fix: the checkpoint's ignore entries do not
        # match SGLang's module names, so passing the quant config down would
        # quantize the tower. This is the behaviour the fix routes around.
        from sglang.srt.layers.quantization.compressed_tensors.utils import (
            should_ignore_layer,
        )

        packed = Gemma4ForConditionalGeneration.packed_modules_mapping
        self.assertFalse(
            should_ignore_layer(
                "model.vision_tower.encoder.layers.0.mlp.gate_up_proj",
                ignore=IGNORE,
                fused_mapping=packed,
            )
        )
        # ... while an entry without the clip wrapper does match.
        self.assertTrue(
            should_ignore_layer(
                "model.embed_vision.embedding_projection",
                ignore=IGNORE,
                fused_mapping=packed,
            )
        )


if __name__ == "__main__":
    unittest.main()
