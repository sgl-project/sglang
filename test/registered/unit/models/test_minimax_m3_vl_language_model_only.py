# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""`--language-model-only` on MiniMax-M3-VL.

The flag serves the text half of a multimodal checkpoint without constructing
the vision tower or assigning vision weights. `handle_language_model_only`
only admits architectures listed in
`ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES`, so the allowlist entry and the
model-side skip belong to the same surface and are checked together here.
"""

import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.arg_groups.model_hook import handle_language_model_only
from sglang.srt.configs.minimax_vl import MiniMaxM3VLConfig
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.models import minimax_m3_vl
from sglang.srt.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration
from sglang.srt.models.minimax_vl_common import MiniMaxVLVisionModel
from sglang.srt.runtime_context import SpawnRanks, get_parallel
from sglang.srt.runtime_context import publish as publish_runtime
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

HIDDEN = 2
VIT_PREFIX = "vision_tower.vision_model.encoder.layers.0.self_attn."
QKV_PARAM = VIT_PREFIX + "qkv_proj.weight"
PROJ_PARAM = VIT_PREFIX + "proj.weight"
EMBED_PARAM = "model.embed_tokens.weight"


def _resolve_language_model_only(architectures):
    """Run just `handle_language_model_only`, since ServerArgs' real constructor
    pulls in a model config."""
    sa = ServerArgs(model_path="dummy")
    msgspec.Struct.__setattr__(sa, "language_model_only", True)
    setattr(
        sa,
        "_model_config",
        SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=architectures,
                model_type="minimax_m3_vl",
            ),
        ),
    )
    handle_language_model_only(sa)


def _raised(call):
    """Return what `call` raised, so a missing guard shows up as a failed
    assertion on the error instead of an unrelated exception."""
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - the assertion inspects it
        return exc
    return None


def _vision_tower_stub():
    """The two vision parameters the M3-VL loader writes: the fused qkv the
    checkpoint's separate q/k/v are merged into, and the attention output."""
    self_attn = nn.Module()
    self_attn.qkv_proj = nn.Module()
    self_attn.qkv_proj.weight = nn.Parameter(
        torch.zeros(3 * HIDDEN, HIDDEN), requires_grad=False
    )
    self_attn.proj = nn.Module()
    self_attn.proj.weight = nn.Parameter(
        torch.zeros(HIDDEN, HIDDEN), requires_grad=False
    )
    layer = nn.Module()
    layer.self_attn = self_attn
    encoder = nn.Module()
    encoder.layers = nn.ModuleList([layer])
    vision_model = nn.Module()
    vision_model.encoder = encoder
    tower = nn.Module()
    tower.vision_model = vision_model
    return tower


def _model_stub(*, language_model_only, vision_tower):
    model = object.__new__(MiniMaxM3SparseForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model_only = language_model_only
    model.use_data_parallel = False
    # Dense, unquantized stand-in: no expert mapping and no Quark scale aliasing.
    model.quant_config = None
    model.num_fused_shared_experts = 0
    model.config = SimpleNamespace(
        encoder_only=False,
        text_config=SimpleNamespace(
            sparse_attention_config=None,
            num_local_experts=0,
        ),
    )
    inner = nn.Module()
    inner.embed_tokens = nn.Module()
    inner.embed_tokens.weight = nn.Parameter(torch.zeros(2, 2), requires_grad=False)
    inner.start_layer = 0
    inner.end_layer = 1
    model.model = inner
    model.vision_tower = vision_tower
    return model


VISION_CONFIG = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_channels=3,
    image_size=28,
    patch_size=14,
    hidden_act="gelu",
    layer_norm_eps=1e-6,
    img_token_compression_config={"temporal_patch_size": 2, "spatial_merge_size": 2},
    position_embedding_type="rope",
    rope_mode="3d",
    rope_theta=1e4,
    vision_segment_max_frames=None,
)


def _hf_config(*, language_model_only):
    """A checkpoint the size of a unit test: a one-layer tower, and a text half
    whose rope scaling carries `mrope_section`."""
    return SimpleNamespace(
        encoder_only=False,
        language_model_only=language_model_only,
        vision_config=dict(VISION_CONFIG),
        projector_hidden_size=None,
        text_config=SimpleNamespace(
            hidden_size=16,
            vocab_size=64,
            rope_theta=1e4,
            rope_scaling={"rope_type": "mrope", "mrope_section": [2, 1, 1]},
        ),
    )


def _checkpoint_weights():
    q = torch.full((HIDDEN, HIDDEN), 1.0)
    k = torch.full((HIDDEN, HIDDEN), 2.0)
    v = torch.full((HIDDEN, HIDDEN), 3.0)
    out = torch.full((HIDDEN, HIDDEN), 4.0)
    embed = torch.full((2, 2), 5.0)
    weights = [
        (VIT_PREFIX + "q_proj.weight", q),
        (VIT_PREFIX + "k_proj.weight", k),
        (VIT_PREFIX + "v_proj.weight", v),
        (VIT_PREFIX + "out_proj.weight", out),
        ("language_model." + EMBED_PARAM, embed),
    ]
    return weights, SimpleNamespace(q=q, k=k, v=v, out=out, embed=embed)


class TestMiniMaxM3VLLanguageModelOnlyAllowlist(CustomTestCase):
    def test_minimax_m3_vl_is_admitted(self):
        exc = _raised(
            lambda: _resolve_language_model_only(
                ["MiniMaxM3SparseForConditionalGeneration"]
            )
        )
        self.assertIsNone(exc)

    def test_an_unlisted_architecture_is_still_refused(self):
        exc = _raised(
            lambda: _resolve_language_model_only(["Qwen2_5_VLForConditionalGeneration"])
        )
        self.assertIn("--language-model-only does not support", str(exc))
        self.assertIsInstance(exc, ValueError)


class TestMiniMaxM3VLLanguageModelOnlyWeights(CustomTestCase):
    def test_the_tower_is_filled_when_the_flag_is_off(self):
        """Baseline: the same checkpoint names do reach the vision parameters,
        so the skip below is the flag's doing and not a naming mismatch."""
        model = _model_stub(
            language_model_only=False, vision_tower=_vision_tower_stub()
        )
        weights, w = _checkpoint_weights()

        model.load_weights(weights)

        params = dict(model.named_parameters())
        self.assertTrue(
            torch.equal(params[QKV_PARAM], torch.cat([w.q, w.k, w.v], dim=0))
        )
        self.assertTrue(torch.equal(params[PROJ_PARAM], w.out))
        self.assertTrue(torch.equal(params[EMBED_PARAM], w.embed))

    def test_vision_weights_are_skipped_under_the_flag(self):
        # A real --language-model-only run builds no tower at all; the tower
        # parameters are kept registered here so that "not written" is visible.
        model = _model_stub(language_model_only=True, vision_tower=_vision_tower_stub())
        weights, w = _checkpoint_weights()

        model.load_weights(weights)

        params = dict(model.named_parameters())
        self.assertTrue(torch.equal(params[QKV_PARAM], torch.zeros(3 * HIDDEN, HIDDEN)))
        self.assertTrue(torch.equal(params[PROJ_PARAM], torch.zeros(HIDDEN, HIDDEN)))
        self.assertTrue(torch.equal(params[EMBED_PARAM], w.embed))

    def test_a_checkpoint_without_a_tower_loads_the_language_half(self):
        model = _model_stub(language_model_only=True, vision_tower=None)
        weights, w = _checkpoint_weights()

        model.load_weights(weights)

        params = dict(model.named_parameters())
        self.assertEqual(set(params), {EMBED_PARAM})
        self.assertTrue(torch.equal(params[EMBED_PARAM], w.embed))


class TestMiniMaxM3VLLanguageModelOnlyConstruction(CustomTestCase):
    """The real `__init__` on CPU. Only the language-side builders are replaced,
    so whether a tower exists is decided by the code under test."""

    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy",
                device="cpu",
                tp_size=1,
                # Pin the vision backend: CPU CI must not pick an accelerator one.
                mm_attention_backend="sdpa",
            ),
            role="test",
        )
        # publish() leaves the pipeline group unset; __init__ reads is_last_rank.
        get_parallel().override_permanently(
            pp_group=SimpleNamespace(
                is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
            )
        )
        for builder in ("MiniMaxM3Model", "ParallelLMHead", "LogitsProcessor"):
            self.enterContext(
                patch.object(minimax_m3_vl, builder, return_value=nn.Identity())
            )

    def _build(self, *, language_model_only):
        return MiniMaxM3SparseForConditionalGeneration(
            _hf_config(language_model_only=language_model_only)
        )

    def test_the_tower_is_not_built_under_the_flag(self):
        model = self._build(language_model_only=True)

        self.assertIsNone(model.vision_tower)
        self.assertFalse(model.is_mrope_enabled)

    def test_the_tower_and_mrope_survive_with_the_flag_off(self):
        """Baseline for the check above: the same config does build a real tower,
        and mrope stays on, when the flag is off."""
        model = self._build(language_model_only=False)

        self.assertIsInstance(model.vision_tower, MiniMaxVLVisionModel)
        self.assertTrue(model.is_mrope_enabled)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestMiniMaxM3VLLanguageModelOnlyParameters(CustomTestCase):
    """Build both complete tiny models with real single-rank parallel groups."""

    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        original_dtype = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, original_dtype)
        torch.set_default_dtype(torch.bfloat16)
        server = ServerArgs(
            model_path="dummy", device="cuda", tp_size=1, mm_attention_backend="sdpa"
        )
        publish_runtime(server, role="test", ranks=SpawnRanks(world_rank=0, gpu_id=0))
        torch.cuda.set_device(0)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        self.addCleanup(destroy_distributed_environment)
        initialize_model_parallel()
        self.addCleanup(destroy_model_parallel)
        initialize_dp_attention(server)

    def _parameters(self, *, language_model_only):
        text = PretrainedConfig(
            model_type="minimax_m2",
            hidden_size=128,
            vocab_size=512,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            hidden_act="silu",
            dense_intermediate_size=256,
            intermediate_size=256,
            moe_layer_freq=[0],
            num_local_experts=0,
            n_shared_experts=0,
            rms_norm_eps=1e-6,
            max_position_embeddings=64,
            rope_theta=10000.0,
            rope_scaling=None,
            pad_token_id=0,
            tie_word_embeddings=False,
            sparse_attention_config=None,
            use_qk_norm=False,
            qk_norm_type="per_layer",
            use_gemma_norm=False,
        )
        config = MiniMaxM3VLConfig(
            vision_config=dict(VISION_CONFIG),
            text_config=text,
            architectures=["MiniMaxM3SparseForConditionalGeneration"],
            encoder_only=False,
            language_model_only=language_model_only,
            projector_hidden_size=None,
        )
        with torch.device("cuda"):
            model = MiniMaxM3SparseForConditionalGeneration(config)
        return {name: (tuple(p.shape), p.dtype) for name, p in model.named_parameters()}

    def test_only_vision_parameters_are_removed(self):
        full = self._parameters(language_model_only=False)
        text_only = self._parameters(language_model_only=True)
        self.assertTrue(any(name.startswith("vision_tower.") for name in full))
        self.assertEqual(
            text_only,
            {
                name: info
                for name, info in full.items()
                if not name.startswith("vision_tower.")
            },
        )


if __name__ == "__main__":
    unittest.main()
