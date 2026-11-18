"""Quark PTPC-FP8 Muse Spark on ROCm: whole-model weight loading.

Builds the real ROCm Muse model at tiny sizes (on CPU tensors), loads a synthetic
checkpoint with the release's tensor names through ``load_weights`` and checks
what each TP = EP rank holds: rope-permuted q/k rows with their scales, EP-local
expert banks, and the load checks that refuse missing or malformed pieces. The
ROCm model imports the ROCm kernel libraries, so this runs on the AMD runner;
the checkpoint contract and the projection loaders are covered on CPU by
``test_quark_ptpc_fp8.py``.
"""

import copy
import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.muse_spark import MuseSparkTextConfig
from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.srt.rocm_muse_spark import model as muse
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.muse_spark_fp8_moe import FP8, quantize_rows
from sglang.test.parallel_groups import parallel_scope, publish
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, stage="base-b", runner_config="1-gpu-small-amd-mi35x")

# Quantizers as the release config.json spells them (Quark 0.12.post1).
_WEIGHT = {
    "dtype": "fp8_e4m3",
    "is_dynamic": False,
    "qscheme": "per_channel",
    "ch_axis": 0,
    "group_size": None,
    "block_size": None,
    "symmetric": True,
    "round_method": "half_even",
    "scale_type": "float",
    "zero_point_type": "int32",
    "scale_format": None,
    "scale_calculation_mode": None,
    "mx_element_dtype": None,
    "observer_cls": "PerChannelMinMaxObserver",
    "is_scale_quant": False,
    "enable_buffer_reuse": False,
    "max_input_numel": 4194304,
}
_INPUT = {**_WEIGHT, "is_dynamic": True, "ch_axis": 1}
NUM_LAYERS = 2
NUM_EXPERTS = 8
HEADS, KV_HEADS, HEAD_DIM, HIDDEN = 8, 2, 16, 64
MOE_HIDDEN, MOE_INTER = 32, 32
ATTN = "model.language_model.layers.{}.self_attn"
EXPERT = "model.language_model.layers.{}.mlp.experts.{}.{}"


def release_quant_config(num_layers: int = 62) -> dict:
    """The release's quantization_config: global PTPC, BF16 router / pooler / lm_head."""
    exclude = ["lm_head"]
    for layer in range(num_layers):
        exclude += [
            f"model.language_model.layers.{layer}.mlp.{m}"
            for m in ("gate", "post_expert_proj", "pre_expert_proj")
        ]
    exclude += ["model.audio_tower.proj", "model.vision_tower.encoder.layers.0.mlp.fc1"]
    return {
        "global_quant_config": {
            "input_tensors": dict(_INPUT),
            "output_tensors": None,
            "weight": dict(_WEIGHT),
            "bias": None,
            "target_device": None,
        },
        "exclude": exclude,
        "algo_config": None,
        "softmax_quant_spec": None,
        "quant_method": "quark",
        "layer_type_quant_config": {},
        "layer_quant_config": {},
        "kv_cache_quant_config": {},
        "kv_cache_post_rope": False,
        "quant_mode": "eager_mode",
        "version": "0.12.post1+f7d8cef",
        "export": {
            "kv_cache_group": [],
            "min_kv_scale": 0.0,
            "pack_method": "reorder",
            "weight_format": "real_quantized",
            "weight_merge_groups": None,
        },
    }


def quark(config: dict) -> QuarkConfig:
    config = copy.deepcopy(config)
    config["packed_modules_mapping"] = (
        muse.MuseSparkTextForCausalLM.packed_modules_mapping
    )
    return QuarkConfig(quant_config=config, is_prequantized=True)


def expert_leaves(layer: int, experts) -> list[str]:
    return [
        EXPERT.format(layer, e, p)
        for e in experts
        for p in ("gate_proj", "up_proj", "down_proj")
    ]


def rope_rows(heads: int) -> torch.Tensor:
    """Checkpoint row of each model q/k row: per head, row 2j+i holds stored row i*D/2+j."""
    j = torch.arange(HEAD_DIM // 2).repeat_interleave(2)
    i = torch.arange(2).repeat(HEAD_DIM // 2)
    within = i * (HEAD_DIM // 2) + j
    return (torch.arange(heads).unsqueeze(1) * HEAD_DIM + within).reshape(-1)


def dequant(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return codes.float() * scale.unsqueeze(-1)


def synthetic_checkpoint(seed: int = 0, bf16_expert_layers=()):
    """Release-named FP8 attention and expert tensors for every layer, as quantized codes + scales."""
    gen = torch.Generator().manual_seed(seed)
    tensors = {}

    def fp8(name, rows, cols):
        codes, scale = quantize_rows(torch.randn(rows, cols, generator=gen))
        tensors[f"{name}.weight"], tensors[f"{name}.weight_scale"] = codes, scale

    for layer in range(NUM_LAYERS):
        attn = ATTN.format(layer)
        fp8(f"{attn}.q_proj", HEADS * HEAD_DIM, HIDDEN)
        fp8(f"{attn}.k_proj", KV_HEADS * HEAD_DIM, HIDDEN)
        fp8(f"{attn}.v_proj", KV_HEADS * HEAD_DIM, HIDDEN)
        fp8(f"{attn}.gate_proj", HEADS * HEAD_DIM, HIDDEN)
        fp8(f"{attn}.o_proj", HIDDEN, HEADS * HEAD_DIM)
        for e in range(NUM_EXPERTS):
            for proj, rows, cols in (
                ("gate_proj", MOE_INTER, MOE_HIDDEN),
                ("up_proj", MOE_INTER, MOE_HIDDEN),
                ("down_proj", MOE_HIDDEN, MOE_INTER),
            ):
                name = EXPERT.format(layer, e, proj)
                if layer in bf16_expert_layers:
                    tensors[f"{name}.weight"] = torch.randn(
                        rows, cols, generator=gen
                    ).bfloat16()
                else:
                    fp8(name, rows, cols)
    return tensors


class _TinyModel:
    """Builds the ROCm model at tiny sizes for one TP = EP rank."""

    def __init__(
        self, test: CustomTestCase, tp_size: int, rank: int, kv_heads=KV_HEADS
    ):
        self.test, self.tp_size, self.rank = test, tp_size, rank
        self.config = MuseSparkTextConfig(
            hidden_size=HIDDEN,
            head_dim=HEAD_DIM,
            num_attention_heads=HEADS,
            num_key_value_heads=kv_heads,
            num_hidden_layers=NUM_LAYERS,
            num_local_experts=NUM_EXPERTS,
            num_experts_per_tok=2,
            moe_hidden_size=MOE_HIDDEN,
            moe_intermediate_size=MOE_INTER,
            intermediate_size=HIDDEN,
            vocab_size=128,
            sliding_window=8,
        )
        # NoPE layers: the loader is what is under test, not the rotary kernels.
        self.config.layer_rope_theta = [0.0] * NUM_LAYERS

    def build(self, quant_config):
        reset_context()
        self.test.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy",
                device="cpu",
                tp_size=self.tp_size,
                ep_size=self.tp_size,
            ),
            role="test",
            ranks=SpawnRanks(world_rank=self.rank),
        )
        with ExitStack() as stack:
            # QuarkW8A8Fp8 needs FP8-capable hardware; this is the device boundary.
            stack.enter_context(
                patch(
                    "sglang.srt.layers.quantization.quark.quark.get_device_capability",
                    return_value=(9, 5),
                )
            )
            stack.enter_context(
                parallel_scope(
                    tp_group=SimpleNamespace(
                        rank_in_group=self.rank,
                        world_size=self.tp_size,
                        cpu_group=None,
                    )
                )
            )
            return muse.MuseSparkTextForCausalLM(self.config, quant_config)


@unittest.skipUnless(is_hip(), "ROCm Muse model")
class TestPtpcFp8Loading(CustomTestCase):
    def load(self, tp_size, rank, tensors, *, kv_heads=KV_HEADS, config=None):
        tiny = _TinyModel(self, tp_size, rank, kv_heads=kv_heads)
        model = tiny.build(quark(config or release_quant_config(NUM_LAYERS)))
        # The GPU-only post-load preparations of the BF16 router / pooler stay off on CPU. Ulysses stays enabled
        # (TP > 1): preparing it would all-gather the FP8 attention weights through the metadata-only TP group.
        with (
            patch.object(muse.router_pooler, "JOINT_ROUTER_POOLER", False),
            patch.object(muse, "_DENSE_TILED", False),
            patch.object(muse, "_PREFILL_DENSE_FP8", False),
        ):
            self.assertEqual(muse.ulysses_enabled(), tp_size > 1)
            model.load_weights(
                list(tensors.items()) if isinstance(tensors, dict) else list(tensors)
            )
        return model

    def test_attention_rows_and_scales_per_tp_rank(self):
        """q/k scales take the rope row permutation with their rows; q/k/v/gate shard per rank, KV replicas
        included; o_proj shards K and keeps every output scale."""
        for tp_size, kv_heads in ((2, 2), (4, 2), (8, 1)):
            tensors = synthetic_checkpoint(seed=tp_size)
            if kv_heads != KV_HEADS:
                for layer in range(NUM_LAYERS):
                    for proj in ("k_proj", "v_proj"):
                        name = f"{ATTN.format(layer)}.{proj}"
                        for suffix in (".weight", ".weight_scale"):
                            tensors[name + suffix] = tensors[name + suffix][
                                : kv_heads * HEAD_DIM
                            ]
            replicas = max(1, tp_size // kv_heads)
            q_rows = HEADS * HEAD_DIM // tp_size
            kv_rows = max(1, kv_heads // tp_size) * HEAD_DIM
            for rank in range(tp_size):
                with self.subTest(tp=tp_size, rank=rank):
                    model = self.load(tp_size, rank, tensors, kv_heads=kv_heads)
                    attn = model.model.layers[1].self_attn
                    name = ATTN.format(1)

                    def full(proj, heads=None):
                        w = dequant(
                            tensors[f"{name}.{proj}.weight"],
                            tensors[f"{name}.{proj}.weight_scale"],
                        )
                        return w if heads is None else w[rope_rows(heads)]

                    q_slice = slice(rank * q_rows, (rank + 1) * q_rows)
                    kv_start = (rank // replicas) * kv_rows
                    kv_slice = slice(kv_start, kv_start + kv_rows)
                    expected = torch.cat(
                        [
                            full("q_proj", HEADS)[q_slice],
                            full("k_proj", kv_heads)[kv_slice],
                            full("v_proj")[kv_slice],
                            full("gate_proj")[q_slice],
                        ]
                    )
                    self.assertEqual(attn.qkv_proj.weight.dtype, FP8)
                    torch.testing.assert_close(
                        dequant(attn.qkv_proj.weight, attn.qkv_proj.weight_scale),
                        expected,
                        rtol=0,
                        atol=0,
                    )
                    k_width = HEADS * HEAD_DIM // tp_size
                    o_codes = tensors[f"{name}.o_proj.weight"]
                    torch.testing.assert_close(
                        attn.o_proj.weight.view(torch.uint8),
                        o_codes[:, rank * k_width : (rank + 1) * k_width].view(
                            torch.uint8
                        ),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        attn.o_proj.weight_scale, tensors[f"{name}.o_proj.weight_scale"]
                    )
                    # Default decode / prefill reads of the BF16 fast-path copies must find "none".
                    self.assertIsNone(attn.qkv_proj.small_m_fp8_weight)
                    self.assertIsNone(attn.o_proj.small_m_fp8_weight)
                    self.assertFalse(muse.ulysses_prepared(attn))

    def test_ep_local_experts_and_scales(self):
        tensors = synthetic_checkpoint(seed=7)
        tp_size = 4
        local = NUM_EXPERTS // tp_size
        for rank in range(tp_size):
            with self.subTest(rank=rank):
                model = self.load(tp_size, rank, tensors)
                experts = model.model.layers[1].mlp.experts
                self.assertIsInstance(experts, muse.MuseSpark1xFp8Experts)
                for i in range(local):

                    def leaf(proj, suffix, expert=rank * local + i):
                        return tensors[f"{EXPERT.format(1, expert, proj)}.{suffix}"]

                    torch.testing.assert_close(
                        dequant(experts.w13_weight[i], experts.w13_weight_scale[i]),
                        torch.cat(
                            [
                                dequant(
                                    leaf("gate_proj", "weight"),
                                    leaf("gate_proj", "weight_scale"),
                                ),
                                dequant(
                                    leaf("up_proj", "weight"),
                                    leaf("up_proj", "weight_scale"),
                                ),
                            ]
                        ),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        dequant(experts.w2_weight[i], experts.w2_weight_scale[i]),
                        dequant(
                            leaf("down_proj", "weight"),
                            leaf("down_proj", "weight_scale"),
                        ),
                        rtol=0,
                        atol=0,
                    )

    def test_missing_or_malformed_pieces_fail_the_load(self):
        """Holes a checkpoint can leave must not load as valid data: an unloaded scale keeps its sentinel, a
        weight without its scale (or the reverse) is unpaired, and BF16 values never land in FP8 parameters."""
        tp_size, rank = 4, 1  # rank 1 owns experts 2 and 3
        base = synthetic_checkpoint(seed=3)
        q_scale = ATTN.format(0) + ".q_proj.weight_scale"
        local_scale = EXPERT.format(1, 3, "down_proj") + ".weight_scale"
        local_weight = EXPERT.format(1, 2, "gate_proj") + ".weight"
        remote = EXPERT.format(1, 6, "down_proj")

        def without(*names):
            return {k: v for k, v in base.items() if k not in names}

        cases = {
            "local expert scale": (
                without(local_scale, local_scale.replace("_scale", "")),
                RuntimeError,
                "scales missing",
            ),
            "local expert weight": (without(local_weight), RuntimeError, "unpaired"),
            "attention scale": (
                without(
                    ATTN.format(1) + ".k_proj.weight_scale",
                    ATTN.format(1) + ".k_proj.weight",
                ),
                RuntimeError,
                "scales missing",
            ),
            "bf16 into fp8": (
                {**base, local_weight: base[local_weight].float().bfloat16()},
                ValueError,
                "does not match parameter dtype",
            ),
            "scalar q scale": (
                {**base, q_scale: base[q_scale][:1].reshape(())},
                ValueError,
                "expected 1-D tensor",
            ),
            "short v scale": (
                {
                    **base,
                    ATTN.format(0) + ".v_proj.weight_scale": base[
                        ATTN.format(0) + ".v_proj.weight_scale"
                    ][:-1],
                },
                ValueError,
                "expected 32 rows",
            ),
            "unexpected scale suffix": (
                {**base, ATTN.format(0) + ".q_proj.weight_scale_inv": base[q_scale]},
                ValueError,
                "outside the Muse Spark text model",
            ),
            # The same local expert piece twice with different values: the second copy must not win silently.
            "duplicate local expert weight": (
                list(base.items()) + [(local_weight, base[remote + ".weight"].clone())],
                RuntimeError,
                "loaded twice",
            ),
        }
        for label, (tensors, error, fragment) in cases.items():
            with self.subTest(label):
                with self.assertRaisesRegex(error, fragment):
                    self.load(tp_size, rank, tensors)
        # A non-local expert's tensors are skipped, so their absence is not an error.
        self.load(tp_size, rank, without(remote + ".weight", remote + ".weight_scale"))

    def test_whole_layer_expert_exclusion_loads_bf16_experts(self):
        config = release_quant_config(NUM_LAYERS)
        config["exclude"] += expert_leaves(0, range(NUM_EXPERTS))
        tensors = synthetic_checkpoint(seed=5, bf16_expert_layers=(0,))
        model = self.load(2, 1, tensors, config=config)
        layers = model.model.layers
        self.assertIsInstance(layers[0].mlp.experts, muse.MuseSpark1xNativeExperts)
        self.assertIsInstance(layers[1].mlp.experts, muse.MuseSpark1xFp8Experts)
        torch.testing.assert_close(
            layers[0].mlp.experts.w2_weight[0].bfloat16(),
            tensors[EXPERT.format(0, 4, "down_proj") + ".weight"],
            rtol=0,
            atol=0,
        )

    def test_bf16_checkpoint_attention_stays_unquantized(self):
        model = _TinyModel(self, 2, 0).build(None)
        attn = model.model.layers[0].self_attn
        self.assertFalse(attn.fp8_serialized)
        self.assertEqual(attn.qkv_proj.weight.dtype, torch.get_default_dtype())
        self.assertEqual(
            type(attn.qkv_proj.quant_method).__name__, "UnquantizedLinearMethod"
        )
        self.assertIsInstance(
            model.model.layers[0].mlp.experts, muse.MuseSpark1xNativeExperts
        )

    def test_mxfp4_checkpoint_keeps_its_experts(self):
        config = release_quant_config(NUM_LAYERS)
        for role in ("weight", "input_tensors"):
            config["global_quant_config"][role].update(
                dtype="fp4",
                qscheme="per_group",
                group_size=32,
                scale_format="e8m0",
                ch_axis=-1,
            )
        config["exclude"] += [
            ATTN.format(layer) + f".{p}"
            for layer in range(NUM_LAYERS)
            for p in ("q_proj", "k_proj", "v_proj", "gate_proj", "o_proj")
        ]
        model = _TinyModel(self, 2, 0).build(quark(config))
        layer = model.model.layers[0]
        self.assertIsInstance(layer.mlp.experts, muse.MuseSpark1xMxfp4Experts)
        self.assertFalse(layer.self_attn.fp8_serialized)

    def test_fp8_experts_need_the_triton_backend(self):
        """MUSE_MOE_BACKEND=flydsl / native have no FP8 expert path; auto resolves to Triton on ROCm."""
        for backend in ("flydsl", "native"):
            with self.subTest(backend=backend):
                with patch.dict(os.environ, {"MUSE_MOE_BACKEND": backend}):
                    with self.assertRaisesRegex(ValueError, "has no FP8 path"):
                        muse.MuseSpark1xFp8Experts(
                            num_local_experts=2,
                            expert_hidden_size=MOE_HIDDEN,
                            intermediate_size=MOE_INTER,
                        )
        from sglang.kernels.ops.moe import fused_moe_triton_kernels as wrapper

        # Backend selection only: an unpadded wrapper, whatever SGLANG_MOE_PADDING this runner sets.
        with (
            patch.dict(os.environ, {"MUSE_MOE_BACKEND": "auto"}),
            patch.object(wrapper, "padding_size", 0),
        ):
            muse.MuseSpark1xFp8Experts(
                num_local_experts=2,
                expert_hidden_size=MOE_HIDDEN,
                intermediate_size=MOE_INTER,
            )

    def test_fp8_experts_refuse_fnuz_fp8(self):
        """gfx942-class GPUs use FNUZ FP8; the serialized OCP e4m3fn codes would need a conversion this path lacks."""
        from sglang.srt.rocm_muse_spark import muse_moe_fp8

        # Hardware capability boundary: the query answers for the GPU, not for the checkpoint.
        with patch.object(muse_moe_fp8, "is_fp8_fnuz", return_value=True):
            with self.assertRaisesRegex(ValueError, "need OCP FP8 e4m3fn"):
                muse.MuseSpark1xFp8Experts(
                    num_local_experts=2,
                    expert_hidden_size=MOE_HIDDEN,
                    intermediate_size=MOE_INTER,
                )

    def test_fp8_experts_refuse_padded_k_without_aiter(self):
        """Without aiter the generic MoE wrapper subtracts SGLANG_MOE_PADDING from K; Muse weights are unpadded."""
        from sglang.kernels.ops.moe import fused_moe_triton_kernels as wrapper

        with (
            patch.object(wrapper, "_use_aiter", False),
            patch.object(wrapper, "padding_size", 128),
        ):
            with self.assertRaisesRegex(ValueError, "not K-padded"):
                muse.MuseSpark1xFp8Experts(
                    num_local_experts=2,
                    expert_hidden_size=MOE_HIDDEN,
                    intermediate_size=MOE_INTER,
                )


if __name__ == "__main__":
    unittest.main()
