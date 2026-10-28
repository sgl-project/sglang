"""M3.1 config resolution, DSpark draft wiring and the deferred MoE handoff."""

import copy
import os
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.arg_groups.model_overrides.minimax_m3 import _minimax_m3_overrides
from sglang.srt.configs.minimax_vl import (
    MiniMaxM3VLConfig,
    configure_minimax_m3_dspark,
)
from sglang.srt.configs.model_config import (
    ModelConfig,
    is_dspark_draft,
    uses_minimax_m3_packed_nvfp4_kv,
)
from sglang.srt.distributed.device_communicators import custom_all_reduce_ops
from sglang.srt.environ import envs
from sglang.srt.layers.layer_boundary.fusions.moe_finalize import MoeDeferredFinalize
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.models import minimax_m3
from sglang.srt.models.dspark_minimax import DSparkMiniMaxDraftModel
from sglang.srt.models.minimax_common import minimax_weight_loader
from sglang.srt.models.minimax_m3 import (
    MiniMaxM3Model,
    get_spec_layer_idx_from_weight_name,
)
from sglang.srt.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration
from sglang.srt.runtime_context import override_platform
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

ARCH = ["MiniMaxM3SparseForConditionalGeneration"]
NVFP4 = SimpleNamespace(
    architectures=ARCH,
    quantization_config={"quant_method": "minimax_m3_nvfp4"},
    num_key_value_heads=4,
)
MXFP8 = SimpleNamespace(
    quantization_config={"quant_method": "mxfp8"}, num_key_value_heads=4
)


QUARK_M31 = SimpleNamespace(
    quantization_config={
        "quant_method": "quark",
        "layer_quant_config": {
            "language_model.model.layers.0.self_attn.q_proj": {
                "weight": {
                    "dtype": "fp8_e4m3",
                    "qscheme": "per_block",
                    "block_size": [1, 32],
                    "scale_type": "float8_e8m0fnu",
                    "is_dynamic": False,
                }
            }
        },
    },
    num_key_value_heads=4,
)
QUARK_M3 = SimpleNamespace(
    quantization_config={"quant_method": "quark", "layer_quant_config": {}},
    num_key_value_heads=4,
)


def draft_config():
    """A small DSpark draft in the layout of the released checkpoint's dspark/ folder."""
    config = MiniMaxM3VLConfig.from_dict(
        {
            "architectures": ["DSparkMiniMaxDraftModel"],
            "quantization_config": {"quant_method": "minimax_m3_nvfp4"},
            "text_config": {
                "model_type": "",
                "num_hidden_layers": 2,
                "dflash_config": {
                    "num_hidden_layers": 2,
                    "num_target_layers": 6,
                    "target_layer_ids": [1, 5],
                    "layer_types": ["sliding_attention"] * 2,
                },
                "dspark_use_dense_ffn": True,
                "dspark_ffn_hidden_size": 256,
                "dspark_block_size": 7,
                "dspark_markov_rank": 16,
                "dspark_noise_token_id": 1,
                "dspark_target_layer_ids": [1, 5],
            },
        }
    )
    # from_dict keeps text_config a dict; tests set draft fields as attributes.
    config.text_config = PretrainedConfig(**config.text_config)
    return config


def server_args(**kwargs):
    args = dict(
        quantization=None,
        _quantization_explicitly_unset=False,
        speculative_algorithm=None,
        moe_a2a_backend="none",
        moe_runner_backend="auto",
        page_size=None,
        kv_cache_dtype="auto",
        attention_backend=None,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        pp_size=1,
        speculative_draft_kv_cache_dtype=None,
        speculative_draft_attention_backend=None,
        dtype="auto",
    )
    return SimpleNamespace(**(args | kwargs))


def packed_moe_hf(**quant_overrides):
    quant = {
        "quant_method": "mxfp8",
        "moe_quant_algo": "NVFP4",
        "moe_quant_format": "nvfp4-pack-quantized",
        "group_size": 16,
    }
    # None drops the field, matching checkpoints that do not carry it.
    quant = {k: v for k, v in (quant | quant_overrides).items() if v is not None}
    return SimpleNamespace(architectures=ARCH, quantization_config=quant)


class TestMiniMaxM31ConfigResolution(CustomTestCase):
    def test_nvfp4_defaults_on_blackwell_and_portable_fallback(self):
        base = dict(quantization="minimax_m3_nvfp4", page_size=128)
        blackwell = base | dict(
            moe_a2a_backend="megamoe",
            moe_runner_backend="deep_gemm",
            kv_cache_dtype="fp8_e4m3",
            attention_backend="trtllm_mha",
        )
        cases = [
            (True, None, blackwell),
            (
                True,
                "DSPARK",
                blackwell
                | dict(
                    speculative_draft_kv_cache_dtype="bf16",
                    speculative_draft_attention_backend="trtllm_mha",
                ),
            ),
            # MegaMoE and the packed NVFP4 KV cache are SM100-only.
            (
                False,
                None,
                base | dict(moe_runner_backend="marlin", attention_backend="fa3"),
            ),
        ]
        for is_blackwell, algorithm, expected in cases:
            with (
                self.subTest(blackwell=is_blackwell, algorithm=algorithm),
                # pin a CUDA host: gfx950 is a supported non-Blackwell platform
                override_platform(is_blackwell=is_blackwell, is_hip=False),
                patch.dict("os.environ", SGLANG_RAGGED_VERIFY_MODE="static"),
            ):
                overrides = _minimax_m3_overrides(
                    server_args(speculative_algorithm=algorithm), NVFP4
                )
                self.assertEqual(overrides, expected)

    def test_packed_moe_mxfp8_resolves_like_nvfp4_and_plain_mxfp8_does_not(self):
        for blackwell in (True, False):
            with (
                self.subTest(blackwell=blackwell),
                override_platform(is_blackwell=blackwell, is_hip=False),
            ):
                self.assertEqual(
                    _minimax_m3_overrides(server_args(), packed_moe_hf()),
                    _minimax_m3_overrides(server_args(), NVFP4),
                )
        for quant_overrides in (
            # The fully MXFP8 checkpoints of this family carry no MoE fields.
            dict(moe_quant_algo=None, moe_quant_format=None),
            dict(moe_quant_format="nvfp4-quantized"),
        ):
            with (
                self.subTest(**quant_overrides),
                override_platform(is_blackwell=True, is_hip=False),
            ):
                overrides = _minimax_m3_overrides(
                    server_args(), packed_moe_hf(**quant_overrides)
                )
                self.assertEqual(overrides["quantization"], "mxfp8")
                self.assertNotIn("moe_a2a_backend", overrides)
        with override_platform(is_blackwell=True, is_hip=False):
            overrides = _minimax_m3_overrides(
                server_args(quantization="mxfp8"), packed_moe_hf()
            )
        self.assertNotIn("quantization", overrides)
        self.assertNotIn("moe_a2a_backend", overrides)

    def test_mxfp8_block_convert_flag_has_a_base_default(self):
        """Configs outside the Fp8 hierarchy also set use_mxfp8 and reach the Fp8
        methods, which read this flag without a defensive lookup."""
        from sglang.srt.layers.quantization.base_config import QuantizationConfig

        self.assertIsNone(QuantizationConfig.convert_mxfp8_to_block_fp8)

    def test_unsupported_combinations_are_rejected(self):
        def capture_ids(ids):
            config = draft_config()
            config.text_config.dflash_config["target_layer_ids"] = ids
            return lambda: configure_minimax_m3_dspark(config)

        def resolve(**kwargs):
            return lambda: _minimax_m3_overrides(server_args(**kwargs), NVFP4)

        cases = {
            "capture id past the target depth": ("static", capture_ids([1, 6])),
            "duplicate capture id": ("static", capture_ids([2, 2])),
            "compact verify": ("compact", resolve(speculative_algorithm="DSPARK")),
            "dspark pp": ("static", resolve(speculative_algorithm="DSPARK", pp_size=2)),
            "eagle": ("static", resolve(speculative_algorithm="EAGLE")),
            "text-only dspark": (
                "static",
                lambda: _minimax_m3_overrides(
                    server_args(speculative_algorithm="DSPARK"),
                    SimpleNamespace(
                        architectures=["MiniMaxM3SparseForCausalLM"],
                        quantization_config=NVFP4.quantization_config,
                    ),
                ),
            ),
        }
        for name, (verify_mode, call) in cases.items():
            with (
                self.subTest(name),
                override_platform(is_blackwell=True, is_hip=False),
                patch.dict("os.environ", SGLANG_RAGGED_VERIFY_MODE=verify_mode),
                self.assertRaises(ValueError),
            ):
                call()


def rocm_server_args(**kwargs):
    rocm = dict(
        tp_size=4,
        attn_dp_size=1,
        attn_cp_size=1,
        ep_size=1,
        enable_aiter_allreduce_fusion=False,
    )
    return server_args(**(rocm | kwargs))


def gfx950():
    return override_platform(
        is_blackwell=False, is_sm100=False, is_hip=True, is_gfx95=True
    )


class TestMiniMaxM31Rocm(CustomTestCase):
    def test_rocm_caps_nvfp4_tp_and_quark_attention_tp(self):
        """Past one KV head per rank the server serves NUL bytes instead of failing, and
        NVFP4 at TP8 serves runaway whitespace even with the attention TP divided."""
        with gfx950():
            for attn_dp_size in (1, 2):
                with (
                    self.subTest(attn_dp_size=attn_dp_size),
                    self.assertRaisesRegex(ValueError, "--tp-size 4 or less"),
                ):
                    _minimax_m3_overrides(
                        rocm_server_args(tp_size=8, attn_dp_size=attn_dp_size), NVFP4
                    )
            quark = dict(quantization="quark_mxfp4", tp_size=8)
            with self.assertRaisesRegex(ValueError, "4 KV heads"):
                _minimax_m3_overrides(rocm_server_args(**quark), MXFP8)
            _minimax_m3_overrides(rocm_server_args(attn_dp_size=2, **quark), MXFP8)

    def test_blackwell_keeps_attention_tp_above_the_kv_heads(self):
        """Blackwell replicates KV heads past one per rank; the ROCm guard must not fire there."""
        with override_platform(is_blackwell=True, is_sm100=True, is_hip=False):
            _minimax_m3_overrides(rocm_server_args(tp_size=8), NVFP4)

    def test_quark_mxfp4_keeps_custom_all_reduce(self):
        """RCCL slows the Quark path's decode on gfx950, where it was measured; MXFP8 and
        other ROCm GPUs keep the default."""
        with gfx950(), envs.SGLANG_M3_ALLOW_CUSTOM_AR.override(False):
            quark = _minimax_m3_overrides(
                rocm_server_args(quantization="quark_mxfp4"), MXFP8
            )
            mxfp8 = _minimax_m3_overrides(rocm_server_args(quantization="mxfp8"), MXFP8)
        with (
            override_platform(
                is_blackwell=False, is_sm100=False, is_hip=True, is_gfx95=False
            ),
            envs.SGLANG_M3_ALLOW_CUSTOM_AR.override(False),
        ):
            quark_gfx942 = _minimax_m3_overrides(
                rocm_server_args(quantization="quark_mxfp4"), MXFP8
            )
        self.assertTrue(quark_gfx942["disable_custom_all_reduce"])
        self.assertNotIn("disable_custom_all_reduce", quark)
        self.assertTrue(mxfp8["disable_custom_all_reduce"])

    def test_m31_quark_export_gets_the_nvfp4_gfx950_defaults(self):
        """The M3.1 Quark export shares the NVFP4 checkpoint's MXFP8 linears and MXFP4
        experts, so it resolves the same gfx950 defaults; the M3 Quark export (bf16
        linears) keeps the MiniMax-M3 ones."""
        block, level = (
            "SGLANG_FORCE_MXFP8_BLOCK_CONVERT_DENSE",
            "ROCM_QUICK_REDUCE_QUANTIZATION",
        )
        with (
            gfx950(),
            envs.SGLANG_M3_ALLOW_CUSTOM_AR.override(False),
            patch.dict("os.environ", SGLANG_RAGGED_VERIFY_MODE="static"),
        ):
            os.environ.pop(block, None)
            os.environ[level] = "INT8"
            spec = dict(speculative_algorithm="DSPARK")
            m3 = _minimax_m3_overrides(rocm_server_args(**spec), QUARK_M3)
            self.assertNotIn(block, os.environ)
            self.assertEqual(os.environ[level], "INT8")
            m31 = _minimax_m3_overrides(rocm_server_args(**spec), QUARK_M31)
            self.assertTrue(envs.SGLANG_FORCE_MXFP8_BLOCK_CONVERT_DENSE.get())
            self.assertEqual(os.environ[level], "INT4")
            os.environ[block] = "0"
            _minimax_m3_overrides(rocm_server_args(**spec), QUARK_M31)
            self.assertFalse(envs.SGLANG_FORCE_MXFP8_BLOCK_CONVERT_DENSE.get())
            explicit = _minimax_m3_overrides(
                rocm_server_args(kv_cache_dtype="bf16"), QUARK_M31
            )
        self.assertTrue(m3["disable_custom_all_reduce"])
        self.assertNotIn("kv_cache_dtype", m3)
        self.assertNotIn("disable_custom_all_reduce", m31)
        self.assertEqual(m31["kv_cache_dtype"], "fp8_e4m3")
        self.assertEqual(m31["speculative_draft_kv_cache_dtype"], "bf16")
        self.assertNotIn("kv_cache_dtype", explicit)
        with gfx950(), self.assertRaisesRegex(ValueError, "4 KV heads"):
            _minimax_m3_overrides(rocm_server_args(tp_size=8), QUARK_M31)

    def test_gfx950_resolves_the_portable_stack(self):
        """CUDA-only backends (fa3, trtllm_mha, MegaMoE) would not start on gfx950."""
        with (
            gfx950(),
            patch.dict("os.environ", SGLANG_RAGGED_VERIFY_MODE="static"),
        ):
            overrides = _minimax_m3_overrides(
                rocm_server_args(speculative_algorithm="DSPARK"), NVFP4
            )
        self.assertEqual(overrides["attention_backend"], "triton")
        self.assertEqual(overrides["speculative_draft_attention_backend"], "triton")
        # the dispatcher keeps global expert ids (EP) only for aiter
        self.assertEqual(overrides["moe_runner_backend"], "aiter")

    def test_gfx950_kv_defaults_to_nvfp4_only_under_dspark(self):
        """Only the DSPARK verify indexer reads an NVFP4 index K: the default keeps plain
        decode on fp8, fp8 opts DSPARK out, and an unknown format fails at startup."""
        cases = {
            (None, "DSPARK"): True,
            (None, None): False,
            (" FP8", "DSPARK"): False,
        }
        key = "SGLANG_MINIMAX_M3_AMD_KV_FORMAT"
        for (kv_format, algorithm), packed in cases.items():
            with (
                self.subTest(kv_format=kv_format, algorithm=algorithm),
                gfx950(),
                patch.dict("os.environ"),
            ):
                os.environ.pop(key, None)
                if kv_format is not None:
                    os.environ[key] = kv_format
                self.assertEqual(
                    uses_minimax_m3_packed_nvfp4_kv(
                        "minimax_m3_nvfp4", speculative_algorithm=algorithm
                    ),
                    packed,
                )
        with (
            gfx950(),
            patch.dict("os.environ", {key: "nvfp4"}),
            self.assertRaisesRegex(ValueError, "must be auto or fp8"),
        ):
            uses_minimax_m3_packed_nvfp4_kv(
                "minimax_m3_nvfp4", speculative_algorithm="DSPARK"
            )

    def test_gfx950_quick_all_reduce_takes_int4_over_the_image_default(self):
        """ROCm images export INT8 for every model: M3.1 still gets INT4, while another
        level the user chose (FP, INT6, NONE) survives, and so does INT8 when the user
        turns the INT4 default off."""
        level = "ROCM_QUICK_REDUCE_QUANTIZATION"
        int4 = envs.SGLANG_ENABLE_MINIMAX_M3_AMD_QUICK_REDUCE_INT4
        cases = (
            (None, True, "INT4"),
            ("INT8", True, "INT4"),
            ("FP", True, "FP"),
            ("INT8", False, "INT8"),
            (None, False, None),
        )
        for preset, enabled, resolved in cases:
            with (
                self.subTest(preset=preset, enabled=enabled),
                gfx950(),
                patch.dict("os.environ"),
                int4.override(enabled),
            ):
                os.environ.pop(level, None)
                if preset is not None:
                    os.environ[level] = preset
                _minimax_m3_overrides(rocm_server_args(), NVFP4)
                self.assertEqual(os.environ.get(level), resolved)

    def test_gfx950_jit_quick_reduce_takes_mid_sizes_only_under_int4(self):
        """The JIT quick-reduce replaces only the INT4 one the override turns on at TP4, and
        only then takes the mid-size all-reduces off custom all-reduce; a build or a cap the
        user set stays."""
        cap, jit = "AITER_CUSTOM_AR_MAX_SIZE", "SGLANG_ROCM_QUICK_ALL_REDUCE_JIT"
        level = "ROCM_QUICK_REDUCE_QUANTIZATION"
        with gfx950(), patch.dict("os.environ"):
            os.environ.pop(cap, None)
            os.environ.pop(jit, None)
            os.environ[level] = "FP"
            _minimax_m3_overrides(rocm_server_args(), NVFP4)
            self.assertNotIn(jit, os.environ)
            self.assertNotIn(cap, os.environ)
            os.environ.pop(level)
            os.environ[jit] = "0"
            _minimax_m3_overrides(rocm_server_args(), NVFP4)
            self.assertFalse(envs.SGLANG_ROCM_QUICK_ALL_REDUCE_JIT.get())
            self.assertNotIn(cap, os.environ)
            del os.environ[jit]
            _minimax_m3_overrides(rocm_server_args(), NVFP4)
            self.assertTrue(envs.SGLANG_ROCM_QUICK_ALL_REDUCE_JIT.get())
            self.assertEqual(os.environ[cap], "2516582")
            os.environ[cap] = "1048576"
            _minimax_m3_overrides(rocm_server_args(), NVFP4)
            self.assertEqual(os.environ[cap], "1048576")
        # an fp16 model, or QuickAllReduce without the fp16 cast, is not the JIT build's
        cast = "ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"
        for dtype, cast_value in (("float16", "1"), ("auto", "0")):
            with (
                self.subTest(dtype=dtype, cast=cast_value),
                gfx950(),
                patch.dict("os.environ", {cast: cast_value}),
            ):
                for name in (cap, jit, level):
                    os.environ.pop(name, None)
                _minimax_m3_overrides(rocm_server_args(dtype=dtype), NVFP4)
                self.assertNotIn(jit, os.environ)
                self.assertNotIn(cap, os.environ)

    def test_jit_quick_reduce_is_gfx950_only(self):
        """The JIT build is validated on gfx950 only; a user-set flag elsewhere keeps
        sgl-kernel's quick all-reduce."""
        uses_jit = custom_all_reduce_ops._uses_jit_quick_all_reduce
        with (
            patch.object(custom_all_reduce_ops, "_is_hip", True),
            envs.SGLANG_ROCM_QUICK_ALL_REDUCE_JIT.override(True),
        ):
            for gfx95, expected in ((True, True), (False, False)):
                with (
                    self.subTest(gfx95=gfx95),
                    patch.object(
                        custom_all_reduce_ops, "is_gfx95_supported", return_value=gfx95
                    ),
                ):
                    self.assertEqual(uses_jit(), expected)


class TestMiniMaxM31DSparkDraft(CustomTestCase):
    def test_draft_target_ids_are_zero_based_and_the_input_is_kept(self):
        config = draft_config()
        before = copy.deepcopy(config.to_dict())
        obj = object.__new__(ModelConfig)
        obj.is_draft_model = True
        obj.hf_config = config
        obj.hf_text_config = config.text_config
        obj.speculative_algorithm = "DSPARK"
        obj.is_draft_quantization_explicit = False
        obj.quantization = None
        obj._config_draft_model()
        self.assertTrue(is_dspark_draft(obj.hf_config))
        self.assertEqual(obj.hf_config.num_hidden_layers, 2)
        parsed = parse_dflash_draft_config(draft_hf_config=obj.hf_config)
        self.assertEqual(parsed.resolve_target_layer_ids(target_num_layers=6), [1, 5])
        self.assertEqual(config.to_dict(), before)

    def test_verify_window_comes_from_the_draft_checkpoint(self):
        from sglang.srt.arg_groups.overrides import resolution_result
        from sglang.srt.arg_groups.speculative_hook import _handle_dspark
        from sglang.srt.server_args import ServerArgs

        config = draft_config()
        config.text_config.dspark_block_size = 5
        args = ServerArgs(
            model_path="dummy",
            speculative_algorithm="DSPARK",
            speculative_draft_model_path="dummy-draft",
            device="cuda",
        )
        args._model_config = SimpleNamespace(hf_config=config)
        with patch(
            "sglang.srt.utils.hf_transformers_utils.get_config", return_value=config
        ):
            _handle_dspark(args)
        self.assertEqual(
            resolution_result(args, "speculative_draft_model_path"), "dummy-draft"
        )
        self.assertEqual(resolution_result(args, "speculative_num_draft_tokens"), 6)

    def test_loader_requires_every_checkpoint_weight_except_runtime_scales(self):
        model = object.__new__(DSparkMiniMaxDraftModel)
        nn.Module.__init__(model)
        model.fc = nn.Linear(2, 2, bias=False)
        model.confidence_head = None
        model.k_scale = nn.Parameter(torch.tensor(-1.0))
        model.k_scale._skip_weight_check = True
        with self.assertRaisesRegex(ValueError, "Missing.*fc.weight"):
            model.load_weights([])
        with self.assertRaisesRegex(ValueError, "Unexpected DSpark"):
            model.load_weights([("dspark.unknown.weight", torch.ones(2))])
        model.attn = nn.Module()
        model.attn.qkv_proj = nn.Linear(2, 6, bias=False)
        model.attn.qkv_proj.weight.weight_loader = lambda param, weight, shard: None
        fc = [("language_model.model.dspark.fc.weight", torch.ones(2, 2))]
        shards = [(f"dspark.attn.{s}_proj.weight", torch.ones(2, 2)) for s in "qkv"]
        with self.assertRaisesRegex(ValueError, "Missing DSpark checkpoint shards"):
            model.load_weights(fc + shards[:2])
        self.assertEqual(
            model.load_weights(fc + shards), {"fc.weight", "attn.qkv_proj.weight"}
        )

    def test_target_loader_skips_bundled_draft_weights(self):
        self.assertEqual(
            get_spec_layer_idx_from_weight_name(
                SimpleNamespace(num_hidden_layers=60), "model.dspark.fc.weight"
            ),
            60,
        )


class TestMiniMaxM31QuarkScaleNames(CustomTestCase):
    """Quark 0.14+eef70b3 names the MXFP8 [1, 32] linear scale weight_scale."""

    def test_alias_redirects_only_an_unmatched_scale(self):
        p = nn.Parameter(torch.zeros(1), requires_grad=False)
        params = {
            "a.qkv_proj.weight_scale_inv": p,
            "b.weight_scale": p,
            "b.weight_scale_inv": p,
        }
        alias = minimax_weight_loader.mxfp8_linear_scale_alias
        self.assertEqual(
            alias("a.qkv_proj.weight_scale", params), "a.qkv_proj.weight_scale_inv"
        )
        self.assertEqual(alias("b.weight_scale", params), "b.weight_scale")
        self.assertEqual(alias("c.weight_scale", params), "c.weight_scale")
        self.assertEqual(alias("a.qkv_proj.weight", params), "a.qkv_proj.weight")

    def test_multimodal_loader_puts_quark_scales_into_weight_scale_inv(self):
        model = object.__new__(MiniMaxM3SparseForConditionalGeneration)
        nn.Module.__init__(model)
        model.model = SimpleNamespace(start_layer=0, end_layer=1)
        model.num_fused_shared_experts = 0
        model.quant_config = None
        model.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=1))
        loaded = {}

        def scale_param(key):
            param = nn.Parameter(torch.zeros(2, dtype=torch.uint8), requires_grad=False)
            param.weight_loader = lambda p, w, shard=None: loaded.__setitem__(
                key, (shard, w)
            )
            return param

        params = {
            "model.layers.0.self_attn.qkv_proj.weight_scale_inv": scale_param("qkv"),
            "model.layers.0.self_attn.o_proj.weight_scale_inv": scale_param("o"),
        }
        mapping = [(".qkv_proj", ".k_proj", "k")]
        k = torch.full((2,), 127, dtype=torch.uint8)
        o = torch.full((2,), 126, dtype=torch.uint8)
        prefix = "model.layers.0.self_attn."
        model._load_llm_weight(prefix + "k_proj.weight_scale", k, params, mapping, [])
        model._load_llm_weight(prefix + "o_proj.weight_scale", o, params, mapping, [])
        self.assertEqual(loaded["qkv"][0], "k")
        self.assertTrue(torch.equal(loaded["qkv"][1], k))
        self.assertIsNone(loaded["o"][0])
        self.assertTrue(torch.equal(loaded["o"][1], o))


class CaptureLayer(nn.Module):
    """Adds ``1 + offset`` to the residual stream, writing it with nothing pending,
    so a capture at the layer's entry reads the full stream value."""

    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset

    def forward(
        self, *, hidden_states, forward_batch, captured_last_layer_outputs, **kwargs
    ):
        if captured_last_layer_outputs is not None:
            captured_last_layer_outputs.capture(hidden_states)
        return residual_batch.set_written(
            hidden_states + 1 + self.offset, forward_batch
        )


class CaptureNorm(nn.Module):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states + 100
        return hidden_states + residual + 100, residual


class TestMiniMaxM31Forward(CustomTestCase):
    def test_capture_includes_final_layer_before_output_norm(self):
        model = object.__new__(MiniMaxM3Model)
        nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        # The first layer's offset stands in for a residual already on the stream.
        model.layers = nn.ModuleList(
            [CaptureLayer(offset=10), CaptureLayer(), CaptureLayer()]
        )
        model.norm = CaptureNorm()
        model.finalize_norm = None
        model.start_layer, model.end_layer = 0, 3
        model.layers_to_capture = []
        wrapper = SimpleNamespace(pp_group=model.pp_group, model=model)
        MiniMaxM3SparseForConditionalGeneration.set_dspark_layers_to_capture(
            wrapper, [0, 2]
        )
        fb = SimpleNamespace(can_run_tbo=False)
        with patch(
            "sglang.srt.models.minimax_m3.check_cuda_graph_backend",
            return_value=True,
        ):
            output, captured = model(
                torch.tensor([0]), torch.tensor([0]), fb, input_embeds=torch.zeros(1, 2)
            )
        torch.testing.assert_close(captured[0], torch.full((1, 2), 11.0))
        torch.testing.assert_close(captured[1], torch.full((1, 2), 13.0))
        torch.testing.assert_close(output, torch.full((1, 2), 113.0))
        self.assertEqual(len(captured), 2)

    def test_deferred_moe_hands_unreduced_expert_output_to_the_boundary(self):
        x = torch.ones(2, 8, dtype=torch.bfloat16)
        shared = x * 2
        deferred = SimpleNamespace(
            gemm2_out=torch.ones(4, 8, dtype=torch.bfloat16),
            expert_weights=torch.ones(2, 2),
            expanded_idx_to_permuted_idx=torch.arange(4, dtype=torch.int32).view(2, 2),
            top_k=2,
        )
        # Not callable: a finalizing expert call would raise instead of passing.
        experts = SimpleNamespace(forward_deferred_finalize=lambda h, topk: deferred)
        moe = SimpleNamespace(
            alt_stream=None,
            shared_experts=object(),
            _forward_shared_experts=lambda h: shared,
            _compute_router_logits=lambda h: None,
            topk=lambda h, logits: None,
            experts=experts,
        )
        moe._forward_router_experts = partial(
            minimax_m3.MiniMaxM3MoE._forward_router_experts, moe
        )
        with patch.object(
            minimax_m3,
            "get_forward",
            return_value=SimpleNamespace(defer_moe_finalize=True),
        ):
            result = minimax_m3.MiniMaxM3MoE.forward_normal(moe, x)
        self.assertIsInstance(result, MoeDeferredFinalize)
        self.assertIs(result.gated_shared_output, shared)
        self.assertEqual(result.routed_output.data_ptr(), deferred.gemm2_out.data_ptr())

    def test_norms_bind_the_gemm_reading_their_fp8(self):
        """The input norm's fp8 feeds the fused qkv + index GEMM, the FFN norm's the
        shared expert's gate_up (the dense gate_up in dense layers); under attention DP
        the norms keep bf16, since the DP gather after them takes a tensor."""
        from sglang.srt.layers.layernorm import GemmaRMSNorm
        from sglang.srt.models.minimax_common.amd import minimax_m3_hip

        def proj(n):
            return SimpleNamespace(weight=torch.empty(n, 64))

        def bound(sparse, shared):
            layer = SimpleNamespace(
                input_layernorm=GemmaRMSNorm(64),
                post_attention_layernorm=GemmaRMSNorm(64),
                self_attn=SimpleNamespace(_fused_qkv_index=proj(2560)),
                is_layer_sparse=sparse,
                mlp=SimpleNamespace(
                    gate_up_proj=proj(6144),
                    shared_experts=(
                        SimpleNamespace(gate_up_proj=proj(1536)) if shared else None
                    ),
                ),
            )
            minimax_m3_hip.Gfx950NormQuant(layer).bind()
            return (
                layer.input_layernorm.fp8_out_gemm_shape,
                layer.post_attention_layernorm.fp8_out_gemm_shape,
            )

        a2a = SimpleNamespace(is_megamoe=lambda: False, is_deepep=lambda: False)
        with patch.object(minimax_m3_hip, "get_moe_a2a_backend", lambda: a2a):
            self.assertEqual(bound(True, True), ((2560, 64), (1536, 64)))
            self.assertEqual(bound(False, False), ((2560, 64), (6144, 64)))
            self.assertEqual(bound(True, False), ((2560, 64), None))
        with (
            envs.SGLANG_USE_AITER.override(True),
            patch.object(minimax_m3_hip, "is_dp_attention_enabled", lambda: True),
        ):
            self.assertIsNone(minimax_m3_hip.norm_quant(object()))

    def test_fused_cache_store_leaves_a_scaled_index_k_to_the_pool(self):
        """The fused qk-norm/rope store writes index K unscaled; a non-unit idx_k scale
        must reach the pool's store, which divides by it before the fp8 cast."""
        from sglang.kernels.ops.attention import minimax_m3_qk_norm_rope
        from sglang.srt.models.minimax_common.amd import minimax_m3_hip

        fp8 = torch.float8_e4m3fn
        q = torch.empty(2, 128, dtype=torch.bfloat16)

        def stored_by_fusion(idx_k_scale, index_dtype):
            pool = SimpleNamespace(
                dtype=fp8,
                index_k_pool=SimpleNamespace(dtype=index_dtype),
                get_kv_buffer=lambda layer_id: (None, None),
                get_index_k_buffer=lambda layer_id: None,
            )
            attn = SimpleNamespace(
                _get_sparse_kv_pool=lambda: pool,
                _enable_hisparse=False,
                _can_use_rocm_sparse_qk_index_norm_rope=lambda *args: True,
                _sparse_qk_index_norm_rope=lambda *args: ("unfused",) * 4,
                _mark_sparse_kv_cached_by_fusion=lambda *args: None,
                attn=SimpleNamespace(
                    layer_id=0,
                    idx_k_scale_float=idx_k_scale,
                    k_scale_float=None,
                    v_scale_float=None,
                ),
                q_norm=SimpleNamespace(weight=q, variance_epsilon=1e-6),
                k_norm=SimpleNamespace(weight=q),
                index_q_norm=SimpleNamespace(weight=q),
                index_k_norm=SimpleNamespace(weight=q),
                rotary_emb=SimpleNamespace(cos_sin_cache=q, is_neox_style=True),
                head_dim=128,
                rotary_dim=64,
            )
            batch = SimpleNamespace(out_cache_loc=torch.zeros(2, dtype=torch.int64))
            with patch.object(
                minimax_m3_qk_norm_rope,
                "sparse_qk_index_gemma_rmsnorm_rope_cache",
                lambda *args, **kwargs: ("fused",) * 4,
            ):
                out = minimax_m3_hip.sparse_qk_index_norm_rope_cache(
                    attn, None, q, q, q, q, q, None, batch
                )
            return out[0] == "fused"

        self.assertTrue(stored_by_fusion(None, fp8))
        self.assertTrue(stored_by_fusion(1.0, fp8))
        self.assertFalse(stored_by_fusion(0.5, fp8))
        # a bf16 index cache ignores the scale on both paths
        self.assertTrue(stored_by_fusion(0.5, torch.bfloat16))


class TestMiniMaxM31Nvfp4Backend(CustomTestCase):
    def test_backend_uses_resolved_verify_window(self):
        from sglang.srt.layers.attention import minimax_sparse_nvfp4_backend as backend
        from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool
        from sglang.srt.runtime_context import get_context

        pool = object.__new__(MiniMaxM3Nvfp4KVPool)
        pool.page_size, pool.head_num = 128, 1
        pool.sparse_layer_id_mapping = {0: 0}
        config = SimpleNamespace(
            num_attention_heads=64,
            num_key_value_heads=4,
            head_dim=128,
            sparse_attention_config={
                "sparse_index_dim": 128,
                "sparse_block_size": 128,
                "sparse_topk_blocks": 16,
                "sparse_num_index_heads": 4,
            },
        )
        runner = SimpleNamespace(
            token_to_kv_pool=pool,
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(1, 128)),
            model_config=SimpleNamespace(
                context_len=128, hf_config=config, hf_text_config=config
            ),
            server_args=SimpleNamespace(speculative_num_draft_tokens=None),
        )
        with (
            get_context().override_server_args(speculative_num_draft_tokens=8),
            patch.object(
                backend, "msa_v1_qmul4_decode_capability", return_value=(True, None)
            ),
        ):
            attention = backend.MiniMaxSparseNvfp4AttnBackend(runner)
        self.assertEqual(attention.speculative_num_draft_tokens, 8)

    def test_verify_replans_after_prefix_growth_without_mutating_input_lengths(self):
        from sglang.srt.layers.attention import minimax_sparse_nvfp4_backend as backend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        class Plan:
            def __init__(self, **kwargs):
                self.query_length = kwargs["query_length"]
                self.fp8_attention = None
                self.page_table = self.final_seq_lens = None
                self.refreshes = 0

            def refresh(
                self,
                *,
                attention_plan,
                req_to_token,
                req_pool_indices,
                seq_lens,
                is_target_verify,
            ):
                self.refreshes += 1
                offset = self.query_length if is_target_verify else 0
                self.final_seq_lens = seq_lens.to(torch.int32) + offset

        obj = object.__new__(backend.MiniMaxSparseNvfp4AttnBackend)
        obj.speculative_num_draft_tokens = 8
        obj.page_size = obj.idx_head_dim = 128
        obj.req_to_token = torch.arange(1024).reshape(2, 512)
        obj._msa_v1_attention_workspace = None
        obj._msa_v1_qmul4_available = True
        obj._msa_v1_num_q_heads, obj._msa_v1_num_kv_heads = 32, 2
        obj._msa_v1_num_index_heads = 2
        obj._msa_v1_index_fp8_scratch = None
        obj._msa_v1_cuda_graph_plans = {}
        obj._msa_v1_cuda_graph_attention_plans = {}
        obj.forward_metadata = SimpleNamespace()
        fb = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            seq_lens=torch.tensor([126, 250]),
            req_pool_indices=torch.tensor([0, 1]),
        )
        with (
            patch.object(backend, "MSAV1DecodeIndexerPlan", Plan),
            patch.object(backend, "MSAV1DecodeAttentionPlan", Plan),
        ):
            first = obj._prepare_msa_v1_decode_plan(
                fb, use_cuda_graph=True, max_seq_len=512
            )
            self.assertEqual(first.query_length, 8)
            torch.testing.assert_close(
                first.final_seq_lens, torch.tensor([134, 258], dtype=torch.int32)
            )
            torch.testing.assert_close(fb.seq_lens, torch.tensor([126, 250]))
            fb.seq_lens += 3
            second = obj._prepare_msa_v1_decode_plan(
                fb, use_cuda_graph=True, max_seq_len=512
            )
            self.assertIs(first, second)
            torch.testing.assert_close(
                second.final_seq_lens, torch.tensor([137, 261], dtype=torch.int32)
            )
            fb.forward_mode = ForwardMode.DECODE
            decode = obj._prepare_msa_v1_decode_plan(
                fb, use_cuda_graph=True, max_seq_len=512
            )
            self.assertEqual(decode.query_length, 1)
            torch.testing.assert_close(
                decode.final_seq_lens, fb.seq_lens.to(torch.int32)
            )
        # One refresh per prepare, on the cached plan.
        self.assertEqual((first.refreshes, decode.refreshes), (2, 1))


if __name__ == "__main__":
    unittest.main()
