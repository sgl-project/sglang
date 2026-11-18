"""Muse Spark Quark PTPC-FP8 attention projections on gfx950.

The ROCm Muse model serves serialized FP8 q/k/v/o-gate and o projections with the
generic Quark W8A8 FP8 linear (per-token dynamic activation quantization, one
static scale per output channel). These cases build the model's own projection
layers at the release's TP8 rank shapes (QKV + gate N = 2304, K = 8192; O
N = 8192, K = 1024), load full FP8 tensors through the layer loaders, run the
Quark post-processing and check that the weight stays FP8 (no BF16 dequant
fallback) and that the GEMM matches an fp32 PTPC reference.
"""

import itertools
import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.muse_spark_fp8_moe import (
    FP8,
    exceeds,
    gemm_row_bounds,
    quantize_rows,
    quantizer_report,
)
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=45, stage="base-b", runner_config="1-gpu-small-amd-mi35x")

TP, RANK = 8, 3
HEADS, KV_HEADS, HEAD_DIM, HIDDEN = 128, 16, 64, 8192
ROWS = (1, 16, 17, 128, 129, 2048)
_QUANTIZER = {
    "dtype": "fp8_e4m3",
    "qscheme": "per_channel",
    "symmetric": True,
    "group_size": None,
    "block_size": None,
    "scale_format": None,
    "scale_type": "float",
    "mx_element_dtype": None,
    "is_scale_quant": False,
}


def _tokens(m: int, k: int, gen: torch.Generator) -> torch.Tensor:
    """Activation rows with magnitudes spread over 0.01..100."""
    magnitude = 10.0 ** (torch.rand(m, 1, generator=gen) * 4 - 2)
    return (torch.randn(m, k, generator=gen) * magnitude).cuda().bfloat16()


def _weight(rows: int, cols: int):
    """FP8 codes and scales of a weight whose output channels span 0.1..10 in magnitude."""
    gen = torch.Generator().manual_seed(rows + cols)
    spread = 10.0 ** (torch.rand(rows, 1, generator=gen) * 2 - 1)
    return quantize_rows(
        (torch.randn(rows, cols, generator=gen) * 0.02 * spread).cuda()
    )


def _quark_config():
    from sglang.srt.layers.quantization.quark.quark import QuarkConfig

    return QuarkConfig(
        quant_config={
            "global_quant_config": {
                "weight": {**_QUANTIZER, "ch_axis": 0, "is_dynamic": False},
                "input_tensors": {**_QUANTIZER, "ch_axis": 1, "is_dynamic": True},
                "output_tensors": None,
                "bias": None,
            },
            "exclude": [],
            "layer_quant_config": {},
            "layer_type_quant_config": {},
            "export": {"weight_format": "real_quantized"},
            "packed_modules_mapping": {"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
        },
        is_prequantized=True,
    )


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 only")
class TestMuseSparkQuarkPtpcLinear(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.runtime_context import SpawnRanks, reset_context
        from sglang.srt.server_args import ServerArgs
        from sglang.test.parallel_groups import publish

        reset_context()
        publish(
            ServerArgs(model_path="dummy", tp_size=TP),
            role="test",
            ranks=SpawnRanks(world_rank=RANK),
        )

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.runtime_context import reset_context

        reset_context()

    def loaded(self, layer, pieces, shard_ids=(None,)):
        """Loads full checkpoint tensors through the layer's loaders, then the Quark post-processing."""
        for shard_id, (codes, scale) in zip(shard_ids, pieces):
            for param, value in ((layer.weight, codes), (layer.weight_scale, scale)):
                args = () if shard_id is None else (shard_id,)
                param.weight_loader(param, value, *args)
        layer.quant_method.process_weights_after_loading(layer)
        self.assertEqual(layer.weight.dtype, FP8)
        self.assertIsNotNone(layer.weight_scale)
        return layer

    def check(self, layer, codes, scale):
        """The GEMM against exact float64 math on the production quantizer's own activation codes and scales
        (what ``apply_fp8_linear`` quantizes with on ROCm), within the rounding bounds of the test helper."""
        from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8

        gen = torch.Generator().manual_seed(0)
        weight = codes.double() * scale.double().unsqueeze(-1)
        for m in ROWS:
            with self.subTest(m=m):
                x = _tokens(m, codes.shape[1], gen)
                out = layer.quant_method.apply(layer, x)
                self.assertEqual(out.shape, (m, codes.shape[0]))
                self.assertEqual(out.dtype, torch.bfloat16)
                xq, xs = per_token_group_quant_fp8(x, group_size=x.shape[1])
                self.assertTrue(bool(torch.isfinite(out).all()), "output not finite")
                bounds = gemm_row_bounds(out, xq.double() * xs.double(), weight)
                for key in ("deterministic", "statistical"):
                    bad = exceeds(bounds["err"], bounds[key])
                    self.assertFalse(
                        bool(bad.any()),
                        f"{key}: err {bounds['err'][bad][:4].tolist()} bound {bounds[key][bad][:4].tolist()}",
                    )

    def test_activation_quantizers_match_ptpc_definition(self):
        """Both production per-token quantizers (the linear's and the MoE wrapper's): scale = absmax/448 within one
        FP32 ulp, every code a neighbour of x/s, and round-to-nearest-even except at rounding midpoints. The
        power-of-two absmax and short-significand rows put many quotients exactly on midpoints."""
        from sglang.kernels.ops.quantization.fp8_kernel import (
            per_token_group_quant_fp8,
            scaled_fp8_quant,
        )

        gen = torch.Generator().manual_seed(1)
        general = _tokens(512, 8192, gen)
        short = (torch.randint(-64, 65, (512, 1024), generator=gen) / 32.0).cuda()
        short[:, 0] = 2.0  # absmax a power of two: x / s = 224 x lands on FP8 midpoints
        cases = {"general": general, "power_of_two_absmax": short.bfloat16()}
        # Scale ulps: the linear's Triton quantizer divides absmax by 448 (one rounding); the MoE wrapper's aiter
        # quantizer multiplies absmax by a rounded 1/448 (csrc/kernels/quant_kernels.cu, two roundings).
        quantizers = {
            "linear": (
                lambda x: per_token_group_quant_fp8(x, group_size=x.shape[1]),
                1.0,
            ),
            "moe": (
                lambda x: scaled_fp8_quant(x, None, use_per_token_if_dynamic=True),
                2.0,
            ),
        }
        for (case, x), (name, (quant, scale_ulps)) in itertools.product(
            cases.items(), quantizers.items()
        ):
            with self.subTest(case=case, quantizer=name):
                codes, scale = quant(x)
                report = quantizer_report(x, codes, scale)
                self.assertLessEqual(report["scale_err_ulps_max"], scale_ulps, report)
                self.assertEqual(report["not_a_neighbour"], 0, report)
                self.assertEqual(
                    report["differs_from_rne_away_from_midpoint"], 0, report
                )

    def test_qkv_gate_projection(self):
        from sglang.srt.rocm_muse_spark.layers import QKVOGateParallelLinear

        with torch.device("cuda"):
            layer = QKVOGateParallelLinear(
                hidden_size=HIDDEN,
                head_size=HEAD_DIM,
                total_num_heads=HEADS,
                total_num_kv_heads=KV_HEADS,
                prefix="model.language_model.layers.0.self_attn.qkv_proj",
                quant_config=_quark_config(),
            )
        q_rows, kv_rows = HEADS * HEAD_DIM // TP, KV_HEADS * HEAD_DIM // TP
        full = {
            shard: _weight(rows * TP, HIDDEN)
            for shard, rows in (
                ("q", q_rows),
                ("k", kv_rows),
                ("v", kv_rows),
                ("g", q_rows),
            )
        }
        self.loaded(layer, list(full.values()), shard_ids=tuple(full))
        local = [
            (
                codes[RANK * (n := codes.shape[0] // TP) : (RANK + 1) * n],
                scale[RANK * n : (RANK + 1) * n],
            )
            for codes, scale in full.values()
        ]
        codes = torch.cat([c for c, _ in local])
        scale = torch.cat([s for _, s in local])
        self.assertEqual(tuple(codes.shape), (2304, HIDDEN))
        self.check(layer, codes, scale)

    def test_o_projection(self):
        from sglang.srt.layers.linear import RowParallelLinear

        with torch.device("cuda"):
            layer = RowParallelLinear(
                input_size=HEADS * HEAD_DIM,
                output_size=HIDDEN,
                bias=False,
                reduce_results=True,
                quant_config=_quark_config(),
                prefix="model.language_model.layers.0.self_attn.o_proj",
                parallel_group="attn_tp",
            )
        codes, scale = quantize_rows(
            torch.randn(HIDDEN, HEADS * HEAD_DIM, device="cuda") * 0.02
        )
        self.loaded(layer, [(codes, scale)])
        k = HEADS * HEAD_DIM // TP
        local = codes[:, RANK * k : (RANK + 1) * k]
        self.assertEqual(tuple(local.shape), (8192, 1024))
        # Per-channel scales follow output rows; a K shard has no scale of its own.
        self.check(layer, local, scale)


if __name__ == "__main__":
    unittest.main()
