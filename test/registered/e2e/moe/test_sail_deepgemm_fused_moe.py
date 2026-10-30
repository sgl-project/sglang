"""Numerical comparison of the SAIL MoE nopad/gather-read paths against Triton,
including a real JIT warmup."""

import gc
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context, get_flags
from sglang.srt.utils import is_sail
from sglang.srt.utils.network import get_free_port
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=560, suite="stage-a-test-1-gpu-ppu")


class TestSailDeepGemmFusedMoE(CustomTestCase):
    NUM_EXPERTS = 256
    HIDDEN_SIZE = 6144
    N_UP = 256
    N_DOWN = 128
    TOP_K = 8
    M_CANDIDATES = (1, 4, 32, 256, 1024)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not is_sail() or not torch.cuda.is_available():
            raise unittest.SkipTest(
                "The vendor fused MoE API is only validated on SAIL devices."
            )
        if torch.distributed.is_initialized():
            raise RuntimeError(
                "This numerical test needs its own process; it must not take over "
                "existing distributed resources."
            )
        cls._stack = ExitStack()
        cls._stack.enter_context(envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.override(True))
        cls._stack.enter_context(
            get_context().override_server_args(
                model_path="dummy",
                moe_runner_backend="deep_gemm",
                moe_a2a_backend="none",
            )
        )
        flags = get_flags().moe
        cls._stack.enter_context(
            flags.override(
                **{name: getattr(flags, name) for name in type(flags).__struct_fields__}
            )
        )
        # Load the layer early to avoid a circular import from the Triton runner.
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from sglang.srt.layers.deep_gemm_wrapper import compile_utils, entrypoint
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
        from sglang.srt.layers.moe.utils import initialize_moe_config

        entrypoint._require_sail_fused_api("m_grouped_gemm_bf16_bf16_bf16_nt_fused")
        initialize_moe_config()
        # Teardown order: model groups, default group, flags, config, env.
        cls._stack.callback(destroy_distributed_environment)
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_free_port()}",
            backend="nccl",
        )
        cls._stack.callback(destroy_model_parallel)
        initialize_model_parallel()
        cls._layer_cls = FusedMoE
        cls._reference = staticmethod(fused_moe)
        # Restrict the warmup shapes while still exercising the real factory,
        # alignment, GEMM and compile-mode restoration.
        warmup_ms = sorted(
            set(cls.M_CANDIDATES) | {m * cls.TOP_K for m in cls.M_CANDIDATES}
        )
        for name, value in (
            ("_BUILTIN_M_LIST", warmup_ms),
            ("_INITIALIZATION_DICT", {}),
            ("_ENABLE_JIT_DEEPGEMM_PRECOMPILE", True),
            ("_DO_COMPILE_ALL", True),
        ):
            cls._stack.enter_context(patch.object(compile_utils, name, value))

    @classmethod
    def tearDownClass(cls):
        try:
            if hasattr(cls, "_stack"):
                cls._stack.close()
        finally:
            super().tearDownClass()

    @staticmethod
    def _random(shape):
        return torch.empty(shape, dtype=torch.bfloat16, device="cuda").normal_(0, 0.2)

    @staticmethod
    def _block_fp8(weight):
        E, N, K = weight.shape
        blocks = weight.float().reshape(E, N // 128, 128, K // 128, 128)
        scale = blocks.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-12) / 448.0
        quant = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        return quant.reshape(E, N, K), scale.squeeze(4).squeeze(2)

    @staticmethod
    def _channel_int8(weight):
        weight = weight.float()
        amax = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        quant = (weight / amax * 127).round().clamp(-128, 127).to(torch.int8)
        return quant, (amax / 127).squeeze(-1)

    def _build_layer(self, w1, w2, mode, quant):
        from sglang.srt.layers.quantization import get_quantization_config

        if mode == "int8":
            config = get_quantization_config("w8a8_int8")(
                {"quant_method": "compressed-tensors"}
            )
        elif mode == "fp8":
            config = get_quantization_config("fp8")(
                is_checkpoint_fp8_serialized=True,
                weight_block_size=[128, 128],
                activation_scheme="dynamic",
                ignored_layers=[],
                use_mxfp8=False,
            )
        elif mode in ("fp4", "w4a16"):
            config = get_quantization_config("mxfp4")(
                is_checkpoint_mxfp4_serialized=True, ignored_layers=[]
            )
        else:
            config = None
        layer = self._layer_cls(
            self.NUM_EXPERTS,
            self.HIDDEN_SIZE,
            self.N_DOWN,
            0,
            top_k=self.TOP_K,
            quant_config=config,
            activation="silu",
            params_dtype=torch.bfloat16,
            uses_expert_location=False,
        )
        layer.w13_weight = torch.nn.Parameter(w1, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        if mode != "bf16":
            suffix = "weight_scale_inv" if mode == "fp8" else "weight_scale"
            setattr(
                layer,
                f"w13_{suffix}",
                torch.nn.Parameter(quant["w1_scale"], requires_grad=False),
            )
            setattr(
                layer,
                f"w2_{suffix}",
                torch.nn.Parameter(quant["w2_scale"], requires_grad=False),
            )
        if mode == "w4a16":
            # Run the real post-load: VALU Marlin repack or MMA scale tile
            # conversion.
            layer.quant_method.process_weights_after_loading(layer)
        return layer

    @staticmethod
    def _fp4(weight, preprocess=True):
        from deep_gemm import preprocess_mxfp4_scales
        from triton_kernels.numerics_details.mxfp import upcast_from_mxfp

        from sglang.kernels.ops.quantization.sail_mxfp4 import downcast_to_mxfp4

        packed, scale = downcast_to_mxfp4(weight, axis=-1, enable_fused_moe=True)
        # The reference must read the logical K-major scale bytes; the vendor
        # MN-major storage must not be treated as the raw value.
        raw_scale = scale.contiguous().view(torch.uint8)
        reference = upcast_from_mxfp(
            packed, raw_scale, target_dtype=torch.bfloat16, axis=-1
        )
        return (
            packed,
            preprocess_mxfp4_scales(raw_scale) if preprocess else raw_scale,
            reference,
        )

    def _compare(self, mode):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        with torch.no_grad():
            for m in self.M_CANDIDATES:
                torch.manual_seed(42)
                a = self._random((m, self.HIDDEN_SIZE))
                w1 = self._random((self.NUM_EXPERTS, self.N_UP, self.HIDDEN_SIZE))
                w2 = self._random((self.NUM_EXPERTS, self.HIDDEN_SIZE, self.N_DOWN))
                quant = {}
                reference_a, reference_w1, reference_w2 = a, w1, w2
                if mode in ("fp4", "w4a16"):
                    from triton_kernels.numerics_details.mxfp import upcast_from_mxfp

                    from sglang.kernels.ops.quantization.sail_mxfp4 import (
                        downcast_to_mxfp4,
                    )

                    w1, s1, reference_w1 = self._fp4(w1, preprocess=mode == "fp4")
                    w2, s2, reference_w2 = self._fp4(w2, preprocess=mode == "fp4")
                    if mode == "fp4":
                        aq, ascale = downcast_to_mxfp4(a, axis=1, enable_fused_moe=True)
                        reference_a = upcast_from_mxfp(
                            aq,
                            ascale.view(torch.uint8),
                            target_dtype=torch.bfloat16,
                            axis=-1,
                        )
                        del aq, ascale
                    quant = dict(w1_scale=s1, w2_scale=s2)
                elif mode != "bf16":
                    quantize = self._block_fp8 if mode == "fp8" else self._channel_int8
                    w1, s1 = quantize(w1)
                    w2, s2 = quantize(w2)
                    quant = dict(
                        w1_scale=s1,
                        w2_scale=s2,
                        use_fp8_w8a8=mode == "fp8",
                        use_int8_w8a8=mode == "int8",
                        per_channel_quant=mode == "int8",
                        block_shape=[128, 128] if mode == "fp8" else None,
                    )
                    reference_w1, reference_w2 = w1, w2
                topk = select_experts(
                    hidden_states=a,
                    router_logits=self._random((m, self.NUM_EXPERTS)),
                    topk_config=TopKConfig(top_k=self.TOP_K, renormalize=True),
                )
                reference = self._reference(
                    hidden_states=reference_a.clone(),
                    w1=reference_w1,
                    w2=reference_w2,
                    topk_output=topk,
                    moe_runner_config=MoeRunnerConfig(
                        inplace=True, routed_scaling_factor=1.0
                    ),
                    **({} if mode in ("fp4", "w4a16") else quant),
                )
                self.assertTrue(torch.isfinite(reference).all().item())
                layer = self._build_layer(w1, w2, mode, quant)
                try:
                    for fused in (False, True):
                        with self.subTest(mode=mode, m=m, fused=fused):
                            with envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED.override(fused):
                                actual = layer(a.clone(), topk)
                            self.assertTrue(torch.isfinite(actual).all().item())
                            if mode == "fp4":
                                # The BF16 reference has no FP4 activation
                                # quantization before GEMM2, so keep the
                                # original threshold.
                                x, y = actual.double(), reference.double()
                                diff = (
                                    1
                                    - 2
                                    * (x * y).sum()
                                    / (x.square() + y.square()).sum()
                                )
                                self.assertLess(diff.item(), 0.15)
                            else:
                                torch.testing.assert_close(
                                    actual, reference, rtol=0.2, atol=1.0
                                )
                finally:
                    del layer
                del a, w1, w2, quant, topk, reference, actual
                del reference_a, reference_w1, reference_w2
                if mode != "bf16":
                    del s1, s2
                if mode == "fp4":
                    del x, y
                gc.collect()
                torch.cuda.empty_cache()

    def test_bf16_unquantized(self):
        self._compare("bf16")

    def test_fp8_blockwise(self):
        self._compare("fp8")

    def test_int8_channelwise(self):
        self._compare("int8")

    def test_fp4(self):
        with (
            envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16.override(False),
            envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA.override(False),
        ):
            self._compare("fp4")

    def test_w4a16_valu_and_mma(self):
        """BF16 activations, real weight conversion and warmup, compared against the
        decoded-weight reference on both TP paths."""
        for mma in (False, True):
            with (
                self.subTest(mma=mma),
                envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16.override(not mma),
                envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA.override(mma),
            ):
                self._compare("w4a16")


if __name__ == "__main__":
    unittest.main()
