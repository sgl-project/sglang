import gc
import unittest

import torch

# Pre-import to break circular import between triton_utils and fused_moe_triton
import sglang.srt.layers.moe.fused_moe_triton  # noqa: F401
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.network import get_free_port
from sglang.test.test_utils import CustomTestCase


class TestFusedMoEDeepGEMMPath(CustomTestCase):
    """
    Tests DeepGEMM MoE path against Triton path as reference.

    In v0.5.12, fused_moe() no longer dispatches to DeepGEMM.
    The DeepGEMM path is accessed through the FusedMoE layer + MoeRunner
    framework with moe_runner_backend='deep_gemm', following MoEOpBench's
    approach.

    Triton reference: fused_moe() direct call (pure Triton path).
    DeepGEMM: FusedMoE layer → MoeRunner → fused_func → deep_moe_impl_fused().

    Supports both nopad and fused paths via external
    SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED env var.

    DeepGemm dependency: commit 53ea8ff (refactor fused MoE API, simplify token addressing).
    """

    NUM_EXPERTS = 256
    HIDDEN_SIZE = 6144
    N_UP = 256  # w1's N dim (= 2 * intermediate_size)
    N_DOWN = 128  # w2's N dim (= intermediate_size)
    TOP_K = 8
    M_CANDIDATES = [1, 4, 32, 256, 1024]

    @classmethod
    def setUpClass(cls):
        # Initialize distributed environment (needed by FusedMoE layer)
        from sglang.srt.distributed.parallel_state import _TP

        world_size = 1
        init_method = f"tcp://127.0.0.1:{get_free_port()}"
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method=init_method,
                world_size=world_size,
                rank=0,
            )
        if _TP is None:
            from sglang.srt.distributed.parallel_state import (
                init_distributed_environment,
                initialize_model_parallel,
            )

            init_distributed_environment(
                world_size=world_size,
                rank=0,
                distributed_init_method=init_method,
                local_rank=0,
                backend="nccl" if torch.cuda.is_available() else "gloo",
            )
            initialize_model_parallel(
                expert_model_parallel_size=world_size,
                moe_data_model_parallel_size=world_size,
            )

        # Enable JIT precompile to test warmup executors
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(True)

        # Set default server args (needed by select_experts and other utilities)
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    @classmethod
    def tearDownClass(cls):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @staticmethod
    def create_random_cuda_tensor(shape, dtype, mean=0, std=0.2):
        return torch.empty(shape, dtype=dtype, device="cuda").normal_(mean, std)

    def get_tolerance(self, dtype):
        if dtype in [torch.float16, torch.bfloat16]:
            return 0.2, 1.0
        return 1e-2, 1e-2

    @staticmethod
    def blockwise_quant_fp8(weight, block_n, block_k):
        fp8_dtype = torch.float8_e4m3fn
        finfo = torch.finfo(fp8_dtype)
        fp8_max = finfo.max
        E, N, K = weight.shape
        assert N % block_n == 0 and K % block_k == 0
        w_blocked = weight.reshape(E, N // block_n, block_n, K // block_k, block_k)
        block_abs_max = w_blocked.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-12)
        scale = block_abs_max / fp8_max
        w_quant = (w_blocked / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        w_fp8 = w_quant.reshape(E, N, K)
        scale_out = scale.squeeze(4).squeeze(2)
        return w_fp8, scale_out

    @staticmethod
    def channelwise_quant_int8(weight):
        E, N, K = weight.shape
        amax = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = (amax / 127.0).squeeze(-1)
        w_int8 = (weight / amax * 127.0).round().clamp(-128, 127).to(torch.int8)
        return w_int8, scale

    @staticmethod
    def calc_diff(x, y):
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum()
        sim = 2 * (x * y).sum() / denominator
        return 1 - sim

    # ------------------------------------------------------------------
    # Triton reference (fused_moe direct call — pure Triton, no DeepGEMM)
    # ------------------------------------------------------------------

    def _run_triton_reference(self, hidden_states, w1, w2, topk_output, **quant_kwargs):
        """Run Triton path as reference using fused_moe() directly."""
        moe_runner_config = MoeRunnerConfig(inplace=True, routed_scaling_factor=1.0)
        return fused_moe(
            hidden_states=hidden_states.clone(),
            w1=w1,
            w2=w2,
            topk_output=topk_output,
            moe_runner_config=moe_runner_config,
            **quant_kwargs,
        )

    # ------------------------------------------------------------------
    # DeepGEMM path (FusedMoE layer + MoeRunner framework)
    # ------------------------------------------------------------------

    def _run_deepgemm(self, hidden_states, w1, w2, topk_output, **quant_kwargs):
        """
        Run DeepGEMM path using FusedMoE layer (MoEOpBench approach).

        Creates a FusedMoE layer with moe_runner_backend='deep_gemm',
        which triggers ppu_deepgemm_moe registration and routes through
        deep_moe_impl_fused().
        """
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization import get_quantization_config

        use_fp8 = quant_kwargs.get("use_fp8_w8a8", False)
        use_int8 = quant_kwargs.get("use_int8_w8a8", False)
        use_mxfp4 = quant_kwargs.get("use_mxfp4", False)
        w1_scale = quant_kwargs.get("w1_scale")
        w2_scale = quant_kwargs.get("w2_scale")
        block_shape = quant_kwargs.get("block_shape")

        # --- Build quant_config ---
        if use_int8:
            quant_config = get_quantization_config("w8a8_int8")(
                {"quant_method": "compressed-tensors"}
            )
        elif use_fp8:
            quant_config = get_quantization_config("fp8")(
                is_checkpoint_fp8_serialized=True,
                weight_block_size=block_shape,
                activation_scheme="dynamic",
                ignored_layers=[],
                use_mxfp8=False,
            )
        elif use_mxfp4:
            quant_config = get_quantization_config("mxfp4")(
                is_checkpoint_mxfp4_serialized=True,
                ignored_layers=[],
            )
        else:
            quant_config = None

        # --- Switch to deep_gemm backend ---
        server_args = ServerArgs(model_path="dummy", moe_runner_backend="deep_gemm")
        set_global_server_args_for_scheduler(server_args)
        initialize_moe_config(server_args)

        # --- Create FusedMoE layer ---
        moe_experts = FusedMoE(
            self.NUM_EXPERTS,
            self.HIDDEN_SIZE,
            self.N_DOWN,  # intermediate_size
            0,  # layer_id
            top_k=self.TOP_K,
            quant_config=quant_config,
            activation="silu",
        )

        # --- Clear pre-allocated weights, then assign test weights ---
        w13_weight = torch.nn.Parameter(w1, requires_grad=False)
        w2_weight = torch.nn.Parameter(w2, requires_grad=False)

        moe_experts.w13_weight = None
        moe_experts.w2_weight = None
        if hasattr(moe_experts, "w13_weight_scale_inv"):
            moe_experts.w13_weight_scale_inv = None
        if hasattr(moe_experts, "w2_weight_scale_inv"):
            moe_experts.w2_weight_scale_inv = None
        if hasattr(moe_experts, "w13_weight_scale"):
            moe_experts.w13_weight_scale = None
        if hasattr(moe_experts, "w2_weight_scale"):
            moe_experts.w2_weight_scale = None
        gc.collect()
        torch.cuda.empty_cache()

        moe_experts.w13_weight = w13_weight
        moe_experts.w2_weight = w2_weight
        if use_int8:
            moe_experts.w13_weight_scale = torch.nn.Parameter(
                w1_scale, requires_grad=False
            )
            moe_experts.w2_weight_scale = torch.nn.Parameter(
                w2_scale, requires_grad=False
            )
        elif use_fp8:
            moe_experts.w13_weight_scale_inv = torch.nn.Parameter(
                w1_scale, requires_grad=False
            )
            moe_experts.w2_weight_scale_inv = torch.nn.Parameter(
                w2_scale, requires_grad=False
            )
        elif use_mxfp4:
            moe_experts.w13_weight_scale = torch.nn.Parameter(
                w1_scale, requires_grad=False
            )
            moe_experts.w2_weight_scale = torch.nn.Parameter(
                w2_scale, requires_grad=False
            )

        # --- Run ---
        output = moe_experts(hidden_states.clone(), topk_output)
        del moe_experts
        gc.collect()
        torch.cuda.empty_cache()
        return output

    # ------------------------------------------------------------------
    # Compare helper
    # ------------------------------------------------------------------

    def _compare_triton_vs_deepgemm(
        self, hidden_states, w1, w2, score, topk, **quant_kwargs
    ):
        """Run Triton (reference) and DeepGEMM, compare outputs."""
        rtol, atol = self.get_tolerance(hidden_states.dtype)

        topk_output = select_experts(
            hidden_states=hidden_states,
            router_logits=score,
            topk_config=TopKConfig(top_k=topk, renormalize=True),
        )

        # --- Triton reference ---
        quant_kwargs_ = quant_kwargs.copy()
        w1_, w2_ = w1, w2
        hidden_states_ = hidden_states
        use_mxfp4 = quant_kwargs_.pop("use_mxfp4", False)
        if use_mxfp4:
            from triton_kernels.numerics_details.mxfp import upcast_from_mxfp

            from sglang.srt.layers.quantization.ppu_mxfp4_utils import downcast_to_mxfp4

            ### Triton fused_moe does NOT support mxfp4. so dequant to bfloat16.
            ### the triton fp4 path in deep_gemm is NOT support(only dequant path still exists)
            w1_ = upcast_from_mxfp(
                w1_,
                quant_kwargs["w1_scale"].contiguous().view(torch.uint8),
                target_dtype=torch.bfloat16,
                axis=-1,
            )
            w2_ = upcast_from_mxfp(
                w2_,
                quant_kwargs["w2_scale"].contiguous().view(torch.uint8),
                target_dtype=torch.bfloat16,
                axis=-1,
            )
            quant_kwargs_.pop("w1_scale")
            quant_kwargs_.pop("w2_scale")
            hidden_states_, hidden_states_scale_ = downcast_to_mxfp4(
                hidden_states_, axis=1
            )
            hidden_states_ = upcast_from_mxfp(
                hidden_states_,
                hidden_states_scale_.contiguous().view(torch.uint8),
                target_dtype=torch.bfloat16,
                axis=-1,
            )

        triton_output = self._run_triton_reference(
            hidden_states_, w1_, w2_, topk_output, **quant_kwargs_
        )

        # --- DeepGEMM ---
        deepgemm_output = self._run_deepgemm(
            hidden_states, w1, w2, topk_output, **quant_kwargs
        )

        assert not torch.isnan(triton_output).any(), "Triton output has NaN"
        assert not torch.isnan(deepgemm_output).any(), "DeepGEMM output has NaN"
        assert not torch.isinf(triton_output).any(), "Triton output has Inf"
        assert not torch.isinf(deepgemm_output).any(), "DeepGEMM output has Inf"

        if use_mxfp4:
            diff = self.calc_diff(triton_output, deepgemm_output)
            ### The threshold is relaxed for mxfp4, because quant ater silu_and_mul is absent during b16 path.
            assert diff < 0.15, (
                f"The difference between MXFP4 and Bfloat16 is too large."
                f"threshold=0.15 but got {diff=}"
            )
            return

        torch.testing.assert_close(deepgemm_output, triton_output, rtol=rtol, atol=atol)

    # ------------------------------------------------------------------
    # Test cases
    # ------------------------------------------------------------------

    def test_bf16_unquantized(self):
        """BF16 unquantized: fused_moe() vs FusedMoE(deep_gemm)."""
        dtype = torch.bfloat16
        E = self.NUM_EXPERTS
        for m in self.M_CANDIDATES:
            with self.subTest(m=m):
                torch.manual_seed(42)
                a = self.create_random_cuda_tensor((m, self.HIDDEN_SIZE), dtype)
                w1 = self.create_random_cuda_tensor(
                    (E, self.N_UP, self.HIDDEN_SIZE), dtype
                )
                w2 = self.create_random_cuda_tensor(
                    (E, self.HIDDEN_SIZE, self.N_DOWN), dtype
                )
                score = self.create_random_cuda_tensor((m, E), dtype)
                self._compare_triton_vs_deepgemm(a, w1, w2, score, self.TOP_K)
                torch.cuda.empty_cache()

    def test_fp8_blockwise(self):
        """FP8 blockwise: fused_moe() vs FusedMoE(deep_gemm)."""
        dtype = torch.bfloat16
        E = self.NUM_EXPERTS
        block_n, block_k = 128, 128
        for m in self.M_CANDIDATES:
            with self.subTest(m=m):
                torch.manual_seed(42)
                a = self.create_random_cuda_tensor((m, self.HIDDEN_SIZE), dtype)
                w1_bf16 = self.create_random_cuda_tensor(
                    (E, self.N_UP, self.HIDDEN_SIZE), dtype
                )
                w2_bf16 = self.create_random_cuda_tensor(
                    (E, self.HIDDEN_SIZE, self.N_DOWN), dtype
                )
                score = self.create_random_cuda_tensor((m, E), dtype)
                w1_fp8, w1_scale = self.blockwise_quant_fp8(
                    w1_bf16.float(), block_n, block_k
                )
                w2_fp8, w2_scale = self.blockwise_quant_fp8(
                    w2_bf16.float(), block_n, block_k
                )
                self._compare_triton_vs_deepgemm(
                    a,
                    w1_fp8,
                    w2_fp8,
                    score,
                    self.TOP_K,
                    use_fp8_w8a8=True,
                    per_channel_quant=False,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    block_shape=[block_n, block_k],
                )
                torch.cuda.empty_cache()

    def test_int8_channelwise(self):
        """INT8 channelwise: fused_moe() vs FusedMoE(deep_gemm)."""
        dtype = torch.bfloat16
        E = self.NUM_EXPERTS
        for m in self.M_CANDIDATES:
            with self.subTest(m=m):
                torch.manual_seed(42)
                a = self.create_random_cuda_tensor((m, self.HIDDEN_SIZE), dtype)
                w1_bf16 = self.create_random_cuda_tensor(
                    (E, self.N_UP, self.HIDDEN_SIZE), dtype
                )
                w2_bf16 = self.create_random_cuda_tensor(
                    (E, self.HIDDEN_SIZE, self.N_DOWN), dtype
                )
                score = self.create_random_cuda_tensor((m, E), dtype)
                w1_int8, w1_scale = self.channelwise_quant_int8(w1_bf16.float())
                w2_int8, w2_scale = self.channelwise_quant_int8(w2_bf16.float())
                self._compare_triton_vs_deepgemm(
                    a,
                    w1_int8,
                    w2_int8,
                    score,
                    self.TOP_K,
                    use_int8_w8a8=True,
                    per_channel_quant=True,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    block_shape=None,
                )
                torch.cuda.empty_cache()

    def test_fp4(self):
        """MXFP4: fused_moe() v.s. FusedMoE(deep_gemm)."""
        from sglang.srt.layers.quantization.ppu_mxfp4_utils import downcast_to_mxfp4

        dtype = torch.bfloat16
        E = self.NUM_EXPERTS
        for m in self.M_CANDIDATES:
            with self.subTest(m=m):
                torch.manual_seed(42)
                a = self.create_random_cuda_tensor((m, self.HIDDEN_SIZE), dtype)
                w1_bf16 = self.create_random_cuda_tensor(
                    (E, self.N_UP, self.HIDDEN_SIZE), dtype
                )
                w2_bf16 = self.create_random_cuda_tensor(
                    (E, self.HIDDEN_SIZE, self.N_DOWN), dtype
                )
                score = self.create_random_cuda_tensor((m, E), dtype)

                w1_uint8, w1_scale_uint16 = downcast_to_mxfp4(
                    w1_bf16,
                    axis=-1,
                )
                w2_uint8, w2_scale_uint16 = downcast_to_mxfp4(
                    w2_bf16,
                    axis=-1,
                )
                from deep_gemm import preprocess_mxfp4_scales

                ### NOTE: the stride of scale derived from downcast_to_mxfp4
                ### are WRONG for 3-axis tensor
                w1_scale_uint16 = preprocess_mxfp4_scales(
                    scale=(w1_scale_uint16.contiguous().view(torch.uint8))
                )
                w2_scale_uint16 = preprocess_mxfp4_scales(
                    scale=(w2_scale_uint16.contiguous().view(torch.uint8))
                )

                self._compare_triton_vs_deepgemm(
                    a,
                    w1_uint8,
                    w2_uint8,
                    score,
                    self.TOP_K,
                    use_mxfp4=True,
                    per_channel_quant=False,
                    w1_scale=w1_scale_uint16,
                    w2_scale=w2_scale_uint16,
                )
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
