import unittest
from unittest import mock

import torch
from tqdm import tqdm

import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fused_moe_module
from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz
from sglang.srt.lora.lora_moe_runners import LoRAHooks
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device, get_device_capability, is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase, empty_gpu_cache

register_cuda_ci(est_time=17, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")

_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()


class TestFusedMOE(CustomTestCase):
    NUM_EXPERTS = [8, 64]
    TOP_KS = [2, 6]

    @staticmethod
    def create_random_gpu_tensor(shape, dtype, mean=0, std=0.01):
        """Create a random Torch(device) tensor

        Args:
            shape: Tensor shape
            dtype: Data type
            mean: Mean value
            std: Standard deviation

        Returns:
            torch.Tensor: Randomly initialized Torch(device) tensor
        """
        return torch.empty(shape, dtype=dtype, device=get_device()).normal_(mean, std)

    def get_tolerance(self, dtype):
        """Get tolerance values for different data types

        Args:
            dtype: Data type

        Returns:
            tuple: (relative tolerance, absolute tolerance)
        """
        if dtype == torch.float32:
            return 1e-3, 1e-5
        elif dtype in [torch.float16, torch.bfloat16]:
            return 1e-1, 1e-2
        else:
            return 1e-2, 1e-2  # Default values for other types

    def torch_naive_moe(
        self,
        a,
        w1,
        w2,
        score,
        topk,
        w1_scale=None,
        w2_scale=None,
        a1_scale=None,
        a2_scale=None,
        routed_scaling_factor=None,
    ):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

        B, D = a.shape
        a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
        out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        topk_weight = topk_weight.view(-1)
        topk_ids = topk_ids.view(-1)

        if w1.dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]:
            w1_compute = w1.to(a.dtype)
            w2_compute = w2.to(a.dtype)

            if w1_scale is not None:
                w1_compute = (w1_compute * w1_scale.view(-1, 1, 1)).to(a.dtype)
            if w2_scale is not None:
                w2_compute = (w2_compute * w2_scale.view(-1, 1, 1)).to(a.dtype)
            if a1_scale is not None:
                a = (a * a1_scale).to(a.dtype)
            if a2_scale is not None:
                a = (a * a2_scale).to(a.dtype)
        else:
            w1_compute = w1
            w2_compute = w2

        for i in range(w1_compute.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = SiluAndMul()(
                    a[mask] @ w1_compute[i].transpose(0, 1)
                ) @ w2_compute[i].transpose(0, 1)

        result = (
            out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
        ).sum(dim=1)
        if routed_scaling_factor is not None:
            result = result * routed_scaling_factor
        return result

    def _test_case(
        self, m, n, k, e, topk, dtype, use_fp8_w8a8=False, routed_scaling_factor=None
    ):
        rtol, atol = self.get_tolerance(dtype)

        if use_fp8_w8a8:
            # AssertionError: fp8e4nv data type is not supported on CUDA arch < 89
            capability = get_device_capability()
            if not _is_hip and not (capability[0] >= 9 or capability == (8, 9)):
                return

            a = self.create_random_gpu_tensor((m, k), dtype)
            w1 = self.create_random_gpu_tensor((e, 2 * n, k), dtype)
            w2 = self.create_random_gpu_tensor((e, k, n), dtype)
            w1 = w1.to(torch.float8_e4m3fn)
            w2 = w2.to(torch.float8_e4m3fn)
            score = self.create_random_gpu_tensor((m, e), dtype)
            w1_scale = self.create_random_gpu_tensor(e, torch.float32)
            w2_scale = self.create_random_gpu_tensor(e, torch.float32)
            a1_scale = self.create_random_gpu_tensor(1, torch.float32)
            a2_scale = self.create_random_gpu_tensor(1, torch.float32)

            # Handle HIP case: normalize float8 weights so fused kernel doesn't break
            # on ROCm.
            if _is_fp8_fnuz:
                # Normalize to e4m3fnuz on HIP
                w1, w1_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=w1,
                    weight_scale=w1_scale,
                    input_scale=a1_scale,
                )
                w2, w2_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=w2,
                    weight_scale=w2_scale,
                    input_scale=a2_scale,
                )

            topk_output = select_experts(
                hidden_states=a,
                router_logits=score,
                topk_config=TopKConfig(top_k=topk, renormalize=False),
            )

            torch_output = self.torch_naive_moe(
                a,
                w1,
                w2,
                score,
                topk,
                w1_scale,
                w2_scale,
                a1_scale,
                a2_scale,
            )

            sglang_output = fused_moe(
                a,
                w1,
                w2,
                topk_output,
                use_fp8_w8a8=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
            )
            torch.testing.assert_close(
                sglang_output, torch_output, rtol=rtol, atol=atol
            )
        else:
            a = self.create_random_gpu_tensor((m, k), dtype)
            w1 = self.create_random_gpu_tensor((e, 2 * n, k), dtype)
            w2 = self.create_random_gpu_tensor((e, k, n), dtype)
            score = self.create_random_gpu_tensor((m, e), dtype)

            topk_output = select_experts(
                hidden_states=a,
                router_logits=score,
                topk_config=TopKConfig(top_k=topk, renormalize=False),
            )

            moe_runner_config = MoeRunnerConfig(
                routed_scaling_factor=routed_scaling_factor
            )
            triton_output = fused_moe(
                a, w1, w2, topk_output, moe_runner_config=moe_runner_config
            )
            torch_output = self.torch_naive_moe(
                a, w1, w2, score, topk, routed_scaling_factor=routed_scaling_factor
            )
            torch.testing.assert_close(
                triton_output, torch_output, rtol=rtol, atol=atol
            )

    def test_various_configurations(self):
        m_values = [1, 33, 64, 222]
        n_values = [128, 1024]
        k_values = [128, 511, 1024]
        dtypes = [torch.float16, torch.bfloat16]
        fp8_modes = [False, True]

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

        # Calculate total number of tests
        total_tests = (
            len(m_values)
            * len(n_values)
            * len(k_values)
            * len(self.NUM_EXPERTS)
            * len(self.TOP_KS)
            * len(dtypes)
            * len(fp8_modes)
        )

        # Create progress bar
        with tqdm(total=total_tests, desc="Running MoE tests") as pbar:
            for m in m_values:
                for n in n_values:
                    for k in k_values:
                        for e in self.NUM_EXPERTS:
                            for topk in self.TOP_KS:
                                for dtype in dtypes:
                                    for use_fp8_w8a8 in fp8_modes:
                                        with self.subTest(
                                            m=m,
                                            n=n,
                                            k=k,
                                            e=e,
                                            topk=topk,
                                            dtype=dtype,
                                            fp8=use_fp8_w8a8,
                                        ):
                                            self._test_case(
                                                m,
                                                n,
                                                k,
                                                e,
                                                topk,
                                                dtype,
                                                use_fp8_w8a8=use_fp8_w8a8,
                                            )
                                            empty_gpu_cache()
                                        pbar.update(1)

    def test_single_expert_routing(self):
        # Cover the topk == 1 fast path (e.g. Llama-4-Scout num_experts_per_tok=1),
        # where the second kernel writes directly into the output when
        # routed_scaling_factor == 1.0 and the reduction is otherwise applied.
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        for routed_scaling_factor in [1.0, 1.5]:
            for m in [1, 33]:
                with self.subTest(m=m, routed_scaling_factor=routed_scaling_factor):
                    self._test_case(
                        m,
                        n=128,
                        k=128,
                        e=8,
                        topk=1,
                        dtype=torch.bfloat16,
                        routed_scaling_factor=routed_scaling_factor,
                    )
                    empty_gpu_cache()


class TestFusedMoeSumAllReduce(CustomTestCase):
    """--enable-fused-moe-sum-all-reduce must match the unfused combine on every
    platform, including with LoRA hooks."""

    E, TOPK, K, N = 32, 6, 512, 256
    NUM_TOKENS = [5, 37, 333]

    def setUp(self):
        torch.manual_seed(0)
        device = get_device()
        self.gen = torch.Generator(device=device).manual_seed(0)
        self.w1 = (
            torch.randn(self.E, 2 * self.N, self.K, device=device) / self.K**0.5
        ).to(torch.bfloat16)
        self.w2 = (torch.randn(self.E, self.K, self.N, device=device) / self.N**0.5).to(
            torch.bfloat16
        )

    def tearDown(self):
        self._set_fused_sum(False)

    @staticmethod
    def _set_fused_sum(enabled):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", enable_fused_moe_sum_all_reduce=enabled)
        )

    @staticmethod
    def _after_down(intermediate_cache2, intermediate_cache3, topk_weights, topk_ids):
        rows = torch.arange(
            intermediate_cache3.shape[0], device=intermediate_cache3.device
        )
        slots = torch.arange(
            intermediate_cache3.shape[1], device=intermediate_cache3.device
        )
        delta = torch.sin(rows[:, None, None] + 3.0 * slots[None, :, None])
        delta = delta * intermediate_cache3.float().sum(dim=1).abs().mean()
        intermediate_cache3.add_(delta.to(intermediate_cache3.dtype))

    def _inputs(self, num_tokens):
        x = torch.randn(
            num_tokens, self.K, generator=self.gen, device=self.w1.device
        ) / (self.K**0.5)
        logits = torch.randn(
            num_tokens, self.E, generator=self.gen, device=self.w1.device
        )
        x = x.to(torch.bfloat16)
        topk = select_experts(
            hidden_states=x,
            router_logits=logits,
            topk_config=TopKConfig(top_k=self.TOPK, renormalize=False),
        )
        return x, topk

    def _run(self, x, topk, routed_scaling_factor, hooks):
        (
            config,
            down_config,
            down_moe_use_tma,
            up_moe_use_tma,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
        ) = fused_moe_module._prepare_fused_moe_run(
            x,
            self.w1,
            self.w2,
            topk.topk_ids,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )
        invoke = fused_moe_module.invoke_fused_moe_kernel
        with mock.patch.object(
            fused_moe_module, "invoke_fused_moe_kernel", wraps=invoke
        ) as spy:
            out = fused_moe_module._fused_moe_kernel_sequence(
                x.clone(),
                self.w1,
                self.w2,
                topk.topk_weights,
                topk.topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                config,
                down_config,
                down_moe_use_tma,
                up_moe_use_tma,
                b1=None,
                b2=None,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                w1_scale=None,
                w2_scale=None,
                w1_zp=None,
                w2_zp=None,
                a1_scale=None,
                a2_scale=None,
                block_shape=None,
                activation="silu",
                is_gated=True,
                no_combine=False,
                inplace=True,
                apply_router_weight_on_input=False,
                routed_scaling_factor=routed_scaling_factor,
                gemm1_alpha=None,
                gemm1_limit=None,
                filter_expert=True,
                hooks=hooks,
            )
        fused = any(c.kwargs.get("fuse_sum_all_reduce") for c in spy.call_args_list)
        return out.float(), fused

    def _check(self, hooks, expect_fused):
        for routed_scaling_factor in [None, 2.5]:
            for num_tokens in self.NUM_TOKENS:
                with self.subTest(
                    num_tokens=num_tokens, routed_scaling_factor=routed_scaling_factor
                ):
                    # Fused first: reusing the reference run's freed buffers could
                    # hide an unwritten intermediate_cache3.
                    x, topk = self._inputs(num_tokens)
                    self._set_fused_sum(True)
                    out, fused = self._run(x, topk, routed_scaling_factor, hooks)
                    self._set_fused_sum(False)
                    ref, _ = self._run(x, topk, routed_scaling_factor, hooks)
                    rel_err = ((out - ref).norm() / ref.norm()).item()
                    self.assertLess(rel_err, 2e-2)
                    self.assertEqual(fused, expect_fused)

    def test_matches_unfused_combine(self):
        for hooks in [None, LoRAHooks()]:
            with self.subTest(hooks=hooks):
                self._check(hooks, expect_fused=True)

    def test_keeps_lora_after_down_delta(self):
        self._check(LoRAHooks(after_down=self._after_down), expect_fused=False)


if __name__ == "__main__":
    unittest.main()
