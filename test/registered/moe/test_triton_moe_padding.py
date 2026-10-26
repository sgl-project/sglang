"""Padded routes must contribute zero even when every expert is local."""

import unittest

import torch
import torch.nn.functional as F
import triton.language as tl

from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import (
    TritonMoeQuantInfo,
    TritonRunnerCore,
    pre_permute_standard_to_triton,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import MoeRunnerBackend, initialize_moe_config
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


class TestTritonMoePadding(CustomTestCase):
    ROWS, EXPERTS, TOP_K, HIDDEN, INTERMEDIATE = 4, 4, 2, 128, 128

    def setUp(self):
        super().setUp()
        self.enterContext(
            get_context().override_server_args(
                model_path="dummy",
                device="cuda",
                moe_runner_backend="triton",
                ep_dispatch_algorithm=None,
                enable_eplb=False,
                enable_waterfill=False,
            )
        )
        moe_flags = get_flags().moe
        scoped_flags = {
            name: getattr(moe_flags, name) for name in moe_flags.__struct_fields__
        }
        scoped_flags["in_speculative_scope"] = False
        self.enterContext(moe_flags.override(**scoped_flags))
        initialize_moe_config()
        self.enterContext(
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_dp_size=1,
                attn_cp_size=1,
                tp_group=None,
            )
        )
        torch.manual_seed(123)

    @staticmethod
    def _expert_view(shape, dtype, *, quantized=False):
        # A normal view with an extra leading record makes selecting an inactive
        # expert numerically observable. Only actual experts contain finite data.
        if quantized:
            storage = torch.randint(
                120, 137, (shape[0] + 1, *shape[1:]), device="cuda", dtype=dtype
            )
        else:
            storage = (
                torch.randn((shape[0] + 1, *shape[1:]), device="cuda", dtype=dtype)
                * 0.05
            )
            storage[0].fill_(float("nan"))
        return storage[1:]

    def _fixture(self, dtype, quantized=False):
        x = torch.randn((self.ROWS, self.HIDDEN), device="cuda", dtype=dtype) * 0.1
        logits = torch.tensor(
            [
                [9.0, 8.0, 1.0, 0.0],
                [0.0, 9.0, 8.0, 1.0],
                [1.0, 0.0, 9.0, 8.0],
                [8.0, 1.0, 0.0, 9.0],
            ],
            device="cuda",
        )
        shapes = [
            (self.EXPERTS, 2 * self.INTERMEDIATE, self.HIDDEN),
            (self.EXPERTS, self.HIDDEN, self.INTERMEDIATE),
        ]
        if quantized:
            weights = [
                self._expert_view(shape, torch.uint8, quantized=True)
                for shape in shapes
            ]
            scales = []
            for shape in shapes:
                storage = torch.full(
                    (self.EXPERTS + 1, shape[1], shape[2] // 32),
                    0.01,
                    device="cuda",
                    dtype=dtype,
                )
                storage[0].fill_(float("nan"))
                scales.append(storage[1:])
            quant = TritonMoeQuantInfo(
                *weights,
                use_int8_w8a16=True,
                w13_scale=scales[0],
                w2_scale=scales[1],
                block_shape=[0, 32],
            )
            dense_weights = [
                (weight.to(dtype) - 128) * scale.repeat_interleave(32, dim=-1)
                for weight, scale in zip(weights, scales)
            ]
        else:
            weights = [self._expert_view(shape, dtype) for shape in shapes]
            quant = TritonMoeQuantInfo(*weights)
            dense_weights = weights
        router = TopK(
            top_k=self.TOP_K,
            layer_id=0,
            output_format=TopKOutputFormat.STANDARD,
            allow_routed_experts_capture=False,
        )
        router.bind_runner_backend(MoeRunnerBackend.TRITON)
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=self.ROWS,
            input_ids=torch.arange(self.ROWS, device="cuda"),
            req_pool_indices=torch.arange(self.ROWS, device="cuda"),
            seq_lens=torch.ones(self.ROWS, device="cuda", dtype=torch.int32),
            out_cache_loc=torch.arange(self.ROWS, device="cuda"),
            seq_lens_sum=self.ROWS,
            num_token_non_padded=torch.tensor(2, device="cuda", dtype=torch.int32),
        )
        config = MoeRunnerConfig(
            num_experts=self.EXPERTS,
            num_local_experts=self.EXPERTS,
            hidden_size=self.HIDDEN,
            intermediate_size_per_partition=self.INTERMEDIATE,
            top_k=self.TOP_K,
            num_fused_shared_experts=0,
            params_dtype=dtype,
            runner_backend=MoeRunnerBackend.TRITON,
            inplace=False,
        )
        return x, logits, quant, dense_weights, router, batch, config

    @staticmethod
    def _prepare(x, routes, quant, config):
        state = {}
        prepared = pre_permute_standard_to_triton(
            StandardDispatchOutput(x, None, routes), quant, config, state
        )
        return prepared, state

    def _run(self, x, routes, quant, config):
        prepared, state = self._prepare(x, routes, quant, config)
        return TritonRunnerCore(config).run(prepared, quant, state).hidden_states

    def _dense_reference(self, x, logits, weights):
        probabilities = torch.softmax(logits, dim=-1)
        scores, ids = probabilities.topk(self.TOP_K, dim=-1)
        scores /= scores.sum(dim=-1, keepdim=True)
        result = torch.zeros_like(x)
        for row in range(x.shape[0]):
            for slot in range(self.TOP_K):
                expert = int(ids[row, slot])
                gate, up = F.linear(x[row], weights[0][expert]).chunk(2)
                activated = (F.silu(gate.float()) * up.float()).to(x.dtype)
                result[row] += (
                    F.linear(activated, weights[1][expert]).float() * scores[row, slot]
                ).to(x.dtype)
        return result

    def test_runner_observes_updated_batch_bound(self):
        """An inactive row must stay finite and zero through both expert GEMMs."""
        for dtype, quantized in [
            (torch.bfloat16, False),
            (torch.float16, False),
            (torch.bfloat16, True),
        ]:
            with self.subTest(dtype=dtype, quantized=quantized):
                x, logits, quant, weights, router, batch, config = self._fixture(
                    dtype, quantized
                )
                dense = self._dense_reference(x, logits, weights)
                # Reuse the batch and its device scalar, as static graph buffers do.
                with forward_context(
                    ForwardContext(attn_backend=None), forward_batch=batch
                ):
                    for live in (2, 3, 0, self.ROWS):
                        with self.subTest(live=live):
                            batch.num_token_non_padded.fill_(live)
                            routes = router.forward_cuda(x, logits)
                            self.assertTrue(torch.all(routes.topk_ids[live:] == -1))
                            output = self._run(x, routes, quant, config)
                            self.assertTrue(torch.isfinite(output).all())
                            torch.testing.assert_close(
                                output[live:],
                                torch.zeros_like(output[live:]),
                                rtol=0,
                                atol=0,
                            )
                            torch.testing.assert_close(
                                output[:live], dense[:live], rtol=0.04, atol=2e-4
                            )
                            if live:
                                # A cropped input has no inactive routes at all.
                                cropped_routes = router.forward_cuda(
                                    x[:live], logits[:live]
                                )
                                cropped = self._run(
                                    x[:live], cropped_routes, quant, config
                                )
                                torch.testing.assert_close(
                                    output[:live], cropped, rtol=0.01, atol=2e-5
                                )

    def test_inactive_routes_preserve_output_layouts(self):
        """Skipping a route must respect sorted, accumulated and half-width outputs."""
        x, logits, quant, _, router, batch, config = self._fixture(torch.bfloat16)
        with forward_context(ForwardContext(attn_backend=None), forward_batch=batch):
            routes = router.forward_cuda(x, logits)
        prepared, state = self._prepare(x, routes, quant, config)
        count = int(prepared.num_tokens_post_padded)
        block = state["config"]["BLOCK_SIZE_M"]
        sorted_ids = prepared.sorted_token_ids[:count].tolist()
        expert_ids = prepared.expert_ids[: (count + block - 1) // block].tolist()
        for layout in ("sorted", "add", "sum", "swiglu"):
            with self.subTest(layout=layout):
                width = (
                    self.INTERMEDIATE if layout == "swiglu" else 2 * self.INTERMEDIATE
                )
                rows = (
                    count
                    if layout == "sorted"
                    else self.ROWS
                    if layout == "sum"
                    else self.ROWS * self.TOP_K
                )
                output = torch.full((rows, width), 0.375, device="cuda", dtype=x.dtype)
                expected = output.clone()
                for position, route in enumerate(sorted_ids):
                    expert = expert_ids[position // block]
                    if expert == -1 or route >= self.ROWS * self.TOP_K:
                        continue
                    value = F.linear(x[route // self.TOP_K], quant.w13_weight[expert])
                    if layout == "swiglu":
                        gate, up = value[::2], value[1::2]
                        value = (F.silu(gate.float()) * up.float()).to(x.dtype)
                    target = (
                        position
                        if layout == "sorted"
                        else route // self.TOP_K
                        if layout == "sum"
                        else route
                    )
                    if layout in ("add", "sum"):
                        expected[target] += value
                    else:
                        expected[target] = value
                invoke_fused_moe_kernel(
                    x,
                    quant.w13_weight,
                    None,
                    output,
                    None,
                    None,
                    None,
                    routes.topk_weights,
                    routes.topk_ids,
                    prepared.sorted_token_ids,
                    prepared.expert_ids,
                    prepared.num_tokens_post_padded,
                    False,
                    self.TOP_K,
                    state["config"],
                    compute_type=tl.bfloat16,
                    use_fp8_w8a8=False,
                    use_int8_w8a8=False,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                    per_channel_quant=False,
                    filter_expert=False,
                    c_sorted=layout == "sorted",
                    fuse_add_to_output=layout == "add",
                    add_output_mask=torch.ones(
                        self.ROWS, device="cuda", dtype=torch.bool
                    ),
                    fuse_sum_all_reduce=layout == "sum",
                    router_topk=self.TOP_K,
                    fuse_swiglu=layout == "swiglu",
                )
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(output, expected, rtol=0.02, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
