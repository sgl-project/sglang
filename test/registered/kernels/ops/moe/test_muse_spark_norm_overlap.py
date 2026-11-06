"""Real Muse Spark BF16 components: overlap versus a serialized execution.

Small random weights, TP=EP=1; no checkpoint or serving-accuracy claim.
Both Triton and (on SM10x) TRT-LLM execute their actual expert kernels.
"""

import contextlib
import unittest
from unittest.mock import patch

import torch

from sglang.srt.configs.muse_spark import MuseSparkTextConfig
from sglang.srt.layers.muse_spark import muse_spark_bf16_moe
from sglang.srt.models import muse_spark
from sglang.srt.runtime_context import get_context, get_forward, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@contextlib.contextmanager
def single_rank():
    with (
        get_context().override_server_args(model_path="dummy"),
        get_parallel().override(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_dp_size=1,
            attn_cp_size=1,
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_dp_size=1,
        ),
        get_forward().scoped(sp_active=False),
    ):
        yield


def make_moe(
    backend,
    projection=True,
    hidden=1024,
    expert_hidden=512,
    intermediate=512,
    experts=16,
    top_k=8,
):
    config = MuseSparkTextConfig(
        hidden_size=hidden if projection else expert_hidden,
        moe_hidden_size=expert_hidden,
        moe_intermediate_size=intermediate,
        num_local_experts=experts,
        num_experts_per_tok=top_k,
    )
    # Construct the production class, including its configured weighted norms.
    with torch.device("cuda"):
        layer = muse_spark.MuseSpark1xMoE(config, 0, False, "test")
    layer.bfloat16()
    layer.gate.float()  # Production router weights and correction bias are FP32.
    with torch.no_grad():
        for name, param in layer.named_parameters():
            param.normal_(0, 0.05 if "norm" in name else param.shape[-1] ** -0.5)
        layer.gate.e_score_correction_bias.uniform_(-0.05, 0.05)
        for norm in (layer.pre_expert_norm, layer.experts.post_expert_norm):
            norm.precompute_effective_weight()
    if backend == "triton":
        with patch.object(
            muse_spark_bf16_moe, "_trtllm_bf16_moe_supported", return_value=False
        ):
            layer.experts.prepare_trtllm_weights(layer.experts.post_expert_norm)
        assert layer.experts.use_triton and not layer.experts.use_trtllm
    else:
        layer.experts.prepare_trtllm_weights(layer.experts.post_expert_norm)
        assert layer.experts.use_trtllm and not layer.experts.use_triton
    return layer


def serialized(layer, x):
    """Same production operators, all on the caller stream (no overlap)."""
    split = (
        layer.router_in_fp32
        and muse_spark._use_split_bf16_router(x.shape[0])
        and layer.gate.weight.dtype == torch.float32
        and x.dtype in (torch.bfloat16, torch.float16)
    )
    projected = (
        x
        if layer.pre_expert_proj is None
        else muse_spark.bf16_gemm_dispatch(x, layer.pre_expert_proj.weight, None)
    )
    logits = layer.gate.forward_split2(x) if split else layer.gate(x.float())
    scores, ids, packed = layer._routing_op()(
        None, logits, layer.top_k, True, want_packed=True
    )
    if layer.pre_expert_norm is not None:
        projected = layer.pre_expert_norm(projected)
    return layer.experts(
        projected, scores, ids, layer.experts.post_expert_norm, packed_topk_ids=packed
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMuseSparkNormOverlap(CustomTestCase):
    @torch.inference_mode()
    def _check_backend(self, backend):
        with single_rank():
            for projection in (False, True):
                torch.manual_seed(42)
                layer = make_moe(backend, projection)
                for tokens in (1, 16, 128, 129):
                    with self.subTest(
                        backend=backend, projection=projection, tokens=tokens
                    ):
                        x = torch.randn(
                            tokens,
                            layer.hidden_size,
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        for _ in range(3):
                            expected = serialized(layer, x)
                            actual = layer._forward_experts(x)
                        torch.cuda.synchronize()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = layer._forward_experts(x)
                        for _ in range(5):
                            x.normal_()
                            expected = serialized(layer, x)
                            graph.replay()
                            torch.cuda.synchronize()
                            torch.testing.assert_close(
                                captured, expected, rtol=0, atol=0
                            )

    def test_triton(self):
        self._check_backend("triton")

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
        "TRT-LLM BF16 requires SM10x",
    )
    def test_trtllm(self):
        self._check_backend("trtllm")


if __name__ == "__main__":
    unittest.main()
