"""Block-FP8 MoE whose TP shard splits checkpoint blocks must load and match TP1.

Qwen3.8-Flash-Next's MTP experts (128x128 blocks, 640 intermediate) put 320 rows
on each of two ranks; the layer used to be rejected. The ranks' outputs must sum
to the unsharded layer's output.
"""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

from sglang.srt import runtime_context
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

E, H, I, TOPK, TP, BLOCK = 8, 512, 640, 2, 2, 128
DEVICE = torch.device("cuda")


def _quantize_blocks(w):
    """FP8 weight and FP32 per-128x128-block scales of a [rows, cols] matrix."""
    rows, cols = w.shape
    tiles = w.view(rows // BLOCK, BLOCK, cols // BLOCK, BLOCK)
    scale = tiles.abs().amax(dim=(1, 3)).clamp(min=1e-6) / 448.0
    q = (tiles / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return q.view(rows, cols), scale.float()


def _random_checkpoint():
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    shards = {}
    for expert_id in range(E):
        for shard_id, shape in (("w1", (I, H)), ("w3", (I, H)), ("w2", (H, I))):
            w = torch.randn(*shape, device=DEVICE, generator=gen) / 16
            # Vary block magnitudes so a scale applied to the wrong block shows.
            blocks = torch.rand(
                shape[0] // BLOCK, shape[1] // BLOCK, device=DEVICE, generator=gen
            )
            w *= (blocks + 0.5).repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
            shards[expert_id, shard_id] = _quantize_blocks(w)
    return shards


def _load(layer, shards):
    for (expert_id, shard_id), (weight, scale) in shards.items():
        prefix = "w2" if shard_id == "w2" else "w13"
        for name, tensor in (
            (f"{prefix}_weight", weight),
            (f"{prefix}_weight_scale_inv", scale),
        ):
            param = getattr(layer, name)
            param.weight_loader(
                param, tensor, name, shard_id=shard_id, expert_id=expert_id
            )


def _dequantize(weight, scale):
    return weight.float() * scale.repeat_interleave(BLOCK, dim=0).repeat_interleave(
        BLOCK, dim=1
    )


def _reference(shards, x, topk_weights, topk_ids):
    """FP32 experts on the dequantized checkpoint, without activation quantization."""
    x = x.float()
    out = torch.zeros_like(x)
    for expert_id in range(E):
        token, slot = (topk_ids == expert_id).nonzero(as_tuple=True)
        w1, w3, w2 = (_dequantize(*shards[expert_id, s]) for s in ("w1", "w3", "w2"))
        h = torch.nn.functional.silu(x[token] @ w1.T) * (x[token] @ w3.T)
        out.index_add_(0, token, topk_weights[token, slot, None] * (h @ w2.T))
    return out


def _relative_error(out, ref):
    return ((out - ref).norm() / ref.norm()).item()


class TestFp8MoETpBlockRefine(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29634)
        torch.set_default_device("cuda")

    @contextmanager
    def _runtime(self, backend, tp_size, tp_rank):
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(runner_backend=backend),
            # One process stands in for each rank in turn; its groups stay at
            # width 1, and the layer does not reduce across ranks.
            patch.object(runtime_context, "_validate_parallel"),
            get_parallel().override(
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=tp_size,
                moe_tp_rank=tp_rank,
                tp_size=tp_size,
                tp_rank=tp_rank,
            ),
        ):
            yield

    def _forward(self, backend, shards, x, topk_output, *, tp_size, tp_rank):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        with self._runtime(backend, tp_size, tp_rank):
            model = torch.nn.Module()
            model.experts = FusedMoE(
                num_experts=E,
                hidden_size=H,
                intermediate_size=I,
                layer_id=0,
                top_k=TOPK,
                params_dtype=torch.bfloat16,
                quant_config=Fp8Config(
                    is_checkpoint_fp8_serialized=True, weight_block_size=[BLOCK, BLOCK]
                ),
                prefix="model.layers.0.mlp.experts",
            ).cuda()
            _load(model.experts, shards)
            DefaultModelLoader.postprocess_weights(model, DEVICE)
            # The layer may write its result into the input.
            out = model.experts.forward(x.clone(), topk_output)
            if not isinstance(out, torch.Tensor):
                out = out[0] if isinstance(out, tuple) else out.hidden_states
            method = model.experts.quant_method
            return (method.weight_block_size, method.w2_weight_block_size), out.float()

    def _assert_ranks_match_tp1(self, backend, *, blocks, num_tokens):
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        torch.manual_seed(0)
        shards = _random_checkpoint()
        x = torch.randn(num_tokens, H, dtype=torch.bfloat16) / 4
        with self._runtime(backend, tp_size=1, tp_rank=0):
            topk_output = select_experts(
                hidden_states=x,
                router_logits=torch.randn(num_tokens, E, dtype=torch.float32),
                topk_config=TopKConfig(top_k=TOPK, renormalize=True),
            )
        ref = _reference(shards, x, topk_output.topk_weights, topk_output.topk_ids)

        _, tp1 = self._forward(backend, shards, x, topk_output, tp_size=1, tp_rank=0)
        summed = torch.zeros_like(tp1)
        for rank in range(TP):
            block_sizes, out = self._forward(
                backend, shards, x, topk_output, tp_size=TP, tp_rank=rank
            )
            self.assertEqual(block_sizes, blocks)
            summed += out
        # Rounding activations to FP8 dominates both errors (5-7%); misplaced
        # weights or scales would be off by about the size of the output.
        tp1_err, tp2_err = _relative_error(tp1, ref), _relative_error(summed, ref)
        self.assertLess(tp2_err, 1.02 * tp1_err)

    def test_triton(self):
        # TP splits w13 along N and w2 along K; only those are refined.
        self._assert_ranks_match_tp1(
            MoeRunnerBackend.TRITON,
            blocks=([64, BLOCK], [BLOCK, 64]),
            num_tokens=33,
        )

    @unittest.skipUnless(
        deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0, "needs DeepGEMM with UE8M0 scales"
    )
    def test_deep_gemm(self):
        # Below and above the token count where SM120 switches to its own
        # contiguous layout.
        for num_tokens in (33, 1100):
            with self.subTest(num_tokens=num_tokens):
                self._assert_ranks_match_tp1(
                    MoeRunnerBackend.DEEP_GEMM,
                    blocks=([64, BLOCK], [BLOCK, 32]),
                    num_tokens=num_tokens,
                )


if __name__ == "__main__":
    unittest.main()
