# SPDX-License-Identifier: Apache-2.0
"""Dense LTX attention replay must consume each request's Q/K/V and key mask."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
    BaseBreakableCudaGraphRunner,
)
from sglang.multimodal_gen.runtime.layers.attention import layer
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")


class _AttentionBlock(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, q, k, v, attn_mask):
        return self.attention(q + 0.25, k, v, attn_mask=attn_mask).sin()


class TestLTX2DenseAttentionBCG(CustomTestCase):
    @torch.no_grad()
    def test_capture_and_eager_break_replay_changed_masks(self):
        with ExitStack() as stack:
            for name, value in (
                ("get_attn_backend", SDPABackend),
                ("get_compute_dtype", torch.bfloat16),
                ("get_sequence_parallel_world_size", 1),
                ("get_ring_parallel_world_size", 1),
                ("_resolve_sp_attention_mode", ("ulysses", False)),
                ("_maybe_install_backend_autotune", None),
            ):
                stack.enter_context(patch.object(layer, name, return_value=value))
            stack.enter_context(set_forward_context(0, None))
            for cls in (layer.LocalAttention, layer.USPAttention):
                for capture_dense in (False, True):
                    with self.subTest(layer=cls.__name__, capture_dense=capture_dense):
                        attention = cls(
                            num_heads=4,
                            head_size=64,
                            is_cross_attention=True,
                            allow_cudnn_sdp=True,
                            capture_dense_attention=capture_dense,
                        )
                        model = _AttentionBlock(attention)
                        runner = BaseBreakableCudaGraphRunner(
                            model, torch.device("cuda:0")
                        )
                        kwargs = dict(
                            q=torch.randn(
                                2, 32, 4, 64, device="cuda", dtype=torch.bfloat16
                            ),
                            k=torch.randn(
                                2, 48, 4, 64, device="cuda", dtype=torch.bfloat16
                            ),
                            v=torch.randn(
                                2, 48, 4, 64, device="cuda", dtype=torch.bfloat16
                            ),
                            attn_mask=torch.ones(
                                2, 48, device="cuda", dtype=torch.int64
                            ),
                        )
                        self.assertTrue(runner.capture(**kwargs))
                        entry = next(iter(runner.entries.values()))
                        self.assertEqual(entry.num_segments, 1 if capture_dense else 2)
                        for start in (0, 16):
                            for key in ("q", "k", "v"):
                                kwargs[key].normal_()
                            kwargs["attn_mask"].fill_(1)
                            kwargs["attn_mask"][:, start : start + 16] = 0
                            expected = model(**kwargs)
                            actual = runner(**kwargs)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        runner.reset()

    def test_request_backend_overrides_bypass_captured_default(self):
        for backend, skip_softmax in (
            (AttentionBackendEnum.TORCH_SDPA, False),
            (None, True),
        ):
            with self.subTest(backend=backend, skip_softmax=skip_softmax):
                stage = object.__new__(LTX2DenoisingStage)
                stage._attention_backend_active_override = backend
                stage._skip_softmax_forced_fa = skip_softmax
                stage._maybe_get_bcg_runner = Mock(
                    side_effect=AssertionError("stale graph")
                )
                model = Mock(return_value="eager")
                result = stage._ltx2_call_current_model(
                    SimpleNamespace(is_warmup=False),
                    SimpleNamespace(current_model=model),
                    {"value": 7},
                )
                self.assertEqual(result, "eager")
                model.assert_called_once_with(value=7)
                stage._maybe_get_bcg_runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
