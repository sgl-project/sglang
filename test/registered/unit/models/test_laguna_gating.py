"""Output-gating allocation tests for LagunaAttention.

Laguna applies a softplus output gate after attention. The gate granularity is
set by ``config.gating``:

  - ``True`` / ``"per-head"`` : one gate per head, broadcast across head_dim
                                (g_proj output dim = num_heads).
  - ``"per-element"``         : one gate per (head, head_dim) channel
                                (g_proj output dim = num_heads * head_dim).

These tests construct ``LagunaAttention`` on CPU (TP=1) and assert that each
``gating`` value sizes ``g_proj`` and sets ``gate_per_head`` correctly. The
dispatch mirrors vLLM's ``gate_per_head`` branch.
"""

import os
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _ensure_dist_initialized() -> None:
    """Minimal single-rank gloo environment plus TP=1/PP=1/EP=1 model-parallel
    groups, required before constructing the parallel linears inside
    ``LagunaAttention.__init__``.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29641")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )


NUM_HEADS = 8
HEAD_DIM = 16


def _make_attention(gating):
    from sglang.srt.models.laguna import LagunaAttention

    try:
        return LagunaAttention(
            hidden_size=64,
            num_heads=NUM_HEADS,
            num_kv_heads=2,
            head_dim=HEAD_DIM,
            layer_id=0,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            rope_scaling=None,
            partial_rotary_factor=1.0,
            max_position_embeddings=2048,
            attention_bias=False,
            sliding_window_size=-1,
            layer_type="full_attention",
            gating=gating,
            quant_config=None,
            prefix="",
        )
    except ModuleNotFoundError as exc:
        # On a bare runner with no CUDA kernels that isn't flagged as sglang's
        # CPU build, rope's fallback path imports vllm, which may be absent.
        # The gate wiring under test is unaffected, so skip where it can't build.
        raise unittest.SkipTest(f"rope kernel backend unavailable: {exc}")


class TestLagunaGating(CustomTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_per_head_true(self):
        """``gating=True`` -> per-head, g_proj dim = num_heads."""
        attn = _make_attention(True)
        self.assertTrue(attn.gate_per_head)
        self.assertEqual(attn.g_proj.weight.shape[0], NUM_HEADS)

    def test_per_head_string(self):
        """``gating="per-head"`` -> per-head, g_proj dim = num_heads."""
        attn = _make_attention("per-head")
        self.assertTrue(attn.gate_per_head)
        self.assertEqual(attn.g_proj.weight.shape[0], NUM_HEADS)

    def test_per_element(self):
        """``gating="per-element"`` -> g_proj dim = num_heads * head_dim."""
        attn = _make_attention("per-element")
        self.assertFalse(attn.gate_per_head)
        self.assertEqual(attn.g_proj.weight.shape[0], NUM_HEADS * HEAD_DIM)

    def test_disabled_gating(self):
        """Falsy ``gating`` builds no g_proj (ungated attention)."""
        for value in (False, None):
            attn = _make_attention(value)
            self.assertFalse(attn.gating)
            self.assertFalse(attn.gate_per_head)
            self.assertIsNone(attn.g_proj)

    def test_invalid_gating_raises(self):
        """An unsupported gating value raises ValueError."""
        with self.assertRaises(ValueError):
            _make_attention("bogus")


if __name__ == "__main__":
    unittest.main()
