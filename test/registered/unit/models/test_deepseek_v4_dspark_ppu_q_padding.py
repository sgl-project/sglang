"""Unit tests for the PPU DSpark attention fixes.

Covers the staged changes in:

- python/sglang/srt/models/deepseek_v4.py
  MqaAttentionBase now gates the wo_a fp8 scale fixup on the resolved
  ``fp8`` flag instead of the global ``_FP8_WO_A_GEMM`` env, so callers
  that force ``wo_a_fp8=False`` (e.g. DSparkAttention) can construct
  cleanly even when SGLANG_OPT_FP8_WO_A_GEMM=1.

- python/sglang/srt/models/deepseek_v4_dspark.py
  DSparkAttention.forward no longer pads q to _PAD_NUM_HEADS=64 on PPU:
  the DeepseekV4AttnBackend slices the full attn_sink per TP rank down
  to n_local_heads, so q must keep its local head count or the backend
  assert ``attn_sink.shape[0] == q.shape[1]`` fails during CUDA graph
  capture.

All tests run on CPU only.
"""

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.model_executor.forward_context as forward_context
import sglang.srt.models.deepseek_v4 as dsv4
import sglang.srt.models.deepseek_v4_dspark as dspark
from sglang.srt.models.deepseek_v4 import MqaAttentionBase
from sglang.srt.models.deepseek_v4_dspark import DSparkAttention
from sglang.srt.runtime_context import get_context, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _StubLinear:
    """Minimal stand-in for the parallel linear layers (CPU-safe)."""

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def __call__(self, x):
        return x, None


class _ChannelwiseQuantStub:
    """quant_config that is not an Fp8Config instance."""


def _fake_config():
    return SimpleNamespace(
        hidden_size=64,
        qk_rope_head_dim=8,
        head_dim=16,
        num_attention_heads=16,
        num_key_value_heads=1,
        o_groups=4,
        q_lora_rank=16,
        o_lora_rank=8,
        rms_norm_eps=1e-6,
        compress_ratios=[0],
        max_position_embeddings=64,
        rope_theta=10000.0,
        rope_scaling=None,
    )


class TestMqaAttentionWoAFp8Gating(unittest.TestCase):
    """The wo_a fp8 scale fixup must follow the resolved ``fp8`` flag."""

    def setUp(self):
        self._saved_server_args = get_context()._server_args
        get_context().set_server_args(ServerArgs(model_path="dummy"))
        patcher = ExitStack()
        self.addCleanup(patcher.close)
        for name in ("ReplicatedLinear", "ColumnParallelLinear", "RowParallelLinear"):
            patcher.enter_context(patch.object(dsv4, name, _StubLinear))

    def tearDown(self):
        if self._saved_server_args is None:
            reset_context()
        else:
            get_context().set_server_args(self._saved_server_args)

    def _make_base(self, quant_config, **kwargs):
        return MqaAttentionBase(
            _fake_config(),
            0,
            quant_config,
            "layers.0.self_attn",
            attn_tp_rank=0,
            attn_tp_size=1,
            compress_ratio=0,
            fuse_wqa_wkv=False,
            wo_b_reduce_results=False,
            rope_original_seq_len=0,
            **kwargs,
        )

    def test_forced_wo_a_fp8_false_skips_fp8_fixup(self):
        # DSparkAttention forces wo_a_fp8=False; with the global
        # SGLANG_OPT_FP8_WO_A_GEMM=1 this used to enter the fp8 fixup and
        # crash on the missing weight_scale of the bf16 wo_a linear.
        with patch.object(dsv4, "_FP8_WO_A_GEMM", True):
            base = self._make_base(None, wo_a_fp8=False, wo_a_keeps_quant_config=False)
        self.assertFalse(hasattr(base.wo_a, "weight_scale"))
        self.assertFalse(hasattr(base.wo_a, "weight_scale_inv"))

    def test_fp8_wo_a_channelwise_requires_weight_scale(self):
        with patch.object(dsv4, "_FP8_WO_A_GEMM", True):
            with self.assertRaisesRegex(AssertionError, "weight_scale"):
                self._make_base(_ChannelwiseQuantStub())

    def test_fp8_wo_a_channelwise_with_weight_scale_builds(self):
        class _StubLinearWithScale(_StubLinear):
            weight_scale = None

        with (
            patch.object(dsv4, "_FP8_WO_A_GEMM", True),
            patch.object(dsv4, "ColumnParallelLinear", _StubLinearWithScale),
        ):
            self._make_base(_ChannelwiseQuantStub())


class _CapturingAttnBackend:
    """Records the q/attn_sink handed to the attention backend."""

    # Must equal head_dim so the wo_a einsum geometry matches the real model.
    v_head_dim = 16

    def __init__(self):
        self.q = None
        self.attn_sink = None

    def forward(self, **kwargs):
        self.q = kwargs["q"]
        self.attn_sink = kwargs["attn_sink"]
        return torch.zeros(
            self.q.shape[0], self.q.shape[1], self.v_head_dim, dtype=self.q.dtype
        )


def _make_fake_dspark_attn():
    n_local_heads, n_local_groups = 8, 2
    head_dim, o_lora_rank, rope_head_dim = 16, 4, 8
    wo_a_weight = torch.zeros(
        n_local_groups * o_lora_rank, n_local_heads * head_dim // n_local_groups
    )

    def _compute_q(x, positions, q_out=None):
        if q_out is not None:
            return q_out
        return torch.zeros(x.shape[0], n_local_heads, head_dim)

    return SimpleNamespace(
        rope_head_dim=rope_head_dim,
        head_dim=head_dim,
        n_local_heads=n_local_heads,
        n_local_groups=n_local_groups,
        o_lora_rank=o_lora_rank,
        alt_streams=None,
        _multi_stream_bs_limit=64,
        _use_fast_kernel=False,
        freqs_cis=torch.complex(
            torch.ones(64, rope_head_dim // 2), torch.zeros(64, rope_head_dim // 2)
        ),
        attn=SimpleNamespace(),
        wo_a=SimpleNamespace(weight=wo_a_weight),
        wo_b=lambda x: (x, None),
        kv_proj_only=lambda x: torch.zeros(x.shape[0], head_dim, dtype=x.dtype),
        _store_block_kv=lambda **kwargs: None,
        _compute_q=_compute_q,
        _local_attn_sink=lambda: torch.zeros(max(n_local_heads, dspark._PAD_NUM_HEADS)),
    )


@contextmanager
def _dspark_runtime(is_ppu: bool):
    """Patch the runtime hooks DSparkAttention.forward resolves globally."""
    backend = _CapturingAttnBackend()
    with ExitStack() as stack:
        stack.enter_context(patch.object(dspark, "_is_ppu", is_ppu))
        stack.enter_context(patch.object(dspark, "_resolve_dspark_pool", lambda: None))
        stack.enter_context(patch.object(dspark, "get_is_capture_mode", lambda: False))
        stack.enter_context(
            patch.object(forward_context, "get_attn_backend", lambda: backend)
        )
        yield backend


def _run_dspark_forward(is_ppu: bool):
    fake = _make_fake_dspark_attn()
    bs = 2
    hidden_states = torch.zeros(bs, fake.head_dim)
    positions = torch.arange(bs)
    with _dspark_runtime(is_ppu) as backend:
        out = DSparkAttention.forward(fake, positions, hidden_states, None)
    return backend, out


class TestDSparkAttentionQPadding(unittest.TestCase):
    """q head padding must match the backend's attn_sink convention."""

    def test_q_padded_to_64_heads_off_ppu(self):
        backend, out = _run_dspark_forward(is_ppu=False)
        self.assertEqual(
            backend.q.shape[1],
            dspark._PAD_NUM_HEADS,
            "off-PPU q must be padded to 64 heads to match the padded sink",
        )
        self.assertEqual(out.shape, (2, 8))

    def test_q_keeps_local_heads_on_ppu(self):
        backend, out = _run_dspark_forward(is_ppu=True)
        self.assertEqual(
            backend.q.shape[1],
            8,
            "on PPU q must keep n_local_heads; the backend slices the full "
            "attn_sink per TP rank instead",
        )
        self.assertEqual(out.shape, (2, 8))

    def test_sink_head_count_matches_q_for_backend_assert(self):
        # The backend asserts attn_sink.shape[0] == q.shape[1]. Reproduce
        # both conventions that feed it:
        # - PPU: the full n_heads sink is sliced per TP rank by the backend.
        # - non-PPU: DSparkAttention._local_attn_sink pads the sink to 64.
        n_heads, tp_size, n_local_heads = 64, 8, 8
        full_sink = torch.arange(n_heads, dtype=torch.float32)

        q_heads_ppu = 8  # from test_q_keeps_local_heads_on_ppu
        for tp_rank in range(tp_size):
            sliced = full_sink[tp_rank * n_local_heads : (tp_rank + 1) * n_local_heads]
            self.assertEqual(sliced.shape[0], q_heads_ppu)

        fake = SimpleNamespace(
            attn_tp_size=tp_size,
            attn_tp_rank=3,
            n_local_heads=n_local_heads,
            attn_sink=full_sink,
            _attn_sink_local=None,  # non-PPU init leaves this as None
        )
        padded_sink = DSparkAttention._local_attn_sink(fake)
        self.assertEqual(padded_sink.shape[0], dspark._PAD_NUM_HEADS)
        self.assertTrue(torch.equal(padded_sink[:n_local_heads], full_sink[24:32]))


if __name__ == "__main__":
    unittest.main()
