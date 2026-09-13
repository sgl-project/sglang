"""Memory limits and attention correctness for K3 expanded MLA under CP."""

import unittest
from contextlib import nullcontext
from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.layers.cp.padding import pad_logical_token_to_physical
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
    DeepseekMHAForwardMixin,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.models.kimi_k3 import KimiK3MLAAttention
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")
_K3 = "sglang.srt.models.kimi_k3."
_MHA = "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha."
_FA = "sglang.srt.layers.attention.flashattention_backend."


def _attention_fixture():
    attention = KimiK3MLAAttention.__new__(KimiK3MLAAttention)
    nn.Module.__init__(attention)
    attention._cp_mha_max_workspace_bytes = 512 * 1024**2
    attention.num_local_heads = 16
    attention.qk_nope_head_dim = 128
    attention.qk_rope_head_dim = 64
    attention.qk_head_dim = 192
    attention.v_head_dim = 128
    attention.kv_lora_rank = 512
    attention.q_lora_rank = 1536
    attention.hidden_size = 4096
    attention.current_attention_backend = "fa4"
    attention.rotary_emb = None
    attention.use_dsa = False
    attention.kv_b_proj = Mock()
    attention._kimi_split_gguf_kv_b = False
    return attention


class TestKimiK3CPMHAWorkspace(CustomTestCase):
    def setUp(self):
        self.enterContext(
            get_context().override_server_args(enable_deterministic_inference=False)
        )
        self.attention = _attention_fixture()
        self.batch = SimpleNamespace(
            seq_lens_cpu=[211, 307],
            attn_cp_metadata=SimpleNamespace(per_rank_actual_token=[40, 48]),
        )
        self.enterContext(get_parallel().override(attn_cp_size=2))
        self.enterContext(patch(_K3 + "is_cp_active", return_value=True))
        self.enterContext(patch(_K3 + "get_is_capture_mode", return_value=False))
        self.enterContext(patch(_K3 + "is_in_breakable_cuda_graph", return_value=False))
        self.enterContext(
            patch(_K3 + "is_in_tc_piecewise_cuda_graph", return_value=False)
        )
        self.pool = SimpleNamespace(dtype=torch.bfloat16)
        self.enterContext(patch(_K3 + "get_token_to_kv_pool", return_value=self.pool))

    def test_workspace_limit_switches_at_exact_byte_boundary(self):
        needed = self.attention._cp_mha_workspace_size(self.batch)
        # It must budget at least full expanded K+V, rather than only the
        # rank-local current query rows (cached prefixes dominate this case).
        expanded_kv = 518 * 16 * (192 + 128) * 2
        self.assertGreater(needed, expanded_kv)
        for budget, eligible in ((0, False), (needed - 1, False), (needed, True)):
            with self.subTest(budget=budget):
                self.attention._cp_mha_max_workspace_bytes = budget
                self.assertEqual(self.attention._can_use_cp_mha(self.batch), eligible)

    def test_default_budget_admits_k3_geometry_up_to_128k(self):
        # The env default (SGLANG_K3_CP_MHA_MAX_WORKSPACE_MB) must let real K3
        # (96 heads, q_lora 1536, hidden 7168) take expanded MHA for a 128K
        # prefill at CP8 and CP4, and stop at 256K/CP8.
        from sglang.srt.environ import envs

        budget = envs.SGLANG_K3_CP_MHA_MAX_WORKSPACE_MB.get() * 1024 * 1024
        self.attention.hidden_size = 7168
        for cp, heads, tokens, expect in (
            (8, 96, 131072, True),
            (4, 48, 131072, True),
            (8, 96, 262144, False),
        ):
            self.attention.num_local_heads = heads
            batch = SimpleNamespace(
                seq_lens_cpu=[tokens],
                attn_cp_metadata=SimpleNamespace(
                    per_rank_actual_token=[tokens // cp] * cp
                ),
            )
            with get_parallel().override(attn_cp_size=cp):
                needed = self.attention._cp_mha_workspace_size(batch)
            with self.subTest(cp=cp, tokens=tokens):
                self.assertEqual(needed <= budget, expect, (needed, budget))

    def test_cached_prefix_growth_can_force_absorbed_fallback(self):
        self.attention._cp_mha_max_workspace_bytes = (
            self.attention._cp_mha_workspace_size(self.batch)
        )
        self.assertTrue(self.attention._can_use_cp_mha(self.batch))
        # The current-token geometry remains unchanged; only cached KV grows.
        self.batch.seq_lens_cpu = [212, 307]
        self.assertFalse(self.attention._can_use_cp_mha(self.batch))

    def test_asymmetric_ranks_make_the_same_memory_decision(self):
        needed = []
        for rank in (0, 1):
            with get_parallel().override(attn_cp_rank=rank):
                needed.append(self.attention._cp_mha_workspace_size(self.batch))
        self.assertEqual(needed[0], needed[1])

    def test_unsupported_paths_retain_existing_dispatch(self):
        for name, value in (
            ("current_attention_backend", "flashinfer"),
            ("rotary_emb", Mock()),
            ("q_lora_rank", None),
            ("use_dsa", True),
            ("_kimi_split_gguf_kv_b", True),
            ("kv_b_proj", None),
        ):
            with self.subTest(name=name), patch.object(self.attention, name, value):
                self.assertFalse(self.attention._can_use_cp_mha(self.batch))
        projection = self.attention.kv_b_proj
        del self.attention.kv_b_proj
        self.assertFalse(self.attention._can_use_cp_mha(self.batch))
        self.attention.kv_b_proj = projection
        for dtype in (torch.float8_e4m3fn, torch.float32):
            with self.subTest(dtype=dtype):
                self.pool.dtype = dtype
                self.assertFalse(self.attention._can_use_cp_mha(self.batch))
        self.pool.dtype = torch.float16
        self.assertTrue(self.attention._can_use_cp_mha(self.batch))
        with patch.object(self.attention, "current_attention_backend", "fa3"):
            self.assertTrue(self.attention._can_use_cp_mha(self.batch))
        with patch(_K3 + "is_cp_active", return_value=False):
            self.assertFalse(self.attention._can_use_cp_mha(self.batch))
        with patch(_K3 + "get_is_capture_mode", return_value=True):
            self.assertFalse(self.attention._can_use_cp_mha(self.batch))
        with get_context().override_server_args(enable_deterministic_inference=True):
            self.assertFalse(self.attention._can_use_cp_mha(self.batch))
        with patch(_K3 + "is_in_breakable_cuda_graph", return_value=True):
            self.assertFalse(self.attention._can_use_cp_mha(self.batch))
        with patch(_K3 + "is_in_tc_piecewise_cuda_graph", return_value=True):
            self.assertFalse(self.attention._can_use_cp_mha(self.batch))
        self.batch.seq_lens_cpu = None
        self.assertFalse(self.attention._can_use_cp_mha(self.batch))

    def test_dispatch_uses_bounded_one_shot_and_preserves_parent_fallback(self):
        with patch.object(
            DeepseekV2AttentionMLA,
            "dispatch_attn_forward_method",
            return_value=AttnForwardMethod.MLA,
        ):
            self.assertEqual(
                self.attention.dispatch_attn_forward_method(self.batch),
                AttnForwardMethod.MHA_ONE_SHOT,
            )
            self.attention._cp_mha_max_workspace_bytes = 0
            self.batch.mha_one_shot = True
            self.batch.attn_attend_prefix_cache = True
            self.batch.mha_return_lse = True
            self.assertEqual(
                self.attention.dispatch_attn_forward_method(self.batch),
                AttnForwardMethod.MLA,
            )
            self.assertFalse(self.batch.mha_one_shot)
            self.assertIsNone(self.batch.attn_attend_prefix_cache)
            self.assertFalse(self.batch.mha_return_lse)


class TestKimiK3CPMHAPreparation(CustomTestCase):
    def test_packing_precedes_query_but_collective_launch_follows_query(self):
        for prefix in (0, 3):
            with self.subTest(prefix=prefix):
                self._check_prepare(prefix, deferred=True)

    def test_missing_producer_event_preserves_immediate_launch_fallback(self):
        for prefix in (0, 3):
            with self.subTest(prefix=prefix):
                self._check_prepare(prefix, deferred=False)

    def test_failed_preparation_uses_synchronous_cache_materialization(self):
        for prefix in (0, 3):
            with self.subTest(prefix=prefix):
                self._check_prepare(prefix, deferred=True, prepare_success=False)

    def test_query_failure_drains_unlaunched_packing_without_cache_write(self):
        for failure in ("q_norm", "q_projection"):
            with self.subTest(failure=failure):
                self._check_prepare(3, deferred=True, query_error=failure)

    def _check_prepare(
        self, prefix, *, deferred, query_error=None, prepare_success=True
    ):
        attention = _attention_fixture()
        attention._cp_kv_overlap = True
        attention.use_output_gate = False
        attention.num_local_heads = 2
        attention.q_lora_rank = 3
        attention.kv_lora_rank = 4
        attention.qk_nope_head_dim = 4
        attention.qk_rope_head_dim = 2
        attention.qk_head_dim = 6
        attention.v_head_dim = 4
        attention.attn_mqa = SimpleNamespace(layer_id=2)
        attention.attn_mha = SimpleNamespace(layer_id=2)
        attention.kv_cache_dtype = "auto"
        strategy = ZigzagCPStrategy(2)
        # Two local tokens, plus a deliberately different peer contribution.
        local_latent = torch.arange(18, dtype=torch.float32).view(2, 9) / 10
        original_latent = local_latent.clone()
        expected_local_kv = local_latent[:, 3:7] * 3
        peer_kv = expected_local_kv + 10
        new_kv = torch.cat([expected_local_kv, peer_kv], dim=0)
        new_pe = torch.cat([local_latent[:, 7:], local_latent[:, 7:] + 10])
        cache = torch.full((prefix + 4, 1, 6), torch.nan)
        cache[:prefix] = 0.25
        all_indices = torch.arange(prefix + 4)
        batch = SimpleNamespace(
            mha_one_shot=True,
            extend_prefix_lens_cpu=[prefix],
            out_cache_loc=torch.arange(prefix, prefix + 4),
            fetch_mha_one_shot_kv_indices=Mock(return_value=all_indices),
        )
        events, pending = [], []
        ready_event = Mock(name="kv_ready_event")
        normalized_kv = []

        def kv_norm(x):
            events.append("kv_norm")
            result = x * 3
            normalized_kv.append(result)
            return result

        def record_ready(fb, kv):
            self.assertIs(fb, batch)
            self.assertEqual(kv.data_ptr(), normalized_kv[0].data_ptr())
            torch.testing.assert_close(kv.reshape(2, 4), expected_local_kv)
            events.append("ready")
            return ready_event if deferred else None

        def prepare_or_start(
            fb, layer, kv, pe, *, producer_event=None, prepare_only=False
        ):
            self.assertIs(fb, batch)
            self.assertEqual(layer.layer_id, 2)
            self.assertIs(producer_event, ready_event if deferred else None)
            torch.testing.assert_close(kv.squeeze(1), expected_local_kv)
            torch.testing.assert_close(pe.squeeze(1), original_latent[:, 7:])
            self.assertEqual(prepare_only, deferred)
            if prepare_only:
                events.append("prepare_only")
                if not prepare_success:
                    return False
                pending.append("prepared")
            else:
                events.append("launch")
                pending.append("launched")
            return True

        def launch_prepared(fb, layer):
            self.assertIs(fb, batch)
            self.assertEqual(layer.layer_id, 2)
            self.assertEqual(pending.pop(), "prepared")
            events.append("launch")
            pending.append("launched")
            return True

        def q_norm(x):
            events.append("q_norm")
            if query_error == "q_norm":
                raise RuntimeError("q_norm failed")
            # Even an in-place Q normalization may only change the Q slice
            # of fused latent storage, leaving deferred K-PE input intact.
            return x.mul_(2)

        q_weight = torch.arange(36, dtype=torch.float32).view(3, 12) / 100

        def q_projection(x):
            events.append("q_projection")
            if query_error == "q_projection":
                raise RuntimeError("q_projection failed")
            return (x @ q_weight).view(2, 2, 6)

        def join(fb, layer, kv, pe):
            self.assertEqual(layer.layer_id, 2)
            if pending:
                self.assertEqual(pending.pop(), "launched")
                events.append("join")
            else:
                self.assertFalse(prepare_success)
                events.append("sync_gather")
            cache[fb.out_cache_loc, 0, :4] = new_kv
            cache[fb.out_cache_loc, 0, 4:] = new_pe

        def finish(fb, layer):
            self.assertIs(fb, batch)
            self.assertEqual(layer.layer_id, 2)
            if pending:
                # The real K3 finally must drain unfinished packing after Q
                # fails, without starting NCCL or publishing any new KV.
                self.assertEqual(pending.pop(), "prepared")
                self.assertIsNotNone(query_error)
                self.assertTrue(torch.isnan(cache[prefix:]).all())
                events.append("drain")
            return False

        def read(layer_id):
            self.assertEqual(layer_id, 2)
            self.assertFalse(pending, "Cache read raced a pending KV transfer")
            self.assertTrue(torch.isfinite(cache).all())
            events.append("read")
            return cache

        kv_weight = torch.arange(64, dtype=torch.float32).view(4, 16) / 100

        def kv_projection(x):
            self.assertEqual(events[-1], "read")
            events.append("kv_projection")
            return x @ kv_weight, None

        attention.kv_a_layernorm = Mock(side_effect=kv_norm)
        attention.q_a_layernorm = Mock(side_effect=q_norm)
        attention.q_b_proj_forward = Mock(side_effect=q_projection)
        attention.kv_b_proj = Mock(side_effect=kv_projection)
        attention.record_mla_cp_kv_producer_event = Mock(side_effect=record_ready)
        pool = SimpleNamespace(get_key_buffer=read)
        context = SimpleNamespace(fetch_qkv_latent=lambda: local_latent)

        def parent_forward(positions, hidden_states, fb, zero_allocator, **kwargs):
            return DeepseekMHAForwardMixin.forward_normal_prepare(
                attention, positions, hidden_states, fb, zero_allocator
            )

        with (
            patch(_K3 + "is_cp_active", return_value=True),
            patch(_K3 + "get_is_capture_mode", return_value=False),
            patch(_K3 + "get_cp_strategy", return_value=strategy),
            patch(_MHA + "is_cp_active", return_value=True),
            patch(_MHA + "get_cp_strategy", return_value=strategy),
            patch(_MHA + "get_attn_tp_context", return_value=context),
            patch(_MHA + "resolve_attn_backend", return_value=SimpleNamespace()),
            patch(_MHA + "get_token_to_kv_pool", return_value=pool),
            patch(_MHA + "_is_cuda", False),
            patch(_MHA + "_is_musa", False),
            patch(_MHA + "_use_aiter_gfx95", False),
            patch.object(
                strategy, "start_mla_kv_materialization", side_effect=prepare_or_start
            ) as prepare_transfer,
            patch.object(
                strategy, "launch_mla_kv_materialization", side_effect=launch_prepared
            ) as launch_transfer,
            patch.object(
                strategy, "materialize_full_mla_kv", side_effect=join
            ) as join_transfer,
            patch.object(
                strategy, "finish_mla_kv_materialization", side_effect=finish
            ) as finish_transfer,
            patch.object(DeepseekV2AttentionMLA, "forward", side_effect=parent_forward),
            get_parallel().override(dcp_enabled=False),
        ):
            if query_error is not None:
                with self.assertRaisesRegex(RuntimeError, f"{query_error} failed"):
                    attention.forward(torch.arange(2), torch.empty(2, 5), batch, None)
                prepare_transfer.assert_called_once()
                self.assertTrue(prepare_transfer.call_args.kwargs["prepare_only"])
                launch_transfer.assert_not_called()
                join_transfer.assert_not_called()
                finish_transfer.assert_called_once_with(batch, attention.attn_mqa)
                batch.fetch_mha_one_shot_kv_indices.assert_not_called()
                self.assertFalse(pending)
                self.assertTrue(torch.isnan(cache[prefix:]).all())
                self.assertEqual(
                    events,
                    ["kv_norm", "ready", "prepare_only", "q_norm"]
                    + (["q_projection"] if query_error == "q_projection" else [])
                    + ["drain"],
                )
                return
            q, k, v, returned_batch = attention.forward(
                torch.arange(2), torch.empty(2, 5), batch, None
            )
            prepare_transfer.assert_called_once()
            if deferred and prepare_success:
                launch_transfer.assert_called_once_with(batch, attention.attn_mqa)
            else:
                launch_transfer.assert_not_called()
            finish_transfer.assert_called_once_with(batch, attention.attn_mqa)
        if deferred:
            launch_order = ["prepare_only", "q_norm", "q_projection"]
            launch_order += ["launch", "join"] if prepare_success else ["sync_gather"]
        else:
            launch_order = ["launch", "q_norm", "q_projection", "join"]
        self.assertEqual(
            events,
            ["kv_norm", "ready"] + launch_order + ["read", "kv_projection"],
        )
        self.assertIs(returned_batch, batch)
        self.assertEqual(q.shape, (2, 2, 6))
        torch.testing.assert_close(
            q, ((original_latent[:, :3] * 2) @ q_weight).view(2, 2, 6)
        )
        # Even without a prefix, expansion must include all four current tokens.
        self.assertEqual(k.shape, (prefix + 4, 2, 6))
        self.assertEqual(v.shape, (prefix + 4, 2, 4))
        expected_expanded = (cache[:, 0, :4] @ kv_weight).view(-1, 2, 8)
        torch.testing.assert_close(k[..., :4], expected_expanded[..., :4])
        torch.testing.assert_close(k[..., 4:], cache[..., 4:].expand(-1, 2, -1))
        torch.testing.assert_close(v, expected_expanded[..., 4:])
        batch.fetch_mha_one_shot_kv_indices.assert_called_once()


def _sdpa(q, k, v, visible_lengths, scale):
    """Reference with explicit absolute visibility, independent of FA causality."""
    if q.shape[0] == 0:
        return v.new_empty((0, q.shape[1], v.shape[-1]))
    mask = torch.arange(k.shape[0], device=q.device)[None, :] < visible_lengths[:, None]
    return (
        F.scaled_dot_product_attention(
            q.transpose(0, 1).float(),
            k.transpose(0, 1).float(),
            v.transpose(0, 1).float(),
            attn_mask=mask,
            dropout_p=0.0,
            scale=scale,
        )
        .transpose(0, 1)
        .to(q.dtype)
    )


def _varlen_reference(
    q,
    k,
    v,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    assert causal
    q_offsets = cu_seqlens_q.tolist()
    k_offsets = cu_seqlens_k.tolist()
    used = (
        seqused_k.tolist()
        if seqused_k is not None
        else [b - a for a, b in zip(k_offsets, k_offsets[1:])]
    )
    results = []
    for i, length in enumerate(used):
        q_i = q[q_offsets[i] : q_offsets[i + 1]]
        k_i = k[k_offsets[i] : k_offsets[i] + length]
        v_i = v[k_offsets[i] : k_offsets[i] + length]
        visible = (
            torch.arange(q_i.shape[0], device=q.device) + length - q_i.shape[0] + 1
        )
        results.append(_sdpa(q_i, k_i, v_i, visible, softmax_scale))
    return torch.cat(results)


# The last case deliberately exercises empty logical chunks directly through
# the helper, beyond the scheduler's minimum-length admission guard.
NUMERICAL_CASES = (
    (2, [8, 13], [0, 0]),
    (4, [9, 14, 8], [7, 3, 17]),
    (4, [1, 0, 3], [5, 2, 4]),
)


def check_cp_mha_numerical(
    test, cp_size, extend_lens, prefix_lens, *, device="cpu", real_flash=False
):
    generator = torch.Generator(device=device).manual_seed(991)
    dtype = torch.bfloat16 if real_flash else torch.float32
    seq_lens = [p + n for p, n in zip(prefix_lens, extend_lens)]
    full_q = torch.randn(
        sum(extend_lens), 4, 192, generator=generator, device=device, dtype=dtype
    )
    full_k = torch.randn(
        sum(seq_lens), 4, 192, generator=generator, device=device, dtype=dtype
    )
    full_v = torch.randn(
        sum(seq_lens), 4, 128, generator=generator, device=device, dtype=dtype
    )
    scale = 192**-0.5
    q_offsets = [0] + list(accumulate(extend_lens))
    k_offsets = [0] + list(accumulate(seq_lens))
    reference = []
    for i, (prefix, length) in enumerate(zip(prefix_lens, extend_lens)):
        reference.append(
            _sdpa(
                full_q[q_offsets[i] : q_offsets[i + 1]],
                full_k[k_offsets[i] : k_offsets[i + 1]],
                full_v[k_offsets[i] : k_offsets[i + 1]],
                torch.arange(length, device=device) + prefix + 1,
                scale,
            )
        )
    expected = torch.cat(reference)
    layer = SimpleNamespace(
        tp_q_head_num=4,
        tp_k_head_num=4,
        tp_v_head_num=4,
        head_dim=192,
        v_head_dim=128,
        scaling=scale,
        logit_cap=0.0,
    )
    backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
    backend.fa_impl_ver = 4
    backend.num_splits = 1
    backend.device = device
    strategy = ZigzagCPStrategy(cp_size)
    batches, outputs = [], []
    init_cp_strategy(enable_prefill_cp=True, cp_size=cp_size, cp_strategy="zigzag")
    try:
        for rank in range(cp_size):
            with get_parallel().override(attn_cp_size=cp_size, attn_cp_rank=rank):
                with patch(
                    "sglang.srt.layers.cp.zigzag.get_device",
                    return_value=SimpleNamespace(device=device),
                ):
                    metadata = strategy.build_metadata(
                        sum(extend_lens), seq_lens, extend_lens
                    )
                pad_logical_token_to_physical(metadata)
                batch = SimpleNamespace(attn_cp_metadata=metadata)
                batches.append(batch)
                local_q = strategy.shard_hidden_states(full_q, batch)
                context = (
                    nullcontext()
                    if real_flash
                    else patch(
                        _FA + "flash_attn_varlen_func", side_effect=_varlen_reference
                    )
                )
                with context:
                    result = backend._forward_cp_mha(
                        local_q,
                        full_k,
                        full_v,
                        layer,
                        batch,
                        cu_seqlens_k=torch.tensor(
                            k_offsets, device=device, dtype=torch.int32
                        ),
                        max_seqlen_k=max(seq_lens),
                    )
                outputs.append(result)
                logical = metadata.per_rank_logical_token[rank]
                test.assertEqual(result.shape[0], local_q.shape[0])
                test.assertEqual(torch.count_nonzero(result[logical:]).item(), 0)
        group = Mock()

        def gather(output, local):
            rows = output.shape[0] // cp_size
            torch.cat([rank_output[:rows] for rank_output in outputs], out=output)

        group.all_gather_into_tensor.side_effect = gather
        with get_parallel().override(attn_cp_rank=0, attn_cp_group=group):
            actual = strategy.gather_hidden_states(outputs[0], batches[0])
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.03 if real_flash else 1e-5,
            atol=0.02 if real_flash else 1e-6,
        )
        return float((actual.float() - expected.float()).abs().max().item())
    finally:
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag")


class TestKimiK3CPMHAReference(CustomTestCase):
    def test_ragged_prefix_and_empty_queries_match_full_causal_attention(self):
        for cp_size, extend, prefix in NUMERICAL_CASES:
            with self.subTest(cp_size=cp_size, extend=extend, prefix=prefix):
                check_cp_mha_numerical(self, cp_size, extend, prefix)


if __name__ == "__main__":
    unittest.main()
