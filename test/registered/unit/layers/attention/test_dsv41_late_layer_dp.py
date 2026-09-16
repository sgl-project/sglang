"""CPU tensor and Gloo regressions for decoder SWA replay's DP MoE layout."""

import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.layers import dp_attention as dp
from sglang.srt.layers.attention.dsv4.late_layer import (
    LateLayerDPLayout,
    scatter_tail_rows,
    select_tail_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _batch(dp_size, rank, mode, non_padded=512):
    return SimpleNamespace(
        global_num_tokens_cpu=[512] * dp_size,
        global_num_tokens_gpu=torch.full((dp_size,), 512, dtype=torch.int64),
        global_dp_buffer_len=512 * dp_size,
        dp_padding_mode=mode,
        dp_local_start_pos=torch.tensor(rank * 512),
        dp_local_num_tokens=torch.tensor(512),
        num_token_non_padded=torch.tensor(non_padded, dtype=torch.int32),
        num_token_non_padded_cpu=non_padded,
    )


class TestLateLayerRows(CustomTestCase):
    def test_model_forward_pads_only_moe_and_restores_residual_rows(self):
        from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer

        for n in (0, 1, 129):
            with self.subTest(local_rows=n):
                batch = _batch(2, 0, dp.DpPaddingMode.MAX_LEN, non_padded=n)
                old = vars(batch).copy()
                layout = LateLayerDPLayout.from_counts(
                    [n, 129], dp_rank=0, attn_tp_size=2, batch=batch, device="cpu"
                )
                hidden = torch.arange(n * 4 * 3).float().reshape(n, 4, 3)
                ids = torch.arange(n, dtype=torch.int32)
                global_ids = torch.arange(sum(layout.counts), dtype=ids.dtype)
                stats = torch.ones(n, 4)
                precomputed = object()

                def attention(*, x, **kwargs):
                    self.assertEqual(x.shape[0], n)
                    self.assertIs(
                        batch.global_num_tokens_cpu, old["global_num_tokens_cpu"]
                    )
                    return x + 1

                def moe(x, forward_batch, *, input_ids, input_ids_global):
                    self.assertEqual(x.shape[0], 130)
                    self.assertEqual(dp.get_global_dp_buffer_len(), 260)
                    self.assertIs(forward_batch.global_num_tokens_cpu, layout.counts)
                    self.assertIs(input_ids_global, global_ids)
                    torch.testing.assert_close(input_ids[:n], ids)
                    return x * 2 + input_ids[:, None]

                def post(x, residual, post, comb):
                    self.assertEqual(x.shape[0], n)
                    self.assertEqual(residual.shape, hidden.shape)
                    self.assertIs(
                        batch.global_num_tokens_cpu, old["global_num_tokens_cpu"]
                    )
                    return residual + x[:, None]

                attn = Mock(side_effect=attention)
                attn.accepts_mxfp8_swizzled_input.return_value = False
                attn.maybe_use_decode_attn_tp.side_effect = lambda _: nullcontext()
                layer = SimpleNamespace(
                    config=SimpleNamespace(model_type="deepseek_v41"),
                    self_attn=attn,
                    input_layernorm=None,
                    post_attention_layernorm=None,
                    hc_attn_fn=None,
                    hc_attn_scale=None,
                    hc_attn_base=None,
                    hc_ffn_fn=None,
                    hc_ffn_scale=None,
                    hc_ffn_base=None,
                    _get_hc_stats_stream=lambda *_: None,
                    _hc_mix_stats=Mock(return_value=(stats, stats, stats)),
                    _hc_combine=Mock(side_effect=lambda x, **_: x[:, 0]),
                    _run_moe_ffn_dp_sync=moe,
                    hc_post=post,
                )
                with dp.dp_buffer_size_scope(
                    1024,
                    512,
                    True,
                    old["global_num_tokens_cpu"],
                    old["global_num_tokens_gpu"],
                ):
                    output, next_pre = DeepseekV4DecoderLayer.forward_hc_pre_from_prev(
                        layer,
                        positions=ids,
                        hidden_states=hidden,
                        input_ids=ids,
                        forward_batch=batch,
                        input_ids_global=global_ids,
                        prev_pre=None,
                        precomputed_attn=precomputed,
                        late_dp_layout=layout,
                    )
                    self.assertEqual(dp.get_global_dp_buffer_len(), 1024)
                after_attn = hidden + (hidden[:, 0] + 1)[:, None]
                expected = after_attn + (after_attn[:, 0] * 2 + ids[:, None])[:, None]
                torch.testing.assert_close(output, expected)
                self.assertIs(next_pre, stats)
                self.assertIs(
                    layer._hc_combine.call_args_list[0].kwargs["precomputed"],
                    precomputed,
                )
                for name, value in old.items():
                    self.assertIs(getattr(batch, name), value, name)

    def test_single_request_excludes_dp_padding_and_restores_positions(self):
        original = torch.arange(132 * 3).reshape(132, 3)
        indices = torch.arange(1, 129)  # 129 real tokens plus three DP padding rows
        tail = SimpleNamespace(token_indices=indices, contiguous_start=1)
        selected = select_tail_rows(original, token_indices=indices, contiguous_start=1)
        torch.testing.assert_close(selected, original[1:129])
        self.assertEqual(
            selected.untyped_storage().data_ptr(), original.untyped_storage().data_ptr()
        )
        restored = scatter_tail_rows(tail, selected + 7, original.shape[0])
        self.assertEqual(restored.shape, original.shape)
        torch.testing.assert_close(restored[indices], original[indices] + 7)

    def test_multiple_requests_keep_each_tail_in_request_order(self):
        original = torch.arange(600)
        indices = torch.cat((torch.arange(128, 256), torch.arange(384, 512)))
        selected = select_tail_rows(
            original, token_indices=indices, contiguous_start=None
        )
        torch.testing.assert_close(
            selected, torch.cat((original[128:256], original[384:512]))
        )
        restored = scatter_tail_rows(
            SimpleNamespace(token_indices=indices, contiguous_start=None),
            selected,
            original.shape[0],
        )
        torch.testing.assert_close(restored[indices], original[indices])

    def test_scope_restores_cached_offsets_masks_and_buffers_on_failure(self):
        batch = _batch(2, 1, dp.DpPaddingMode.MAX_LEN, non_padded=0)
        old = vars(batch).copy()
        layout = LateLayerDPLayout.from_counts(
            [129, 0], dp_rank=1, attn_tp_size=2, batch=batch, device="cpu"
        )
        self.assertEqual(layout.counts, [130, 130])
        self.assertEqual(layout.num_token_non_padded_cpu, 0)
        with dp.dp_buffer_size_scope(
            1024, 512, True, old["global_num_tokens_cpu"], old["global_num_tokens_gpu"]
        ):
            with self.assertRaisesRegex(RuntimeError, "MoE failure"):
                with layout.activate(batch):
                    self.assertIsNone(batch.dp_local_start_pos)
                    self.assertIsNone(batch.dp_local_num_tokens)
                    self.assertEqual(dp.get_global_dp_buffer_len(), 260)
                    self.assertEqual(dp.get_local_dp_buffer_len(), 130)
                    self.assertEqual(dp.get_dp_global_num_tokens(), [130, 130])
                    batch.dp_local_start_pos = torch.tensor(130)
                    batch.dp_local_num_tokens = torch.tensor(130)
                    raise RuntimeError("MoE failure")
            self.assertEqual(dp.get_global_dp_buffer_len(), 1024)
            self.assertEqual(dp.get_local_dp_buffer_len(), 512)
            self.assertIs(dp.get_dp_global_num_tokens(), old["global_num_tokens_cpu"])
        for name, value in old.items():
            self.assertIs(getattr(batch, name), value, name)

    def test_dp_attention_allowed_but_prefill_graph_still_rejected(self):
        from sglang.srt.arg_groups import deepseek_v4_hook as hook
        from sglang.srt.model_executor.cuda_graph_config import Backend

        prefill = SimpleNamespace(backend=Backend.DISABLED, max_seq_len=4096)
        cfg = SimpleNamespace(
            enable_encoder_swa_bounded_replay=False,
            enable_decoder_swa_bounded_replay=True,
            enable_dp_attention=True,
            speculative_algorithm=None,
            enable_hisparse=False,
            dsv4_attn_backend="auto",
            enable_two_batch_overlap=False,
            pp_size=1,
            disaggregation_mode="null",
            cuda_graph_config=SimpleNamespace(prefill=prefill),
        )
        model = SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41"))
        with (
            patch.object(hook, "resolving_view", return_value=cfg),
            patch.object(hook, "model_config_of", return_value=model),
            patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                return_value=False,
            ),
        ):
            hook.validate_deepseek_v41_features(object())
            prefill.backend = Backend.BREAKABLE
            with self.assertRaisesRegex(ValueError, "prefill CUDA graph"):
                hook.validate_deepseek_v41_features(object())


class _GlooGroup:
    """CPU implementations of the coordinator collectives used by DP gathering."""

    def __init__(self, group, rank, size):
        self.group, self.rank_in_group, self.world_size = group, rank, size
        self.unique_name = "late-layer-gloo"

    def all_gather_into_tensor(self, output, tensor):
        dist.all_gather_into_tensor(output, tensor.contiguous(), group=self.group)

    def reduce_scatter_tensor(self, output, tensor):
        # Gloo versions without reduce_scatter support can still exercise the
        # same attention-TP reduction and partition using all_reduce.
        tensor = tensor.clone()
        dist.all_reduce(tensor, group=self.group)
        output.copy_(tensor.tensor_split(self.world_size)[self.rank_in_group])

    def all_gatherv(self, tensor, *, sizes, output):
        assert tensor.shape[0] == sizes[self.rank_in_group]
        padded = tensor.new_zeros((max(sizes), *tensor.shape[1:]))
        padded[: tensor.shape[0]].copy_(tensor)
        gathered = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(gathered, padded, group=self.group)
        output.copy_(torch.cat([x[:n] for x, n in zip(gathered, sizes)]))


def _all_reduce(tensor, **kwargs):
    dist.all_reduce(tensor)
    return tensor


def _distributed_worker(rank, world_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=45),
    )
    try:
        tp_group = _GlooGroup(dist.group.WORLD, rank, world_size)
        for attn_tp_size in (1, 2):
            dp_size, dp_rank = world_size // attn_tp_size, rank // attn_tp_size
            attn_rank = rank % attn_tp_size
            for start in range(0, world_size, attn_tp_size):
                group = dist.new_group(list(range(start, start + attn_tp_size)))
                if start <= rank < start + attn_tp_size:
                    attn_group = _GlooGroup(group, attn_rank, attn_tp_size)
            with ExitStack() as stack:
                for name, value in dict(
                    get_attention_dp_rank=lambda: dp_rank,
                    get_attention_dp_size=lambda: dp_size,
                    get_attn_tensor_model_parallel_rank=lambda: attn_rank,
                    get_attn_tensor_model_parallel_world_size=lambda: attn_tp_size,
                    get_tensor_model_parallel_world_size=lambda: world_size,
                    get_tp_group=lambda: tp_group,
                    get_attn_tp_group=lambda: attn_group,
                    world_dp_gather_enabled=lambda: False,
                    tensor_model_parallel_all_reduce=_all_reduce,
                    memcpy_func=dp.memcpy_cpu,
                    _use_dp_gather_fp8=lambda: False,
                ).items():
                    stack.enter_context(patch.object(dp, name, value))
                stack.enter_context(
                    patch("sglang.srt.distributed.get_tp_group", return_value=tp_group)
                )
                stack.enter_context(
                    patch(
                        "sglang.srt.distributed.parallel_state.inplace_all_reduce",
                        _all_reduce,
                    )
                )
                stack.enter_context(
                    patch(
                        "sglang.srt.runtime_context.get_parallel",
                        return_value=SimpleNamespace(attn_tp_size=attn_tp_size),
                    )
                )
                scenarios = (
                    ([128, 7, 1, 0], [129, 3, 0, 1])
                    if dp_size == 4
                    else ([128, 7], [129, 0], [128, 1])
                )
                for real_counts in scenarios:
                    for mode, gatherv in (
                        (dp.DpPaddingMode.SUM_LEN, False),
                        (dp.DpPaddingMode.MAX_LEN, False),
                        (dp.DpPaddingMode.SUM_LEN, True),
                    ):
                        with patch.object(dp, "_USE_DP_GATHERV", gatherv):
                            n = real_counts[dp_rank]
                            batch = _batch(dp_size, dp_rank, mode, non_padded=n)
                            old = vars(batch).copy()
                            ids = torch.arange(n, dtype=torch.int32) + dp_rank * 10000
                            original_ids = ids.clone()
                            layout = LateLayerDPLayout.prepare(ids, batch)
                            expected_parts = []
                            for r, count in enumerate(real_counts):
                                part = torch.zeros(layout.counts[r], dtype=ids.dtype)
                                part[:count] = torch.arange(count) + r * 10000
                                expected_parts.append(part)
                            global_ids = layout.gather_input_ids(
                                ids, batch, gather=True
                            )
                            torch.testing.assert_close(
                                global_ids, torch.cat(expected_parts)
                            )
                            torch.testing.assert_close(ids, original_ids)
                            torch.testing.assert_close(
                                layout.gather_input_ids(ids, batch, gather=False),
                                expected_parts[dp_rank],
                            )
                            local_hidden = torch.stack(
                                (ids.float() + 1, ids.float() + 2), dim=-1
                            )
                            expected_local = local_hidden * 2 + ids[:, None]
                            with layout.activate(batch):
                                gathered = local_hidden.new_empty(
                                    (sum(layout.counts), 2)
                                )
                                dp.dp_gather_replicate(
                                    gathered, layout.pad(local_hidden), batch
                                )
                                # A per-token expert using token IDs makes ordering
                                # errors visible, even when all lengths are equal.
                                expert_output = gathered * 2 + global_ids[:, None]
                                local_output = local_hidden.new_empty(
                                    (layout.counts[dp_rank], 2)
                                )
                                dp.dp_scatter(local_output, expert_output, batch)
                                torch.testing.assert_close(
                                    local_output[:n], expected_local
                                )
                            for name, value in old.items():
                                assert getattr(batch, name) is value, name
    finally:
        dist.destroy_process_group()


class TestLateLayerDPCollectives(CustomTestCase):
    @unittest.skipUnless(dist.is_gloo_available(), "Gloo is required")
    def test_uneven_idle_and_decode_ranks_with_attention_tp(self):
        with tempfile.TemporaryDirectory() as temp:
            rendezvous = (Path(temp) / "gloo").as_uri()
            mp.spawn(_distributed_worker, args=(4, rendezvous), nprocs=4, join=True)


if __name__ == "__main__":
    unittest.main()
