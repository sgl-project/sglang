import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_ENV_GATE = "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate"
_PREFILL = SimpleNamespace(
    forward_mode=SimpleNamespace(
        is_decode_or_idle=lambda: False,
        is_target_verify=lambda: False,
    )
)


def _layer(**attrs):
    layer = MQALayer.__new__(MQALayer)
    layer.__dict__.update(
        indexer=None,
        compressor=None,
        compress_ratio=4,
        _normalize_q_lora=lambda value: (value, value),
        **attrs,
    )
    return layer


@contextlib.contextmanager
def _cp_prefill(*, unified, gathered, kv_pool, **module_attrs):
    """Patch a CP prefill; yields the gather mock."""
    with contextlib.ExitStack() as stack:
        for name, value in module_attrs.items():
            stack.enter_context(mock.patch.object(deepseek_v4, name, value))
        stack.enter_context(
            mock.patch.object(deepseek_v4, "is_cp_active", return_value=True)
        )
        stack.enter_context(
            mock.patch.object(deepseek_v4, "get_token_to_kv_pool", return_value=kv_pool)
        )
        stack.enter_context(
            mock.patch(f"{_ENV_GATE}.is_unified_kv_triton", return_value=unified)
        )
        stack.enter_context(
            mock.patch(f"{_ENV_GATE}.is_unified_kv_fp8", return_value=False)
        )
        stack.enter_context(
            mock.patch.object(torch.cuda, "current_stream", return_value=object())
        )
        yield stack.enter_context(
            mock.patch.object(
                deepseek_v4, "cp_materialize_global_token_order", return_value=gathered
            )
        )


class TestDeepseekV4CPKVStore(CustomTestCase):
    def test_unified_fp8_pair_gathers_one_raw_byte_row(self):
        # Rows 0 and 2 are local; row 1 comes from the peer rank.
        global_nope = torch.arange(24, dtype=torch.uint8).view(3, 8)
        global_rope = torch.arange(6, dtype=torch.bfloat16).view(3, 2)
        local_nope, local_rope = global_nope[[0, 2]], global_rope[[0, 2]]
        gathered = torch.cat((global_nope, global_rope.view(torch.uint8)), dim=-1)
        forward_batch, stream = object(), object()

        with mock.patch.object(
            deepseek_v4, "cp_materialize_global_token_order", return_value=gathered
        ) as materialize:
            nope, rope = deepseek_v4._materialize_cp_unified_fp8_kv(
                local_nope.view(torch.float8_e4m3fn), local_rope, forward_batch, stream
            )

        packed, gathered_batch, gathered_stream = materialize.call_args.args
        self.assertTrue(
            torch.equal(
                packed, torch.cat((local_nope, local_rope.view(torch.uint8)), dim=-1)
            )
        )
        self.assertIs(gathered_batch, forward_batch)
        self.assertIs(gathered_stream, stream)
        self.assertEqual(nope.dtype, torch.float8_e4m3fn)
        self.assertTrue(nope.is_contiguous() and rope.is_contiguous())
        self.assertTrue(torch.equal(nope.view(torch.uint8), global_nope))
        self.assertTrue(torch.equal(rope, global_rope))

    def test_zigzag_pair_gather_restores_global_rows_before_swa_store(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )
        from sglang.srt.layers.cp.base import init_cp_strategy
        from sglang.srt.layers.cp.padding import pad_logical_token_to_physical
        from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
        from sglang.srt.runtime_context import get_parallel

        init_cp_strategy(enable_prefill_cp=True, cp_size=4, cp_strategy="zigzag")
        self.addCleanup(
            init_cp_strategy, enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag"
        )
        global_nope = torch.arange(22 * 8, dtype=torch.uint8).view(22, 8)
        global_rope = torch.arange(44, dtype=torch.bfloat16).view(22, 2)
        global_packed = torch.cat((global_nope, global_rope.view(torch.uint8)), dim=-1)
        strategy = ZigzagCPStrategy(4)
        batches, shards = [], []
        for rank in range(4):
            with (
                get_parallel().override(attn_cp_rank=rank, attn_cp_size=4),
                mock.patch(
                    "sglang.srt.layers.cp.zigzag.get_device",
                    return_value=SimpleNamespace(device="cpu"),
                ),
            ):
                metadata = strategy.build_metadata(22, [14, 142], [9, 13])
                pad_logical_token_to_physical(metadata)
                batch = SimpleNamespace(
                    input_ids=torch.arange(22),
                    extend_seq_lens_cpu=[9, 13],
                    forward_mode=SimpleNamespace(
                        is_context_parallel_extend=lambda: True, is_idle=lambda: False
                    ),
                    attn_cp_metadata=metadata,
                    out_cache_loc=torch.arange(22),
                )
                batches.append(batch)
                shards.append(strategy.shard_hidden_states(global_packed, batch))

        def gather(output, _):
            wire_rows = output.shape[0] // 4
            torch.cat([shard[:wire_rows] for shard in shards], out=output)

        locations = torch.arange(22).flip(0)
        for rank, batch in enumerate(batches):
            with get_parallel().override(
                attn_cp_rank=rank,
                attn_cp_size=4,
                attn_cp_group=SimpleNamespace(all_gather_into_tensor=gather),
            ):
                nope, rope = deepseek_v4._materialize_cp_unified_fp8_kv(
                    shards[rank][:, :8].contiguous().view(torch.float8_e4m3fn),
                    shards[rank][:, 8:].contiguous().view(torch.bfloat16),
                    batch,
                    None,
                )
            torch.testing.assert_close(nope.view(torch.uint8), global_nope)
            torch.testing.assert_close(rope, global_rope)
            stored = torch.zeros_like(global_packed)

            def store(*, layer_id, swa_loc, cache_k):
                stored[swa_loc] = cache_k

            backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
            backend.token_to_kv_pool = SimpleNamespace(
                request_window=None, set_swa_key_buffer_radix_fused=store
            )
            backend.forward_metadata = SimpleNamespace(
                core_attn_metadata=SimpleNamespace(swa_out_cache_loc=locations)
            )
            backend.store_cache(
                0,
                torch.cat((nope.view(torch.uint8), rope.view(torch.uint8)), dim=-1),
                batch,
            )
            torch.testing.assert_close(stored[locations], global_packed)

    def test_unified_cp_gathers_current_chunk_for_two_source_attention(self):
        q = torch.ones(2, 1, 2)
        local_kv = torch.arange(8, dtype=torch.float32).view(2, 4)
        global_kv = torch.arange(12, dtype=torch.float32).view(3, 4)
        layer = _layer(
            fuse_wqa_wkv=False,
            wq_a=mock.Mock(return_value=(torch.ones(2, 2), None)),
            _compute_kv_bf16=mock.Mock(return_value=local_kv),
        )

        with _cp_prefill(
            unified=True,
            gathered=global_kv,
            kv_pool=object(),
            _is_hip=True,
            _is_npu=False,
            # CPU CI never imports the ROCm helpers.
            _hip=SimpleNamespace(compute_q_b=lambda *_: (q, None, False)),
        ) as materialize:
            returned_q, returned_kv = layer._forward_prepare(
                torch.zeros(2, 4), torch.arange(2), _PREFILL, object()
            )

        self.assertIs(returned_q, q)
        self.assertIs(returned_kv, global_kv)
        self.assertIs(materialize.call_args.args[0], local_kv)
        self.assertIs(materialize.call_args.args[1], _PREFILL)

    def test_fused_cp_store_reuses_transformed_kv(self):
        qkv_a = torch.arange(24, dtype=torch.float32).view(4, 6)
        expected_kv = qkv_a[:, 2:].clone() + 100
        layer = _layer(
            fuse_wqa_wkv=True,
            q_lora_rank=2,
            use_fused_qk_norm_rope=True,
            layer_id=3,
            eps=1e-6,
            qk_rope_head_dim=2,
            cos_cache=torch.ones(1),
            sin_cache=torch.zeros(1),
            wqkv_a=mock.Mock(return_value=(qkv_a, None)),
            kv_norm=SimpleNamespace(weight=torch.ones(4)),
            wq_b=lambda value: (value, None),
            _compute_kv_bf16=mock.Mock(
                side_effect=AssertionError("fused KV must not be transformed twice")
            ),
        )
        kv_pool = SimpleNamespace(
            get_swa_raw_buffer=mock.Mock(return_value=object()),
            swa_page_size=256,
        )
        attn_backend = SimpleNamespace(
            get_swa_out_cache_loc=mock.Mock(return_value=torch.arange(4)),
            store_cache=mock.Mock(),
        )

        def fused_store(*, kv, q, **_):
            kv.add_(100)
            return q

        gathered_kv = object()
        with (
            _cp_prefill(
                unified=False,
                gathered=gathered_kv,
                kv_pool=kv_pool,
                _is_gfx95_supported=False,
            ) as materialize,
            mock.patch(
                "sglang.kernels.ops.attention.fused_qk_norm_rope_store.fused_qk_norm_rope_swa_store",
                side_effect=fused_store,
            ),
        ):
            _, returned_kv = layer._forward_prepare(
                torch.zeros(4, 6), torch.arange(4), _PREFILL, attn_backend
            )

        # The gather and store see the kv the fused kernel already transformed.
        torch.testing.assert_close(materialize.call_args.args[0], expected_kv)
        attn_backend.store_cache.assert_called_once_with(
            layer_id=3, swa_k=gathered_kv, forward_batch=_PREFILL
        )
        self.assertIs(returned_kv, gathered_kv)


if __name__ == "__main__":
    unittest.main()
