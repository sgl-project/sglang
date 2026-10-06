import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.test.ci.ci_register import register_cpu_ci

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


class TestDeepseekV4CPKVStore(unittest.TestCase):
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
