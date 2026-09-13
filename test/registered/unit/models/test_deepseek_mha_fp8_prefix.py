from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_rocm_fp8_prefix_reads_hybrid_full_attention_page_table():
    slot_indices = torch.tensor([4, 8, 12])
    translated_indices = torch.tensor([1, 2, 3])
    full_attention_backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(page_table_1_flattened=slot_indices)
    )
    translator = Mock()
    translator.translate_dcp_read_ids.return_value = translated_indices
    hybrid_backend = SimpleNamespace(
        full_attn_backend=full_attention_backend,
        kv_index_translator=translator,
    )

    kv_a = torch.ones(3, 1, 4, dtype=torch.bfloat16)
    k_pe = torch.ones(3, 1, 2, dtype=torch.bfloat16)
    pool = Mock()
    pool.get_mla_kv_buffer.return_value = kv_a, k_pe
    layer = SimpleNamespace(attn_mha=object())
    forward_batch = SimpleNamespace(forward_mode=object())

    with (
        patch.object(forward_mha, "get_attn_backend", return_value=hybrid_backend),
        patch.object(forward_mha, "get_token_to_kv_pool", return_value=pool),
        patch.object(forward_mha, "_use_aiter_gfx95", True),
        patch.object(
            forward_mha, "filter_dcp_local_kv_indices", return_value=slot_indices
        ),
    ):
        actual_kv_a, actual_k_pe = (
            forward_mha.DeepseekMHAForwardMixin._get_mla_kv_buffer_from_fp8_for_dsa(
                layer, forward_batch
            )
        )

    translator.translate_dcp_read_ids.assert_called_once_with(slot_indices)
    assert pool.get_mla_kv_buffer.call_args.args[0] is layer.attn_mha
    torch.testing.assert_close(
        pool.get_mla_kv_buffer.call_args.args[1], translated_indices
    )
    assert pool.get_mla_kv_buffer.call_args.args[2] == torch.bfloat16
    torch.testing.assert_close(actual_kv_a, kv_a.squeeze(1))
    assert actual_k_pe is k_pe
