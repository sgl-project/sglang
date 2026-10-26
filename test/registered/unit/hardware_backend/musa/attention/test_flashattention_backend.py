import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

with patch.dict(
    sys.modules,
    {
        module: MagicMock()
        for module in (
            "flash_attn_interface",
            "sgl_kernel",
            "sgl_kernel.quantization",
            "sgl_kernel.scalar_type",
        )
    },
):
    from sglang.srt.hardware_backend.musa.attention import (
        flashattention_backend as musa_fa,
    )
    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionBackend,
    )
    from sglang.srt.layers.radix_attention import AttentionType

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PAGE_SIZE = 1
_BATCH_SIZE = 2


def _layer(*, layer_id, num_heads, head_dim):
    return SimpleNamespace(
        layer_id=layer_id,
        tp_q_head_num=num_heads,
        tp_k_head_num=num_heads,
        tp_v_head_num=num_heads,
        head_dim=head_dim,
        qk_head_dim=head_dim,
        v_head_dim=head_dim,
        is_cross_attention=False,
        sliding_window_size=None,
        attn_type=AttentionType.DECODER,
        scaling=1.0,
        logit_cap=0.0,
        k_scale=None,
    )


def _backend(*, num_hidden_layers, num_heads, head_dim, max_seq_len_k):
    model_runner = SimpleNamespace(
        model_config=SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            first_k_dense_replace=None,
            full_attention_interval=None,
        )
    )
    with patch.object(FlashAttentionBackend, "__init__", return_value=None):
        backend = musa_fa.MusaFlashAttentionBackend(model_runner)

    kv_cache = torch.zeros(8 * _PAGE_SIZE, num_heads, head_dim)
    backend.use_mla = False
    backend.page_size = _PAGE_SIZE
    backend.has_local_attention = False
    backend.attention_chunk_size = None
    backend.topk = 1
    backend.kv_cache_dtype_str = "auto"
    backend.use_sliding_window_kv_pool = False
    backend.num_splits = 0
    backend.token_to_kv_pool = SimpleNamespace(
        get_kv_buffer=lambda layer_id: (kv_cache, kv_cache)
    )
    backend.forward_metadata = SimpleNamespace(
        page_table=torch.zeros(_BATCH_SIZE, 4, dtype=torch.int32),
        cache_seqlens_int32=torch.full((_BATCH_SIZE,), 4, dtype=torch.int32),
        cu_seqlens_q=torch.arange(_BATCH_SIZE + 1, dtype=torch.int32),
        cu_seqlens_k=torch.arange(0, 4 * _BATCH_SIZE + 1, 4, dtype=torch.int32),
        max_seq_len_q=1,
        max_seq_len_k=max_seq_len_k,
        local_attn_metadata=None,
    )
    return backend


def _decode(backend, layer):
    forward_batch = SimpleNamespace(spec_info=None, can_run_tbo=False)
    q = torch.zeros(_BATCH_SIZE, layer.tp_q_head_num * layer.head_dim)
    backend.forward_decode(
        q=q, k=None, v=None, layer=layer, forward_batch=forward_batch
    )


class TestMusaFlashAttentionSchedulerMetadata(CustomTestCase):
    def test_target_metadata_ignores_later_built_draft_backend(self):
        """Under speculative decoding the draft backend is built after the target;
        the target's kernel calls must still get scheduler metadata from its own state."""
        kernel_metadata = []

        def kernel(**kwargs):
            kernel_metadata.append(kwargs["scheduler_metadata"])
            return torch.zeros_like(kwargs["q"])

        def scheduler_metadata(**kwargs):
            return (kwargs["num_heads_q"], kwargs["max_seqlen_k"])

        parallel = SimpleNamespace(
            pp_group=SimpleNamespace(rank_in_group=0, world_size=1)
        )
        with (
            patch.object(musa_fa, "mate_flash_attn_with_kvcache", side_effect=kernel),
            patch.object(
                musa_fa, "get_scheduler_metadata", side_effect=scheduler_metadata
            ),
            patch.object(musa_fa, "get_parallel", return_value=parallel),
            patch.dict(musa_fa._MATE_NO_MLA_SCHEDULER_METADATA_DICT, clear=True),
        ):
            target = _backend(
                num_hidden_layers=4, num_heads=8, head_dim=16, max_seq_len_k=100
            )
            draft = _backend(
                num_hidden_layers=2, num_heads=4, head_dim=16, max_seq_len_k=7
            )
            for layer_id in range(2):
                _decode(draft, _layer(layer_id=layer_id, num_heads=4, head_dim=16))
            _decode(target, _layer(layer_id=0, num_heads=8, head_dim=16))

        self.assertEqual(kernel_metadata, [(4, 7), (4, 7), (8, 100)])


if __name__ == "__main__":
    unittest.main()
