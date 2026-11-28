"""CPU contracts for A5 publication and mixed-prefill metadata reuse."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


def _load_fused_attention(*, cp_enabled=False):
    module_path = (
        Path(__file__).parents[4]
        / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload"
        / "fused_attention.py"
    )
    stubs = {
        name: ModuleType(name)
        for name in (
            "torch_npu",
            "sglang",
            "sglang.srt",
            "sglang.srt.layers",
            "sglang.srt.layers.attention",
            "sglang.srt.layers.attention.dsa",
            "sglang.srt.layers.attention.dsa.utils",
            "sglang.srt.model_executor",
            "sglang.srt.model_executor.forward_mode",
        )
    }
    stubs["sglang.srt.layers.attention.dsa.utils"].is_dsa_enable_prefill_cp = lambda: (
        cp_enabled
    )
    stubs["sglang.srt.model_executor.forward_mode"].ForwardMode = object
    spec = importlib.util.spec_from_file_location(
        "a5_fused_attention_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def test_publication_rows_are_built_once_per_target_forward():
    module = _load_fused_attention()
    publication_rows = tuple(torch.empty(2) for _ in range(4))
    plan = SimpleNamespace(query_width=1)
    metadata = SimpleNamespace(sparse_kv_lim_plan=plan, sparse_kv_offload_rows=None)
    manager = SimpleNamespace(
        start_layer=0,
        layer_num=2,
        lim_adapter=Mock(),
        prepare_offload_rows=Mock(return_value=publication_rows),
        offload_v2=Mock(
            return_value=(
                torch.empty(2, 656, dtype=torch.int8),
                torch.arange(2),
                torch.ones(2, dtype=torch.bool),
            )
        ),
        copy_sfa_attention=Mock(return_value=torch.empty(2, 512)),
    )
    backend = SimpleNamespace(
        sparse_kv_manager=manager,
        forward_metadata=metadata,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    forward_batch = SimpleNamespace(
        batch_size=2,
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: False),
    )
    tensors = {
        "q": torch.empty(2, 1, 512),
        "k": torch.empty(2, 1, 512),
        "v": torch.empty(2, 1, 512),
        "q_rope": torch.empty(2, 1, 64),
        "k_rope": torch.empty(2, 1, 64),
    }

    for layer_id in (0, 1):
        module.forward_fused_sparse_kv_offload(
            backend,
            layer=SimpleNamespace(layer_id=layer_id, tp_k_head_num=1),
            forward_batch=forward_batch,
            **tensors,
        )

    manager.prepare_offload_rows.assert_called_once_with(2, forward_batch)
    manager.lim_adapter.commit_forward.assert_called_once_with(2)
    assert metadata.sparse_kv_offload_rows is None
    assert metadata.sparse_kv_lim_plan is None
    assert manager.offload_v2.call_count == 2
    for call in manager.offload_v2.call_args_list:
        assert call.kwargs["offload_rows"] is publication_rows

    metadata.sparse_kv_lim_plan = plan
    module.forward_fused_sparse_kv_offload(
        backend,
        layer=SimpleNamespace(layer_id=0, tp_k_head_num=1),
        forward_batch=forward_batch,
        **tensors,
    )
    assert manager.prepare_offload_rows.call_count == 2
    manager.lim_adapter.commit_forward.assert_called_once()


def test_packed_prefill_dispatches_before_the_generic_cp_path():
    module = _load_fused_attention(cp_enabled=True)
    packed_prefill = Mock(return_value=torch.empty((2, 1, 512)))
    manager = SimpleNamespace(
        packed_c8=True,
        start_layer=0,
        layer_num=1,
        prepare_offload_rows=Mock(
            return_value=tuple(torch.arange(2) for _ in range(4))
        ),
        offload_v2=Mock(
            return_value=(
                torch.empty((2, 656), dtype=torch.int8),
                torch.arange(2),
                torch.ones(2, dtype=torch.bool),
            )
        ),
    )
    metadata = SimpleNamespace(sparse_kv_offload_rows=None)
    backend = SimpleNamespace(
        sparse_kv_manager=manager,
        forward_metadata=metadata,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        do_cp_balance_attn=Mock(),
    )
    batch = SimpleNamespace(
        batch_size=2,
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
        attn_cp_metadata=object(),
    )
    c8_prefill = SimpleNamespace(forward_packed_prefill=packed_prefill)
    with patch.dict(
        sys.modules,
        {
            "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.c8_prefill": c8_prefill
        },
    ):
        result = module.forward_fused_sparse_kv_offload(
            backend,
            q=torch.zeros((2, 1, 512)),
            k=torch.zeros((2, 1, 512)),
            v=torch.zeros((2, 1, 512)),
            q_rope=torch.zeros((2, 1, 64)),
            k_rope=torch.zeros((2, 1, 64)),
            topk_indices=torch.zeros((2, 3), dtype=torch.int32),
            layer=SimpleNamespace(layer_id=0, tp_k_head_num=1),
            forward_batch=batch,
        )

    assert result.shape == (2, 1, 512)
    packed_prefill.assert_called_once()
    backend.do_cp_balance_attn.assert_not_called()


def test_mixed_packed_prefill_builds_one_page_table_for_short_and_long_rows():
    module_path = (
        Path(__file__).parents[4]
        / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload"
        / "c8_prefill.py"
    )
    copy_packed = Mock()
    quant_attention = Mock(
        side_effect=lambda **kwargs: torch.zeros(
            (kwargs["query"].shape[0], kwargs["query"].shape[1], 512)
        )
    )
    stubs = {
        "torch_npu": SimpleNamespace(
            npu_kv_quant_sparse_flash_attention=quant_attention
        ),
        "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.c8_manager": SimpleNamespace(
            _copy_packed=copy_packed
        ),
        "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget": SimpleNamespace(
            C8_ROW_BYTES=656
        ),
    }
    spec = importlib.util.spec_from_file_location("a5_c8_prefill_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)

    manager = SimpleNamespace(
        page_size=128,
        start_layer=0,
        sparse_topk=3,
        budget=SimpleNamespace(copy_sfa_query_heads=2),
        host_kv_buffer=[torch.zeros((3, 128, 1, 656), dtype=torch.int8)],
        host_source_indices=Mock(
            side_effect=lambda rows, positions: rows * 1024 + positions
        ),
    )
    backend = SimpleNamespace(
        sparse_kv_manager=manager,
        forward_metadata=SimpleNamespace(
            actual_seq_lengths_q=torch.tensor([1, 2, 999])
        ),
    )
    batch = SimpleNamespace(
        batch_size=2,
        req_pool_indices=torch.tensor([1, 2, -1]),
        seq_lens=torch.tensor([1, 129, 0]),
        seq_lens_cpu=torch.tensor([1, 129, 0]),
        extend_seq_lens=torch.tensor([1, 1, 0]),
    )
    result = module.forward_packed_prefill(
        backend,
        torch.zeros((2, 1, 512)),
        torch.zeros((2, 1, 64)),
        SimpleNamespace(layer_id=0, tp_q_head_num=1, scaling=0.1),
        batch,
        torch.zeros((2, 3), dtype=torch.int32),
    )

    assert result.shape == (2, 1, 512)
    copy_packed.assert_called_once()
    kwargs = quant_attention.call_args.kwargs
    assert kwargs["actual_seq_lengths_query"].tolist() == [1, 2]
    assert kwargs["actual_seq_lengths_kv"].tolist() == [1, 129]
    assert kwargs["block_table"].tolist() == [[0, 0], [1, 2]]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
