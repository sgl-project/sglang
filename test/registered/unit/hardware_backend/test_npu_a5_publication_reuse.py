"""CPU contract for reusing A5 publication row metadata across target layers."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


def _load_fused_attention():
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
    dsa_utils = stubs["sglang.srt.layers.attention.dsa.utils"]
    dsa_utils.is_dsa_enable_prefill_cp = lambda: False
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
    publication_rows = tuple(object() for _ in range(4))
    plan = SimpleNamespace(query_width=1)
    metadata = SimpleNamespace(
        sparse_kv_lim_plan=plan,
        sparse_kv_publication_rows=None,
    )
    manager = SimpleNamespace(
        packed_c8=True,
        start_layer=0,
        prepare_publication_rows=Mock(return_value=publication_rows),
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
        forward_mode=SimpleNamespace(
            is_extend_without_speculative=lambda: False,
        ),
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

    manager.prepare_publication_rows.assert_called_once()
    assert metadata.sparse_kv_publication_rows is publication_rows
    assert manager.offload_v2.call_count == 2
    for call in manager.offload_v2.call_args_list:
        assert call.kwargs["publication_rows"] is publication_rows

    # Capture warmup may execute again with the same graph ForwardMetadata.
    # Its first layer must still record a fresh producer in that invocation.
    module.forward_fused_sparse_kv_offload(
        backend,
        layer=SimpleNamespace(layer_id=0, tp_k_head_num=1),
        forward_batch=forward_batch,
        **tensors,
    )
    assert manager.prepare_publication_rows.call_count == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
