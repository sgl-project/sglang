"""CPU checks for QSA's zigzag-only context parallel entrypoints."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.layers.attention import qwen_sparse_attn_backend
from sglang.srt.layers.attention.qsa import qsa_indexer
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.layers.cp.base import ContextParallelStrategyKind
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "strategy",
    [
        None,
        SimpleNamespace(name="interleave", kind=ContextParallelStrategyKind.INTERLEAVE),
    ],
)
@pytest.mark.parametrize("entrypoint", ["indexer", "attention"])
def test_qsa_cp_rejects_unsupported_strategy_before_metadata_or_collectives(
    monkeypatch, strategy, entrypoint
):
    module = qsa_indexer if entrypoint == "indexer" else qwen_sparse_attn_backend
    monkeypatch.setattr(module, "get_cp_strategy", lambda: strategy)
    gather = Mock(side_effect=AssertionError("unsupported CP must not communicate"))
    monkeypatch.setattr(module, "cp_materialize_global_token_order", gather)
    # None metadata and tensors ensure rejection precedes zigzag-specific access.
    with pytest.raises(
        NotImplementedError, match="QSA prefill CP only supports the zigzag strategy"
    ):
        if entrypoint == "indexer":
            QSAIndexer.forward_cuda_cp(None, None, None, None, None, None)
        else:
            QwenSparseAttnBackend._forward_extend_cp(
                None, None, None, None, None, None, None, False
            )
    gather.assert_not_called()


def test_cuda_indexer_forwards_global_rope_to_cp_path():
    hidden, local_rope, metadata, global_rope = (object() for _ in range(4))
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    expected = object()
    indexer = SimpleNamespace(forward_cuda_cp=Mock(return_value=expected))
    indexer._forward_impl = lambda *args: QSAIndexer._forward_impl(indexer, *args)

    actual = QSAIndexer.forward_cuda(
        indexer,
        hidden,
        local_rope,
        batch,
        metadata,
        cp_global_rope_positions=global_rope,
    )

    assert actual is expected
    indexer.forward_cuda_cp.assert_called_once_with(
        hidden, local_rope, batch, metadata, global_rope
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
