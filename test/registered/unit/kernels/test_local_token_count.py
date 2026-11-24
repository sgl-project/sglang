"""CPU-only contracts for local token-count dispatch."""

import sys

import pytest
import torch

import sglang.kernels.fused_op as fused_op
import sglang.kernels.ops.elementwise.local_token_count as local_count
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_local_token_count_uses_registered_compile_forward(monkeypatch):
    op = local_count.LocalTokenCountOp()
    monkeypatch.setattr(local_count, "_LOCAL_TOKEN_COUNT", op)
    monkeypatch.setattr(fused_op, "_forced_backend", None)
    monkeypatch.setattr(fused_op, "_oot_dispatch_key", lambda: "test_platform")
    monkeypatch.setattr(fused_op.BaseFusedOp, "_oot_compile_forward_registry", {})

    def compile_forward(
        op, global_count, rank_offset, tokens_per_rank, *, global_count_cpu=None
    ):
        native = op.forward_native(
            global_count=global_count,
            rank_offset=rank_offset,
            tokens_per_rank=tokens_per_rank,
        )
        return native + 100

    local_count.LocalTokenCountOp.register_oot_compile_forward(
        op_cls=local_count.LocalTokenCountOp,
        fn=compile_forward,
        platform_key="test_platform",
    )
    op.enter_torch_compile(num_tokens=1)
    try:
        compiled = torch.compile(
            local_count.local_token_count, backend="eager", fullgraph=True
        )
        result = compiled(
            global_count=torch.tensor(12, dtype=torch.int32),
            rank_offset=8,
            tokens_per_rank=8,
            global_count_cpu=12,
        )
        assert result.item() == 104
    finally:
        op.leave_torch_compile()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
