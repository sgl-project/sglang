"""Native executor defaults and caller-owned role contexts use the same ordering."""

from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.batch_overlap import operations
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("caller_owned", [False, True])
def test_two_batch_context_and_operation_order(monkeypatch, caller_owned):
    trace = []

    @contextmanager
    def context(index):
        trace.append(("enter", index))
        yield
        trace.append(("exit", index))

    dp = Mock()
    monkeypatch.setattr(operations, "set_dp_buffer_len", dp)
    resolve = Mock(return_value=(0, 1))
    monkeypatch.setattr(operations, "_resolve_tbo_child_contexts", resolve)
    monkeypatch.setattr(operations, "forward_context", context)

    def operation(*, state, index, layer, **inputs):
        trace.append(("op", index, layer))
        state.update({f"layer_{layer}": index})
        return inputs

    batches = [
        SimpleNamespace(
            global_dp_buffer_len=8,
            tbo_padded_len=4,
            global_num_tokens_cpu=[8],
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: True),
        )
        for _ in range(2)
    ]
    operations.execute_overlapped_operations(
        inputs_arr=[{} if caller_owned else {"forward_batch": b} for b in batches],
        operations_arr=[
            [
                op
                for layer in range(3)
                for op in (
                    partial(operation, index=index, layer=layer),
                    operations.YieldOperation(),
                )
            ]
            for index in range(2)
        ],
        delta_stages=[0, 0],
        stage_contexts=[partial(context, i) for i in range(2)]
        if caller_owned
        else None,
    )
    assert trace == [
        event
        for layer in range(3)
        for i in range(2)
        for event in (
            ("enter", i),
            ("op", i, layer),
            ("exit", i),
        )
    ]
    assert resolve.call_count == (0 if caller_owned else 1)
    assert dp.call_count == (0 if caller_owned else 6)
    if not caller_owned:
        dp.assert_called_with(8, 4, True, [8])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
