"""Independent all-shard oracle for uneven A/F groups and empty F ranks."""

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.srt import runtime_context
from sglang.srt.afd.cache import make_shape
from sglang.srt.afd.config import AFDConfig
from sglang.srt.afd.contracts import AFDRole
from sglang.srt.afd.model_adapters import base
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("a,f", [(20, 32), (3, 4), (4, 4), (24, 4)])
def test_all_expert_shards_contribute_and_return_each_source_row_once(
    monkeypatch, a, f
):
    rows = tuple((i % 4, (i + 2) % 4) for i in range(a))
    config = AFDConfig(lanes=f, attention_lanes=a)
    shapes = [
        make_shape(
            lane=j,
            lane_rows=rows,
            hidden_size=4,
            dtype="bfloat16",
            config=config,
            capture_sizes=(4,),
        )
        for j in range(f)
    ]
    inputs = [
        [
            torch.arange(n * 4, dtype=torch.float32).reshape(n, 4) + 10 * i + stage
            for stage, n in enumerate(vector)
        ]
        for i, vector in enumerate(rows)
    ]
    barrier = threading.Barrier(f, timeout=10)
    values, calls = [None] * f, []
    state = threading.local()

    class Group:
        def __init__(self, rank):
            self.rank_in_group = rank
            self.step = 0

        def collect(self, tensor, *, reduce=False):
            values[self.rank_in_group] = tensor
            barrier.wait()
            result = torch.stack(values).sum(0) if reduce else torch.cat(values)
            # Finish reading every rank before the next collective overwrites slots.
            barrier.wait()
            return result

        def all_gatherv(self, local, sizes):
            assert local.shape == (sizes[self.rank_in_group], 4)
            return [self.collect(local)]

        def reduce_scatterv(self, partial, sizes):
            result = self.collect(partial, reduce=True).narrow(
                0, sum(sizes[: self.rank_in_group]), sizes[self.rank_in_group]
            )
            self.step += 1
            return result

    class Forward:
        @contextmanager
        def scoped(self, **kwargs):
            assert kwargs == {"mlp_reduce_scatter": True}
            yield

    monkeypatch.setattr(
        base, "get_parallel", lambda: SimpleNamespace(tp_group=state.group)
    )
    monkeypatch.setattr(runtime_context, "get_forward", lambda: Forward())

    def run(rank):
        state.group = Group(rank)
        shape = shapes[rank]
        capacities = shape.group_bucket_rows(ffn_ordinal=rank)
        buffers = tuple(torch.empty(n, 4) for n in capacities)
        adapter = object.__new__(base.AFDDecoderAdapter)
        adapter.role = AFDRole.FFN

        def compute(x, batch):
            calls.append((rank, state.group.step))
            # Each rank owns a distinct expert selected by global row ID.
            chosen = torch.arange(x.shape[0]) % f == rank
            return x * chosen[:, None] * (rank + 1)

        adapter.inner = SimpleNamespace(
            layers=[SimpleNamespace(compute_ffn_output=compute)]
        )
        descriptor = SimpleNamespace(stage_rows=lambda *, lane: rows[lane])
        stages, views = adapter.make_ffn_stages(
            descriptor=descriptor, buffers=buffers, lane=rank, shape=shape
        )
        group = shape.group_lanes(ffn_ordinal=rank)
        for stage in range(2):
            for tensor, lane in zip(views[stage], group):
                tensor.copy_(inputs[lane][stage])
        return [
            adapter.local_compute(
                layer=0, stage=stage, hidden_states=view, residual=None
            )[0]
            for stage, view in zip(stages, views)
        ]

    with ThreadPoolExecutor(max_workers=f) as executor:
        outputs = list(executor.map(run, range(f)))
    assert sorted(calls) == [(j, s) for j in range(f) for s in range(2)]
    for stage in range(2):
        # Independent serial oracle in global A row order, including bucket gaps.
        packed = torch.cat(
            [
                torch.nn.functional.pad(inputs[i][stage], (0, 0, 0, 4 - rows[i][stage]))
                for i in range(a)
            ]
        )
        oracle = packed * (torch.arange(a * 4) % f + 1)[:, None]
        for rank, shape in enumerate(shapes):
            lanes = shape.group_lanes(ffn_ordinal=rank)
            assert len(outputs[rank][stage]) == len(lanes)
            for output, lane in zip(outputs[rank][stage], lanes):
                torch.testing.assert_close(
                    output, oracle[lane * 4 : lane * 4 + rows[lane][stage]]
                )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
