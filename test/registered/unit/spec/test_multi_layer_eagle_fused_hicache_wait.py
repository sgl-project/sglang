"""A multi-layer EAGLE draft fused into the target's entries reads rows that a
HiCache reload is still writing, so its worker must wait for the reload's last
layer before it dispatches anything, CUDA graph replay included: a replayed
graph cannot wait on a host event once dispatched.

    python -m pytest test/registered/unit/spec/test_multi_layer_eagle_fused_hicache_wait.py -v
"""

from types import SimpleNamespace

import pytest

from sglang.srt.speculative import base_spec_worker as spec
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import MultiLayerEagleWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Dispatched(Exception):
    pass


def _worker(order, mode, *, num_layers=9):
    events = {3: SimpleNamespace(wait=lambda layer: order.append(("wait", layer)))}

    def dispatch(*args, **kwargs):
        order.append("dispatch")
        raise _Dispatched

    worker = MultiLayerEagleWorkerV2.__new__(MultiLayerEagleWorkerV2)
    worker._target_worker = SimpleNamespace(
        hicache_layer_transfer_counter=SimpleNamespace(
            num_layers=num_layers, events=events
        ),
        # The first dispatch of a prefill.
        forward_batch_generation=dispatch,
    )
    # The first dispatch of a decode: the draft's rounds, graph replay included.
    worker._draft_worker = SimpleNamespace(
        last_draft_extend_staged=True, draft=dispatch
    )
    worker._hicache_draft_plan = spec.HiCacheDraftPlan(mode=mode)
    worker.speculative_algorithm = SimpleNamespace(is_standalone=lambda: False)
    return worker


def _batch(extend, consumer_index=3):
    return SimpleNamespace(
        hicache_consumer_index=consumer_index,
        is_extend_in_batch=extend,
        forward_mode=SimpleNamespace(is_extend=lambda: extend),
        spec_info=object(),
    )


@pytest.mark.parametrize("extend", [True, False])
def test_a_fused_draft_waits_for_the_last_layer_before_any_dispatch(extend):
    order = []
    worker = _worker(order, spec.HiCacheDraftMode.FUSED)
    with pytest.raises(_Dispatched):
        worker.forward_batch_generation(_batch(extend))
    assert order == [("wait", 8), "dispatch"]


@pytest.mark.parametrize("extend", [True, False])
@pytest.mark.parametrize(
    "mode,consumer_index",
    [
        # Packed layers wait layer by layer, through the target's own events.
        (spec.HiCacheDraftMode.PACKED, 3),
        (spec.HiCacheDraftMode.NONE, 3),
        # Nothing was reloaded for this batch.
        (spec.HiCacheDraftMode.FUSED, -1),
    ],
)
def test_nothing_else_waits(extend, mode, consumer_index):
    order = []
    worker = _worker(order, mode)
    with pytest.raises(_Dispatched):
        worker.forward_batch_generation(_batch(extend, consumer_index))
    assert order == ["dispatch"]
