"""CPU ownership and layer-readiness checks for CP replay cache restoration."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from test_dsv41_cp_replay_layout import definitions


@pytest.mark.parametrize("cp_enabled,replayed", [(False, True), (True, False)])
def test_legacy_cache_does_not_match_or_reduce_for_cp_replay(cp_enabled, replayed):
    functions = definitions(
        "srt/mem_cache/unified_radix_cache.py",
        ["_replay_overlaps_unadopted_cache"],
        klass="UnifiedRadixCache",
    )
    cache = NS(
        cp_encoder_replay=cp_enabled,
        tree_core=NS(match_prefix=Mock()),
        _all_reduce=Mock(),
    )
    assert not functions["_replay_overlaps_unadopted_cache"](
        cache, NS(kv=NS(encoder_swa_replayed=replayed)), object()
    )
    cache.tree_core.match_prefix.assert_not_called()
    cache._all_reduce.assert_not_called()


@pytest.mark.parametrize(
    "local_hit,peer_overlap,expected",
    [(4, False, False), (8, False, True), (4, True, True)],
)
def test_unadopted_snapshot_overlap_is_a_group_decision(
    local_hit, peer_overlap, expected
):
    functions = definitions(
        "srt/mem_cache/unified_radix_cache.py",
        ["_replay_overlaps_unadopted_cache"],
        {"MatchPrefixParams": NS},
        klass="UnifiedRadixCache",
    )
    actions = object()

    def reduce(value, op):
        assert op == torch.distributed.ReduceOp.MAX
        if peer_overlap:
            value.fill_(1)

    cache = NS(
        cp_encoder_replay=True,
        tree_core=NS(
            match_prefix=Mock(
                return_value=NS(full_kv_hit_length=local_hit, cache_actions=actions)
            )
        ),
        _apply_cache_actions=Mock(),
        _all_reduce=reduce,
    )
    req = NS(kv=NS(encoder_swa_replayed=True, cache_protected_len=4))
    assert (
        functions["_replay_overlaps_unadopted_cache"](cache, req, object()) is expected
    )
    cache._apply_cache_actions.assert_called_once_with(actions)


@pytest.mark.parametrize(
    "getter,args",
    [
        ("get_extra_key_buffer", [3]),
        ("get_low_ratio_index_k_dequant", [3, torch.tensor([1, 2])]),
        ("get_low_ratio_index_k_fp4", [3, torch.tensor([1, 2])]),
    ],
)
def test_restored_kv_and_indexer_wait_before_reading_the_layer(getter, args):
    order = []
    value = object()
    pool = NS(
        get_key_buffer=lambda *a: order.append("read") or value,
        get_index_k_dequant=lambda *a: order.append("read") or value,
        get_index_k_fp4=lambda *a: order.append("read") or value,
    )
    functions = definitions(
        "srt/mem_cache/deepseek_v4_memory_pool.py",
        [getter],
        klass="DeepSeekV4TokenToKVPool",
    )
    target = NS(
        wait_layer_transfer=lambda layer: order.append(("wait", layer)),
        layer_mapping={3: (1, 0, pool)},
        _indexer_pool=lambda ratio: pool,
    )
    assert functions[getter](target, *args) is value
    assert order == [("wait", 3), "read"]


def test_layer_readiness_uses_existing_event_without_device_synchronize():
    calls = []
    counter_functions = definitions(
        "srt/managers/cache_controller.py", ["wait_until"], klass="LayerDoneCounter"
    )
    event_functions = definitions(
        "srt/managers/cache_controller.py",
        ["wait"],
        {
            "device_module": NS(
                current_stream=lambda: NS(wait_event=lambda event: calls.append(event))
            )
        },
        klass="LayerLoadingEvent",
    )
    events = [object(), object(), object()]
    event = NS(load_events=events)
    counter = NS(
        consumer_index=0,
        events=[NS(wait=lambda layer: event_functions["wait"](event, layer))],
    )
    counter_functions["wait_until"](counter, 1)
    assert calls == [events[1]]
