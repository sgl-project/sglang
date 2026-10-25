"""Before-work peer exchange through the native CPU TP group.

This entry point does not enable the public ADP2 factory, qualify joint costs,
advance a clock, publish native results or commit request history.
"""

import json
from dataclasses import replace

from sglang_simulator.simulation.sglang.dspark import (
    DSparkGroup,
    DSparkStep,
    DSparkTopology,
)
from sglang_simulator.simulation.sglang.dspark_runtime import canonical
from sglang_simulator.time_predictor.base import PredictorError


def _require(valid, message):
    if not valid:
        raise PredictorError("invalid_dspark_step", message)


def capture_native_dp_peer_step(
    runtime, batch, *, tp_group, attention_dp_rank, native_forward_ct, reference=False
):
    """Capture both real partitions without using aggregate token-count columns.

    A native scheduler/worker adapter supplies its actual ``forward_ct`` and
    ``get_parallel().tp_group`` after native DP batch selection. Logical model
    FID and native scheduler FID are independently checked across both ranks.
    The caller still needs a joint-cost and publication/clock adapter before it
    can execute this unresolved step. One rank's local buffer is never copied
    into its peer's state.
    """
    # New native dependencies remain lazy: ADP1 and the library protocol do not
    # require torch, GroupCoordinator or the new SDK DP policy at import time.
    from sglang.srt.distributed.parallel_state import GroupCoordinator

    _require(
        type(tp_group) is GroupCoordinator
        and type(tp_group.world_size) is int
        and tp_group.world_size == 2
        and type(tp_group.ranks) is list
        and all(type(rank) is int for rank in tp_group.ranks)
        and len(tp_group.ranks) == len(set(tp_group.ranks)) == 2
        and type(tp_group.rank_in_group) is int
        and tp_group.rank_in_group in (0, 1)
        and tp_group.cpu_group is not None,
        "peer exchange requires the actual two-rank native CPU TP group",
    )
    payload = dict.fromkeys(
        ("forward_id", "native_forward_ct", "runtime_profile_json", "group")
    )
    capture_error = None
    try:
        _require(
            runtime.topology == DSparkTopology(2, 2)
            and type(attention_dp_rank) is int
            and attention_dp_rank == tp_group.rank_in_group
            and type(native_forward_ct) is int
            and native_forward_ct >= 0
            and type(runtime._forward_id) is int
            and runtime._forward_id >= 0
            and type(runtime._draft_buffer_capacity) is int
            and runtime._draft_buffer_capacity >= 0,
            "peer exchange requires TP2/ADP2/PP1 and explicit before-work state",
        )
        payload.update(
            forward_id=runtime._forward_id,
            native_forward_ct=native_forward_ct,
            runtime_profile_json=runtime.runtime_profile_json,
        )
        local = runtime._capture_local_group(
            batch, attention_dp_rank=attention_dp_rank, reference=reference
        )
        local = replace(
            local,
            execution_json=canonical(
                {
                    "selection": "native_peer_before_work_unresolved",
                    "draft_buffer_capacity": runtime._draft_buffer_capacity,
                    "draft_buffer_device_matches": runtime._draft_buffer_capacity > 0,
                }
            ),
        )
        payload["group"] = local
    except Exception as error:  # noqa: BLE001 - rendezvous, then re-raise unchanged
        # Both real ranks must reach the rendezvous even if one local capture
        # fails. A missing group makes every participant reject before costs.
        capture_error = error
    peers = tp_group.all_gather_object(payload)
    if capture_error is not None:
        raise capture_error
    _require(type(peers) is list and len(peers) == 2, "missing native DP peer")
    for rank, peer in enumerate(peers):
        _require(
            type(peer) is dict
            and set(peer) == set(payload)
            and type(peer["forward_id"]) is int
            and peer["forward_id"] == runtime._forward_id
            and type(peer["native_forward_ct"]) is int
            and peer["native_forward_ct"] == native_forward_ct
            and peer["runtime_profile_json"] == runtime.runtime_profile_json
            and type(peer["group"]) is DSparkGroup
            and peer["group"].attention_dp_rank == rank,
            "native DP peer has different FID, runtime or ordered partition",
        )
        execution = json.loads(peer["group"].execution_json)
        _require(
            set(execution)
            == {
                "selection",
                "draft_buffer_capacity",
                "draft_buffer_device_matches",
            }
            and execution["selection"] == "native_peer_before_work_unresolved"
            and type(execution["draft_buffer_capacity"]) is int
            and execution["draft_buffer_capacity"] >= 0
            and type(execution["draft_buffer_device_matches"]) is bool
            and execution["draft_buffer_device_matches"]
            == (execution["draft_buffer_capacity"] > 0),
            "native DP peer omits its independent before-work buffer state",
        )
    _require(
        peers[attention_dp_rank] == payload,
        "native DP exchange changed its local captured partition",
    )
    step = DSparkStep(
        runtime._forward_id,
        runtime.topology,
        tuple(peer["group"] for peer in peers),
        runtime.runtime_profile_json,
    )
    _require(
        {group.forward_mode for group in step.groups} - {"IDLE"}
        in ({"EXTEND"}, {"TARGET_VERIFY"}),
        "native DP mixed-phase coordination remains unsupported",
    )
    return step
