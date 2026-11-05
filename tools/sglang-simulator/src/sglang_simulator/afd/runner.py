"""Decode orchestration and result accounting, independent of native objects."""

from .contracts import ExecutionAdapter
from .costs import CostTable


def critical_path(events, provider):
    by = {event["id"]: event for event in events}
    # Preserve the existing max-ID tie rule without scanning the complete
    # multi-rank graph at every resource wait on the critical path.
    resource_end = {}
    for event in events:
        if event["duration"] > 0:
            for resource in event["resources"]:
                key = event["finish_ns"], resource
                resource_end[key] = (
                    max(resource_end[key], event["id"])
                    if key in resource_end
                    else event["id"]
                )
    current = max(events, key=lambda event: (event["finish_ns"], event["id"]))
    path, seen = [], set()
    while True:
        if current["id"] in seen:
            raise ValueError("AFD_CRITICAL_PATH_CYCLE")
        seen.add(current["id"])
        path.append(current)
        deps = [
            by[key]
            for key in current["deps"]
            if by[key]["finish_ns"] == current["start_ns"]
        ]
        if deps:
            current = max(deps, key=lambda event: event["id"])
            continue
        if current["start_ns"] == 0:
            break
        previous = [
            resource_end[key]
            for resource in current["resources"]
            if (key := (current["start_ns"], resource)) in resource_end
        ]
        if not previous:
            raise ValueError("AFD_UNEXPLAINED_RESOURCE_WAIT")
        current = by[max(previous)]
    total = sum(event["duration"] for event in path)
    assert total == max(event["finish_ns"] for event in events)
    contributions = dict(exact=0.0, interpolation=0.0, formula=0.0, fixture=0.0)
    for event in path:
        if not event["duration"]:
            continue
        records = provider.event_components(event)
        denominator = sum(record["duration_ns"] for _, record in records)
        for _, record in records:
            contributions[record["basis"]] += (
                event["duration"] * record["duration_ns"] / denominator
            )
    return {
        "duration_ns": total,
        "contributions_ns": contributions,
        "declared_exact_interpolation_fraction": (
            contributions["exact"] + contributions["interpolation"]
        )
        / total,
    }


def simulate(document, *, source_root=None, adapter: ExecutionAdapter | None = None):
    config = document["deployment"]
    if adapter is None:
        from .adapters import create_adapter

        adapter = create_adapter(
            source_root, config["mode"], routing=config.get("routing", "f-side")
        )
    provider = CostTable(document, adapter.cost_mapping(config))
    config = provider.config
    trace = adapter.execute(provider)
    devices = trace.total_devices
    path = critical_path(trace.events, provider)
    end = path["duration_ns"]
    return {
        "kind": (
            "control_fixture" if provider.synthetic else "conditional_decode_estimate"
        ),
        "deployment": config,
        "source_files": trace.source_files,
        "total_devices": devices,
        "control_completion_ns": end,
        "step_ms": None if provider.synthetic else end / 1e6,
        "total_tgs": None if provider.synthetic else config["batch_size"] * 1e9 / end,
        "per_device_tgs": (
            None if provider.synthetic else config["batch_size"] * 1e9 / end / devices
        ),
        "capacity_fit": "unknown",
        "service_admitted": False,
        "service_accuracy_verified": False,
        "critical_path": path,
        "cost_ledger": document["costs"],
        "events": trace.events,
        "queries": provider.queries,
        "limits": [
            "Fixed steady decode only, not an HTTP/Scheduler request simulation or prefill/TTFT estimate",
            "Single module costs and explicit dependency/resource assumptions do not prove service overlap or contention",
            "Timing is conditioned on capacity fit; cost provenance is caller-declared, not independently certified",
        ],
    }
