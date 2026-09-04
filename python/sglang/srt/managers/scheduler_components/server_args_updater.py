"""Runtime (hot) updates of allowlisted server args.

`POST /set_internal_state` fans out to every scheduler, which validates the
request here and applies it through `get_context().override`, the sanctioned
post-publish mutation entry. `GET /server_info` reads the resolved config
back.

A field is admitted to `HOT_UPDATABLE_SERVER_ARGS` only when its readers see
the new value: either every reader reads the config-bag leaf live (e.g.
`stream_interval`, read per output batch), or `Scheduler.set_internal_state`
refreshes the derived state after the override (e.g.
`schedule_conservativeness` rebuilds the new-token-ratio tracker).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sglang.srt.runtime_context import get_context

logger = logging.getLogger(__name__)

HOT_UPDATABLE_SERVER_ARGS = frozenset(
    {
        "pp_max_micro_batch_size",
        "speculative_accept_threshold_single",
        "speculative_accept_threshold_acc",
        "dspark_force_budget_frac",
        "dspark_clear_info_records",
        "schedule_conservativeness",
        "stream_interval",
    }
)

# DSpark control keys are worker commands, not server args; they are routed
# to the draft worker and kept out of the override.
_DSPARK_WORKER_COMMANDS = frozenset(
    {"dspark_force_budget_frac", "dspark_clear_info_records"}
)


def validate_server_args_update(
    server_args_dict: Dict[str, Any],
    *,
    max_running_requests: int,
    pp_size: int,
    spec_algorithm,
    draft_worker,
) -> Optional[str]:
    """Return the rejection reason, or None when every entry is applicable."""
    for k, v in server_args_dict.items():
        if k not in HOT_UPDATABLE_SERVER_ARGS:
            return f"Updating {k} is not supported."
        if k == "pp_max_micro_batch_size" and (
            v > max_running_requests // pp_size or v < 1
        ):
            return (
                f"Updating {k} to {v} is rejected because it is out of the "
                f"valid range [1, {max_running_requests // pp_size}]."
            )
        if k == "dspark_force_budget_frac":
            if not spec_algorithm.is_dspark() or not hasattr(
                draft_worker, "set_dspark_forced_budget_frac"
            ):
                return "dspark_force_budget_frac requires a DSpark draft worker."
            if v is not None and not (0.0 < float(v) <= 1.0):
                return f"dspark_force_budget_frac must be in (0, 1] or null, got {v}."
        if k == "dspark_clear_info_records":
            if not spec_algorithm.is_dspark() or not hasattr(
                draft_worker, "clear_info_records"
            ):
                return "dspark_clear_info_records requires a DSpark draft worker."
        if k == "schedule_conservativeness" and not (
            isinstance(v, (int, float)) and v > 0
        ):
            return f"schedule_conservativeness must be a positive number, got {v}."
        if k == "stream_interval" and not (isinstance(v, int) and v >= 1):
            return f"stream_interval must be an integer >= 1, got {v}."
    return None


def apply_server_args_update(
    server_args_dict: Dict[str, Any], *, draft_worker
) -> Dict[str, Any]:
    """Route worker commands and override the rest onto the config bags.

    Returns the fields written to the bags, for the caller to refresh any
    derived state from.
    """
    remaining = dict(server_args_dict)
    frac = remaining.pop("dspark_force_budget_frac", None)
    if "dspark_force_budget_frac" in server_args_dict:
        draft_worker.set_dspark_forced_budget_frac(
            None if frac is None else float(frac)
        )
    if remaining.pop("dspark_clear_info_records", None):
        draft_worker.clear_info_records()
    if remaining:
        get_context().override(source="update_server_args", **remaining)
    return remaining
