# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class UGSRTSchedulerExecutorError(RuntimeError):
    """Raised when UG cannot execute an SRT-backed request."""


class UGSRTRequestBoundaryExecutor:
    """Records materialized UG SRT requests at the execution boundary.

    The lean UG contract PR intentionally stops here. Real scheduler execution
    and temporary visual-step KV batches are model-specific work and should land
    with the first native model adapter.
    """

    finish_request_after_execute = True

    def __init__(self) -> None:
        self.events: list[tuple[str, str, int]] = []

    def execute_ug_request(self, *, record, req, state) -> None:
        del record
        self.events.append((state.value, req.rid, len(req.origin_input_ids)))


class UGSRTSchedulerExecutor(UGSRTRequestBoundaryExecutor):
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        raise UGSRTSchedulerExecutorError(
            "Native UG scheduler execution is split out of the lean interleave "
            "contract PR"
        )
