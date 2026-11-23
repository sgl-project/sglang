"""Per-tokenizer-worker rid namespacing.

Each tokenizer worker rejects a duplicate rid only among its own requests, so with
several workers one client rid can be in flight on two of them at once. The scheduler
and the detokenizer key per-request state by rid, so two such requests would share it
and one could receive the other's output. A worker therefore appends its own suffix to
every rid it sends on, and strips it from every rid it returns to a client.

With a single worker the suffix is empty and rids pass through unchanged.
"""

import uuid
from typing import List, Union

# Same separator as the Rust server's uniquified rids (rust/sglang-server/src/message/ids.rs).
WORKER_RID_SEP = "#"


def make_worker_rid_suffix(*, multi_worker: bool) -> str:
    return f"{WORKER_RID_SEP}{uuid.uuid4().hex[:8]}" if multi_worker else ""


def add_worker_rid_suffix(
    rid: Union[str, List[str]], suffix: str
) -> Union[str, List[str]]:
    if isinstance(rid, list):
        return [r + suffix for r in rid]
    return rid + suffix


def strip_worker_rid_suffix(rid: str, suffix: str) -> str:
    return rid.removesuffix(suffix)
