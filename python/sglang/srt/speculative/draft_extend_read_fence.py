"""Where the draft-extend graph publishes its shared-read-done event.

With EAGLE spec v2 the draft-extend graph is the last phase of a round that
reads scheduler-shared buffers, so the scheduler's WAR barrier waits on the
event it publishes, or on the whole forward when it publishes none. The graph's
attention metadata kernel (``init_forward_metadata_in_graph``) runs first and
reads ``req_to_token``; an external event recorded right after it marks where
those reads end.

Policies (every switch defaults off; off keeps the stock path exactly):

- ``in_graph``: record the marker at capture and hand it to the barrier at
  replay when the attention backend declares IN_REPLAY reads for this mode.
- ``pre_replay``: publish a plain event before the replay. It does not cover
  the in-graph metadata reads, so it exists only to measure the fence cost.
- ``stock``: no event; the barrier waits for the whole forward.

An A/B control file can switch the policy between rounds of one server.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)

# Seconds between reads of the A/B control file. A daemon thread does the reads so a
# slow (e.g. network filesystem) read never stalls the scheduler loop.
_CTL_REFRESH_S = 0.2


class DraftExtendFencePolicy(enum.Enum):
    STOCK = "stock"
    IN_GRAPH = "in_graph"
    PRE_REPLAY = "pre_replay"


def draft_extend_in_graph_fence_enabled() -> bool:
    """Whether the draft-extend graph captures its in-graph read-done marker."""
    return envs.SGLANG_Q35MF_PDDEC_FENCE.get() or bool(
        envs.SGLANG_Q35MF_PDDEC_FENCE_AB_CTL.get()
    )


class DraftExtendReadFence:
    """Per-runner policy state; exists only when one of the switches is set."""

    def __init__(
        self,
        *,
        capture_marker: bool,
        allow_pre_replay: bool,
        ctl_path: Optional[str],
        start_poller: bool = True,
    ):
        self.capture_marker = capture_marker
        self.allow_pre_replay = allow_pre_replay
        self.ctl_path = ctl_path
        if ctl_path:
            self._policy = DraftExtendFencePolicy.STOCK
        elif allow_pre_replay:
            self._policy = DraftExtendFencePolicy.PRE_REPLAY
        else:
            self._policy = DraftExtendFencePolicy.IN_GRAPH
        self._ctl_policy = DraftExtendFencePolicy.STOCK
        self._warned: set[str] = set()
        self.round_ct = 0
        # in_graph_unavailable: in_graph was requested but the backend did not
        # resolve to IN_REPLAY, so that round kept the backend's own fence.
        self.policy_round_ct = {
            "stock": 0,
            "in_graph": 0,
            "pre_replay": 0,
            "in_graph_unavailable": 0,
        }
        _ACTIVE_FENCES.append(self)
        if ctl_path and start_poller:
            threading.Thread(
                target=self._poll_ctl_forever,
                name="draft-extend-fence-ctl",
                daemon=True,
            ).start()
        logger.info(
            "Draft-extend read fence: capture_marker=%s pre_replay_allowed=%s "
            "ab_ctl=%s initial_policy=%s",
            capture_marker,
            allow_pre_replay,
            ctl_path,
            self._policy.value,
        )

    @classmethod
    def from_env(
        cls, *, allocator: BaseTokenToKVPoolAllocator
    ) -> Optional[DraftExtendReadFence]:
        capture_marker = draft_extend_in_graph_fence_enabled()
        allow_pre_replay = envs.SGLANG_Q35MF_PDDEC_FENCE_PREREPLAY_UNSAFE.get()
        if not (capture_marker or allow_pre_replay):
            return None
        # Under PD (need_sort) a fence that ends before the replay's KV writes is
        # sound only if freed pages wait for them before reuse.
        if allocator.need_sort and not allocator.supports_reuse_guard:
            logger.warning(
                "Draft-extend read fence off: %s has no KV page reuse guard",
                type(allocator).__name__,
            )
            return None
        return cls(
            capture_marker=capture_marker,
            allow_pre_replay=allow_pre_replay,
            ctl_path=envs.SGLANG_Q35MF_PDDEC_FENCE_AB_CTL.get() or None,
        )

    def _warn_once(self, key: str, msg: str, *args) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg, *args)

    def _read_ctl(self) -> DraftExtendFencePolicy:
        try:
            with open(self.ctl_path) as f:
                text = f.read().strip()
        except OSError:
            return DraftExtendFencePolicy.STOCK
        try:
            policy = DraftExtendFencePolicy(text)
        except ValueError:
            self._warn_once(
                f"ctl:{text}", "Draft-extend read fence: unknown policy %r", text
            )
            return DraftExtendFencePolicy.STOCK
        if policy is DraftExtendFencePolicy.PRE_REPLAY and not self.allow_pre_replay:
            self._warn_once(
                "pre_replay",
                "Draft-extend read fence: pre_replay needs "
                "SGLANG_Q35MF_PDDEC_FENCE_PREREPLAY_UNSAFE=1; using stock",
            )
            return DraftExtendFencePolicy.STOCK
        return policy

    def _poll_ctl_once(self) -> None:
        self._ctl_policy = self._read_ctl()

    def _poll_ctl_forever(self) -> None:
        while True:
            self._poll_ctl_once()
            time.sleep(_CTL_REFRESH_S)

    def policy_for_round(self) -> DraftExtendFencePolicy:
        if self.ctl_path:
            policy = self._ctl_policy
            if policy is not self._policy:
                logger.info(
                    "Draft-extend read fence policy %s -> %s at round %d",
                    self._policy.value,
                    policy.value,
                    self.round_ct,
                )
                self._policy = policy
        return self._policy

    def resolve(self, *, backend_ends: SharedReadEnds) -> SharedReadEnds:
        """This round's read-done placement, given the runner's own resolution
        of the backend declaration (IN_REPLAY only when a marker was captured)."""
        self.round_ct += 1
        policy = self.policy_for_round()
        if policy is DraftExtendFencePolicy.PRE_REPLAY:
            self.policy_round_ct["pre_replay"] += 1
            return SharedReadEnds.PRE_REPLAY
        if policy is DraftExtendFencePolicy.STOCK:
            self.policy_round_ct["stock"] += 1
            return SharedReadEnds.UNKNOWN
        if backend_ends is SharedReadEnds.IN_REPLAY:
            self.policy_round_ct["in_graph"] += 1
        else:
            self.policy_round_ct["in_graph_unavailable"] += 1
            self._warn_once(
                f"unavailable:{backend_ends.name}",
                "Draft-extend read fence: in_graph requested, backend resolves "
                "to %s; keeping that fence",
                backend_ends.name,
            )
        return backend_ends

    def snapshot(self) -> dict:
        return {
            "policy": self._policy.value,
            "round_ct": self.round_ct,
            **self.policy_round_ct,
        }


_ACTIVE_FENCES: list[DraftExtendReadFence] = []


def draft_extend_read_fence_stats() -> dict:
    """Counters of every fence in this process (diagnostics only)."""
    return {f"runner{i}": fence.snapshot() for i, fence in enumerate(_ACTIVE_FENCES)}
