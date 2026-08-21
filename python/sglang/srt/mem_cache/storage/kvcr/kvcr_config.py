# SPDX-License-Identifier: Apache-2.0
"""Configuration for the KVCR HiCacheStorage backend.

Parsed from ``HiCacheStorageConfig.extra_config`` (the ``--hicache-storage-backend
-extra-config`` JSON blob), mirroring how Mooncake/UMBP read their own settings.
"""

from __future__ import annotations

from typing import Optional

import msgspec

# Bind wildcards: legal to bind, impossible to dial. A peer handed one of these
# as a source endpoint would connect to itself.
_UNROUTABLE_HOSTS = frozenset({"0.0.0.0", "::"})


class KVCRBackendConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Draft config surface for KVCR-as-HiCacheStorage.

    Only the fields the draft actually consumes are here; the KVCR core exposes
    many more knobs (``KVCRConfig``) that we pass through with defaults for now.
    """

    # Size of KVCR's own local DRAM tier (the "buffer-only L3" pool), in bytes.
    # deposit() copies engine host KV into slots carved from this region.
    local_dram_bytes: int = 1 << 30  # 1 GiB placeholder

    # Number of slots the local DRAM region is divided into. slot_size =
    # local_dram_bytes // local_dram_slots and MUST equal the host KV page byte
    # size, so this is validated against the real page size at registration.
    local_dram_slots: int = 0  # 0 => derive from page size at registration

    # Control-plane (ZMQ peer channel) bind host/port for cross-worker P2P.
    # control_port 0 means "ask the OS", which is fine local-only and refused
    # with enable_remote_hint -- see __post_init__.
    control_host: str = "0.0.0.0"
    control_port: int = 0
    control_advertise_host: Optional[str] = None

    # KVCR core knobs. enable_telemetry / operation_timeout_ms feed KVCRConfig;
    # eager_ctrl_connect / opportunistic_query / metadata_retry_interval_ms feed
    # KVCRBackendConfigs.remote_fw_dram (RemoteFWDramOptions) in the wheel core.
    enable_telemetry: bool = False
    # Budget for one KVCR operation end to end. The core's own default is 1000ms,
    # which is too tight here: the *source* clamps its pin deadline to this
    # value, and a single prefetch fans a 64-token page out into many block keys
    # (192 for a 96-page request), all of which must be pinned and written
    # before it expires. At 1000ms that reliably force-failed a fetch that had
    # every key resident on the source.
    operation_timeout_ms: int = 20000
    eager_ctrl_connect: bool = True
    opportunistic_query: bool = False
    metadata_retry_interval_ms: int = 100

    # Placement/eviction policy for KVCR's local DRAM tier. Named explicitly
    # rather than left to the core's default, because that default is not
    # stable: it was FIFO through kvcc abb13bf and is LRU as of e3a816e, so
    # leaving it unset silently re-tunes the tier under us between core bumps
    # and invalidates any benchmark taken before one. Accepts a builtin name
    # ("fifo"/"lru"/"g3_fifo"/"g3_lru") or a fully qualified module.Class path
    # to an external KVCachePolicy. The g3_* policies require a configured G3
    # tier, which this backend does not expose, so selecting one raises.
    policy: str = "lru"

    # --- Workstream B seam (router hint). Placeholder only. ---
    # When True, the backend expects per-request router hints (source control
    # endpoint + block hashes) to arrive via HiCacheStorageExtraInfo.extra_info
    # and drives remote P2P fetches. When False, the backend is local-only
    # (deposit/local-get against KVCR's own DRAM), which is all the pinned KVCR
    # commit can serve today.
    enable_remote_hint: bool = False

    # Wall-clock budget for one deposit/deliver to report completion on the
    # HiCache prefetch daemon thread. A remote fetch crosses the control plane
    # plus a NIXL transfer, so this is generously above operation_timeout_ms;
    # exceeding it is reported as a miss and HiCache recomputes.
    get_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        """Refuse a remote-hint config whose control endpoint cannot be dialed.

        A hinted fetch reaches the source by dialing the endpoint the source
        registered, so the source must know its own endpoint *before* it binds.
        Two defaults make that impossible and both are silent:

        - ``control_port = 0`` binds an OS-assigned port that only exists inside
          the scheduler process, so there is nothing to register; and it is not
          stable across a restart even if there were.
        - ``control_host`` is a bind address -- commonly the ``"0.0.0.0"``
          wildcard, which no peer can connect to -- so it cannot double as the
          advertised address. ``control_advertise_host`` names the interface
          peers should dial, and remote hints require it explicitly.

        Neither breaks startup. The worker comes up, offloads, gets indexed by
        the router, and receives hints -- every fetch just quietly fails to
        reach it, which reads as "P2P does not work" rather than as a config
        error. Raising here turns that into a startup failure naming the knob.

        Local-only (``enable_remote_hint = False``) keeps both defaults: nothing
        dials this worker, so an ephemeral port is the better choice -- it
        cannot collide with a neighbour.

        The dynamo bridge applies the same rule when it builds the endpoints it
        registers, but that only covers workers launched through it; this is the
        check for the backend's own config surface.
        """
        if not self.enable_remote_hint:
            return
        if self.control_port <= 0:
            raise ValueError(
                "KVCR enable_remote_hint requires an explicit control_port: "
                "port 0 binds an OS-assigned port that cannot be registered "
                "with the router, so no peer can fetch from this worker. Set a "
                "positive control_port in --hicache-storage-backend-extra-config "
                "(each scheduler offsets it by its own rank, so one base port "
                "per engine is enough)."
            )
        advertise = self.control_advertise_host
        if not advertise or advertise in _UNROUTABLE_HOSTS:
            raise ValueError(
                f"KVCR enable_remote_hint cannot advertise {advertise!r}. Set "
                "control_advertise_host to an address peers can dial in "
                "--hicache-storage-backend-extra-config; control_host is the "
                "bind address and is not a substitute."
            )

    @classmethod
    def from_extra_config(cls, extra_config: Optional[dict]) -> "KVCRBackendConfig":
        if not extra_config:
            return cls()
        known = {f for f in cls.__struct_fields__}
        filtered = {k: v for k, v in extra_config.items() if k in known}
        return msgspec.convert(filtered, cls)
