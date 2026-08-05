# SPDX-License-Identifier: Apache-2.0
"""Configuration for the KVCC HiCacheStorage backend.

Parsed from ``HiCacheStorageConfig.extra_config`` (the ``--hicache-storage-backend
-extra-config`` JSON blob), mirroring how Mooncake/UMBP read their own settings.
"""

from __future__ import annotations

from typing import Optional

import msgspec


class KVCCBackendConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Draft config surface for KVCC-as-HiCacheStorage.

    Only the fields the draft actually consumes are here; the KVCC core exposes
    many more knobs (``KVCCConfig``) that we pass through with defaults for now.
    """

    # Size of KVCC's own local DRAM tier (the "buffer-only L3" pool), in bytes.
    # deposit() copies engine host KV into slots carved from this region.
    local_dram_bytes: int = 1 << 30  # 1 GiB placeholder

    # Number of slots the local DRAM region is divided into. slot_size =
    # local_dram_bytes // local_dram_slots and MUST equal the host KV page byte
    # size, so this is validated against the real page size at registration.
    local_dram_slots: int = 0  # 0 => derive from page size at registration

    # Control-plane (ZMQ peer channel) bind host/port for cross-worker P2P.
    control_host: str = "0.0.0.0"
    control_port: int = 0  # 0 => ephemeral
    control_advertise_host: Optional[str] = None

    # KVCC core knobs. enable_telemetry / operation_timeout_ms feed KVCCConfig;
    # eager_ctrl_connect / opportunistic_query / metadata_retry_interval_ms feed
    # KVCCBackendConfigs.remote_fw_dram (RemoteFWDramOptions) in the wheel core.
    enable_telemetry: bool = False
    # Budget for one KVCC operation end to end. The core's own default is 1000ms,
    # which is too tight here: the *source* clamps its pin deadline to this
    # value, and a single prefetch fans a 64-token page out into many block keys
    # (192 for a 96-page request), all of which must be pinned and written
    # before it expires. At 1000ms that reliably force-failed a fetch that had
    # every key resident on the source.
    operation_timeout_ms: int = 20000
    eager_ctrl_connect: bool = True
    opportunistic_query: bool = False
    metadata_retry_interval_ms: int = 100

    # --- Workstream B seam (router hint). Placeholder only. ---
    # When True, the backend expects per-request router hints (source control
    # endpoint + block hashes) to arrive via HiCacheStorageExtraInfo.extra_info
    # and drives remote P2P fetches. When False, the backend is local-only
    # (deposit/local-get against KVCC's own DRAM), which is all the pinned KVCC
    # commit can serve today.
    enable_remote_hint: bool = False

    # Wall-clock budget for one deposit/deliver to report completion on the
    # HiCache prefetch daemon thread. A remote fetch crosses the control plane
    # plus a NIXL transfer, so this is generously above operation_timeout_ms;
    # exceeding it is reported as a miss and HiCache recomputes.
    get_timeout_s: float = 30.0

    @classmethod
    def from_extra_config(cls, extra_config: Optional[dict]) -> "KVCCBackendConfig":
        if not extra_config:
            return cls()
        known = {f for f in cls.__struct_fields__}
        filtered = {k: v for k, v in extra_config.items() if k in known}
        return msgspec.convert(filtered, cls)
