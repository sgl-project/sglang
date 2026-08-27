from __future__ import annotations

import dataclasses
import logging
import os
from typing import TYPE_CHECKING

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def handle_pd_disaggregation(server_args: ServerArgs) -> None:
    """Validate and normalize PD-disaggregation server args."""
    # "mooncake_tcp" is mooncake with the TCP transport forced: set MC_FORCE_TCP
    # so mooncake installs TcpTransport instead of RDMA, rewrite the backend to
    # mooncake, and skip RDMA HCA selection. Must run before backend-name checks.
    if server_args.disaggregation_transfer_backend == "mooncake_tcp":
        os.environ.setdefault("MC_FORCE_TCP", "1")
        server_args.disaggregation_transfer_backend = "mooncake"
        server_args.disaggregation_ib_device = None
        logger.info(
            "disaggregation transfer backend 'mooncake_tcp' -> mooncake "
            "with MC_FORCE_TCP=1 (TCP transport, no RDMA)"
        )

    if server_args.disaggregation_mode == "prefill" and server_args.dcp_size > 1:
        logger.warning(
            "DCP on a PD prefill server is supported when prefill and decode "
            "use the same DCP layout, but it usually adds communication "
            "overhead without improving prefill performance."
        )

    if server_args.disaggregation_mode == "decode" and server_args.dcp_size > 1:
        if server_args.disaggregation_transfer_backend not in ("mooncake", "nixl"):
            raise ValueError(
                "PD decode DCP requires --disaggregation-transfer-backend "
                "mooncake or nixl, got "
                f"{server_args.disaggregation_transfer_backend!r}."
            )
        if server_args.disaggregation_decode_enable_radix_cache:
            raise ValueError(
                "PD decode DCP currently requires chunk cache; "
                "--disaggregation-decode-enable-radix-cache is not supported."
            )
        if server_args.enable_hierarchical_cache:
            raise ValueError(
                "PD decode DCP currently requires chunk cache; "
                "--enable-hierarchical-cache is not supported."
            )

    if server_args.disaggregation_mode == "decode":
        if server_args.disaggregation_decode_enable_radix_cache:
            if server_args.enable_hisparse:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with --enable-hisparse"
                )
            if server_args.disaggregation_transfer_backend == "fake":
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with --disaggregation-transfer-backend fake"
                )
            if server_args.speculative_algorithm is not None:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with speculative decoding "
                    f"(--speculative-algorithm {server_args.speculative_algorithm})"
                )
            from sglang.srt.arg_groups.overrides import resolved_view

            if resolved_view(server_args).enable_dp_attention:
                logger.warning(
                    "EXPERIMENTAL: Decode radix cache with DP attention. "
                    "Requires prefix-aware DP rank routing for optimal cache hits."
                )
            server_args.disable_radix_cache = False
            logger.warning("EXPERIMENTAL: Radix cache is enabled for decode server")
        else:
            server_args.disable_radix_cache = True
            logger.warning("KV cache is forced as chunk cache for decode server")

        # Default the number of *extra* decode req_to_token slots reserved for
        # in-transfer (being-received-from-prefill) requests, on top of the
        # max_running_requests-derived pool. Large batches get none; small
        # per-worker batches reserve 2x the batch as cheap overlap headroom.
        if server_args.disaggregation_decode_extra_slots is None:
            extra_slots = 0
            if server_args.max_running_requests is not None:
                per_worker = server_args.max_running_requests // max(
                    1, server_args.dp_size
                )
                if per_worker <= 32:
                    extra_slots = per_worker * 2
            server_args.disaggregation_decode_extra_slots = extra_slots

    elif server_args.disaggregation_mode == "prefill":
        assert (
            server_args.disaggregation_transfer_backend != "fake"
        ), "Prefill server does not support 'fake' as the transfer backend"

        if envs.SGLANG_RUST_SERVER.get():
            _alias_bootstrap_port_to_api_port(server_args)

    if server_args.disaggregation_mode in ("prefill", "decode"):
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and server_args.disaggregation_transfer_backend not in ("mooncake", "nixl")
        ):
            raise ValueError(
                f"SGLANG_DISAGG_STAGING_BUFFER requires "
                f"disaggregation_transfer_backend='mooncake' or 'nixl', "
                f"got '{server_args.disaggregation_transfer_backend}'."
            )


def _alias_bootstrap_port_to_api_port(server_args: ServerArgs) -> None:
    """Rust-server prefill serves the KV bootstrap registry on the api listener
    itself, so the resolved bootstrap port must BE the api port — every internal
    consumer (KVManager registration, PrefillBootstrapQueue) reads the resolved
    field and agrees automatically. Decode is untouched: there the field names
    the PREFILL side's bootstrap port and must stay as the operator set it.
    """
    default_port = next(
        f.default
        for f in dataclasses.fields(server_args)
        if f.name == "disaggregation_bootstrap_port"
    )
    if server_args.disaggregation_bootstrap_port not in (
        default_port,
        server_args.port,
    ):
        raise ValueError(
            "SGLANG_RUST_SERVER serves the PD KV bootstrap registry on the api "
            "port itself; --disaggregation-bootstrap-port "
            f"{server_args.disaggregation_bootstrap_port} conflicts with --port "
            f"{server_args.port}. Drop --disaggregation-bootstrap-port (decode "
            "nodes and the PD router must then target the prefill api port)."
        )
    if server_args.disaggregation_bootstrap_port != server_args.port:
        logger.info(
            "SGLANG_RUST_SERVER: KV bootstrap registry is served on the api "
            "port; disaggregation_bootstrap_port %d -> %d",
            server_args.disaggregation_bootstrap_port,
            server_args.port,
        )
        server_args.disaggregation_bootstrap_port = server_args.port
