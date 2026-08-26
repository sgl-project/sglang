from __future__ import annotations

import dataclasses
import logging
import os
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def handle_pd_disaggregation(server_args: ServerArgs) -> None:
    """Validate and normalize PD-disaggregation server args."""
    cfg = resolving_view(server_args)
    # "mooncake_tcp" is mooncake with the TCP transport forced: set MC_FORCE_TCP
    # so mooncake installs TcpTransport instead of RDMA, rewrite the backend to
    # mooncake, and skip RDMA HCA selection. Must run before backend-name checks.
    if cfg.disaggregation_transfer_backend == "mooncake_tcp":
        os.environ.setdefault("MC_FORCE_TCP", "1")
        declare_resolution(
            server_args,
            "handle_pd_disaggregation",
            disaggregation_transfer_backend="mooncake",
        )
        declare_resolution(
            server_args,
            "handle_pd_disaggregation",
            disaggregation_ib_device=None,
        )
        logger.info(
            "disaggregation transfer backend 'mooncake_tcp' -> mooncake "
            "with MC_FORCE_TCP=1 (TCP transport, no RDMA)"
        )

    if cfg.disaggregation_mode == "prefill" and cfg.dcp_size > 1:
        logger.warning(
            "DCP on a PD prefill server is supported when prefill and decode "
            "use the same DCP layout, but it usually adds communication "
            "overhead without improving prefill performance."
        )

    if cfg.disaggregation_mode == "decode" and cfg.dcp_size > 1:
        # Fake transfer moves no KV and is only used for synthetic decode
        # benchmarks, so it does not need the DCP relayout from Mooncake/NIXL.
        if cfg.disaggregation_transfer_backend not in (
            "mooncake",
            "nixl",
            "fake",
        ):
            raise ValueError(
                "PD decode DCP requires --disaggregation-transfer-backend "
                "mooncake, nixl, or fake for synthetic benchmarking, got "
                f"{cfg.disaggregation_transfer_backend!r}."
            )
        if cfg.disaggregation_decode_enable_radix_cache:
            raise ValueError(
                "PD decode DCP currently requires chunk cache; "
                "--disaggregation-decode-enable-radix-cache is not supported."
            )
        if cfg.enable_hierarchical_cache:
            raise ValueError(
                "PD decode DCP currently requires chunk cache; "
                "--enable-hierarchical-cache is not supported."
            )

    if cfg.disaggregation_mode == "decode":
        if cfg.disaggregation_decode_enable_radix_cache:
            if cfg.enable_hisparse:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with --enable-hisparse"
                )
            if cfg.disaggregation_transfer_backend == "fake":
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with --disaggregation-transfer-backend fake"
                )
            if cfg.speculative_algorithm is not None:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with speculative decoding "
                    f"(--speculative-algorithm {cfg.speculative_algorithm})"
                )
            from sglang.srt.arg_groups.overrides import resolved_view

            if resolved_view(server_args).enable_dp_attention:
                logger.warning(
                    "EXPERIMENTAL: Decode radix cache with DP attention. "
                    "Requires prefix-aware DP rank routing for optimal cache hits."
                )
            declare_resolution(
                server_args,
                "handle_pd_disaggregation",
                disable_radix_cache=False,
            )
            logger.warning("EXPERIMENTAL: Radix cache is enabled for decode server")
        else:
            declare_resolution(
                server_args,
                "handle_pd_disaggregation",
                disable_radix_cache=True,
            )
            logger.warning("KV cache is forced as chunk cache for decode server")

        # Default the number of *extra* decode req_to_token slots reserved for
        # in-transfer (being-received-from-prefill) requests, on top of the
        # max_running_requests-derived pool. Large batches get none; small
        # per-worker batches reserve 2x the batch as cheap overlap headroom.
        if cfg.disaggregation_decode_extra_slots is None:
            extra_slots = 0
            if cfg.max_running_requests is not None:
                per_worker = cfg.max_running_requests // max(1, cfg.dp_size)
                if per_worker <= 32:
                    extra_slots = per_worker * 2
            declare_resolution(
                server_args,
                "handle_pd_disaggregation",
                disaggregation_decode_extra_slots=extra_slots,
            )

    elif cfg.disaggregation_mode == "prefill":
        assert (
            cfg.disaggregation_transfer_backend != "fake"
        ), "Prefill server does not support 'fake' as the transfer backend"

        if envs.SGLANG_RUST_SERVER.get():
            _alias_bootstrap_port_to_api_port(server_args)

    if cfg.disaggregation_mode in ("prefill", "decode"):
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and cfg.disaggregation_transfer_backend not in ("mooncake", "nixl")
        ):
            raise ValueError(
                f"SGLANG_DISAGG_STAGING_BUFFER requires "
                f"disaggregation_transfer_backend='mooncake' or 'nixl', "
                f"got '{cfg.disaggregation_transfer_backend}'."
            )


def _alias_bootstrap_port_to_api_port(server_args: ServerArgs) -> None:
    """Rust-server prefill serves the KV bootstrap registry on the api listener
    itself, so the resolved bootstrap port must BE the api port — every internal
    consumer (KVManager registration, PrefillBootstrapQueue) reads the resolved
    field and agrees automatically. Decode is untouched: there the field names
    the PREFILL side's bootstrap port and must stay as the operator set it.
    """
    cfg = resolving_view(server_args)
    default_port = next(
        f.default
        for f in dataclasses.fields(server_args)
        if f.name == "disaggregation_bootstrap_port"
    )
    if cfg.disaggregation_bootstrap_port not in (
        default_port,
        cfg.port,
    ):
        raise ValueError(
            "SGLANG_RUST_SERVER serves the PD KV bootstrap registry on the api "
            "port itself; --disaggregation-bootstrap-port "
            f"{cfg.disaggregation_bootstrap_port} conflicts with --port "
            f"{cfg.port}. Drop --disaggregation-bootstrap-port (decode "
            "nodes and the PD router must then target the prefill api port)."
        )
    if cfg.disaggregation_bootstrap_port != cfg.port:
        logger.info(
            "SGLANG_RUST_SERVER: KV bootstrap registry is served on the api "
            "port; disaggregation_bootstrap_port %d -> %d",
            cfg.disaggregation_bootstrap_port,
            cfg.port,
        )
        declare_resolution(
            server_args,
            "_alias_bootstrap_port_to_api_port",
            disaggregation_bootstrap_port=cfg.port,
        )
