from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def handle_pd_disaggregation(server_args: ServerArgs) -> None:
    """Validate and normalize PD-disaggregation server args."""
    for env, attr in (
        (envs.SGLANG_DISAGG_LAYER_PIPELINE, "enable_disagg_layer_pipeline"),
        (envs.SGLANG_DISAGG_LAYER_GROUP_SIZE, "disagg_layer_group_size"),
        (
            envs.SGLANG_DISAGG_LAYER_PIPELINE_MIN_PREFILL_LEN,
            "disagg_layer_pipeline_min_prefill_len",
        ),
    ):
        if env.is_set():
            setattr(server_args, attr, env.get())

    if server_args.enable_disagg_layer_pipeline:
        server_args.enable_disagg_draft_layout_validation = True
        if server_args.disaggregation_mode not in ("prefill", "decode"):
            raise ValueError(
                "--enable-disagg-layer-pipeline requires --disaggregation-mode "
                "prefill or decode."
            )
        if server_args.disagg_layer_group_size < 0:
            raise ValueError("--disagg-layer-group-size must be >= 0.")
        if server_args.disagg_layer_pipeline_min_prefill_len < 0:
            raise ValueError(
                "--disagg-layer-pipeline-min-prefill-len must be >= 0."
            )
        if server_args.enable_hisparse:
            raise ValueError(
                "--enable-disagg-layer-pipeline does not currently support "
                "--enable-hisparse. Disable one of these flags."
            )

    # "mooncake_tcp" forces mooncake's TCP transport: set MC_FORCE_TCP, rewrite
    # the backend to mooncake, and skip RDMA HCA selection. Must run before
    # backend-name checks.
    if server_args.disaggregation_transfer_backend == "mooncake_tcp":
        os.environ.setdefault("MC_FORCE_TCP", "1")
        server_args.disaggregation_transfer_backend = "mooncake"
        server_args.disaggregation_ib_device = None
        logger.info(
            "disaggregation transfer backend 'mooncake_tcp' -> mooncake "
            "with MC_FORCE_TCP=1 (TCP transport, no RDMA)"
        )

    if (
        server_args.enable_disagg_layer_pipeline
        and server_args.disaggregation_transfer_backend != "mooncake"
    ):
        raise ValueError(
            "--enable-disagg-layer-pipeline currently requires "
            "--disaggregation-transfer-backend=mooncake."
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
