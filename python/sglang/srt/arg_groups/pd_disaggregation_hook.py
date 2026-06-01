import logging
import os
from typing import TYPE_CHECKING

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def handle_pd_disaggregation(server_args: "ServerArgs") -> None:
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

    if server_args.disaggregation_mode == "decode":
        if server_args.disaggregation_decode_enable_radix_cache:
            if server_args.enable_hisparse:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with --enable-hisparse"
                )
            if server_args.disaggregation_transfer_backend not in ("nixl", "mooncake"):
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache currently "
                    "requires --disaggregation-transfer-backend in "
                    "('nixl', 'mooncake'), but got "
                    f"{server_args.disaggregation_transfer_backend!r}"
                )
            if server_args.speculative_algorithm is not None:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with speculative decoding "
                    f"(--speculative-algorithm {server_args.speculative_algorithm})"
                )
            if server_args.enable_dp_attention:
                logger.warning(
                    "EXPERIMENTAL: Decode radix cache with DP attention. "
                    "Requires prefix-aware DP rank routing for optimal cache hits."
                )
            server_args.disable_radix_cache = False
            logger.warning("EXPERIMENTAL: Radix cache is enabled for decode server")

            if server_args.disaggregation_decode_enable_hicache:
                server_args.enable_hierarchical_cache = True
                if server_args.disaggregation_decode_hicache_size > 0:
                    server_args.hicache_size = (
                        server_args.disaggregation_decode_hicache_size
                    )
                else:
                    server_args.hicache_ratio = (
                        server_args.disaggregation_decode_hicache_ratio
                    )
                server_args.hicache_write_policy = (
                    server_args.disaggregation_decode_hicache_write_policy
                )
                logger.warning(
                    "EXPERIMENTAL: HiRadixCache is enabled for decode server "
                    f"(ratio={server_args.hicache_ratio}, "
                    f"write_policy={server_args.hicache_write_policy})"
                )
        else:
            if server_args.disaggregation_decode_enable_hicache:
                raise ValueError(
                    "--disaggregation-decode-enable-hicache requires "
                    "--disaggregation-decode-enable-radix-cache to be enabled."
                )
            server_args.disable_radix_cache = True
            logger.warning("KV cache is forced as chunk cache for decode server")
            if server_args.enable_mamba_extra_buffer():
                logger.warning(
                    "Mamba extra_buffer is disabled because decode disaggregation "
                    "currently forces chunk cache. Falling back to no_buffer."
                )
                server_args.mamba_scheduler_strategy = "no_buffer"

    elif server_args.disaggregation_mode == "prefill":
        assert (
            server_args.disaggregation_transfer_backend != "fake"
        ), "Prefill server does not support 'fake' as the transfer backend"

        server_args.disable_cuda_graph = True

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
