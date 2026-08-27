# SPDX-License-Identifier: Apache-2.0
"""Disaggregated diffusion CLI arguments and helper methods.

All disagg-related dataclass fields, argparse registration, and endpoint
derivation logic live here.  ``ServerArgs`` inherits from
``DisaggArgsMixin`` so the fields appear on the top-level config object.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

if TYPE_CHECKING:
    pass

# ── Port offsets for disagg result endpoints (deterministic convention) ──
DISAGG_RESULT_PORT_OFFSETS: dict[RoleType, int] = {
    RoleType.ENCODER: 1,
    RoleType.DENOISER: 2,
    RoleType.DECODER: 3,
}


class DisaggArgsMixin:
    """Methods for disaggregated diffusion, mixed into ``ServerArgs``.

    The dataclass **fields** remain in ``ServerArgs`` (to avoid MRO
    ordering issues with ``@dataclass`` inheritance).  This mixin only
    provides the methods that operate on those fields.
    """

    def get_role_parallelism(self, role_type: RoleType) -> dict[str, int | None]:
        """Return per-role parallelism overrides for the given role.

        Returns a dict with keys tp_size, sp_degree, ulysses_degree,
        ring_degree.  Values are ``None`` when not explicitly set
        (auto-derive from ``num_gpus``).
        """
        _none: dict[str, int | None] = {
            "tp_size": None,
            "sp_degree": None,
            "ulysses_degree": None,
            "ring_degree": None,
        }
        if role_type == RoleType.ENCODER:
            return {**_none, "tp_size": self.encoder_tp}
        elif role_type == RoleType.DENOISER:
            return {
                "tp_size": self.denoiser_tp,
                "sp_degree": self.denoiser_sp,
                "ulysses_degree": self.denoiser_ulysses,
                "ring_degree": self.denoiser_ring,
            }
        elif role_type == RoleType.DECODER:
            return {**_none, "tp_size": self.decoder_tp}
        return _none

    def derive_pool_result_endpoint(self) -> str:
        """Derive the result PUSH endpoint from ``disagg_server_addr`` + role.

        Convention: DS binds result PULL on ``scheduler_port + {1,2,3}``
        for encoder / denoiser / decoder.
        """
        if self.disagg_server_addr is None:
            raise ValueError("disagg_server_addr is required for per-role launch")
        addr = self.disagg_server_addr
        if addr.startswith("tcp://"):
            addr = addr[len("tcp://") :]
        host, port_str = addr.rsplit(":", 1)
        base_port = int(port_str)
        offset = DISAGG_RESULT_PORT_OFFSETS[self.disagg_role]
        return f"tcp://{host}:{base_port + offset}"

    def derive_pool_work_endpoint(self) -> str:
        """Derive the work PULL bind endpoint for a standalone role instance."""
        return f"tcp://0.0.0.0:{self.scheduler_port}"


# ── CLI registration ─────────────────────────────────────────────────


def add_disagg_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register all disaggregated-diffusion CLI arguments as a group."""

    g = parser.add_argument_group(
        "Disaggregated diffusion",
        "Split the pipeline into independent Encoder / Denoiser / Decoder "
        "roles, each on its own GPU(s).  A DiffusionServer head node routes "
        "requests.  See docs/disaggregation.md for details.",
    )

    # Core
    g.add_argument(
        "--base-gpu-id",
        type=int,
        default=0,
        help="Starting GPU ID for this instance.  Used with --disagg-role "
        "to place role instances on specific GPUs without CUDA_VISIBLE_DEVICES.",
    )
    g.add_argument(
        "--disagg-role",
        type=str,
        default=RoleType.MONOLITHIC.value,
        choices=RoleType.choices(),
        help="Role for disaggregated pipeline.  "
        "'monolithic' (default): single server.  "
        "'encoder' / 'denoiser' / 'decoder': role instance.  "
        "'server': DiffusionServer head node (no GPU).  "
        "Role instances require --disagg-server-addr.  "
        "Server requires --encoder-urls, --denoiser-urls, --decoder-urls.",
    )
    g.add_argument(
        "--disagg-server-addr",
        type=str,
        default=None,
        help="DiffusionServer head node address (tcp://HOST:PORT).  "
        "Required for role instances.",
    )
    g.add_argument(
        "--disagg-timeout",
        type=int,
        default=600,
        help="Timeout in seconds for pending disagg requests (default: 600).",
    )
    g.add_argument(
        "--disagg-dispatch-policy",
        type=str,
        default="round_robin",
        choices=["round_robin", "max_free_slots"],
        help="Dispatch policy: 'round_robin' or 'max_free_slots' (default: round_robin).",
    )

    # Server head: remote instance URLs
    g.add_argument(
        "--encoder-urls",
        type=str,
        default=None,
        help="Encoder work endpoints (semicolon-separated).  "
        "Example: 'tcp://10.0.0.1:35000;tcp://10.0.0.2:35000'.",
    )
    g.add_argument(
        "--denoiser-urls",
        type=str,
        default=None,
        help="Denoiser work endpoints (semicolon-separated).",
    )
    g.add_argument(
        "--decoder-urls",
        type=str,
        default=None,
        help="Decoder work endpoints (semicolon-separated).",
    )

    # Per-role parallelism
    g.add_argument("--encoder-tp", type=int, default=None, help="Encoder TP degree.")
    g.add_argument("--denoiser-tp", type=int, default=None, help="Denoiser TP degree.")
    g.add_argument("--denoiser-sp", type=int, default=None, help="Denoiser SP degree.")
    g.add_argument(
        "--denoiser-ulysses", type=int, default=None, help="Denoiser Ulysses degree."
    )
    g.add_argument(
        "--denoiser-ring", type=int, default=None, help="Denoiser Ring degree."
    )
    g.add_argument("--decoder-tp", type=int, default=None, help="Decoder TP degree.")

    # P2P transfer engine
    g.add_argument(
        "--disagg-transfer-pool-size",
        type=int,
        default=256 * 1024 * 1024,
        help="P2P transfer buffer pool size in bytes (default: 256 MiB).",
    )
    g.add_argument(
        "--disagg-p2p-hostname",
        type=str,
        default="127.0.0.1",
        help="RDMA-reachable hostname/IP of this instance (default: 127.0.0.1).",
    )
    g.add_argument(
        "--disagg-ib-device",
        type=str,
        default=None,
        help="InfiniBand device for RDMA transfers (e.g., mlx5_0).",
    )


def convert_disagg_role_string(kwargs: dict) -> None:
    """Convert ``disagg_role`` from string to ``RoleType`` enum in-place."""
    if "disagg_role" in kwargs and isinstance(kwargs["disagg_role"], str):
        kwargs["disagg_role"] = RoleType.from_string(kwargs["disagg_role"])
