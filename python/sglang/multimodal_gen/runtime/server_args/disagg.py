# SPDX-License-Identifier: Apache-2.0
"""Disaggregated diffusion server argument helpers."""

from __future__ import annotations

from typing import ClassVar, Literal

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.utils.common import (
    format_tcp_endpoint,
    parse_tcp_host_port,
)
from sglang.multimodal_gen.utils import FlexibleArgumentParser


class DisaggServerArgsMixin:
    DISAGG_RESULT_PORT_OFFSETS: ClassVar[dict[RoleType, int]] = {
        RoleType.ENCODER: 1,
        RoleType.DENOISER: 2,
        RoleType.DECODER: 3,
    }

    def get_role_parallelism(self, role_type: RoleType) -> dict[str, int | None]:
        _none = {
            "tp_size": None,
            "sp_degree": None,
            "ulysses_degree": None,
            "ring_degree": None,
        }
        if role_type == RoleType.ENCODER:
            return {**_none, "tp_size": self.encoder_tp}
        if role_type == RoleType.DENOISER:
            return {
                "tp_size": self.denoiser_tp,
                "sp_degree": self.denoiser_sp,
                "ulysses_degree": self.denoiser_ulysses,
                "ring_degree": self.denoiser_ring,
            }
        if role_type == RoleType.DECODER:
            return {**_none, "sp_degree": self.decoder_sp}
        return _none

    def derive_pool_result_endpoint(self) -> str:
        host, base_port = parse_tcp_host_port(
            self.disagg_server_addr, "disagg_server_addr"
        )
        role = (
            self.disagg_role
            if isinstance(self.disagg_role, RoleType)
            else RoleType.from_string(self.disagg_role)
        )
        try:
            offset = self.DISAGG_RESULT_PORT_OFFSETS[role]
        except KeyError as exc:
            raise ValueError(
                "pool result endpoints are only defined for encoder, denoiser, "
                f"and decoder roles, got {role.value!r}"
            ) from exc
        return format_tcp_endpoint(host, base_port + offset, "pool_result_endpoint")

    def derive_pool_work_endpoint(self) -> str:
        return format_tcp_endpoint("0.0.0.0", self.scheduler_port, "pool_work_endpoint")

    def derive_pool_control_endpoint(self) -> str:
        return format_tcp_endpoint(
            "0.0.0.0", self.scheduler_port + 1, "pool_control_endpoint"
        )

    def derive_pool_control_advertised_endpoint(self) -> str:
        host = self.host or self.disagg_p2p_hostname or "127.0.0.1"
        if host == "0.0.0.0":
            host = self.disagg_p2p_hostname or "127.0.0.1"
        return format_tcp_endpoint(
            host, self.scheduler_port + 1, "pool_control_advertised_endpoint"
        )

    def dag_result_endpoint(self, node_index: int) -> str:
        """Result endpoint for the ``node_index``-th node of a custom DAG.

        Mirrors ``DISAGG_RESULT_PORT_OFFSETS`` for classic roles: node ``i``
        gets ``base_port + i + 1``, leaving the base port for the frontend.

        Workers learn the base address from ``--disagg-server-addr``; the head
        node is that address, so it falls back to its own host and scheduler
        port and both sides derive the same endpoints.
        """
        if self.disagg_server_addr:
            host, base_port = parse_tcp_host_port(
                self.disagg_server_addr, "disagg_server_addr"
            )
        else:
            host = self.host or "127.0.0.1"
            base_port = self.scheduler_port
        return format_tcp_endpoint(
            host, base_port + node_index + 1, "dag_result_endpoint"
        )

    def resolve_execution_plan(self):
        """Compile ``--disagg-dag`` into an ``ExecutionPlan``, or None.

        Returns None for classic three-role deployments, which the runtime
        treats as an implicit linear DAG.  The compiled plan is cached because
        both the scheduler and the pipeline builder ask for it.
        """
        from sglang.multimodal_gen.runtime.disaggregation.dag import (
            DagSpec,
            ExecutionPlan,
        )

        if not getattr(self, "disagg_dag", None):
            return None

        cached = getattr(self, "_execution_plan", None)
        if cached is not None:
            return cached

        spec = DagSpec.load(self.disagg_dag)
        for index, role in enumerate(spec.roles):
            pool = spec.get_pool(role.name)
            if pool is not None and not pool.result_endpoint:
                pool.result_endpoint = self.dag_result_endpoint(index)

        plan = ExecutionPlan.compile(spec)
        self._execution_plan = plan
        return plan

    def resolved_role_device(self) -> Literal["cpu", "cuda"]:
        if self.disagg_role_device == "auto":
            return "cpu" if self.num_gpus <= 0 else "cuda"
        return self.disagg_role_device

    @classmethod
    def add_disagg_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        role_default = (
            cls.disagg_role.value
            if isinstance(cls.disagg_role, RoleType)
            else cls.disagg_role
        )
        parser.add_argument(
            "--disagg-role",
            type=str,
            default=role_default,
            choices=RoleType.choices(),
            help="Role for disaggregated pipeline.",
        )
        parser.add_argument(
            "--disagg-dag",
            type=str,
            default=cls.disagg_dag,
            help=(
                "Path to a JSON/YAML DAG topology, or an inline JSON object. "
                "Replaces the fixed encoder/denoiser/decoder roles with named "
                "nodes, conditional routes, and independently scaled pools."
            ),
        )
        parser.add_argument(
            "--dag-node",
            type=str,
            default=cls.dag_node,
            help="Which node of --disagg-dag this worker process serves.",
        )
        parser.add_argument(
            "--dag-validate",
            action="store_true",
            default=cls.dag_validate,
            help=(
                "Compile --disagg-dag, print the execution plan JSON, and exit "
                "without loading model weights. Plan structure is validated at "
                "compile time; stage coverage vs the live pipeline is checked "
                "in unit/E2E tests."
            ),
        )
        parser.add_argument(
            "--disagg-timeout",
            type=int,
            default=cls.disagg_timeout,
            help="Timeout in seconds for pending disagg requests. "
            f"Default: {cls.disagg_timeout}.",
        )
        parser.add_argument(
            "--disagg-downstream-wait-timeout",
            type=int,
            default=cls.disagg_downstream_wait_timeout,
            help="Timeout in seconds while waiting for a downstream role slot. "
            f"Default: {cls.disagg_downstream_wait_timeout}.",
        )
        parser.add_argument(
            "--disagg-dispatch-policy",
            type=str,
            default=cls.disagg_dispatch_policy,
            choices=["round_robin", "max_free_slots"],
            help="Dispatch policy for pool mode disagg routing.",
        )
        parser.add_argument(
            "--disagg-instance-id",
            type=int,
            default=cls.disagg_instance_id,
            help="Stable per-role instance ID used by DiffusionServer registration.",
        )
        parser.add_argument(
            "--disagg-max-slots-per-instance",
            type=int,
            default=cls.disagg_max_slots_per_instance,
            help="Maximum concurrent transfer/computation slots tracked per instance.",
        )
        parser.add_argument(
            "--disagg-transfer-redundancy",
            type=float,
            default=cls.disagg_transfer_redundancy,
            help="Redundancy factor used when sizing transfer buffers from warmup.",
        )
        parser.add_argument(
            "--disagg-role-device",
            type=str,
            default=cls.disagg_role_device,
            choices=["auto", "cpu", "cuda"],
            help=(
                "Per-role device override. 'cpu' is intended for same-machine "
                "encoder roles."
            ),
        )
        parser.add_argument(
            "--disagg-transfer-backend",
            type=str,
            default=cls.disagg_transfer_backend,
            choices=["auto", "mock", "mooncake"],
            help="Transfer backend for multimodal diffusion disaggregation.",
        )
        parser.add_argument(
            "--disagg-transfer-pool-size",
            type=int,
            default=cls.disagg_transfer_pool_size,
            help="Size of the P2P transfer buffer pool in bytes.",
        )
        parser.add_argument(
            "--disagg-transfer-pin-memory",
            type=str,
            default=cls.disagg_transfer_pin_memory,
            choices=["auto", "off", "required"],
            help="CUDA host-register same-host shared-memory transfer buffers.",
        )
        parser.add_argument(
            "--disagg-p2p-hostname",
            type=str,
            default=cls.disagg_p2p_hostname,
            help="Hostname for P2P transfer engine.",
        )
        parser.add_argument(
            "--disagg-ib-device",
            type=str,
            default=cls.disagg_ib_device,
            help="InfiniBand device for P2P RDMA transfers.",
        )
        parser.add_argument(
            "--disagg-server-addr",
            type=str,
            default=cls.disagg_server_addr,
            help="DiffusionServer head node address for per-role launch mode.",
        )
        parser.add_argument(
            "--encoder-urls",
            type=str,
            default=cls.encoder_urls,
            help="Encoder instance work endpoints for DiffusionServer head mode.",
        )
        parser.add_argument(
            "--denoiser-urls",
            type=str,
            default=cls.denoiser_urls,
            help="Denoiser instance work endpoints for DiffusionServer head mode.",
        )
        parser.add_argument(
            "--decoder-urls",
            type=str,
            default=cls.decoder_urls,
            help="Decoder instance work endpoints for DiffusionServer head mode.",
        )
        parser.add_argument(
            "--encoder-tp",
            type=int,
            default=cls.encoder_tp,
            help="Tensor parallelism for encoder role.",
        )
        parser.add_argument(
            "--denoiser-tp",
            type=int,
            default=cls.denoiser_tp,
            help="Tensor parallelism for denoiser role.",
        )
        parser.add_argument(
            "--denoiser-sp",
            type=int,
            default=cls.denoiser_sp,
            help="Sequence parallelism for denoiser role.",
        )
        parser.add_argument(
            "--denoiser-ulysses",
            type=int,
            default=cls.denoiser_ulysses,
            help="Ulysses SP degree for denoiser role.",
        )
        parser.add_argument(
            "--denoiser-ring",
            type=int,
            default=cls.denoiser_ring,
            help="Ring SP degree for denoiser role.",
        )
        parser.add_argument(
            "--decoder-sp",
            type=int,
            default=cls.decoder_sp,
            help="Sequence parallelism for decoder role.",
        )
        parser.add_argument(
            "--decoder-tp",
            type=int,
            default=cls.decoder_tp,
            help="Deprecated alias for --decoder-sp.",
        )
