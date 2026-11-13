# SPDX-License-Identifier: Apache-2.0
"""Disaggregated diffusion server argument helpers."""

from __future__ import annotations

from dataclasses import fields
from typing import ClassVar

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.utils.argparse import FlexibleArgumentParser
from sglang.multimodal_gen.runtime.utils.common import (
    format_tcp_endpoint,
    parse_tcp_host_port,
)


class DisaggServerArgsMixin:
    DISAGG_RESULT_PORT_OFFSETS: ClassVar[dict[RoleType, int]] = {
        RoleType.ENCODER: 1,
        RoleType.DENOISER: 2,
        RoleType.DECODER: 3,
        RoleType.PROMPT_ENCODER: 4,
        RoleType.IMAGE_ENCODER: 5,
    }

    def get_role_parallelism(self, role_type: RoleType) -> dict[str, int | None]:
        _none = {
            "tp_size": None,
            "sp_degree": None,
            "ulysses_degree": None,
            "ring_degree": None,
        }
        if role_type.is_input_encoder or role_type == RoleType.IMAGE_ENCODER:
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

    def for_disagg_role(self, role_type: RoleType, **overrides):
        """Derive a worker's arguments without inheriting resolved parent topology."""
        values = vars(self)
        kwargs = {
            f.name: values[f.name]
            for f in fields(self)
            if f.init and f.name != "pipeline_config"
        }
        explicit = self._explicit_arg_names.copy()
        # re-run automatic choices for this role's device count, not the parent's
        for name in (
            "tp_size",
            "sp_degree",
            "ulysses_degree",
            "ring_degree",
            "kv_gather_degree",
            "enable_cfg_parallel",
            "cfg_parallel_degree",
            "hsdp_shard_dim",
        ):
            if name not in explicit:
                kwargs.pop(name)
        kwargs["sp_split_auto"] = False
        # role-specific flags override explicit generic flags; validation stays shared
        for name, value in self.get_role_parallelism(role_type).items():
            if value is not None:
                kwargs[name] = value
                explicit.add(name)
        kwargs.update(overrides)
        kwargs.update(
            disagg_role=role_type, disagg_mode=True, _explicit_arg_names=explicit
        )
        return type(self).from_kwargs(**kwargs)

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
                f"pool result endpoints are only defined for worker roles, got {role.value!r}"
            ) from exc
        return format_tcp_endpoint(host, base_port + offset, "pool_result_endpoint")

    def derive_pool_work_endpoint(self) -> str:
        return format_tcp_endpoint("0.0.0.0", self.scheduler_port, "pool_work_endpoint")

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
            "--disagg-timeout",
            type=int,
            default=cls.disagg_timeout,
            help="Timeout in seconds for pending disagg requests. "
            f"Default: {cls.disagg_timeout}.",
        )
        parser.add_argument(
            "--disagg-dispatch-policy",
            type=str,
            default=cls.disagg_dispatch_policy,
            choices=["round_robin", "max_free_slots"],
            help="Dispatch policy for pool mode disagg routing.",
        )
        parser.add_argument(
            "--disagg-transfer-pool-size",
            type=int,
            default=cls.disagg_transfer_pool_size,
            help="Size of the P2P transfer buffer pool in bytes.",
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
            "--prompt-encoder-urls",
            type=str,
            default=cls.prompt_encoder_urls,
            help="Semicolon-separated prompt encoder endpoints; use with --image-encoder-urls instead of --encoder-urls.",
        )
        parser.add_argument(
            "--image-encoder-urls",
            type=str,
            default=cls.image_encoder_urls,
            help="Semicolon-separated image encoder endpoints for split encoding.",
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
