from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

DSV4_DCP_DIRECT_A2A_HEADS = 128
DSV4_DCP_DIRECT_A2A_HEAD_DIM = 512
DSV4_DCP_DIRECT_A2A_WORLD_SIZES = (2, 4, 8)
DSV4_DCP_DESTINATION_PUSH_PLANES = 2
_BF16_PER_FP32 = 2


@dataclass(frozen=True)
class DCPA2AOutputWorkspace:
    """Stable storage for the existing packed DCP A2A transport.

    Both tensors have physical layout ``[destination, batch, local_head,
    D + 2]`` in BF16. The first ``D`` elements hold BF16 attention output and
    the final two BF16 elements hold one raw FP32 LSE. After all-to-all, axis 0
    of ``recv_combined`` is the source-rank axis consumed by the local LSE
    combine.
    """

    send_combined: torch.Tensor
    recv_combined: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        world_size: int,
        max_batch_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str,
    ) -> DCPA2AOutputWorkspace:
        local_heads = num_heads // world_size
        shape = (
            world_size,
            max_batch_size,
            local_heads,
            head_dim + _BF16_PER_FP32,
        )
        send_combined = torch.empty(shape, dtype=torch.bfloat16, device=device)
        return cls(
            send_combined=send_combined,
            recv_combined=torch.empty_like(send_combined),
        )

    def supports(
        self,
        *,
        world_size: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        if (
            world_size not in DSV4_DCP_DIRECT_A2A_WORLD_SIZES
            or batch_size != 1
            or num_heads != DSV4_DCP_DIRECT_A2A_HEADS
            or head_dim != DSV4_DCP_DIRECT_A2A_HEAD_DIM
            or num_heads % world_size
            or dtype != torch.bfloat16
        ):
            return False

        expected_tail = (num_heads // world_size, head_dim + _BF16_PER_FP32)
        send = self.send_combined
        recv = self.recv_combined
        if (
            send.dtype != torch.bfloat16
            or recv.dtype != torch.bfloat16
            or send.device != device
            or recv.device != device
            or send.ndim != 4
            or recv.shape != send.shape
            or send.shape[0] != world_size
            or send.shape[1] < batch_size
            or tuple(send.shape[2:]) != expected_tail
            or not send.is_contiguous()
            or not recv.is_contiguous()
        ):
            return False

        # A BF16 view can be reinterpreted as FP32 only at an even element
        # offset. Guarded test workspaces deliberately use non-zero offsets.
        return (
            send.storage_offset() % _BF16_PER_FP32 == 0
            and recv.storage_offset() % _BF16_PER_FP32 == 0
        )

    def send_output_and_lse_views(
        self,
        *,
        batch_size: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return producer views without allocating or changing physical order."""
        output = self.send_combined[:, :batch_size, :, :head_dim]
        lse = self.send_combined.view(torch.float32)[
            :, :batch_size, :, head_dim // _BF16_PER_FP32
        ]
        return output, lse


@dataclass(frozen=True)
class DCPA2APackedOutput:
    """Typed marker that the producer already populated the A2A send buffer."""

    workspace: DCPA2AOutputWorkspace
    batch_size: int
    num_heads: int
    head_dim: int
    is_lse_base_on_e: bool

    def is_valid_for(
        self,
        *,
        world_size: int,
        device: torch.device,
    ) -> bool:
        return self.is_lse_base_on_e and self.workspace.supports(
            world_size=world_size,
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )


@dataclass(frozen=True)
class DCPDestinationPushWorkspace:
    """AITER-owned fixed double receive planes for peer destination push."""

    recv_planes: torch.Tensor
    peer_recv_ptrs: torch.Tensor
    epoch: torch.Tensor
    source_rank: int

    def supports(
        self,
        *,
        world_size: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        if (
            world_size not in DSV4_DCP_DIRECT_A2A_WORLD_SIZES
            or not 0 <= self.source_rank < world_size
            or batch_size != 1
            or num_heads != DSV4_DCP_DIRECT_A2A_HEADS
            or head_dim != DSV4_DCP_DIRECT_A2A_HEAD_DIM
            or num_heads % world_size
            or dtype != torch.bfloat16
        ):
            return False

        expected_shape = (
            DSV4_DCP_DESTINATION_PUSH_PLANES,
            world_size,
            1,
            num_heads // world_size,
            head_dim + _BF16_PER_FP32,
        )
        recv = self.recv_planes
        ptrs = self.peer_recv_ptrs
        epoch = self.epoch
        return (
            recv.dtype == torch.bfloat16
            and recv.device == device
            and tuple(recv.shape) == expected_shape
            and recv.is_contiguous()
            and recv.storage_offset() % _BF16_PER_FP32 == 0
            and recv.data_ptr() % 16 == 0
            and ptrs.dtype == torch.uint64
            and ptrs.device == device
            and tuple(ptrs.shape) == (world_size,)
            and ptrs.is_contiguous()
            and ptrs.data_ptr() % 8 == 0
            and epoch.dtype == torch.int32
            and epoch.device == device
            and tuple(epoch.shape) == (1,)
            and epoch.is_contiguous()
            and epoch.data_ptr() % 4 == 0
        )

    def receive_output_and_lse_views(
        self,
        *,
        batch_size: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both receive planes without selecting one on the host."""
        output = self.recv_planes[:, :, :batch_size, :, :head_dim]
        lse = self.recv_planes.view(torch.float32)[
            :, :, :batch_size, :, head_dim // _BF16_PER_FP32
        ]
        return output, lse


@dataclass(frozen=True)
class DCPDestinationPushOutput:
    """Marker that the producer wrote its source plane directly to peers."""

    workspace: DCPDestinationPushWorkspace
    batch_size: int
    num_heads: int
    head_dim: int
    is_lse_base_on_e: bool

    def is_valid_for(
        self,
        *,
        world_size: int,
        device: torch.device,
    ) -> bool:
        return self.is_lse_base_on_e and self.workspace.supports(
            world_size=world_size,
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )


def can_use_dsv4_dcp_direct_a2a_output(
    *,
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    workspace: Optional[DCPA2AOutputWorkspace],
    feature_enabled: bool,
    is_gfx950: bool,
    dcp_size: int,
    comm_backend: str,
    is_plain_decode: bool,
    batch_size: int,
    speculative: bool,
    tbo: bool,
    memory_saver: bool,
    hisparse: bool,
    state_capture: bool,
    piecewise_graph: bool,
    breakable_graph: bool,
) -> bool:
    """Pure routing gate for the first producer-integrated A2A milestone."""
    if (
        not feature_enabled
        or not is_gfx950
        or dcp_size not in DSV4_DCP_DIRECT_A2A_WORLD_SIZES
        or comm_backend != "a2a"
        or not is_plain_decode
        or batch_size != 1
        or speculative
        or tbo
        or memory_saver
        or hisparse
        or state_capture
        or piecewise_graph
        or breakable_graph
        or workspace is None
        or q.dtype != torch.bfloat16
        or unified_kv.dtype != torch.bfloat16
        or unified_kv.ndim != 2
        or unified_kv.shape[1] != DSV4_DCP_DIRECT_A2A_HEAD_DIM
        or not unified_kv.is_contiguous()
    ):
        return False

    if q.ndim == 3:
        supported_q_layout = (
            q.shape
            == (
                1,
                DSV4_DCP_DIRECT_A2A_HEADS,
                DSV4_DCP_DIRECT_A2A_HEAD_DIM,
            )
            and q.is_contiguous()
        )
    elif q.ndim == 4:
        local_heads = DSV4_DCP_DIRECT_A2A_HEADS // dcp_size
        supported_q_layout = (
            q.shape
            == (
                1,
                dcp_size,
                local_heads,
                DSV4_DCP_DIRECT_A2A_HEAD_DIM,
            )
            and q.stride(3) == 1
            and q.stride(2) == DSV4_DCP_DIRECT_A2A_HEAD_DIM
            and q.stride(1) >= local_heads * DSV4_DCP_DIRECT_A2A_HEAD_DIM
        )
    else:
        supported_q_layout = False
    if not supported_q_layout:
        return False

    return workspace.supports(
        world_size=dcp_size,
        batch_size=batch_size,
        num_heads=DSV4_DCP_DIRECT_A2A_HEADS,
        head_dim=DSV4_DCP_DIRECT_A2A_HEAD_DIM,
        dtype=q.dtype,
        device=q.device,
    )


def can_use_dsv4_dcp_destination_push_output(
    *,
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    workspace: DCPDestinationPushWorkspace | None,
    communicator: Any,
    feature_enabled: bool,
    is_gfx950: bool,
    dcp_size: int,
    comm_backend: str,
    is_plain_decode: bool,
    batch_size: int,
    speculative: bool,
    tbo: bool,
    memory_saver: bool,
    hisparse: bool,
    state_capture: bool,
    piecewise_graph: bool,
    breakable_graph: bool,
) -> bool:
    """Preflight the strict peer-push path before any remote store."""
    if (
        not feature_enabled
        or not is_gfx950
        or dcp_size not in DSV4_DCP_DIRECT_A2A_WORLD_SIZES
        or comm_backend != "a2a"
        or not is_plain_decode
        or batch_size != 1
        or speculative
        or tbo
        or memory_saver
        or hisparse
        or state_capture
        or piecewise_graph
        or breakable_graph
        or workspace is None
        or communicator is None
        or workspace.source_rank != getattr(communicator, "rank_in_group", -1)
        or q.dtype != torch.bfloat16
        or unified_kv.dtype != torch.bfloat16
        or unified_kv.ndim != 2
        or unified_kv.shape[1] != DSV4_DCP_DIRECT_A2A_HEAD_DIM
        or not unified_kv.is_contiguous()
    ):
        return False

    if q.ndim == 3:
        supported_q_layout = (
            q.shape
            == (
                1,
                DSV4_DCP_DIRECT_A2A_HEADS,
                DSV4_DCP_DIRECT_A2A_HEAD_DIM,
            )
            and q.is_contiguous()
        )
    elif q.ndim == 4:
        local_heads = DSV4_DCP_DIRECT_A2A_HEADS // dcp_size
        supported_q_layout = (
            q.shape
            == (
                1,
                dcp_size,
                local_heads,
                DSV4_DCP_DIRECT_A2A_HEAD_DIM,
            )
            and q.stride(3) == 1
            and q.stride(2) == DSV4_DCP_DIRECT_A2A_HEAD_DIM
            and q.stride(1) >= local_heads * DSV4_DCP_DIRECT_A2A_HEAD_DIM
        )
    else:
        supported_q_layout = False
    if not supported_q_layout or not workspace.supports(
        world_size=dcp_size,
        batch_size=batch_size,
        num_heads=DSV4_DCP_DIRECT_A2A_HEADS,
        head_dim=DSV4_DCP_DIRECT_A2A_HEAD_DIM,
        dtype=q.dtype,
        device=q.device,
    ):
        return False

    return bool(
        hasattr(communicator, "should_dsv4_dcp_destination_push")
        and communicator.should_dsv4_dcp_destination_push(
            workspace.recv_planes,
            workspace.peer_recv_ptrs,
            workspace.epoch,
        )
    )


def prepare_dsv4_dcp_destination_push_workspace(
    *,
    communicator: Any,
    device: torch.device,
    agreement_group: Any = None,
    candidate_name: str = "unknown",
) -> DCPDestinationPushWorkspace | None:
    """Prepare, validate, and collectively agree before graph capture."""
    agreement_group = communicator if agreement_group is None else agreement_group
    workspace = None
    ca_comm = getattr(communicator, "ca_comm", None)
    if ca_comm is None:
        logger.warning(
            "DSV4 destination-push candidate=%s ca_comm missing", candidate_name
        )
    else:
        ca_type = type(ca_comm).__name__
        ca_disabled = bool(getattr(ca_comm, "disabled", True))
        has_prepare = hasattr(ca_comm, "prepare_dsv4_destination_push_workspace")
        has_should = hasattr(ca_comm, "should_dsv4_destination_push")
        has_ready = hasattr(ca_comm, "dsv4_destination_push_ready")
        if ca_disabled or not (has_prepare and has_should and has_ready):
            logger.warning(
                "DSV4 destination-push candidate=%s ca_comm type=%s "
                "disabled=%s methods prepare=%s should=%s ready=%s",
                candidate_name,
                ca_type,
                ca_disabled,
                has_prepare,
                has_should,
                has_ready,
            )
        else:
            try:
                registered_workspace = (
                    communicator.prepare_dsv4_dcp_destination_push_workspace()
                )
            except Exception as exc:
                logger.warning(
                    "DSV4 destination-push candidate=%s registration exception=%r",
                    candidate_name,
                    exc,
                )
            else:
                if registered_workspace is None:
                    logger.warning(
                        "DSV4 destination-push candidate=%s AITER prepare returned None",
                        candidate_name,
                    )
                else:
                    try:
                        recv_planes, peer_recv_ptrs, epoch = registered_workspace
                        candidate = DCPDestinationPushWorkspace(
                            recv_planes=recv_planes,
                            peer_recv_ptrs=peer_recv_ptrs,
                            epoch=epoch,
                            source_rank=communicator.rank_in_group,
                        )
                        requested_device = torch.device(device)
                        registered_device = recv_planes.device
                        candidate_supported = (
                            requested_device.type == registered_device.type
                            and (
                                requested_device.index is None
                                or requested_device.index == registered_device.index
                            )
                            and candidate.supports(
                                world_size=8,
                                batch_size=1,
                                num_heads=DSV4_DCP_DIRECT_A2A_HEADS,
                                head_dim=DSV4_DCP_DIRECT_A2A_HEAD_DIM,
                                dtype=torch.bfloat16,
                                device=registered_device,
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            "DSV4 destination-push candidate=%s candidate "
                            "supports exception=%r",
                            candidate_name,
                            exc,
                        )
                    else:
                        if not candidate_supported:
                            logger.warning(
                                "DSV4 destination-push candidate=%s candidate "
                                "supports rejected requested_device=%s "
                                "registered_device=%s",
                                candidate_name,
                                device,
                                recv_planes.device,
                            )
                        else:
                            try:
                                should_push = (
                                    communicator.should_dsv4_dcp_destination_push(
                                        recv_planes, peer_recv_ptrs, epoch
                                    )
                                )
                            except Exception as exc:
                                logger.warning(
                                    "DSV4 destination-push candidate=%s should "
                                    "exception=%r",
                                    candidate_name,
                                    exc,
                                )
                            else:
                                if should_push:
                                    workspace = candidate
                                else:
                                    logger.warning(
                                        "DSV4 destination-push candidate=%s "
                                        "should rejected",
                                        candidate_name,
                                    )

    if agreement_group is None or not hasattr(agreement_group, "all_reduce"):
        logger.warning(
            "DSV4 destination-push candidate=%s ready all-reduce unavailable",
            candidate_name,
        )
        return None
    try:
        ready_count = torch.tensor(
            [workspace is not None], dtype=torch.int32, device=device
        )
        ready_count = agreement_group.all_reduce(ready_count)
        ready_value = int(ready_count.item())
    except Exception as exc:
        logger.warning(
            "DSV4 destination-push candidate=%s ready all-reduce exception=%r",
            candidate_name,
            exc,
        )
        return None
    if ready_value != 8:
        logger.warning(
            "DSV4 destination-push candidate=%s ready_count=%d/8",
            candidate_name,
            ready_value,
        )
        return None
    logger.info("DSV4 destination-push candidate=%s ready_count=8/8", candidate_name)
    return workspace


def dsv4_dcp_destination_push_groups_equivalent(
    *, dcp_group: Any, push_group: Any, dcp_rank: int
) -> bool:
    """Require identical DCP/candidate rank order before sharing peer ranks."""
    if dcp_group is None or push_group is None:
        return False
    try:
        dcp_ranks = tuple(dcp_group.ranks)
        push_ranks = tuple(push_group.ranks)
        return bool(
            dcp_group.world_size == 8
            and push_group.world_size == 8
            and len(dcp_ranks) == 8
            and dcp_ranks == push_ranks
            and dcp_group.rank_in_group == push_group.rank_in_group == dcp_rank
        )
    except (AttributeError, TypeError):
        return False


def prepare_dsv4_dcp_full_model_destination_push(
    *,
    dcp_group: Any,
    candidate_groups: list[tuple[str, Any]],
    dcp_rank: int,
    device: torch.device,
) -> tuple[DCPDestinationPushWorkspace | None, Any]:
    """Select one equivalent group with capable AITER destination push."""
    if dcp_group is None or not hasattr(dcp_group, "all_reduce"):
        logger.warning(
            "DSV4 destination-push bootstrap DCP agreement group unavailable"
        )
        return None, None

    selected_name = None
    selected_group = None
    for candidate_name, push_group in candidate_groups:
        equivalent = dsv4_dcp_destination_push_groups_equivalent(
            dcp_group=dcp_group, push_group=push_group, dcp_rank=dcp_rank
        )
        if not equivalent:
            logger.warning(
                "DSV4 destination-push candidate=%s group equivalence mismatch "
                "dcp_ranks=%r candidate_ranks=%r dcp_rank_in_group=%r "
                "candidate_rank_in_group=%r dcp_rank=%r",
                candidate_name,
                getattr(dcp_group, "ranks", None),
                getattr(push_group, "ranks", None),
                getattr(dcp_group, "rank_in_group", None),
                getattr(push_group, "rank_in_group", None),
                dcp_rank,
            )

        ca_comm = getattr(push_group, "ca_comm", None)
        ca_type = type(ca_comm).__name__ if ca_comm is not None else "None"
        ca_disabled = (
            bool(getattr(ca_comm, "disabled", True)) if ca_comm is not None else True
        )
        has_prepare = (
            ca_comm is not None
            and hasattr(ca_comm, "prepare_dsv4_destination_push_workspace")
            and hasattr(push_group, "prepare_dsv4_dcp_destination_push_workspace")
        )
        has_should = (
            ca_comm is not None
            and hasattr(ca_comm, "should_dsv4_destination_push")
            and hasattr(push_group, "should_dsv4_dcp_destination_push")
        )
        has_ready = (
            ca_comm is not None
            and hasattr(ca_comm, "dsv4_destination_push_ready")
            and hasattr(push_group, "dsv4_dcp_destination_push_ready")
        )
        locally_capable = bool(
            equivalent
            and ca_comm is not None
            and not ca_disabled
            and has_prepare
            and has_should
            and has_ready
        )
        if not locally_capable:
            logger.warning(
                "DSV4 destination-push candidate=%s capability rejected "
                "ca_comm type=%s disabled=%s methods prepare=%s should=%s ready=%s",
                candidate_name,
                ca_type,
                ca_disabled,
                has_prepare,
                has_should,
                has_ready,
            )

        try:
            capability_count = torch.tensor(
                [locally_capable], dtype=torch.int32, device=device
            )
            capability_count = dcp_group.all_reduce(capability_count)
            capability_value = int(capability_count.item())
        except Exception as exc:
            logger.warning(
                "DSV4 destination-push candidate=%s capability all-reduce exception=%r",
                candidate_name,
                exc,
            )
            return None, None
        logger.info(
            "DSV4 destination-push candidate=%s capability_count=%d/8 "
            "ca_comm type=%s disabled=%s methods prepare=%s should=%s ready=%s",
            candidate_name,
            capability_value,
            ca_type,
            ca_disabled,
            has_prepare,
            has_should,
            has_ready,
        )
        if selected_group is None and capability_value == 8:
            selected_name = candidate_name
            selected_group = push_group

    if selected_group is None:
        logger.warning("DSV4 destination-push bootstrap no equivalent capable group")
        return None, None

    logger.info("DSV4 destination-push selected candidate=%s", selected_name)
    workspace = prepare_dsv4_dcp_destination_push_workspace(
        communicator=selected_group,
        device=device,
        agreement_group=dcp_group,
        candidate_name=selected_name,
    )
    if workspace is None:
        return None, None
    return workspace, selected_group


def dsv4_dcp_decode_reducer_communicator(
    decode_output: Any,
    *,
    destination_push_communicator: Any,
    dcp_group: Any,
) -> Any:
    """Select the push group only for pushed output; fallback remains DCP."""
    if isinstance(decode_output, DCPDestinationPushOutput):
        if destination_push_communicator is None:
            raise RuntimeError(
                "DSV4 destination-push output has no retained push communicator"
            )
        return destination_push_communicator
    return dcp_group


def select_dsv4_dcp_full_model_output_workspaces(
    *,
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    destination_push_workspace: DCPDestinationPushWorkspace | None,
    direct_output_workspace: DCPA2AOutputWorkspace | None,
    communicator: Any,
    destination_push_enabled: bool,
    direct_output_enabled: bool,
    is_gfx950: bool,
    direct_platform_supported: bool | None = None,
    dcp_size: int,
    comm_backend: str,
    is_plain_decode: bool,
    batch_size: int,
    speculative: bool,
    tbo: bool,
    memory_saver: bool,
    hisparse: bool,
    state_capture: bool,
    piecewise_graph: bool,
    breakable_graph: bool,
) -> tuple[DCPDestinationPushWorkspace | None, DCPA2AOutputWorkspace | None]:
    """Choose strict DCP8 destination push before accepted local direct output."""
    route_kwargs = dict(
        q=q,
        unified_kv=unified_kv,
        dcp_size=dcp_size,
        comm_backend=comm_backend,
        is_plain_decode=is_plain_decode,
        batch_size=batch_size,
        speculative=speculative,
        tbo=tbo,
        memory_saver=memory_saver,
        hisparse=hisparse,
        state_capture=state_capture,
        piecewise_graph=piecewise_graph,
        breakable_graph=breakable_graph,
    )
    if dcp_size == 8 and can_use_dsv4_dcp_destination_push_output(
        workspace=destination_push_workspace,
        communicator=communicator,
        feature_enabled=destination_push_enabled,
        is_gfx950=is_gfx950,
        **route_kwargs,
    ):
        return destination_push_workspace, None
    if can_use_dsv4_dcp_direct_a2a_output(
        workspace=direct_output_workspace,
        feature_enabled=direct_output_enabled,
        is_gfx950=(
            is_gfx950
            if direct_platform_supported is None
            else direct_platform_supported
        ),
        **route_kwargs,
    ):
        return None, direct_output_workspace
    return None, None


def prepare_dsv4_dcp_direct_a2a_output(
    workspace: Optional[DCPA2AOutputWorkspace],
    *,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Return exact packed send views when the narrow DSV4 layout is supported."""
    if workspace is None:
        return None
    world_size = workspace.send_combined.shape[0]
    if not workspace.supports(
        world_size=world_size,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    ):
        return None
    return workspace.send_output_and_lse_views(
        batch_size=batch_size,
        head_dim=head_dim,
    )


def prepare_dsv4_dcp_destination_push_output(
    workspace: DCPDestinationPushWorkspace | None,
    *,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> DCPDestinationPushWorkspace | None:
    """Return the fixed registered workspace when its exact layout matches."""
    if workspace is None:
        return None
    world_size = workspace.recv_planes.shape[1]
    if not workspace.supports(
        world_size=world_size,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    ):
        return None
    return workspace
