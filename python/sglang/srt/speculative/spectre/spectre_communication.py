import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, List

from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spectre.cpp_zmq import (
    DealerEndpoint,
    RouterEndpoint,
    set_spectre_log_level,
)
from sglang.srt.speculative.spectre.spectre_protocol import SpectreRequest

logger = logging.getLogger(__name__)


def get_spectre_log_level() -> int:
    return logger.getEffectiveLevel()


def sync_spectre_log_level() -> None:
    set_spectre_log_level(get_spectre_log_level())


def spectre_info(msg: str) -> None:
    logger.info(msg)


def spectre_warning(msg: str) -> None:
    logger.warning(msg)


def spectre_debug(msg: str) -> None:
    logger.debug(msg)


@dataclass
class SpectreConfig:
    role: str = "target"
    zmq_addr: str = "127.0.0.1"
    zmq_port: str = "30009"
    num_draft_tokens: int = 5
    topk: int = 1
    promote_interval: int = 50
    page_size: int = 1
    tp_size: int = 1
    enable_cuda_graph: bool = False
    zmq_transport: str = "tcp"

    @classmethod
    def from_server_args(cls, server_args: ServerArgs) -> "SpectreConfig":
        zmq_addr = server_args.spectre_zmq_addr or "127.0.0.1"
        if zmq_addr in ["127.0.0.1", "0.0.0.0"]:
            zmq_transport = "ipc"
        else:
            zmq_transport = "tcp"

        return cls(
            role=server_args.spectre_role,
            zmq_addr=zmq_addr,
            zmq_port=server_args.spectre_zmq_port or "30009",
            num_draft_tokens=server_args.speculative_num_steps or 5,
            topk=server_args.speculative_eagle_topk or 1,
            page_size=server_args.page_size,
            tp_size=server_args.tp_size,
            enable_cuda_graph=not server_args.disable_cuda_graph,
            zmq_transport=zmq_transport,
        )

    @property
    def is_target(self) -> bool:
        return self.role == "target"

    @property
    def is_draft(self) -> bool:
        return self.role == "draft"

    @property
    def supports_tree_draft(self) -> bool:
        return self.topk > 1

    @property
    def supports_paged_kv(self) -> bool:
        return self.page_size > 1

    def _get_ipc_base_path(self) -> str:
        if self.zmq_addr.startswith("ipc://"):
            return self.zmq_addr[len("ipc://") :]

        if "/" in self.zmq_addr or self.zmq_addr.startswith("."):
            return self.zmq_addr

        safe_addr = "".join(ch if ch.isalnum() else "_" for ch in self.zmq_addr)
        return f"/tmp/{safe_addr}_{self.zmq_port}"

    def get_addr(self) -> str:
        if self.zmq_transport == "tcp":
            return f"tcp://{self.zmq_addr}:{self.zmq_port}"
        if self.zmq_transport == "ipc":
            return f"ipc://{self._get_ipc_base_path()}"
        raise ValueError(f"Unsupported zmq transport: {self.zmq_transport}")

    def validate(self) -> None:
        if self.role not in ("target", "draft"):
            raise ValueError(f"Invalid role: {self.role}. Must be 'target' or 'draft'")

        if self.zmq_transport not in ("tcp", "ipc"):
            raise ValueError(
                f"Invalid zmq_transport: {self.zmq_transport}. Must be 'tcp' or 'ipc'"
            )

        if self.num_draft_tokens < 1:
            raise ValueError(
                f"num_draft_tokens must be >= 1, got {self.num_draft_tokens}"
            )

        if self.topk < 1:
            raise ValueError(f"topk must be >= 1, got {self.topk}")

        if self.page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {self.page_size}")

        if self.tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {self.tp_size}")

    def __repr__(self) -> str:
        return (
            f"SpectreConfig("
            f"role={self.role}, "
            f"zmq_addr={self.zmq_addr}, "
            f"num_draft_tokens={self.num_draft_tokens}, "
            f"topk={self.topk}, "
            f"page_size={self.page_size}, "
            f"tp_size={self.tp_size})"
        )


class SpectreZMQCommunicator:
    def __init__(
        self,
        config: SpectreConfig,
    ):
        sync_spectre_log_level()
        self.config = config
        self.debug_log_enabled = logger.isEnabledFor(logging.DEBUG)
        self.zmq_endpoint = config.get_addr()
        self.bind = not config.is_target
        self._running = False
        self.identity = self.config.role + "-" + self.generate_identity()

        if self.config.role == "draft":
            self.zmq_communicator = DealerEndpoint(
                self.zmq_endpoint, self.identity, False
            )
        else:
            self.zmq_communicator = RouterEndpoint(self.zmq_endpoint, True)

    def generate_identity(self, bits=8) -> str:
        id_string = str(uuid.uuid4().hex[:bits])
        return id_string

    def get_all_drafts_identity(self) -> List[str]:
        assert self.config.role == "target"
        return self.zmq_communicator.get_all_dealers()

    def get_endpoint(self) -> str:
        return self.zmq_endpoint

    def start(self) -> None:
        sync_spectre_log_level()
        self.debug_log_enabled = logger.isEnabledFor(logging.DEBUG)
        if self._running:
            spectre_debug("ZMQCommunicator already started")
            return

        if not self._running:
            self.zmq_communicator.start()
            self._running = True
            spectre_info(
                f"ZMQ Communicator Started for {self.config.role} with identity {self.identity}"
            )

    def stop(self) -> None:
        if self._running:
            self.zmq_communicator.stop()
            self._running = False
            spectre_info(f"ZMQ Communicator Stopped for {self.config.role}")

    def _process_data(self, data: Any):
        if isinstance(data, SpectreRequest):
            return data.to_dict()
        return data

    def send_obj(self, request: SpectreRequest, identity: str = "DRAFT") -> None:
        self.send_objs([request], identity)

    def send_objs(
        self, requests: List[SpectreRequest], identity: str = "DRAFT"
    ) -> None:

        if not self._running:
            spectre_warning("Cannot send: communicator not running")
            return
        try:
            if self.debug_log_enabled:
                t1 = time.perf_counter()

            msgs = [self._process_data(request) for request in requests]

            if self.debug_log_enabled:
                t2 = time.perf_counter()
                t_process = t2 - t1

            if self.config.role == "draft":
                self.zmq_communicator.send_objs(msgs)
            else:
                self.zmq_communicator.send_objs(identity, msgs)

            if self.debug_log_enabled:
                t3 = time.perf_counter()
                spectre_debug(
                    f"[ZMQ LOG Pyt][SEND] msgs nums:{len(msgs)}, time_us:{(t3-t2)*1e6:.1f}-process time {t_process*1e6:.1f} us"
                )

        except Exception as e:
            spectre_warning(f"Failed to send: {e}")

    def recv_all_objs(self) -> List[SpectreRequest]:
        if not self._running:
            return []
        try:
            if self.debug_log_enabled:
                t1 = time.perf_counter()

            received = self.zmq_communicator.get_received_objs()

            if self.config.role == "target":
                _msgs = [msg for _, msg in received]
            else:
                _msgs = received

            if self.debug_log_enabled and _msgs:
                t2 = time.perf_counter()
                spectre_debug(
                    f"[ZMQ LOG Pyt][RECV] msgs nums:{len(_msgs)}, time_us:{(t2-t1)*1e6:.1f}"
                )

            if not _msgs:
                return []
            else:
                msgs = []
                for _msg in _msgs:
                    if isinstance(_msg, dict):
                        msgs.append(SpectreRequest.from_dict(_msg))
                    else:
                        msgs.append(_msg)
                return msgs

        except Exception as e:
            spectre_warning(f"Failed to receive: {e}")
            return []

    def is_running(self) -> bool:
        return self._running
