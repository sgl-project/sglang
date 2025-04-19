from __future__ import annotations

import logging
import pickle
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import zmq

from sglang.srt.disaggregation.base.conn import (
    BaseKVBootstrapServer,
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
)
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.disaggregation.pyverbs.transfer_engine import RdmaEndpoint
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    find_best_rdma_ib_device,
    groups_kv_indices_continuity_by_numpy,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    get_free_port,
    get_ip,
    get_local_ip_by_remote,
    get_open_port,
)

logger = logging.getLogger(__name__)


class PyverbsKVManager(BaseKVManager):
    """Manager class for handling KV (Key-Value) operations using Pyverbs RDMA.

    This class manages the memory regions and addresses for KV data transfer between
    different components of the system using RDMA (Remote Direct Memory Access).
    """

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
    ):
        # Initialize manager with configuration arguments
        self.args = args
        self.engine_rank = args.engine_rank
        self.kv_data_ptrs = args.kv_data_ptrs  # Pointers to KV data memory regions
        self.kv_data_lens = args.kv_data_lens  # Lengths of KV data memory regions
        self.kv_item_lens = args.kv_item_lens  # Length of each KV item
        self.aux_data_ptrs = (
            args.aux_data_ptrs
        )  # Pointers to auxiliary data memory regions
        self.aux_data_lens = (
            args.aux_data_lens
        )  # Lengths of auxiliary data memory regions
        self.aux_item_lens = args.aux_item_lens  # Length of each auxiliary data item

        self.active_sessions = {}
        # Find the best RDMA IB device for the specified GPU
        self.args.ib_device, net_card = find_best_rdma_ib_device(self.args.gpu_id)
        if self.args.ib_device:
            logger.info(
                "Current Process Using the  gpu id: {}, ib_device: {} net:{}".format(
                    self.args.gpu_id, self.args.ib_device, net_card
                )
            )
        else:
            raise Exception("No ROCE IB device found...")

    def calculate_token_kv_address(self, layer_id: int, token_index: int):
        """Calculate the memory address for a specific token's KV data in a given layer.

        Args:
            layer_id: The layer index in the model
            token_index: The index of the token

        Returns:
            tuple: (token_kv_address, offset) - The calculated memory address and its offset
        """
        # Get base address for the layer's KV data
        base_address = self.args.kv_data_ptrs[layer_id]
        # Get size of KV data for each token
        token_kv_size = self.args.kv_item_lens[layer_id]
        # Calculate offset from base address
        offset = token_kv_size * token_index
        # Final address = base address + offset
        token_kv_address = base_address + offset
        return token_kv_address, offset

    def calculate_all_token_kv_addresses(self, token_indices: list[int]):
        """Calculate memory addresses for all tokens' KV data across all layers.

        Args:
            token_indices: List of token indices to calculate addresses for

        Returns:
            tuple: (addresses_by_layer, offsets_by_layer, addresses_base_and_len)
                - addresses_by_layer: List of token addresses for each layer
                - offsets_by_layer: List of offsets for each layer
                - addresses_base_and_len: List of (base_address, total_length) for each layer
        """
        # Initialize result storage
        addresses_by_layer = []
        offsets_by_layer = []
        addresses_base_and_len = []
        # Calculate for each layer
        for layer_id in range(len(self.args.kv_data_ptrs)):
            token_addresses = []
            token_offsets = []

            # Calculate address and offset for each token
            for token_index in token_indices:
                address, offset = self.calculate_token_kv_address(layer_id, token_index)
                token_addresses.append(address)
                token_offsets.append(offset)

            addresses_by_layer.append(token_addresses)
            offsets_by_layer.append(token_offsets)
            addresses_base_and_len.append(
                (
                    token_addresses[0],
                    self.args.kv_item_lens[layer_id] * len(token_indices),
                )
            )
        return addresses_by_layer, offsets_by_layer, addresses_base_and_len

    def caculate_layer_kv_addresses(self, token_indices: list[int]):
        """Calculate base addresses and lengths for KV data in each layer.

        Args:
            token_indices: List of token indices to calculate addresses for

        Returns:
            list: List of (base_address, total_length) tuples for each layer
        """
        addresses_base_and_len = []
        for layer_id in range(len(self.args.kv_data_ptrs)):
            # Get size of KV data for each token
            token_kv_size = self.args.kv_item_lens[layer_id]
            # Calculate offset from base address
            offset = token_kv_size * (token_indices[0])
            token_kv_layer_base_address = self.args.kv_data_ptrs[layer_id] + offset
            addresses_base_and_len.append(
                (token_kv_layer_base_address, token_kv_size * (len(token_indices)))
            )
        return addresses_base_and_len

    def caculate_layer_kv_base_and_offsets(self, token_indices: list[int]):
        """Calculate base addresses, layer lengths, and offsets for KV data.

        Args:
            token_indices: List of token indices to calculate addresses for

        Returns:
            tuple: (addresses_bases, layer_lens, offsets)
                - addresses_bases: List of base addresses for each layer
                - layer_lens: List of total lengths for each layer
                - offsets: List of offsets for each token
        """
        addresses_bases = []
        offsets = []
        layer_lens = []
        for layer_id in range(len(self.args.kv_data_ptrs)):
            token_kv_size = self.args.kv_item_lens[layer_id]
            token_kv_layer_base_address = self.args.kv_data_ptrs[layer_id]
            # Record base address for each layer, used for initial metadata connection
            addresses_bases.append((token_kv_layer_base_address))
            layer_lens.append(self.args.kv_data_lens[layer_id])
        for token_indice in token_indices:
            # Record offset for each token, used for subsequent address calculation
            offsets.append(token_kv_size * token_indice)
        return addresses_bases, layer_lens, offsets


class PyverbsKVSender(BaseKVSender):

    def __init__(self, mgr: PyverbsKVManager, bootstrap_addr: str, bootstrap_room: int):
        self.bs_client = None
        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.session_id = str(uuid.uuid4())
        self.initialized = False
        self.state = KVPoll.Bootstrapping
        self.transfer_complete = False
        self.metadata_sent = False
        self.data_to_send = None

        logger.info(f"Sender registered with room {self.bootstrap_room}")
        # Initialize transfer metadata
        self.num_tokens = 0
        self.aux_idx = -1

        # target ip
        self.target_ip = None
        # endpoint
        self.ep = None
        # 传输状态
        self.current_indices = None
        self.current_layer = 0
        self.rdma_ep = RdmaEndpoint(max_send_wr=1500, ib_device=self.mgr.args.ib_device)

        self.mrs_to_send = []  # 数据段待发送的内存区域
        self.meta_has_sent = False  # meta 还没有发送
        self.transfer_meta = {}
        self.metadata_mr = None
        self.register(
            "",
            self.rdma_ep.qp.qp_num,
            self.rdma_ep.port_attr.lid,
            str(self.rdma_ep.gid),
            "",
            0,
            0,
            [],
        )

    def register(
        self,
        local_ip,
        qp_num,
        lid,
        gid,
        meta_buff_addr,
        meta_buff_len,
        meta_buff_rkey,
        groups_mrs_info,
    ):
        zmp_start = time.time()
        bootstrap_server = f"tcp://{self.bootstrap_addr}"
        self.bs_client = PyverbsKVBootstrapClient(bootstrap_server)
        resp = self.bs_client.register(
            self.bootstrap_room,
            "prefill",
            self.mgr.engine_rank,
            self.mgr.args.engine_rank,
            local_ip,
            qp_num,
            lid,
            gid,
            meta_buff_addr,
            meta_buff_len,
            meta_buff_rkey,
            groups_mrs_info,
        )
        zmp_end = time.time()

        logger.debug(f"ZMQ Request time: {zmp_end - zmp_start}")
        if resp["status"] != "ok":
            self.state = KVPoll.Failed

    def query_room(self):
        resp = self.bs_client.query_room(self.bootstrap_room)
        if resp["status"] == "ok" and "decode" in resp["clients"]:
            clients = resp["clients"]["decode"]
            if self.mgr.engine_rank in clients:
                return clients.get(self.mgr.engine_rank, None)
        return None

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        """Initialize sender with metadata only

        Args:
            num_tokens: Number of tokens to transfer
            aux_idx: Index for auxiliary data

        Returns:
            bool: True if metadata sent successfully
        """
        self.num_tokens = num_kv_indices
        self.aux_idx = aux_index
        self.metadata_ptr = self.mgr.aux_data_ptrs[0] + (
            aux_index * self.mgr.aux_item_lens[0]
        )
        self.metadata_ptr_length = self.mgr.aux_item_lens[0]
        self.metadata_mr = self.rdma_ep.create_mr(
            self.metadata_ptr, self.metadata_ptr_length
        )

        try:
            self.rdma_ep.set_to_transfer(
                self.transfer_meta.get("qp_num"),
                self.transfer_meta.get("lid"),
                self.transfer_meta.get("gid"),
            )
            logger.debug("Transferring...")
            self.state = KVPoll.Transferring
        except Exception as e:
            logger.error(e)
            self.state = KVPoll.Bootstrapping

    def poll(self) -> KVPoll:
        """Poll transfer status"""
        if self.state == KVPoll.Bootstrapping:
            data = self.query_room()
            if not data:
                self.state = KVPoll.Bootstrapping
            else:
                logger.debug(data)
                self.transfer_meta = data
                # self.target_port = data.get(str(self.mgr.engine_rank), {}).get('port', None)
                if not self.transfer_meta.get("gid"):
                    self.state = KVPoll.Bootstrapping
                else:
                    self.state = KVPoll.WaitingForInput
        if self.state == KVPoll.Failed:
            return KVPoll.Failed

        if self.state == KVPoll.Transferring:
            self.rdma_ep.check_send_complete()

            """
            completed wrs + metadata wrs
            """
            if (
                self.rdma_ep.completed_wrs == len(self.rdma_ep.wrs_to_send) + 1
                and self.meta_has_sent
            ):
                # write remote metadata //todo
                self.state = KVPoll.Success
            elif (
                self.rdma_ep.completed_wrs == len(self.rdma_ep.wrs_to_send)
                and not self.meta_has_sent
            ):
                self.rdma_ep.send_metadata_wrs(
                    self.metadata_ptr,
                    self.metadata_ptr_length,
                    self.metadata_mr,
                    self.transfer_meta,
                )
                self.meta_has_sent = True

        return self.state

    def send(
        self,
        kv_indices: npt.NDArray[np.int64],
        index_slice: slice,
        is_last: bool,
    ):
        """Send actual data synchronously"""
        # 收集要传输的数据
        groups_mrs_info = []
        continuousindices = groups_kv_indices_continuity_by_numpy(kv_indices)
        for group_id, continue_kv_indices in enumerate(continuousindices):
            mrs_info = []
            logger.debug("Sending continuity indices {}".format(continue_kv_indices))
            address_lengths = self.mgr.caculate_layer_kv_addresses(continue_kv_indices)
            for layer_id, (address, length) in enumerate(address_lengths):
                mr = self.rdma_ep.create_mr(address, length)
                self.mrs_to_send.append(mr)
                mrs_info.append(
                    {
                        "address": address,
                        "length": length,
                        "rkey": mr.rkey,
                        "lkey": mr.lkey,
                    }
                )
            groups_mrs_info.append(mrs_info)
        self.rdma_ep.send_wrs(groups_mrs_info, self.transfer_meta["groups_mrs_info"])

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class PyverbsKVReceiver(BaseKVReceiver):
    """Receiver class for KV data transfer using Pyverbs RDMA.

    This class handles the receiving of KV data from remote senders using RDMA.
    It manages the connection setup, metadata reception, and actual data reception.
    """

    def __init__(
        self,
        mgr: PyverbsKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.mgr = mgr
        self.kv_layers_mrs = []  # Memory regions for KV layers

        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.session_id = str(uuid.uuid4())
        self.initialized = False
        self.ep = None
        self.state = KVPoll.Bootstrapping
        self.transfer_complete = False
        self.num_tokens = 0
        self.aux_idx = -1
        self.kv_indices = None
        # Initialize RDMA endpoint
        self.rdma_ep = RdmaEndpoint(ib_device=self.mgr.args.ib_device)
        self.rdma_port = get_open_port()

        self.start_time = time.time()
        # Get local IP address
        self.ip = get_local_ip_by_remote()

        self.transfer_meta = None
        self.mrs_to_receive = []  # Memory regions to receive data into

    def register(
        self,
        local_ip,
        qp_num,
        lid,
        gid,
        meta_buff_addr,
        meta_buff_len,
        meta_buff_rkey,
        groups_mrs_info,
    ):
        """Register the receiver with the bootstrap server.

        Args:
            local_ip: Local IP address
            qp_num: Queue pair number
            lid: Local ID
            gid: Global ID
            meta_buff_addr: Metadata buffer address
            meta_buff_len: Metadata buffer length
            meta_buff_rkey: Metadata buffer remote key
            groups_mrs_info: Information about memory regions to receive
        """
        zmp_start = time.time()
        bootstrap_server = f"tcp://{self.bootstrap_addr}"
        bs_client = PyverbsKVBootstrapClient(bootstrap_server)
        resp = bs_client.register(
            self.bootstrap_room,
            "decode",
            self.mgr.engine_rank,
            self.mgr.args.engine_rank,
            local_ip,
            qp_num,
            lid,
            gid,
            meta_buff_addr,
            meta_buff_len,
            meta_buff_rkey,
            groups_mrs_info,
        )
        zmp_end = time.time()

        logger.debug(f"ZMQ Request time: {zmp_end - zmp_start}")
        if resp["status"] != "ok":
            self.state = KVPoll.Failed

        logger.debug("boostraped success.. qp_num={}, lid={}".format(qp_num, lid))

    def query_room(self):
        """Query the bootstrap server for room information.

        Returns:
            dict: Information about the room if found, None otherwise
        """
        zmq_client = PyverbsKVBootstrapClient(f"tcp://{self.bootstrap_addr}")
        resp = zmq_client.query_room(self.bootstrap_room)
        if resp["status"] == "ok" and "prefill" in resp["clients"]:
            clients = resp["clients"]["prefill"]
            if self.mgr.engine_rank in clients:
                return clients.get(self.mgr.engine_rank, None)
        return None

    def init(self, kv_indices: np.ndarray[np.int32], aux_index: Optional[int] = None):
        """Initialize receiver with KV indices and auxiliary data index.

        Args:
            kv_indices: Array of KV indices to receive
            aux_index: Optional index for auxiliary data

        Returns:
            bool: True if initialization successful
        """
        # Create memory region for metadata
        metadata_ptr = self.mgr.aux_data_ptrs[0] + (
            aux_index * self.mgr.aux_item_lens[0]
        )
        metadata_length = self.mgr.aux_item_lens[0]
        self.meta_mr = self.rdma_ep.create_mr(metadata_ptr, metadata_length)

        # Create MR for each layer and get corresponding key to pass to client
        rkeys = []

        for layer_id, base_addr in enumerate(self.mgr.kv_data_ptrs):
            layer_mr = self.rdma_ep.create_mr(
                base_addr, self.mgr.kv_data_lens[layer_id]
            )
            rkeys.append(layer_mr.rkey)
            self.kv_layers_mrs.append(layer_mr)

        # Group continuous indices to optimize memory region creation
        groups_mrs_info = []
        continous_indices = groups_kv_indices_continuity_by_numpy(kv_indices)
        for group_id, continue_kv_indices in enumerate(continous_indices):
            mrs_info = []
            address_lengths = self.mgr.caculate_layer_kv_addresses(continue_kv_indices)
            for layer_id, (address, length) in enumerate(address_lengths):
                mrs_info.append(
                    {"address": address, "length": length, "rkey": rkeys[layer_id]}
                )
            groups_mrs_info.append(mrs_info)

        try:
            # Register with bootstrap server
            self.register(
                self.ip,
                self.rdma_ep.qp.qp_num,
                self.rdma_ep.port_attr.lid,
                str(self.rdma_ep.gid),
                metadata_ptr,
                metadata_length,
                self.meta_mr.rkey,
                groups_mrs_info,
            )

            # Set up metadata reception
            self.rdma_ep.recv_metadata_mr(
                metadata_ptr, metadata_length, self.meta_mr.lkey
            )
            self.state = KVPoll.Transferring

        except Exception as e:
            self.state = KVPoll.Bootstrapping
            return

    def poll(self) -> KVPoll:
        """Poll receive status and handle state transitions.

        Returns:
            KVPoll: Current state of the transfer
        """
        if self.state == KVPoll.Bootstrapping:
            data = self.query_room()
            if not data:
                self.state = KVPoll.Bootstrapping
            else:
                logger.debug(data)
                self.transfer_meta = data
                if not self.transfer_meta.get("gid"):
                    self.state = KVPoll.Bootstrapping
                else:
                    # Set up RDMA transfer parameters
                    self.rdma_ep.set_to_transfer(
                        self.transfer_meta["qp_num"],
                        self.transfer_meta["lid"],
                        self.transfer_meta["gid"],
                    )
                    self.state = KVPoll.WaitingForInput
        if self.state == KVPoll.Transferring:
            # Check metadata reception status
            self.rdma_ep.check_meta_recv_complete()
            if self.rdma_ep.metadata_mr_complete_num == 1:
                logger.debug("Decode Transferring complete...")
                return KVPoll.Success

        if self.state == KVPoll.Failed:
            return KVPoll.Failed

        return self.state

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, "loop") and self.loop:
            self.loop.close()

    def failure_exception(self):
        """Raise an exception to indicate transfer failure."""
        raise Exception("Fake KVReceiver Exception")


class PyverbsKVBootstrapServer(BaseKVBootstrapServer):
    """Bootstrap server for managing RDMA connections between senders and receivers.

    This server handles registration and discovery of RDMA endpoints,
    allowing senders and receivers to establish direct connections.
    """

    def __init__(self, port: int):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{port}")
        self.registry = {}  # Store registered clients
        self.lock = threading.Lock()  # Thread safety for registry access

        self.run_in_thread()

    def handle_client(self, identity, message):
        """Handle client messages and manage registration.

        Args:
            identity: Client identity
            message: Message from client

        Returns:
            list: Response to client [identity, empty_frame, response_data]
        """
        try:
            request = pickle.loads(message)
            msg_type = request.get("type")

            if msg_type == "register":
                room_id = request["room_id"]
                tp_rank = request["tp_rank"]
                role = request["pd_role"]
                with self.lock:
                    if room_id not in self.registry:
                        self.registry[room_id] = {}
                    if role not in self.registry[room_id]:
                        self.registry[room_id][role] = {}
                    # Store client registration information
                    self.registry[room_id][role][tp_rank] = {
                        "bootstrap_port": request.get("bootstrap_port"),
                        "session_id": request.get("session_id"),
                        "pd_role": role,
                        "tp_rank": tp_rank,
                        "tp_size": request["tp_size"],
                        "ip": request["ip"],
                        "qp_num": request["qp_num"],
                        "lid": request["lid"],
                        "gid": request["gid"],
                        "meta_buff": request["meta_buff"],
                        "groups_mrs_info": request["groups_mrs_info"],
                    }
                    logger.debug(f"Registering new: {self.registry[room_id][role]}")

                response = {"status": "ok"}

            elif msg_type == "query_room":
                room_id = request["room_id"]
                with self.lock:
                    clients = self.registry.get(room_id, {})
                response = {"status": "ok", "clients": clients}

            else:
                response = {"status": "error", "message": "unknown message type"}

        except Exception as e:
            response = {"status": "error", "message": str(e)}

        # Return three-part message: identity, empty frame, actual data
        return [identity, b"", pickle.dumps(response)]

    def run(self):
        """Main server loop to handle client connections."""
        logger.info(f"[Registry Server] Started on tcp://*:{self.port}")
        while True:
            # Receive three-part message: identity, empty frame, actual data
            identity, empty, message = self.socket.recv_multipart()
            response = self.handle_client(identity, message)
            self.socket.send_multipart(response)

    def run_in_thread(self):
        """Start server in a separate thread."""
        server_thread = threading.Thread(target=self.run, daemon=True)
        server_thread.start()
        return server_thread


class PyverbsKVBootstrapClient:
    """Client for interacting with the bootstrap server.

    This client handles registration and querying of RDMA endpoints
    through the bootstrap server.
    """

    def __init__(self, server_address="tcp://localhost:8898"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        # Generate unique client identity
        self.identity = str(uuid.uuid4()).encode()
        self.socket.setsockopt(zmq.IDENTITY, self.identity)
        self.socket.connect(server_address)
        self.server_address = server_address

    def register(
        self,
        room_id,
        pd_role,
        tp_rank,
        tp_size,
        ip,
        qp_num,
        lid,
        gid,
        meta_buff_addr,
        meta_buff_len,
        meta_buff_rkey,
        groups_mrs_info,
    ):
        """Register client with the bootstrap server.

        Args:
            room_id: Room identifier
            pd_role: Role of the client (prefill/decode)
            tp_rank: Tensor parallel rank
            tp_size: Tensor parallel size
            ip: IP address
            qp_num: Queue pair number
            lid: Local ID
            gid: Global ID
            meta_buff_addr: Metadata buffer address
            meta_buff_len: Metadata buffer length
            meta_buff_rkey: Metadata buffer remote key
            groups_mrs_info: Information about memory regions

        Returns:
            dict: Server response
        """
        registration_info = {
            "type": "register",
            "pd_role": pd_role,
            "room_id": room_id,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "ip": ip,
            "qp_num": qp_num,
            "lid": lid,
            "gid": gid,
            "meta_buff": {
                "meta_buff_addr": meta_buff_addr,
                "meta_buff_length": meta_buff_len,
                "meta_buff_rkey": meta_buff_rkey,
            },
            "groups_mrs_info": groups_mrs_info,
            "bootstrap_port": None,
            "session_id": str(uuid.uuid4()),
        }

        # Send empty frame and actual data
        self.socket.send_multipart([b"", pickle.dumps(registration_info)])

        # Receive empty frame and response data
        empty, response = self.socket.recv_multipart()
        return pickle.loads(response)

    def query_room(self, room_id):
        """Query information about a room from the bootstrap server.

        Args:
            room_id: Room identifier

        Returns:
            dict: Room information
        """
        query_info = {"type": "query_room", "room_id": room_id}

        # Send empty frame and actual data
        self.socket.send_multipart([b"", pickle.dumps(query_info)])

        # Receive empty frame and response data
        empty, response = self.socket.recv_multipart()
        return pickle.loads(response)
