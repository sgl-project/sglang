from __future__ import annotations

import time

import zmq
import logging

import pickle
import threading

from pyverbs.device import Context
from pyverbs.pd import PD
from pyverbs.cq import CQ, WC
from pyverbs.qp import QPInitAttr, QPCap, QPAttr, QP
from pyverbs.mr import MR
from pyverbs.addr import GID, AHAttr
from pyverbs.wr import RecvWR, SGE, SendWR
from pyverbs.enums import *
from sglang.srt.disaggregation.ib_devices import find_best_roce_for_gpu
from sglang.srt.disaggregation.conn import KVPoll
import uuid
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

from sglang.srt.utils import get_open_port, get_local_ip_by_remote
from sglang.srt.disaggregation.group_indics import groups_by_continuity_numpy
from sglang.srt.disaggregation.utils import DisaggregationMode

# Registry
registry = {}


class RdmaEndpoint(object):
    def __init__(self, ib_device='mlx5_bond_0', max_send_wr=10, max_recv_wr=10, max_send_sge=30, max_recv_sge=30,
                 rcq_num=1000, scq_num=1600, debug=True):
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.ib_device = ib_device
        self.ctx = Context(name=self.ib_device)
        self.pd = PD(self.ctx)
        self.recv_cq = CQ(self.ctx, rcq_num)
        self.send_cq = CQ(self.ctx, scq_num)
        # Create QP
        cap = QPCap(max_send_wr=max_send_wr, max_recv_wr=max_recv_wr, max_send_sge=max_send_sge,
                    max_recv_sge=max_recv_sge)
        init_attr = QPInitAttr(qp_type=2, scq=self.send_cq, rcq=self.recv_cq, cap=cap)
        self.qp = QP(self.pd, init_attr)

        self.port_attr, self.gid = self.get_gid()

        self.initial_wr_index = 0
        self.completed_wrs = 0
        self.metadata_mr_complete_num = 0
        self.wrs_to_send = []

    def get_gid(self):
        port_attr = self.ctx.query_port(1)
        gid = self.ctx.query_gid(1, 3)
        return port_attr, gid

    def set_to_transfer(self, qp_num, lid, gid):
        attr = QPAttr()
        attr.qp_state = 2
        attr.pkey_index = 0
        attr.port_num = 1
        attr.qp_access_flags = 0b111

        # init attr
        self.qp.to_init(attr)
        #
        attr.qp_state = 3
        attr.path_mtu = self.port_attr.active_mtu
        attr.dest_qp_num = qp_num
        attr.rq_psn = 0
        attr.max_dest_rd_atomic = 1
        attr.min_rnr_timer = 1
        attr.ah_attr.port_num = 1
        if self.port_attr.lid != 0:
            attr.ah_attr.dlid = lid
            attr.ah_attr.is_global = 0
        else:
            ah_attr = AHAttr()
            ah_attr.dlid = 0
            ah_attr.is_global = 1
            ah_attr.dgid = gid
            ah_attr.sgid_index = 3
            ah_attr.hop_limit = 1
            attr.ah_attr = ah_attr

        self.qp.to_rtr(attr)

        # RTS
        attr.qp_state = 4
        attr.sq_psn = 0
        attr.timeout = 14
        attr.retry_cnt = 7
        attr.rnr_retry = 7
        attr.max_rd_atomic = 1
        self.qp.to_rts(attr)

    def send_wrs(self, groups_mrs_info, remote_groups_mrs_info):
        self.add_to_sending_wrs(groups_mrs_info, remote_groups_mrs_info)
        for wr in self.wrs_to_send:
            self.qp.post_send(wr)
        logger.debug("Sending Request posted ...")

    def send_metadata_wrs(self, meta_ptr, meta_len, meta_mr, transfer_meta):
        remote_metadata_addr = transfer_meta['meta_buff']["meta_buff_addr"]
        remote_metadata_len = transfer_meta['meta_buff']["meta_buff_length"]
        remote_metadata_rkey = transfer_meta['meta_buff']["meta_buff_rkey"]
        logger.debug(f"Sending metadata:{remote_metadata_addr}:{remote_metadata_len}:{remote_metadata_rkey}")
        local_sge = SGE(addr=meta_ptr,
                        length=meta_len,
                        lkey=meta_mr.lkey)  # 发送要本地轮询cq，服务端轮询即可
        wr = SendWR(wr_id=self.initial_wr_index + 1, sg=[local_sge], num_sge=1, opcode=IBV_WR_RDMA_WRITE_WITH_IMM,
                    send_flags=IBV_SEND_FENCE)
        wr.set_wr_rdma(addr=remote_metadata_addr,
                       rkey=remote_metadata_rkey)
        self.qp.post_send(wr)
        logger.debug(f"Sending metadata sent ...")

    def add_to_sending_wrs(self, group_mrs_info, group_remote_mrs_info):
        # By default, decode allocated memory is contiguous
        if len(group_remote_mrs_info) != 1:
            raise Exception("decode allocate non-contiguous wrs")

        for group_id, mrs_info in enumerate(group_mrs_info):
            for layer_id, item in enumerate(mrs_info):
                self.initial_wr_index += 1
                local_sge = SGE(addr=item["address"],
                                length=item["length"],
                                lkey=item['lkey'])
                wr = SendWR(wr_id=self.initial_wr_index, sg=[local_sge], num_sge=1, opcode=IBV_WR_RDMA_WRITE,
                            send_flags=IBV_SEND_SIGNALED)
                wr.set_wr_rdma(addr=group_remote_mrs_info[0][layer_id]['address'],
                               rkey=group_remote_mrs_info[0][layer_id]['rkey'])
                self.wrs_to_send.append(wr)
                group_remote_mrs_info[0][layer_id]["address"] += item["length"]

    def recv_metadata_mr(self, meta_ptr, meta_len, meta_lkey):
        recv_sge = SGE(addr=meta_ptr,
                       length=meta_len,
                       lkey=meta_lkey)
        # Sender doesn't need to poll local cq, server will poll
        recv_wr = RecvWR(wr_id=self.initial_wr_index + 1, sg=[recv_sge], num_sge=1)
        self.qp.post_recv(recv_wr)
        logger.debug("Waiting metadata ...")

    def check_send_complete(self):
        npolled, wc_list = self.send_cq.poll()
        if npolled > 0:
            for wc in wc_list:
                if wc.status != IBV_WC_SUCCESS:
                    logger.debug(f"Send failed: {wc.status}")
                    logger.error(wc)
                else:
                    self.completed_wrs += 1

    def check_meta_recv_complete(self):
        npolled, wc_list = self.recv_cq.poll()
        if npolled > 0:
            for wc in wc_list:
                if wc.status != IBV_WC_SUCCESS:
                    logger.debug(f"Recv failed: {wc.status}")
                else:
                    self.metadata_mr_complete_num += 1
                    logger.debug(f"Metadata Received completed! Bytes: {wc.byte_len}, wr_id: {wc.wr_id}")

    def create_mr(self, address, length, access=0b111):
        """
        Pre-allocate GPU memory region and pass MR at once
        """
        return MR(self.pd, address=address,
                  length=length, access=access)


class KVBootstrapServer:
    def __init__(self, port: int):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{port}")
        self.registry = {}
        self.lock = threading.Lock()

        self.run_in_thread()

    def handle_client(self, identity, message):
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
                        "groups_mrs_info": request["groups_mrs_info"]
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

        # 返回三部分消息：身份标识、空帧、实际数据
        return [identity, b"", pickle.dumps(response)]

    def run(self):
        logger.info(f"[Registry Server] Started on tcp://*:{self.port}")
        while True:
            # 接收三部分消息：身份标识、空帧、实际数据
            identity, empty, message = self.socket.recv_multipart()
            response = self.handle_client(identity, message)
            self.socket.send_multipart(response)

    def run_in_thread(self):
        server_thread = threading.Thread(target=self.run, daemon=True)
        server_thread.start()
        return server_thread


class KVBootstrapClient:
    def __init__(self, server_address="tcp://localhost:8898"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        # 生成唯一的客户端标识
        self.identity = str(uuid.uuid4()).encode()
        self.socket.setsockopt(zmq.IDENTITY, self.identity)
        self.socket.connect(server_address)
        self.server_address = server_address

    def register(self, room_id, pd_role, tp_rank, tp_size, ip, qp_num, lid, gid, meta_buff_addr, meta_buff_len,
                 meta_buff_rkey, groups_mrs_info):
        """
        向服务端发送注册信息
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
                "meta_buff_rkey": meta_buff_rkey
            },
            "groups_mrs_info": groups_mrs_info,
            "bootstrap_port": None,
            "session_id": str(uuid.uuid4())
        }

        # 发送空帧和实际数据
        self.socket.send_multipart([b"", pickle.dumps(registration_info)])

        # 接收空帧和响应数据
        empty, response = self.socket.recv_multipart()
        return pickle.loads(response)

    def query_room(self, room_id):
        """
        查询房间内客户端信息
        """
        query_info = {
            "type": "query_room",
            "room_id": room_id
        }

        # 发送空帧和实际数据
        self.socket.send_multipart([b"", pickle.dumps(query_info)])

        # 接收空帧和响应数据
        empty, response = self.socket.recv_multipart()
        return pickle.loads(response)


class KVArgs:
    """Arguments for KV cache management"""
    engine_rank: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str = "all"
    gpu_id: int


class KVManager:
    def __init__(self, args: KVArgs, disaggregation_mode: DisaggregationMode):

        self.args = args
        self.engine_rank = args.engine_rank
        self.kv_data_ptrs = args.kv_data_ptrs
        self.kv_data_lens = args.kv_data_lens
        self.kv_item_lens = args.kv_item_lens
        self.aux_data_ptrs = args.aux_data_ptrs
        self.aux_data_lens = args.aux_data_lens
        self.aux_item_lens = args.aux_item_lens

        self.active_sessions = {}
        self.args.ib_device, net_card = find_best_roce_for_gpu(self.args.gpu_id)
        if self.args.ib_device:
            logger.info(
                "Current Process Using the  gpu id: {}, ib_device: {} net:{}".format(self.args.gpu_id,
                                                                                     self.args.ib_device,
                                                                                     net_card))
        else:
            raise Exception("No ROCE IB device found...")

    def calculate_token_kv_address(self, layer_id: int, token_index: int):
        # 获取基础地址 - 每层的KV数据指针
        base_address = self.args.kv_data_ptrs[layer_id]
        # 每个token的KV数据大小
        token_kv_size = self.args.kv_item_lens[layer_id]
        # 计算偏移量
        offset = token_kv_size * token_index
        # 最终地址 = 基址 + 偏移量
        token_kv_address = base_address + offset
        return token_kv_address, offset

    def calculate_all_token_kv_addresses(self, token_indices: list[int]):
        # 结果存储
        addresses_by_layer = []
        offsets_by_layer = []
        addresses_base_and_len = []
        # 对每一层计算
        for layer_id in range(len(self.args.kv_data_ptrs)):
            token_addresses = []
            token_offsets = []

            # 计算每个token的地址和偏移量
            for token_index in token_indices:
                address, offset = self.calculate_token_kv_address(layer_id, token_index)
                token_addresses.append(address)
                token_offsets.append(offset)

            addresses_by_layer.append(token_addresses)
            offsets_by_layer.append(token_offsets)
            addresses_base_and_len.append((token_addresses[0], self.args.kv_item_lens[layer_id] * len(token_indices)))
        return addresses_by_layer, offsets_by_layer, addresses_base_and_len

    def caculate_layer_kv_addresses(self, token_indices: list[int]):
        addresses_base_and_len = []
        for layer_id in range(len(self.args.kv_data_ptrs)):
            # 每个token的KV数据大小
            token_kv_size = self.args.kv_item_lens[layer_id]
            # 计算偏移量
            offset = token_kv_size * (token_indices[0])
            token_kv_layer_base_address = self.args.kv_data_ptrs[layer_id] + offset
            addresses_base_and_len.append((token_kv_layer_base_address,
                                           token_kv_size * (len(token_indices))))
        return addresses_base_and_len

    def caculate_layer_kv_base_and_offsets(self, token_indices: list[int]):
        addresses_bases = []
        offsets = []
        layer_lens = []
        for layer_id in range(len(self.args.kv_data_ptrs)):
            token_kv_size = self.args.kv_item_lens[layer_id]
            token_kv_layer_base_address = self.args.kv_data_ptrs[layer_id]
            # 记录每层首地址 ，用于首次metadata建联
            addresses_bases.append((token_kv_layer_base_address))
            layer_lens.append(self.args.kv_data_lens[layer_id])
        for token_indice in token_indices:
            # 记录一层的 偏移，用于后续计算地址
            offsets.append(token_kv_size * token_indice)

            # addresses_base_and_len.append((token_kv_layer_base_address,
            #                               token_kv_size * len(token_indices)))
        return addresses_bases, layer_lens, offsets


class KVSender:
    def __init__(self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: int):
        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.session_id = str(uuid.uuid4())
        self.initialized = False
        self.state = KVPoll.Bootstrapping
        self.transfer_complete = False
        self.metadata_sent = False
        self.data_to_send = None

        # self.ucx_server.register_sender(self.session_id, self.bootstrap_room)
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
        self.register("", self.rdma_ep.qp.qp_num, self.rdma_ep.port_attr.lid, str(self.rdma_ep.gid), "", 0, 0, [])

    def register(self, local_ip, qp_num, lid, gid, meta_buff_addr, meta_buff_len,
                 meta_buff_rkey, groups_mrs_info):
        zmp_start = time.time()
        bootstrap_server = f"tcp://{self.bootstrap_addr}"
        mclient = KVBootstrapClient(bootstrap_server)
        resp = mclient.register(self.bootstrap_room, "prefill",
                                self.mgr.engine_rank, self.mgr.args.engine_rank,
                                local_ip, qp_num, lid, gid,
                                meta_buff_addr, meta_buff_len, meta_buff_rkey,
                                groups_mrs_info)
        zmp_end = time.time()

        logger.debug(f"ZMQ Request time: {zmp_end - zmp_start}")
        if resp["status"] != "ok":
            self.state = KVPoll.Failed

    def query_room(self):
        zmq_client = KVBootstrapClient(f"tcp://{self.bootstrap_addr}")
        resp = zmq_client.query_room(self.bootstrap_room)
        if resp['status'] == "ok" and "decode" in resp['clients']:
            clients = resp['clients']['decode']
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
        self.metadata_ptr = self.mgr.aux_data_ptrs[0] + (aux_index * self.mgr.aux_item_lens[0])
        self.metadata_ptr_length = self.mgr.aux_item_lens[0]
        self.metadata_mr = self.rdma_ep.create_mr(self.metadata_ptr, self.metadata_ptr_length)

        try:
            self.rdma_ep.set_to_transfer(self.transfer_meta.get("qp_num"), self.transfer_meta.get("lid"),
                                         self.transfer_meta.get("gid"))
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

            '''
            completed wrs + metadata wrs
            '''
            if self.rdma_ep.completed_wrs == len(self.rdma_ep.wrs_to_send) + 1 and self.meta_has_sent:
                # write remote metadata //todo
                self.state = KVPoll.Success
            elif self.rdma_ep.completed_wrs == len(self.rdma_ep.wrs_to_send) and not self.meta_has_sent:
                self.rdma_ep.send_metadata_wrs(self.metadata_ptr, self.metadata_ptr_length, self.metadata_mr,
                                               self.transfer_meta)
                self.meta_has_sent = True

        return self.state

    def send(self, kv_indices: np.ndarray[np.int32]):
        """Send actual data synchronously"""
        # 收集要传输的数据
        groups_mrs_info = []
        continous_indices = groups_by_continuity_numpy(kv_indices)
        for group_id, continue_kv_indices in enumerate(continous_indices):
            mrs_info = []
            logger.debug("Sending continuity indices {}".format(continue_kv_indices))
            address_lengths = self.mgr.caculate_layer_kv_addresses(continue_kv_indices)
            for layer_id, (address, length) in enumerate(address_lengths):
                mr = self.rdma_ep.create_mr(address, length)
                self.mrs_to_send.append(mr)
                mrs_info.append({
                    "address": address,
                    "length": length,
                    "rkey": mr.rkey,
                    'lkey': mr.lkey
                })
            groups_mrs_info.append(mrs_info)
        self.rdma_ep.send_wrs(groups_mrs_info, self.transfer_meta['groups_mrs_info'])


class KVReceiver:
    def __init__(self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: Optional[int] = None):
        self.mgr = mgr
        self.kv_layers_mrs = []

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
        self.rdma_ep = RdmaEndpoint(ib_device=self.mgr.args.ib_device)
        self.rdma_port = get_open_port()

        self.start_time = time.time()
        # todo ip

        self.ip = get_local_ip_by_remote(self.bootstrap_addr)

        self.transfer_meta = None
        # self.register()
        self.mrs_to_receive = []  # 数据段待接收的内存区域

    def register(self, local_ip, qp_num, lid, gid, meta_buff_addr, meta_buff_len,
                 meta_buff_rkey, groups_mrs_info):
        zmp_start = time.time()
        bootstrap_server = f"tcp://{self.bootstrap_addr}"
        mclient = KVBootstrapClient(bootstrap_server)
        resp = mclient.register(self.bootstrap_room, "decode",
                                self.mgr.engine_rank, self.mgr.args.engine_rank,
                                local_ip, qp_num, lid, gid,
                                meta_buff_addr, meta_buff_len, meta_buff_rkey,
                                groups_mrs_info)
        zmp_end = time.time()

        logger.debug(f"ZMQ Request time: {zmp_end - zmp_start}")
        if resp["status"] != "ok":
            self.state = KVPoll.Failed

        logger.debug("boostraped success.. qp_num={}, lid={}".format(qp_num, lid))

    def query_room(self):
        zmq_client = KVBootstrapClient(f"tcp://{self.bootstrap_addr}")
        resp = zmq_client.query_room(self.bootstrap_room)
        if resp['status'] == "ok" and "prefill" in resp['clients']:
            clients = resp['clients']['prefill']
            if self.mgr.engine_rank in clients:
                return clients.get(self.mgr.engine_rank, None)
        return None

    def init(self, kv_indices: np.ndarray[np.int32], aux_index: Optional[int] = None):
        """Initialize receiver with KV indices and auxiliary data index

        Args:
            kv_indices: Array of KV indices to receive
            aux_index: Optional index for auxiliary data

        Returns:
            bool: True if initialization successful
        """

        metadata_ptr = self.mgr.aux_data_ptrs[0] + (aux_index * self.mgr.aux_item_lens[0])
        metadata_length = self.mgr.aux_item_lens[0]
        self.meta_mr = self.rdma_ep.create_mr(metadata_ptr, metadata_length)

        # Create MR for each layer and get corresponding key to pass to client
        rkeys = []

        for layer_id, base_addr in enumerate(self.mgr.kv_data_ptrs):
            layer_mr = self.rdma_ep.create_mr(base_addr, self.mgr.kv_data_lens[layer_id])
            rkeys.append(layer_mr.rkey)
            self.kv_layers_mrs.append(layer_mr)
        # TODO: Determine address continuity based on kv_indices to dynamically create larger MRs
        groups_mrs_info = []
        continous_indices = groups_by_continuity_numpy(kv_indices)
        for group_id, continue_kv_indices in enumerate(continous_indices):
            mrs_info = []
            address_lengths = self.mgr.caculate_layer_kv_addresses(continue_kv_indices)
            for layer_id, (address, length) in enumerate(address_lengths):
                mrs_info.append({
                    "address": address,
                    "length": length,
                    "rkey": rkeys[layer_id]
                })
            groups_mrs_info.append(mrs_info)

        try:

            self.register(self.ip, self.rdma_ep.qp.qp_num, self.rdma_ep.port_attr.lid, str(self.rdma_ep.gid),
                          metadata_ptr, metadata_length, self.meta_mr.rkey, groups_mrs_info)

            self.rdma_ep.recv_metadata_mr(metadata_ptr, metadata_length, self.meta_mr.lkey)
            self.state = KVPoll.Transferring

        except Exception as e:
            self.state = KVPoll.Bootstrapping
            return

    def poll(self) -> KVPoll:
        """Poll receive status"""
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
                    self.rdma_ep.set_to_transfer(self.transfer_meta['qp_num'], self.transfer_meta['lid'],
                                                 self.transfer_meta['gid'])
                    self.state = KVPoll.WaitingForInput
        if self.state == KVPoll.Transferring:
            self.rdma_ep.check_meta_recv_complete()
            # 轮询
            if self.rdma_ep.metadata_mr_complete_num == 1:
                logger.debug("Decode Transferring complete...")
                return KVPoll.Success

        if self.state == KVPoll.Failed:
            return KVPoll.Failed

        return self.state

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'loop') and self.loop:
            self.loop.close()


if __name__ == '__main__':
    a = KVBootstrapServer(8998)
    c = KVBootstrapClient("tcp://127.0.0.1:8998")
    c.register(1, "decode", 1, 1, "11", 2, "333", "xx", "ddd", "ddd", "ddd", [])
