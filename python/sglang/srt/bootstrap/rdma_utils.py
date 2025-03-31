#!/usr/bin/env python
# coding:utf-8
"""
@author: nivic ybyang7
@license: Apache Licence
@file: rdma_utils
@time: 2025/03/29
@contact: ybyang7@iflytek.com
@site:
@software: PyCharm

# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import logging
import queue
import socket

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

#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class RdmaServer():
    def __init__(self, ib_device='mlx5_bond_0', host_ip='0.0.0.0', socket_port=17777, debug=True):
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.socket_port: None
        self.listen_sock = socket.socket()
        self.listen_sock.bind((host_ip, socket_port))
        self.listen_sock.listen(1)
        logger.debug(f"RDMA KvCache Decode Server listening on port {socket_port}")

        self.qp_s = []  # Only one QP in first phase

        self.ib_device = ib_device
        self.ctx = Context(name=self.ib_device)
        self.pd = PD(self.ctx)
        # Wait for client connection
        self.connections = queue.Queue()
        threading.Thread(target=self.listen, args=(self.connections,), daemon=True).start()
        logger.debug(f"RDMA KvCache run")

    def listen(self, connections):
        retry = 0
        total_retry = 3
        while retry < total_retry:
            try:
                conn, _ = self.listen_sock.accept()
                connections.put(conn)
                return
            except Exception as e:
                logger.error(e)
                retry = retry + 1

    def init(self, mrs, mrs_info, metadata_ptr, metadata_len, rcq_num=1, scq_num=1, max_send_wr=10, max_recv_wr=10,
             max_send_sge=10,
             max_recv_sge=10):
        """Initialize RDMA resources"""
        # Create context

        # Create completion queue
        self.recv_cq = CQ(self.ctx, rcq_num)
        self.send_cq = CQ(self.ctx, scq_num)
        # Create QP
        cap = QPCap(max_send_wr=max_send_wr, max_recv_wr=max_recv_wr, max_send_sge=max_send_sge,
                    max_recv_sge=max_recv_sge)
        init_attr = QPInitAttr(qp_type=2, scq=self.send_cq, rcq=self.recv_cq, cap=cap)
        self.qp = QP(self.pd, init_attr)

        self.port_attr, self.gid = self.get_gid()
        attr = QPAttr()
        attr.qp_state = 2
        attr.pkey_index = 0
        attr.port_num = 1
        attr.qp_access_flags = 0b111

        # init attr
        self.qp.to_init(attr)

        # Register metadata mr
        self.metadata_mr = self.create_mr(metadata_ptr, metadata_len)
        self.metadata_mr_ptr = metadata_ptr
        self.metadata_mr_len = metadata_len
        self.metadata_mr_complete_num = 0
        # Process pending mrs to be sent
        logger.debug("Waiting for incoming connections...")
        self.conn = self.connections.get()
        self.remote_info = self.exchange(mrs_info, metadata_ptr, metadata_len, self.metadata_mr.rkey)

        logger.debug(f"remote decode: {self.remote_info}")

        # Set QP state to ready
        attr.qp_state = 3
        attr.path_mtu = self.port_attr.active_mtu
        attr.dest_qp_num = self.remote_info['qp_num']
        attr.rq_psn = 0
        attr.max_dest_rd_atomic = 1
        attr.min_rnr_timer = 1
        attr.ah_attr.port_num = 1
        if self.port_attr.lid != 0:
            attr.ah_attr.dlid = self.remote_info['lid']
            attr.ah_attr.is_global = 0
        else:
            ah_attr = AHAttr()
            ah_attr.dlid = 0
            ah_attr.is_global = 1
            ah_attr.dgid = self.remote_info['gid']
            ah_attr.sgid_index = 3
            ah_attr.hop_limit = 1
            attr.ah_attr = ah_attr

        self.qp.to_rtr(attr)

        # Set QP state to ready to send
        attr.qp_state = 4
        attr.sq_psn = 0
        attr.timeout = 14
        attr.retry_cnt = 7
        attr.rnr_retry = 7
        attr.max_rd_atomic = 1
        self.qp.to_rts(attr)

        self.initial_wr_index = 0
        #self.recv_metadata_mr()

    def create_mr(self, address, length, access=0b111):
        """
        Register GPU memory area and pass mr at once
        """
        return MR(self.pd, address=address,
                  length=length, access=access)

    def recv_metadata_mr(self):
        recv_sge = SGE(addr=self.metadata_mr_ptr,
                       length=self.metadata_mr_len,
                       lkey=self.metadata_mr.lkey)
        # Server polls for local CQ, client doesn't need to poll
        recv_wr = RecvWR(wr_id=self.initial_wr_index+1, sg=[recv_sge], num_sge=1)
        self.qp.post_recv(recv_wr)
        logger.debug("Waiting metadata ...")


    def get_gid(self):
        port_attr = self.ctx.query_port(1)
        gid = self.ctx.query_gid(1, 3)
        return port_attr, gid

    def exchange(self, mrs_info, metadata_addr, metadata_len, metadata_rkey):
        local_info = {
            'qp_num': self.qp.qp_num,
            'lid': self.port_attr.lid,
            'gid': str(self.gid),
            'metadata_rkey': metadata_rkey,
            'metadata_addr': metadata_addr,
            "metadata_length": metadata_len,
            'mrs_info': mrs_info
        }
        self.conn.sendall(pickle.dumps(local_info))
        # Receive remote information , 4096 is ok ?
        return pickle.loads(self.conn.recv(4096))

    def check_complete(self):
        npolled, wc_list = self.recv_cq.poll()
        if npolled > 0:
            for wc in wc_list:
                if wc.status != IBV_WC_SUCCESS:
                    logger.debug(f"Recv failed: {wc.status}")
                else:
                    self.metadata_mr_complete_num +=1
                    logger.debug(f"Metadata Received completed! Bytes: {wc.byte_len}, wr_id: {wc.wr_id}")


class RdmaClient:
    def __init__(self, ib_device='mlx5_bond_1', host_ip='127.0.0.1', socket_port=17777, debug=True):
        """
        Initialize RDMA client

        Args:
            device_name: RDMA device name
            server_host: Server host address
            server_port: Server port
        """
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        # Create TCP connection
        self.sock = socket.socket()
        # Set connection timeout (in seconds)
        self.sock.connect((host_ip, socket_port))
        # Create context
        self.ib_device = ib_device
        self.ctx = Context(name=self.ib_device)
        self.pd = PD(self.ctx)
        logger.debug(f"RDMA KvCache Prefill Client Connected at port {host_ip}:{socket_port}")

        self.qp_s = []  # Only one QP in first phase
        self.completed_wrs = 0
        self.completed_meta_wrs = 0

    def init(self, meta_buff_addr, meta_buff_len, rcq_num=100, scq_num=100, max_send_wr=100, max_recv_wr=10,
             max_send_sge=10,
             max_recv_sge=10):
        """Initialize RDMA resources"""

        # Create completion queue
        self.rcq_num = rcq_num
        self.scq_num = scq_num
        self.recv_cq = CQ(self.ctx, rcq_num)
        self.send_cq = CQ(self.ctx, scq_num)
        # Create QP
        cap = QPCap(max_send_wr=max_send_wr, max_recv_wr=max_recv_wr, max_send_sge=max_send_sge,
                    max_recv_sge=max_recv_sge)
        init_attr = QPInitAttr(qp_type=2, scq=self.send_cq, rcq=self.recv_cq, cap=cap)
        self.qp = QP(self.pd, init_attr)

        self.port_attr, self.gid = self.get_gid()
        attr = QPAttr()
        attr.qp_state = 2
        attr.pkey_index = 0
        attr.port_num = 1
        attr.qp_access_flags = 0b111

        # init attr
        self.qp.to_init(attr)

        self.metadata_buff_ptr = meta_buff_addr
        self.metadata_buff_len = meta_buff_len
        self.metadata_mr = self.create_mr(meta_buff_addr, meta_buff_len)
        self.remote_info = self.exchange(meta_buff_addr, meta_buff_len, self.metadata_mr.rkey)

        logger.debug(f"remote decode: {self.remote_info}")
        #
        attr.qp_state = 3
        attr.path_mtu = self.port_attr.active_mtu
        attr.dest_qp_num = self.remote_info['qp_num']
        attr.rq_psn = 0
        attr.max_dest_rd_atomic = 1
        attr.min_rnr_timer = 1
        attr.ah_attr.port_num = 1
        if self.port_attr.lid != 0:
            attr.ah_attr.dlid = self.remote_info['lid']
            attr.ah_attr.is_global = 0
        else:
            ah_attr = AHAttr()
            ah_attr.dlid = 0
            ah_attr.is_global = 1
            ah_attr.dgid = self.remote_info['gid']
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

        # receive meta sge
        self.initial_wr_index = 0

        return True

    def send_metadata_wrs(self):
        remote_metadata_addr = self.remote_info["metadata_addr"]
        remote_metadata_len = self.remote_info["metadata_length"]
        remote_metadata_rkey = self.remote_info["metadata_rkey"]
        logger.debug(f"Sending metadata:{remote_metadata_addr}:{remote_metadata_len}:{remote_metadata_rkey}")
        local_sge = SGE(addr=self.metadata_buff_ptr,
                        length=self.metadata_buff_len,
                        lkey=self.metadata_mr.lkey)  # Client needs to poll local CQ, server doesn't need to poll
        wr = SendWR(wr_id=self.initial_wr_index + 1, sg=[local_sge], num_sge=1, opcode=IBV_WR_RDMA_WRITE_WITH_IMM,
                    send_flags=IBV_SEND_FENCE)
        wr.set_wr_rdma(addr=remote_metadata_addr,
                       rkey=remote_metadata_rkey)
        self.qp.post_send(wr)
        logger.debug(f"Sending metadata sent ...")

    def send_wrs(self, mrs, mrs_info):
        remote_mrs_info = self.remote_info['mrs_info']
        wrs = self.get_wrs(mrs, mrs_info, remote_mrs_info)
        for wr in wrs:
            self.qp.post_send(wr)
        logger.debug("Sending Request posted ...")

    def get_wrs(self, mrs, mrs_info, remote_mrs_info):
        wrs = []
        for layer_id, item in enumerate(remote_mrs_info):
            self.initial_wr_index += 1
            local_sge = SGE(addr=mrs_info[layer_id]["address"],
                            length=mrs_info[layer_id]["length"],
                            lkey=mrs[layer_id].lkey)
            wr = SendWR(wr_id=self.initial_wr_index, sg=[local_sge], num_sge=1, opcode=IBV_WR_RDMA_WRITE,
                        send_flags=IBV_SEND_SIGNALED)
            wr.set_wr_rdma(addr=item['address'],
                           rkey=item['rkey'])
            wrs.append(wr)
        return wrs

    def create_mr(self, address, length, access=0b111):
        """
        Pre-allocate GPU memory area and pass mr at once
        """
        return MR(self.pd, address=address,
                  length=length, access=access)

    def get_gid(self):
        port_attr = self.ctx.query_port(1)
        gid = self.ctx.query_gid(1, 3)
        return port_attr, gid

    def exchange(self, meta_buff_addr, meta_buff_len, meta_buff_rkey):
        local_info = {
            'qp_num': self.qp.qp_num,
            'lid': self.port_attr.lid,
            'gid': str(self.gid),
            'metadata_rkey': meta_buff_rkey,
            'metadata_addr': meta_buff_addr,
            "metadata_length": meta_buff_len,

        }
        self.sock.sendall(pickle.dumps(local_info))
        # Receive remote information
        return pickle.loads(self.sock.recv(4096))

    def create_sge(self, ptr, length, mr_key):
        sge = SGE(addr=ptr, length=length, lkey=mr_key)
        return sge

    def check_send_complete(self):
        npolled, wc_list = self.send_cq.poll()
        if npolled > 0:
            for wc in wc_list:
                if wc.status != IBV_WC_SUCCESS:
                    logger.debug(f"Send failed: {wc.status}")
                    logger.error(wc)
                else:
                    self.completed_wrs += 1
                    logger.debug(f"Send completed! Bytes: {wc.byte_len}, wr_id: {wc.wr_id}")

    def check_recv_complete(self):
        npolled, wc_list = self.recv_cq.poll()
        if npolled > 0:
            for wc in wc_list:
                if wc.status != IBV_WC_SUCCESS:
                    logger.debug(f"Recv meata: {wc.status}")
                    logger.error(wc)
                else:
                    logger.debug(f"Send completed Meta! Bytes: {wc.byte_len}, wr_id: {wc.wr_id}")
