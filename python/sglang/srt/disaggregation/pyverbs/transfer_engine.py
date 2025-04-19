import logging

from pyverbs.addr import GID, AHAttr
from pyverbs.cq import CQ, WC
from pyverbs.device import Context
from pyverbs.enums import *
from pyverbs.mr import MR
from pyverbs.pd import PD
from pyverbs.qp import QP, QPAttr, QPCap, QPInitAttr
from pyverbs.wr import SGE, RecvWR, SendWR

logger = logging.getLogger(__name__)


class RdmaEndpoint(object):
    def __init__(
        self,
        ib_device="mlx5_bond_0",
        max_send_wr=10,
        max_recv_wr=10,
        max_send_sge=30,
        max_recv_sge=30,
        rcq_num=1000,
        scq_num=1600,
        debug=True,
    ):
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.ib_device = ib_device
        self.ctx = Context(name=self.ib_device)
        self.pd = PD(self.ctx)
        self.recv_cq = CQ(self.ctx, rcq_num)
        self.send_cq = CQ(self.ctx, scq_num)
        # Create QP
        cap = QPCap(
            max_send_wr=max_send_wr,
            max_recv_wr=max_recv_wr,
            max_send_sge=max_send_sge,
            max_recv_sge=max_recv_sge,
        )
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
        remote_metadata_addr = transfer_meta["meta_buff"]["meta_buff_addr"]
        remote_metadata_len = transfer_meta["meta_buff"]["meta_buff_length"]
        remote_metadata_rkey = transfer_meta["meta_buff"]["meta_buff_rkey"]
        logger.debug(
            f"Sending metadata:{remote_metadata_addr}:{remote_metadata_len}:{remote_metadata_rkey}"
        )
        local_sge = SGE(
            addr=meta_ptr, length=meta_len, lkey=meta_mr.lkey
        )  # 发送要本地轮询cq，服务端轮询即可
        wr = SendWR(
            wr_id=self.initial_wr_index + 1,
            sg=[local_sge],
            num_sge=1,
            opcode=IBV_WR_RDMA_WRITE_WITH_IMM,
            send_flags=IBV_SEND_FENCE,
        )
        wr.set_wr_rdma(addr=remote_metadata_addr, rkey=remote_metadata_rkey)
        self.qp.post_send(wr)
        logger.debug(f"Sending metadata sent ...")

    def add_to_sending_wrs(self, group_mrs_info, group_remote_mrs_info):
        # By default, decode allocated memory is contiguous
        if len(group_remote_mrs_info) != 1:
            raise Exception("decode allocate non-contiguous wrs")

        for group_id, mrs_info in enumerate(group_mrs_info):
            for layer_id, item in enumerate(mrs_info):
                self.initial_wr_index += 1
                local_sge = SGE(
                    addr=item["address"], length=item["length"], lkey=item["lkey"]
                )
                wr = SendWR(
                    wr_id=self.initial_wr_index,
                    sg=[local_sge],
                    num_sge=1,
                    opcode=IBV_WR_RDMA_WRITE,
                    send_flags=IBV_SEND_SIGNALED,
                )
                wr.set_wr_rdma(
                    addr=group_remote_mrs_info[0][layer_id]["address"],
                    rkey=group_remote_mrs_info[0][layer_id]["rkey"],
                )
                self.wrs_to_send.append(wr)
                group_remote_mrs_info[0][layer_id]["address"] += item["length"]

    def recv_metadata_mr(self, meta_ptr, meta_len, meta_lkey):
        recv_sge = SGE(addr=meta_ptr, length=meta_len, lkey=meta_lkey)
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
                    logger.debug(
                        f"Metadata Received completed! Bytes: {wc.byte_len}, wr_id: {wc.wr_id}"
                    )

    def create_mr(self, address, length, access=0b111):
        """
        Pre-allocate GPU memory region and pass MR at once
        """
        return MR(self.pd, address=address, length=length, access=access)
