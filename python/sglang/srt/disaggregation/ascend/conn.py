import logging
from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.utils import get_local_ip_by_remote
from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine

logger = logging.getLogger(__name__)


class AscendKVManager(MooncakeKVManager):
    def init_engine(self):
        # TransferEngine initialized on ascend.
        local_ip = get_local_ip_by_remote()
        self.engine = AscendTransferEngine(
            hostname=local_ip,
            ascend_url=self.kv_args.ascend_url,
            npu_id=self.kv_args.gpu_id,
            disaggregation_mode=self.disaggregation_mode,
            ib_device=self.kv_args.ib_device,
            ascend_mooncake=self.kv_args.ascend_mooncake
        )

    def caculate_kv_addr(self, src_ptr: int, dst_ptr: int, prefill_index: list[int], decode_index: list[int],
                         item_len: int, data_len: int):
        return src_ptr, dst_ptr, data_len

    def caculate_aux_addr(self, src_ptr: int, dst_ptr: int, prefill_aux_index: int, dst_aux_index: int,
                          item_len: int, data_len: int):
        return src_ptr, dst_ptr, data_len

    def register_buffer_to_engine(self):
        self.engine.register(self.kv_args.kv_data_ptrs[0], sum(self.kv_args.kv_data_lens))
        if not self.kv_args.ascend_mooncake:
            self.engine.batch_register(self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens)
        else:
            for aux_data_ptr, aux_data_len in zip(
                self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
            ):
                self.engine.register(aux_data_ptr, aux_data_len)


class AscendKVSender(MooncakeKVSender):
    pass


class AscendKVReceiver(MooncakeKVReceiver):
    pass


class AscendKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
