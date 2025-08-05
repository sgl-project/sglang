import logging

from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.utils import get_local_ip_by_remote

logger = logging.getLogger(__name__)


class AscendKVManager(MooncakeKVManager):
    def init_engine(self):
        # TransferEngine initialized on ascend.
        local_ip = get_local_ip_by_remote()
        self.engine = AscendTransferEngine(
            hostname=local_ip,
            npu_id=self.kv_args.gpu_id,
            disaggregation_mode=self.disaggregation_mode,
        )

    def register_buffer_to_engine(self):
        self.engine.register(
            self.kv_args.kv_data_ptrs[0], sum(self.kv_args.kv_data_lens)
        )
        # The Ascend backend optimize batch registration for small memory blocks.
        self.engine.batch_register(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        )


class AscendKVSender(MooncakeKVSender):
    pass


class AscendKVReceiver(MooncakeKVReceiver):
    pass


class AscendKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
