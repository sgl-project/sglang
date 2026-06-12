import threading
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVSender


class _FakeMooncakeManager:
    def __init__(self):
        self.is_dummy_cp_rank = False
        self.enable_all_cp_ranks_for_transfer = False
        self.enable_trace = False
        self.bootstrap_timeout = 60
        self.kv_item_lens_sum = 0
        self.state_item_lens_sum = 0
        self.attn_dp_rank = 0
        self.server_args = SimpleNamespace(dp_size=1, load_balance_method=None)
        self.request_status = {}
        self.req_to_decode_prefix_len = {}
        self.transfer_infos = {}
        self.failure_records = {}
        self.failure_lock = threading.Lock()

    def update_status(self, bootstrap_room, status):
        self.request_status[bootstrap_room] = status

    def check_status(self, bootstrap_room):
        return self.request_status[bootstrap_room]

    def record_failure(self, bootstrap_room, failure_reason):
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason


def test_mooncake_sender_poll_handles_missing_request_status():
    mgr = _FakeMooncakeManager()
    sender = MooncakeKVSender(mgr, "127.0.0.1:30000", 42, [0], 0)
    mgr.request_status.pop(42)

    assert sender.poll() == KVPoll.Failed
    assert sender.poll() == KVPoll.Failed
    assert "status disappeared" in mgr.failure_records[42]
