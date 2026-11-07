# Regression for #37590: with DP attention and both attn_tp_size > 1 and
# attn_cp_size > 1, the first broadcast hop runs along attn_tp, so its source
# in every attn_cp_rank != 0 slice is a rank that holds no requests yet.
# broadcast_pyobj reads len(data) on the source, which raised
# "object of type NoneType has no len()" and killed the scheduler.
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MOD = "sglang.srt.managers.scheduler_components.request_receiver"
LEADER_REQS = ["req-a", "req-b"]


def _fake_broadcast_pyobj(data, rank, dist_group, src=0, **_):
    # Mirror the real function on the source rank: it serializes data, so it
    # must be a list. Non-source ranks receive the list held by the leader.
    if rank == src:
        len(data)
        return data
    return list(LEADER_REQS)


def _receiver(attn_tp_rank, attn_cp_rank, attn_tp_size=2, attn_cp_size=2):
    def group(rank):
        return SimpleNamespace(rank=rank, ranks=[0], cpu_group=object())

    return SchedulerRequestReceiver(
        recv_from_tokenizer=None,
        recv_from_rpc=None,
        recv_skipper=None,
        input_blocker=None,
        mm_receiver=None,
        ps=SimpleNamespace(
            pp_rank=0,
            tp_size=attn_tp_size * attn_cp_size,
            attn_tp_rank=attn_tp_rank,
            attn_cp_rank=attn_cp_rank,
            attn_tp_size=attn_tp_size,
            attn_cp_size=attn_cp_size,
            attn_dp_rank=0,
        ),
        tp_group=group(0),
        tp_cpu_group=group(0),
        attn_tp_group=group(attn_tp_rank),
        attn_tp_cpu_group=group(attn_tp_rank),
        attn_cp_group=group(attn_cp_rank),
        attn_cp_cpu_group=group(attn_cp_rank),
        world_group=group(0),
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(is_multimodal=False),
        max_recv_per_poll=-1,
        stream_output=lambda *args, **kwargs: None,
        get_last_batch=lambda: None,
    )


class TestDpAttentionRequestBroadcast(unittest.TestCase):
    def _run(self, receiver, recv_reqs):
        parallel = SimpleNamespace(
            enable_dp_attention=True,
            enable_dp_attention_local_control_broadcast=False,
        )
        with (
            patch(f"{MOD}.get_parallel", return_value=parallel),
            patch(f"{MOD}.is_ep_scale_joiner", return_value=False),
            patch(f"{MOD}.broadcast_pyobj", side_effect=_fake_broadcast_pyobj),
        ):
            return receiver._broadcast_reqs_across_ranks(recv_reqs)

    def test_attn_tp_source_in_nonzero_cp_slice_does_not_crash(self):
        # attn_tp_rank 0 in attn_cp slice 1: the source of the first hop for
        # its slice, holding nothing. This is the rank that raised TypeError.
        receiver = _receiver(attn_tp_rank=0, attn_cp_rank=1)
        self.assertEqual(self._run(receiver, None), LEADER_REQS)

    def test_plain_follower_receives_leader_requests(self):
        receiver = _receiver(attn_tp_rank=1, attn_cp_rank=1)
        self.assertEqual(self._run(receiver, None), LEADER_REQS)

    def test_leader_still_splits_and_forwards_its_own_requests(self):
        receiver = _receiver(attn_tp_rank=0, attn_cp_rank=0)
        # The receiver is a frozen slots dataclass, so stub the split on the
        # class rather than the instance.
        with patch.object(
            SchedulerRequestReceiver,
            "_split_work_and_control_reqs",
            lambda self, reqs: (["work"], ["ctrl"]),
        ):
            self.assertEqual(self._run(receiver, ["work", "ctrl"]), ["work", "ctrl"])


if __name__ == "__main__":
    unittest.main()
