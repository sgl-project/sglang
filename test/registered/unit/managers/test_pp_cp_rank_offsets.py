import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.distributed.parallel_state_wrapper import ParallelState  # noqa: E402
from sglang.srt.managers.scheduler_components.request_receiver import (  # noqa: E402
    SchedulerRequestReceiver,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_ps(**overrides) -> ParallelState:
    defaults = dict(
        tp_size=8,
        pp_rank=1,
        pp_size=2,
        dp_rank=None,
        attn_tp_size=2,
        attn_cp_size=2,
        attn_dp_rank=1,
        attn_dp_size=2,
        moe_dp_rank=None,
    )
    defaults.update(overrides)
    return ParallelState.trivial(**defaults)


def _fake_group() -> SimpleNamespace:
    return SimpleNamespace(rank=0, ranks=[0], cpu_group=object())


def _make_receiver(ps: ParallelState) -> SchedulerRequestReceiver:
    tp_group = _fake_group()
    attn_tp_group = _fake_group()
    attn_cp_group = _fake_group()
    world_group = _fake_group()
    return SchedulerRequestReceiver(
        recv_from_tokenizer=None,
        recv_from_rpc=None,
        recv_skipper=None,
        input_blocker=None,
        mm_receiver=None,
        ps=ps,
        tp_group=tp_group,
        tp_cpu_group=tp_group,
        attn_tp_group=attn_tp_group,
        attn_tp_cpu_group=attn_tp_group,
        attn_cp_group=attn_cp_group,
        attn_cp_cpu_group=attn_cp_group,
        world_group=world_group,
        server_args=SimpleNamespace(
            enable_dp_attention=True,
            enable_dp_attention_local_control_broadcast=False,
        ),
        model_config=SimpleNamespace(is_multimodal=False),
        max_recv_per_poll=-1,
        stream_output=lambda *args, **kwargs: None,
        get_last_batch=lambda: None,
    )


class TestRequestReceiverBroadcast(unittest.TestCase):
    def test_local_control_skips_full_tp_broadcast_for_decode_dp(self):
        # Decode uses pure DP attention (attn_tp=attn_cp=1). The DP controller
        # sends control requests to every local leader, so no per-tick Gloo
        # broadcast should remain in SchedulerRequestReceiver.
        ps = SimpleNamespace(
            attn_tp_rank=0,
            attn_cp_rank=0,
            attn_tp_size=1,
            attn_cp_size=1,
            tp_size=32,
        )
        receiver = _make_receiver(ps)
        control_req = SimpleNamespace(kind="control")
        parallel = SimpleNamespace(
            config=SimpleNamespace(
                enable_dp_attention=True,
                enable_dp_attention_local_control_broadcast=True,
            )
        )

        with (
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver."
                "get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver."
                "broadcast_pyobj"
            ) as broadcast,
        ):
            result = receiver._broadcast_reqs_across_ranks([control_req])

        self.assertEqual(result, [control_req])
        broadcast.assert_not_called()

    def test_default_control_uses_full_tp_broadcast(self):
        ps = SimpleNamespace(
            attn_tp_rank=0,
            attn_cp_rank=0,
            attn_tp_size=1,
            attn_cp_size=1,
            tp_size=32,
        )
        receiver = _make_receiver(ps)
        control_req = SimpleNamespace(kind="control")
        parallel = SimpleNamespace(
            config=SimpleNamespace(
                enable_dp_attention=True,
                enable_dp_attention_local_control_broadcast=False,
            )
        )

        with (
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver."
                "get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver."
                "is_ep_scale_joiner",
                return_value=False,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver."
                "broadcast_pyobj",
                side_effect=lambda requests, *_args, **_kwargs: requests,
            ) as broadcast,
        ):
            result = receiver._broadcast_reqs_across_ranks([control_req])

        self.assertEqual(result, [control_req])
        broadcast.assert_called_once_with(
            [control_req],
            receiver.tp_group.rank,
            receiver.tp_cpu_group,
            src=receiver.tp_group.ranks[0],
        )


class TestPPCPRankOffsets(unittest.TestCase):
    def test_request_receiver_uses_cp_size_for_pp_recv_rank(self):
        ps = _make_ps()
        calls = []

        def fake_point_to_point_pyobj(data, rank, group, src, dst, **kwargs):
            calls.append((rank, src, dst))
            return ["req"]

        receiver = _make_receiver(ps)
        with patch(
            "sglang.srt.managers.scheduler_components.request_receiver."
            "point_to_point_pyobj",
            side_effect=fake_point_to_point_pyobj,
        ):
            self.assertEqual(receiver._pull_raw_reqs(), ["req"])

        self.assertEqual(calls, [(12, 4, 12)])

    def test_pp_mixin_uses_cp_size_for_pyobj_send_and_recv_rank(self):
        ps = _make_ps()
        scheduler = SchedulerPPMixin()
        scheduler.ps = ps
        scheduler.world_group = _fake_group()
        scheduler.attn_tp_group = _fake_group()
        scheduler.attn_tp_cpu_group = _fake_group()
        scheduler.attn_cp_group = _fake_group()
        scheduler.attn_cp_cpu_group = _fake_group()
        calls = []

        def fake_point_to_point_pyobj(data, rank, group, src, dst, **kwargs):
            calls.append((rank, src, dst, kwargs.get("async_send", False)))
            return ["work"]

        with (
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.point_to_point_pyobj",
                side_effect=fake_point_to_point_pyobj,
            ),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.broadcast_pyobj",
                side_effect=lambda data, *args, **kwargs: data,
            ),
        ):
            self.assertEqual(
                scheduler._pp_send_pyobj_to_next_stage(["data"], async_send=True),
                ["work"],
            )
            self.assertEqual(scheduler._pp_recv_pyobj_from_prev_stage(), ["work"])

        self.assertEqual(
            calls,
            [
                (12, 12, 4, True),
                (12, 4, 12, False),
            ],
        )


if __name__ == "__main__":
    unittest.main()
