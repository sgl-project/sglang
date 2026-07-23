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
        tp_rank=0,
        tp_size=8,
        pp_rank=1,
        pp_size=2,
        dp_rank=None,
        dp_size=1,
        attn_tp_rank=0,
        attn_tp_size=2,
        attn_cp_rank=0,
        attn_cp_size=2,
        attn_dp_rank=1,
        attn_dp_size=2,
        moe_ep_rank=0,
        moe_ep_size=1,
        moe_dp_rank=None,
        moe_dp_size=1,
        gpu_id=0,
    )
    defaults.update(overrides)
    return ParallelState(**defaults)


def _fake_group() -> SimpleNamespace:
    return SimpleNamespace(rank=0, ranks=[0], cpu_group=object())


def _make_receiver(ps: ParallelState) -> SchedulerRequestReceiver:
    group = _fake_group()
    return SchedulerRequestReceiver(
        recv_from_tokenizer=None,
        recv_from_rpc=None,
        recv_skipper=None,
        input_blocker=None,
        mm_receiver=None,
        ps=ps,
        tp_group=group,
        tp_cpu_group=group,
        attn_tp_group=group,
        attn_tp_cpu_group=group,
        attn_cp_group=group,
        attn_cp_cpu_group=group,
        world_group=group,
        server_args=SimpleNamespace(
            enable_dp_attention=True,
            enable_dp_attention_local_control_broadcast=False,
        ),
        model_config=SimpleNamespace(is_multimodal=False),
        max_recv_per_poll=-1,
        stream_output=lambda *args, **kwargs: None,
        get_last_forward_mode=lambda: None,
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
