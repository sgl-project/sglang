import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

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


class TestPPProfileCleanup(unittest.TestCase):
    def test_profile_cleanup_releases_full_allocated_range(self) -> None:
        """Profiling cleanup releases through the request's allocated KV length."""
        req = SimpleNamespace(
            prefix_indices=[],
            req_pool_idx=1,
            kv=SimpleNamespace(kv_allocated_len=8),
        )

        def set_extend_range(start: int, end: int) -> None:
            req.extend_range = SimpleNamespace(start=start, end=end)

        req.set_extend_range = set_extend_range
        batch = SimpleNamespace(
            input_ids=torch.empty(0),
            prefill_input_ids_cpu=None,
            prepare_for_extend=Mock(),
            forward_mode=SimpleNamespace(is_extend=lambda: True),
        )
        model_config = SimpleNamespace(
            hc_hidden_size=None,
            hidden_size=2,
            dtype=torch.float32,
        )
        model_runner = SimpleNamespace(
            model_config=model_config,
            get_pp_proxy_topk_size=Mock(return_value=None),
            forward=Mock(),
        )
        allocator = SimpleNamespace(free=Mock())
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(32).reshape(2, 16),
            free=Mock(),
        )
        predictor = SimpleNamespace(
            fit=Mock(),
            set_target_latency=Mock(),
            target_latency=0.0,
            is_ready=False,
        )
        scheduler = SchedulerPPMixin()
        scheduler.pp_group = SimpleNamespace(is_first_rank=True)
        scheduler.tp_worker = SimpleNamespace(model_runner=model_runner)
        scheduler.model_config = model_config
        scheduler.chunked_prefill_size = 4
        scheduler.req_to_token_pool = req_to_token_pool
        scheduler.token_to_kv_pool_allocator = allocator
        scheduler.tree_cache = object()
        scheduler.spec_algorithm = object()
        scheduler.device = torch.device("cpu")
        scheduler.ps = SimpleNamespace(attn_tp_size=1, attn_cp_size=1, pp_rank=0)

        with (
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.tqdm",
                side_effect=lambda values, **_: list(values)[:1],
            ),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.Req",
                return_value=req,
            ),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.ScheduleBatch.init_new",
                return_value=batch,
            ),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.ForwardBatch.init_new",
                return_value=object(),
            ),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.get_device_module",
                return_value=SimpleNamespace(synchronize=Mock()),
            ),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.is_dp_attention_enabled",
                return_value=False,
            ),
            patch("sglang.srt.managers.scheduler_pp_mixin.set_is_extend_in_batch"),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.ChunkSizePredictor",
                return_value=predictor,
            ),
            patch("torch.distributed.is_available", return_value=False),
        ):
            scheduler.profile_and_init_predictor()

        allocator.free.assert_called_once()
        self.assertTrue(
            torch.equal(
                allocator.free.call_args.args[0],
                req_to_token_pool.req_to_token[1, :8],
            )
        )
        req_to_token_pool.free.assert_called_once_with(req)
        self.assertIsNone(req.kv)


if __name__ == "__main__":
    unittest.main()
