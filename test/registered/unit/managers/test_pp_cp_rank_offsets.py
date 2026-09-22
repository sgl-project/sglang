import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.request_receiver import (  # noqa: E402
    SchedulerRequestReceiver,
)
from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    SchedulerPPMixin,
    _pp_exchange_outputs_before_forward,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _published_topology():
    """Publish WORLD rank 12 with TP=8 and PP=2.

    This gives TP rank 4, PP rank 1, attention-DP rank 1, and attention-TP rank 0.
    """
    return published_topology(
        role="scheduler",
        ranks={"world_rank": 12, "dp_rank": 1},
        tp_size=8,
        pp_size=2,
        dp_size=2,
        attn_cp_size=2,
        enable_dp_attention=True,
    )


def _fake_group() -> SimpleNamespace:
    return SimpleNamespace(rank=0, ranks=[0], cpu_group=object())


def _make_receiver() -> SchedulerRequestReceiver:
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
        receiver = _make_receiver()
        control_req = SimpleNamespace(kind="control")
        parallel = SimpleNamespace(
            enable_dp_attention=True,
            enable_dp_attention_local_control_broadcast=True,
            attn_tp_rank=0,
            attn_cp_rank=0,
            attn_tp_size=1,
            attn_cp_size=1,
            tp_size=32,
        )

        with (
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver."
                "get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver."
                "attn_cp_tp_broadcast_pyobj",
                side_effect=lambda requests: requests,
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
        receiver = _make_receiver()
        control_req = SimpleNamespace(kind="control")
        parallel = SimpleNamespace(
            enable_dp_attention=True,
            enable_dp_attention_local_control_broadcast=False,
            attn_tp_rank=0,
            attn_cp_rank=0,
            attn_tp_size=1,
            attn_cp_size=1,
            tp_size=32,
        )

        with (
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver."
                "get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(is_ep_scale_joiner=False)
                ),
            ),
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver."
                "attn_cp_tp_broadcast_pyobj",
                side_effect=lambda requests: requests,
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
        enter_scope(self, _published_topology())
        calls = []

        def fake_point_to_point_pyobj(data, rank, group, src, dst, **kwargs):
            calls.append((rank, src, dst))
            return ["req"]

        receiver = _make_receiver()
        with patch(
            "sglang.srt.managers.scheduler_components.request_receiver."
            "point_to_point_pyobj",
            side_effect=fake_point_to_point_pyobj,
        ):
            self.assertEqual(receiver._pull_raw_reqs(), ["req"])

        self.assertEqual(calls, [(12, 4, 12)])

    def test_pp_mixin_uses_cp_size_for_pyobj_send_and_recv_rank(self):
        enter_scope(self, _published_topology())
        scheduler = SchedulerPPMixin()
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
                "sglang.srt.managers.scheduler_pp_mixin.attn_cp_tp_broadcast_pyobj",
                side_effect=lambda data: data,
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


class TestDSparkPPOutput(CustomTestCase):
    def test_output_ring_rebinds_dspark_state_on_each_stage(self):
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
        from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
        from sglang.srt.speculative.dspark_components.dspark_draft import (
            make_next_draft_input,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        payloads = []
        scheduler = SimpleNamespace(
            _pp_spec_relay=False,
            pp_group=SimpleNamespace(is_first_rank=False),
            future_map=SimpleNamespace(
                stash=lambda indices, value: payloads.append(value)
            ),
        )
        tokens = torch.tensor([13, 29])
        batch = SimpleNamespace(
            return_logprob=False,
            req_pool_indices=torch.tensor([0, 1]),
            seq_lens=torch.tensor([8, 15]),
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=object(),
        )
        wire = SchedulerPPMixin._pp_prepare_tensor_dict(
            scheduler,
            SimpleNamespace(
                next_token_ids=tokens,
                next_draft_input=make_next_draft_input(
                    bonus_tokens=tokens, new_seq_lens=batch.seq_lens
                ),
                logits_output=None,
            ),
            batch,
        )
        self.assertNotIn("draft_topk_p", wire)
        result = SchedulerPPMixin._pp_prep_batch_result(
            scheduler,
            batch,
            SimpleNamespace(can_run_cuda_graph=False),
            PPProxyTensors(wire),
        )
        self.assertIsInstance(result.next_draft_input, DFlashDraftInputV2)
        self.assertIs(batch.spec_info, result.next_draft_input)
        torch.testing.assert_close(batch.spec_info.bonus_tokens, tokens)
        torch.testing.assert_close(batch.spec_info.new_seq_lens, batch.seq_lens)
        torch.testing.assert_close(payloads[0].bonus_tokens, tokens)
        self.assertEqual(payloads[0].hidden_states.numel(), 0)


class TestPPSpecExchangeOrder(unittest.TestCase):
    def test_extend_launches_before_the_relay_exchange(self):
        kwargs = dict(spec_relay=True, is_last_rank=False, async_batch_depth=0)
        extend = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND, is_extend_in_batch=False
        )
        decode = SimpleNamespace(
            forward_mode=ForwardMode.DECODE, is_extend_in_batch=False
        )
        self.assertFalse(
            _pp_exchange_outputs_before_forward(cur_batch=extend, **kwargs)
        )
        self.assertTrue(_pp_exchange_outputs_before_forward(cur_batch=decode, **kwargs))


if __name__ == "__main__":
    unittest.main()
