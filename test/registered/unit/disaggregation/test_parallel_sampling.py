import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.common import conn
from sglang.srt.disaggregation.common.conn import CommonKVSender
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _TimeStats:
    def set_finished_time(self):
        pass


class _FakeTokenizerManager:
    def __init__(self, fake_bootstrap_room=None):
        self.rid_to_state = {}
        self.sent_rooms = []
        self.sent_max_new_tokens = []
        self.fake_bootstrap_room = fake_bootstrap_room

    async def _tokenize_one_request(self, obj):
        return SimpleNamespace(
            mm_inputs=None,
            rid=obj.rid,
            input_ids=list(obj.input_ids),
            sampling_params=SimpleNamespace(max_new_tokens=8),
            stream=obj.stream,
            bootstrap_host=obj.bootstrap_host,
            bootstrap_port=obj.bootstrap_port,
            bootstrap_room=(
                obj.bootstrap_room
                if obj.bootstrap_room is not None
                else self.fake_bootstrap_room
            ),
            bootstrap_pair_key=obj.bootstrap_pair_key,
            decode_tp_size=obj.decode_tp_size,
            time_stats=None,
        )

    def _init_req_state(self, obj):
        self.rid_to_state[obj.rid] = SimpleNamespace(time_stats=_TimeStats())

    def _send_one_request(self, obj):
        self.sent_rooms.append(obj.bootstrap_room)
        self.sent_max_new_tokens.append(obj.sampling_params.max_new_tokens)

    async def _wait_one_response(self, obj, request):
        yield {"rid": obj.rid}


class TestParallelSamplingDisaggregation(CustomTestCase):
    """Regression tests for parallel sampling request expansion."""

    def test_parallel_samples_keep_unique_bootstrap_rooms(self):
        """Each disaggregated sample must retain its normalized room."""

        async def run_test():
            request = GenerateReqInput(
                input_ids=[[1, 2], [3, 4]],
                sampling_params={"n": 2, "max_new_tokens": 8},
                rid="request",
                stream=False,
                bootstrap_host="127.0.0.1",
                bootstrap_port=12345,
                bootstrap_room=1000,
            )
            request.normalize_batch_and_arguments()

            manager = _FakeTokenizerManager()
            for i in range(request.batch_size):
                manager._init_req_state(request[i])

            result = TokenizerManager._handle_batch_request(manager, request, None)
            await result.__anext__()

            self.assertEqual(request.bootstrap_room, [1000, 1001, 1002, 1003])
            self.assertEqual(manager.sent_rooms, [1000, 1002, 1001, 1003])

        asyncio.run(run_test())

    def test_non_disaggregated_parallel_sampling_caches_prefix(self):
        """Non-disaggregated sampling must keep the zero-token cache primer."""

        async def run_test():
            request = GenerateReqInput(
                input_ids=[1, 2],
                sampling_params={"n": 2, "max_new_tokens": 8},
                rid="request",
                stream=False,
            )
            request.normalize_batch_and_arguments()

            manager = _FakeTokenizerManager()
            manager._init_req_state(request[0])

            result = TokenizerManager._handle_batch_request(manager, request, None)
            await result.__anext__()

            self.assertEqual(manager.sent_rooms, [None, None, None])
            self.assertEqual(manager.sent_max_new_tokens, [0, 8, 8])

        asyncio.run(run_test())

    def test_fake_transfer_room_is_not_overwritten(self):
        """An auto-assigned fake-transfer room must survive sample cloning."""

        async def run_test():
            request = GenerateReqInput(
                input_ids=[1, 2],
                sampling_params={"n": 2, "max_new_tokens": 8},
                rid="request",
                stream=False,
            )
            request.normalize_batch_and_arguments()

            manager = _FakeTokenizerManager(fake_bootstrap_room=9000)
            manager._init_req_state(request[0])

            result = TokenizerManager._handle_batch_request(manager, request, None)
            await result.__anext__()

            self.assertEqual(manager.sent_rooms, [9000, 9000, 9000])

        asyncio.run(run_test())


class TestParallelSamplingDpRankRegistration(CustomTestCase):
    """Regression tests for forced prefill DP-rank discovery."""

    @staticmethod
    def _manager(attn_dp_rank=1):
        return SimpleNamespace(
            is_dummy_cp_rank=False,
            update_status=MagicMock(),
            record_failure=MagicMock(),
            server_args=SimpleNamespace(
                dp_size=4, load_balance_method="follow_bootstrap_room"
            ),
            attn_dp_rank=attn_dp_rank,
        )

    def test_forced_query_registers_modulo_aligned_room(self):
        """Forced discovery registers even when room modulo matches DP rank."""

        manager = self._manager()

        with (
            conn.envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.override(True),
            patch.object(CommonKVSender, "_register_prefill_dp_rank") as register,
        ):
            CommonKVSender(
                manager,
                bootstrap_addr="127.0.0.1:12345",
                bootstrap_room=5,
                dest_tp_ranks=[],
                pp_rank=0,
            )

        register.assert_called_once_with()

    def test_default_query_skips_modulo_aligned_room(self):
        """Default routing retains the modulo fast path."""

        manager = self._manager()
        with (
            conn.envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.override(False),
            patch.object(CommonKVSender, "_register_prefill_dp_rank") as register,
        ):
            CommonKVSender(
                manager,
                bootstrap_addr="127.0.0.1:12345",
                bootstrap_room=5,
                dest_tp_ranks=[],
                pp_rank=0,
            )

        register.assert_not_called()
        manager.record_failure.assert_not_called()

    def test_default_query_rejects_modulo_mismatch(self):
        """Default routing still rejects externally overridden DP placement."""

        manager = self._manager(attn_dp_rank=2)
        with (
            conn.envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.override(False),
            patch.object(CommonKVSender, "_register_prefill_dp_rank") as register,
        ):
            CommonKVSender(
                manager,
                bootstrap_addr="127.0.0.1:12345",
                bootstrap_room=5,
                dest_tp_ranks=[],
                pp_rank=0,
            )

        register.assert_not_called()
        manager.record_failure.assert_called_once()
        manager.update_status.assert_any_call(5, conn.KVPoll.Failed)


if __name__ == "__main__":
    unittest.main()
