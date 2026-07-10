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
        self.sent_metadata = []
        self.sent_rooms = []
        self.sent_max_new_tokens = []
        self.sent_streams = []
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
        rids = obj.rid if isinstance(obj.rid, list) else [obj.rid]
        for rid in rids:
            self.rid_to_state[rid] = SimpleNamespace(time_stats=_TimeStats())

    def _send_one_request(self, obj):
        self.sent_metadata.append(
            (
                obj.bootstrap_host,
                obj.bootstrap_port,
                obj.bootstrap_room,
                obj.bootstrap_pair_key,
                obj.decode_tp_size,
            )
        )
        self.sent_rooms.append(obj.bootstrap_room)
        self.sent_max_new_tokens.append(obj.sampling_params.max_new_tokens)
        self.sent_streams.append(obj.stream)

    async def _wait_one_response(self, obj, request):
        self.rid_to_state.pop(obj.rid)
        yield {"rid": obj.rid, "meta_info": {"id": obj.rid}}


class TestParallelSamplingDisaggregation(CustomTestCase):
    """Regression tests for parallel sampling request expansion."""

    @staticmethod
    async def _run_request(request, manager):
        request.normalize_batch_and_arguments()
        manager._init_req_state(request)
        outputs = []
        async for output in TokenizerManager._handle_batch_request(
            manager, request, None
        ):
            outputs.append(output)
        return outputs

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
            manager = _FakeTokenizerManager()
            await self._run_request(request, manager)

            self.assertEqual(request.bootstrap_room, [1000, 1001, 1002, 1003])
            self.assertEqual(manager.sent_rooms, [1000, 1002, 1001, 1003])
            self.assertEqual(manager.rid_to_state, {})

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
            manager = _FakeTokenizerManager()
            await self._run_request(request, manager)

            self.assertEqual(manager.sent_rooms, [None, None, None])
            self.assertEqual(manager.sent_max_new_tokens, [0, 8, 8])
            self.assertEqual(manager.rid_to_state, {})

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
            manager = _FakeTokenizerManager(fake_bootstrap_room=9000)
            await self._run_request(request, manager)

            self.assertEqual(manager.sent_rooms, [9000, 9000, 9000])
            self.assertEqual(manager.rid_to_state, {})

        asyncio.run(run_test())

    def test_per_batch_rooms_expand_without_collisions(self):
        """Per-batch rooms expand uniquely even when candidates collide."""

        async def run_test():
            request = GenerateReqInput(
                input_ids=[[1, 2], [3, 4]],
                sampling_params={"n": 3, "max_new_tokens": 8},
                rid=["request-a", "request-b"],
                stream=False,
                bootstrap_host=["host-a", "host-b"],
                bootstrap_port=[1001, 1002],
                bootstrap_room=[100, 102],
                bootstrap_pair_key=["pair-a", "pair-b"],
                decode_tp_size=[2, 4],
            )

            manager = _FakeTokenizerManager()
            await self._run_request(request, manager)

            self.assertEqual(request.rid, ["request-a", "request-b"])
            self.assertEqual(request.bootstrap_room, [100, 102, 104, 106, 108, 110])
            self.assertEqual(
                manager.sent_rooms,
                [100, 104, 108, 102, 106, 110],
            )
            self.assertEqual(
                manager.sent_metadata[1],
                ("host-a", 1001, 104, "pair-a", 2),
            )
            self.assertEqual(manager.rid_to_state, {})

        asyncio.run(run_test())

    def test_already_expanded_bootstrap_metadata_is_preserved(self):
        """Routers may provide one complete bootstrap tuple per sample."""
        request = GenerateReqInput(
            input_ids=[1, 2],
            sampling_params={"n": 3, "max_new_tokens": 8},
            bootstrap_host=["host-a", "host-b", "host-c"],
            bootstrap_port=[1001, 1002, 1003],
            bootstrap_room=[10, 11, 12],
            bootstrap_pair_key=["pair-a", "pair-b", "pair-c"],
            decode_tp_size=[1, 2, 4],
        )

        request.normalize_batch_and_arguments()

        self.assertEqual(request.bootstrap_host, ["host-a", "host-b", "host-c"])
        self.assertEqual(request.bootstrap_port, [1001, 1002, 1003])
        self.assertEqual(request.bootstrap_room, [10, 11, 12])
        self.assertEqual(request.bootstrap_pair_key, ["pair-a", "pair-b", "pair-c"])
        self.assertEqual(request.decode_tp_size, [1, 2, 4])

    def test_duplicate_already_expanded_rooms_are_rejected(self):
        """An already-expanded request must not reintroduce room collisions."""
        request = GenerateReqInput(
            input_ids=[1, 2],
            sampling_params={"n": 3, "max_new_tokens": 8},
            bootstrap_room=[10, 11, 10],
        )

        with self.assertRaisesRegex(ValueError, "must be unique"):
            request.normalize_batch_and_arguments()

    def test_duplicate_per_batch_rooms_are_rejected(self):
        """Original batch items must not share a transfer room."""
        request = GenerateReqInput(
            input_ids=[[1, 2], [3, 4]],
            sampling_params={"n": 2, "max_new_tokens": 8},
            bootstrap_room=[10, 10],
        )

        with self.assertRaisesRegex(ValueError, "must be unique"):
            request.normalize_batch_and_arguments()

    def test_any_expanded_room_skips_prefix_cache_primer(self):
        """The primer must not consume a room from a later sample slice."""

        async def run_test():
            request = GenerateReqInput(
                input_ids=[[1, 2], [3, 4]],
                sampling_params={"n": 2, "max_new_tokens": 8},
                stream=False,
                bootstrap_room=[None, None, 10, 11],
            )
            manager = _FakeTokenizerManager()

            await self._run_request(request, manager)

            self.assertEqual(manager.sent_rooms, [None, 10, None, 11])
            self.assertEqual(manager.sent_max_new_tokens, [8, 8, 8, 8])
            self.assertEqual(manager.rid_to_state, {})

        asyncio.run(run_test())

    def test_invalid_bootstrap_list_lengths_are_rejected(self):
        """Bootstrap lists must describe either originals or all samples."""
        invalid_fields = {
            "bootstrap_host": ["host"],
            "bootstrap_port": [1001, 1002, 1003, 1004],
            "bootstrap_room": [10, 11, 12],
            "bootstrap_pair_key": ["a"],
            "decode_tp_size": [1, 2, 4, 8],
        }

        for field, value in invalid_fields.items():
            with self.subTest(field=field):
                request = GenerateReqInput(
                    input_ids=[[1, 2], [3, 4]],
                    sampling_params={"n": 3, "max_new_tokens": 8},
                    **{field: value},
                )
                with self.assertRaisesRegex(ValueError, field):
                    request.normalize_batch_and_arguments()

    def test_explicit_list_rids_work_with_parallel_sampling(self):
        """Expanded routing must not index beyond original list rids."""

        async def run_test():
            request = GenerateReqInput(
                input_ids=[[1, 2], [3, 4]],
                sampling_params={"n": 2, "max_new_tokens": 8},
                rid=["request-a", "request-b"],
                bootstrap_host="127.0.0.1",
                bootstrap_port=12345,
                bootstrap_room=1000,
            )
            manager = _FakeTokenizerManager()

            await self._run_request(request, manager)

            self.assertEqual(request.rid, ["request-a", "request-b"])
            self.assertEqual(manager.sent_rooms, [1000, 1002, 1001, 1003])
            self.assertEqual(manager.rid_to_state, {})

        asyncio.run(run_test())

    def test_streaming_and_non_streaming_samples_keep_unique_rooms(self):
        """Both response paths use the same unique room expansion."""

        async def run_test(stream):
            request = GenerateReqInput(
                input_ids=[1, 2],
                sampling_params={"n": 2, "max_new_tokens": 8},
                stream=stream,
                bootstrap_host="127.0.0.1",
                bootstrap_port=12345,
                bootstrap_room=1000,
            )
            manager = _FakeTokenizerManager()

            outputs = await self._run_request(request, manager)

            self.assertEqual(manager.sent_rooms, [1000, 1001])
            self.assertEqual(manager.sent_streams, [stream, stream])
            self.assertEqual(len(outputs), 2 if stream else 1)
            self.assertEqual(manager.rid_to_state, {})

        asyncio.run(run_test(False))
        asyncio.run(run_test(True))


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

    def test_forced_query_registers_modulo_mismatch(self):
        """Forced discovery still supports externally overridden placement."""

        manager = self._manager(attn_dp_rank=2)
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
        manager.record_failure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
