import multiprocessing
import os
import queue
import tempfile
import unittest
from array import array
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import call, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.distributed import communication_op  # noqa: E402
from sglang.srt.managers import scheduler as scheduler_module  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    MMInputsProcessError,
    MMInputsProcessMode,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.scheduler_components import (  # noqa: E402
    request_receiver as receiver_module,
)
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402

register_cpu_ci(est_time=50, suite="base-a-test-cpu")


class _Request:
    def __init__(self, mm_inputs, mode=MMInputsProcessMode.NONE):
        self.mm_inputs = mm_inputs
        self.mm_inputs_process_mode = mode


class _EmbeddingRequest(_Request):
    pass


class _BatchRequest:
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        return iter(self.batch)


class _EmbeddingBatchRequest(_BatchRequest):
    pass


def _group(name, rank=0, ranks=(0, 1)):
    return SimpleNamespace(
        name=name, rank=rank, ranks=list(ranks), world_size=len(ranks)
    )


def _receiver(*, tp_rank=0, attn_tp_rank=0, attn_cp_rank=0):
    receiver = object.__new__(receiver_module.SchedulerRequestReceiver)
    object.__setattr__(
        receiver,
        "ps",
        SimpleNamespace(
            tp_size=2,
            attn_tp_size=2,
            attn_cp_size=2,
            attn_tp_rank=attn_tp_rank,
            attn_cp_rank=attn_cp_rank,
        ),
    )
    object.__setattr__(receiver, "tp_group", _group("tp", rank=tp_rank))
    object.__setattr__(receiver, "tp_cpu_group", "tp-cpu")
    object.__setattr__(receiver, "attn_tp_group", _group("attn-tp", rank=attn_tp_rank))
    object.__setattr__(receiver, "attn_tp_cpu_group", f"attn-tp-cp{attn_cp_rank}")
    object.__setattr__(receiver, "attn_cp_group", _group("attn-cp", rank=attn_cp_rank))
    object.__setattr__(receiver, "attn_cp_cpu_group", f"attn-cp-tp{attn_tp_rank}")
    return receiver


def _run_gloo_mm_broadcast_regression(rank, world_size, init_method, result_queue):
    """Exercise the real TP collective; this function must remain spawn-picklable."""
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=8),
    )
    try:
        receiver = _receiver(tp_rank=rank)
        object.__setattr__(receiver, "tp_cpu_group", dist.group.WORLD)
        request = _Request(
            "rank-0-raw-mm-input" if rank == 0 else None,
            MMInputsProcessMode.BROADCAST,
        )
        with (
            patch.object(receiver_module, "TokenizedGenerateReqInput", _Request),
            patch.object(
                receiver_module, "TokenizedEmbeddingReqInput", _EmbeddingRequest
            ),
            patch.object(
                receiver_module,
                "get_parallel",
                return_value=SimpleNamespace(enable_dp_attention=False),
            ),
            patch.object(receiver_module, "unwrap_shm_features"),
            patch.object(
                receiver_module.MultimodalInputs,
                "from_processor_output",
                side_effect=lambda raw: scheduler_module.MultimodalInputs(
                    mm_items=[{"materialized_from": raw}]
                ),
            ),
        ):
            receiver._set_mm_process_mode([request], only_if_none=True)
            receiver._materialize_broadcast_mm_inputs([request])

        gathered = [None] * world_size
        dist.all_gather_object(gathered, request.mm_inputs.mm_items)
        if rank == 0:
            result_queue.put(("ok", gathered))
    finally:
        dist.destroy_process_group()


def _run_gloo_mm_broadcast_regression_with_watchdog(
    world_size, init_method, result_queue
):
    """Contain a broken collective schedule so the parent test cannot hang."""
    import torch.multiprocessing as torch_mp

    try:
        torch_mp.spawn(
            _run_gloo_mm_broadcast_regression,
            args=(world_size, init_method, result_queue),
            nprocs=world_size,
            join=True,
        )
    except BaseException as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_gloo_dp_attention_mm_broadcast_regression(
    rank, world_size, init_method, result_queue
):
    """Exercise the shared TP-then-CP fanout with real TP2 x CP2 Gloo groups."""
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=8),
    )
    try:
        # All ranks create the same orthogonal subgroups in the same order.
        cp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        tp_rank, cp_rank = rank % 2, rank // 2
        receiver = _receiver(tp_rank=rank, attn_tp_rank=tp_rank, attn_cp_rank=cp_rank)
        object.__setattr__(receiver, "attn_cp_cpu_group", cp_groups[tp_rank])
        object.__setattr__(receiver, "attn_tp_cpu_group", tp_groups[cp_rank])
        object.__setattr__(
            receiver, "attn_cp_group", _group("cp", rank, (tp_rank, tp_rank + 2))
        )
        object.__setattr__(
            receiver,
            "attn_tp_group",
            _group("tp", rank, (cp_rank * 2, cp_rank * 2 + 1)),
        )

        request = _Request(
            "rank-0-raw-mm-input" if rank == 0 else None,
            MMInputsProcessMode.BROADCAST,
        )
        trace = []
        original_broadcast = communication_op.broadcast_pyobj

        def traced_broadcast(data, group_rank, group, src):
            trace.append(
                ("cp" if group is cp_groups[tp_rank] else "tp", group_rank, src)
            )
            return original_broadcast(data, group_rank, group, src)

        with (
            patch.object(receiver_module, "TokenizedGenerateReqInput", _Request),
            patch.object(
                receiver_module, "TokenizedEmbeddingReqInput", _EmbeddingRequest
            ),
            patch.object(
                receiver_module,
                "get_parallel",
                return_value=SimpleNamespace(
                    enable_dp_attention=True,
                    enable_dp_attention_local_control_broadcast=True,
                ),
            ),
            patch.object(receiver_module, "unwrap_shm_features"),
            patch.object(
                receiver_module.MultimodalInputs,
                "from_processor_output",
                side_effect=lambda raw: scheduler_module.MultimodalInputs(
                    mm_items=[{"materialized_from": raw}]
                ),
            ),
            patch.object(
                communication_op, "broadcast_pyobj", side_effect=traced_broadcast
            ),
            patch.object(
                communication_op,
                "get_attn_tp_group",
                return_value=SimpleNamespace(
                    rank=rank,
                    ranks=receiver.attn_tp_group.ranks,
                    world_size=2,
                    cpu_group=tp_groups[cp_rank],
                ),
            ),
            patch.object(
                communication_op,
                "get_attn_cp_group",
                return_value=SimpleNamespace(
                    rank=rank,
                    ranks=receiver.attn_cp_group.ranks,
                    world_size=2,
                    cpu_group=cp_groups[tp_rank],
                ),
            ),
        ):
            requests = receiver._broadcast_reqs_across_ranks(
                [request] if rank == 0 else None
            )
            request = requests[0]
            if rank != 0:
                request.mm_inputs = None
            receiver._materialize_broadcast_mm_inputs(requests)

        gathered = [None] * world_size
        dist.all_gather_object(gathered, (request.mm_inputs.mm_items, trace))
        if rank == 0:
            result_queue.put(("ok", gathered))
    finally:
        dist.destroy_process_group()


def _run_gloo_dp_attention_mm_broadcast_regression_with_watchdog(
    world_size, init_method, result_queue
):
    """Contain a broken CP/TP collective schedule so the parent cannot hang."""
    import torch.multiprocessing as torch_mp

    try:
        torch_mp.spawn(
            _run_gloo_dp_attention_mm_broadcast_regression,
            args=(world_size, init_method, result_queue),
            nprocs=world_size,
            join=True,
        )
    except BaseException as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


class TestMmBroadcastProtocol(unittest.TestCase):
    def test_tp2_gloo_ordered_protocol_materializes_root_raw_input_on_all_ranks(self):
        """A local ``None`` peer must still join the root's ordered MM broadcast.

        The outer process is a watchdog for the historical mismatched-collective
        regression; Gloo's own short timeout supplies a useful child-side error.
        """
        fd, store_path = tempfile.mkstemp(prefix="sglang-mm-broadcast-")
        os.close(fd)
        os.unlink(store_path)
        result_queue = multiprocessing.get_context("spawn").Queue()
        watchdog = multiprocessing.get_context("spawn").Process(
            target=_run_gloo_mm_broadcast_regression_with_watchdog,
            args=(2, f"file://{store_path}", result_queue),
        )
        try:
            watchdog.start()
            # Spawn imports the scheduler in both watchdog and workers before
            # Gloo's separate 8-second collective timeout begins.
            watchdog.join(timeout=60)
            if watchdog.is_alive():
                watchdog.terminate()
                watchdog.join(timeout=5)
                self.fail("2-process Gloo MM broadcast exceeded its watchdog timeout")
            self.assertEqual(watchdog.exitcode, 0)
            try:
                status, payload = result_queue.get(timeout=2)
            except queue.Empty:
                self.fail("2-process Gloo MM broadcast produced no result")
            self.assertEqual(status, "ok", payload)
            self.assertEqual(
                payload,
                [
                    [{"materialized_from": "rank-0-raw-mm-input"}],
                    [{"materialized_from": "rank-0-raw-mm-input"}],
                ],
            )
        finally:
            if watchdog.is_alive():
                watchdog.terminate()
                watchdog.join(timeout=5)
            result_queue.close()
            result_queue.join_thread()
            if os.path.exists(store_path):
                os.unlink(store_path)

    def test_dp_attention_tp2_cp2_gloo_shared_fanout_completes(self):
        """Real TP2 x CP2 groups must fan out one envelope without a hang.

        The shared helper distributes the root envelope along TP, then CP.
        Empty non-root envelopes keep every collective source serializable.
        """
        fd, store_path = tempfile.mkstemp(prefix="sglang-mm-dp-attn-broadcast-")
        os.close(fd)
        os.unlink(store_path)
        result_queue = multiprocessing.get_context("spawn").Queue()
        watchdog = multiprocessing.get_context("spawn").Process(
            target=_run_gloo_dp_attention_mm_broadcast_regression_with_watchdog,
            args=(4, f"file://{store_path}", result_queue),
        )
        try:
            watchdog.start()
            watchdog.join(timeout=60)
            if watchdog.is_alive():
                watchdog.terminate()
                watchdog.join(timeout=5)
                self.fail("TP2 x CP2 Gloo MM fanout exceeded its watchdog timeout")
            self.assertEqual(watchdog.exitcode, 0)
            try:
                status, payload = result_queue.get(timeout=2)
            except queue.Empty:
                self.fail("TP2 x CP2 Gloo MM fanout produced no result")
            self.assertEqual(status, "ok", payload)
            self.assertEqual(
                [mm_items for mm_items, _ in payload],
                [[{"materialized_from": "rank-0-raw-mm-input"}]] * 4,
            )
            self.assertEqual(
                [trace for _, trace in payload],
                [
                    [("tp", 0, 0), ("cp", 0, 0)] * 3,
                    [("tp", 1, 0), ("cp", 1, 1)] * 3,
                    [("tp", 2, 2), ("cp", 2, 0)] * 3,
                    [("tp", 3, 2), ("cp", 3, 1)] * 3,
                ],
            )
        finally:
            if watchdog.is_alive():
                watchdog.terminate()
                watchdog.join(timeout=5)
            result_queue.close()
            result_queue.join_thread()
            if os.path.exists(store_path):
                os.unlink(store_path)

    def test_source_stamps_shared_modes_for_generate_embedding_and_batches(self):
        receiver = object.__new__(receiver_module.SchedulerRequestReceiver)
        image = _Request(object())
        text = _Request(None)
        embedding = _EmbeddingRequest(object())
        batch = _BatchRequest([image])
        embedding_batch = _EmbeddingBatchRequest([embedding])

        with (
            patch.object(receiver_module, "TokenizedGenerateReqInput", _Request),
            patch.object(
                receiver_module, "TokenizedEmbeddingReqInput", _EmbeddingRequest
            ),
            patch.object(
                receiver_module, "BatchTokenizedGenerateReqInput", _BatchRequest
            ),
            patch.object(
                receiver_module,
                "BatchTokenizedEmbeddingReqInput",
                _EmbeddingBatchRequest,
            ),
            patch.object(
                receiver_module,
                "get_mm",
                return_value=SimpleNamespace(
                    enable_broadcast_mm_inputs_process=True,
                    mm_feature_transport="cpu",
                ),
            ),
        ):
            receiver._set_mm_process_mode([text, batch, embedding_batch])

        self.assertIs(text.mm_inputs_process_mode, MMInputsProcessMode.NONE)
        self.assertIs(image.mm_inputs_process_mode, MMInputsProcessMode.BROADCAST)
        self.assertIs(embedding.mm_inputs_process_mode, MMInputsProcessMode.BROADCAST)

    def test_cuda_vmm_uses_local_mode_even_when_broadcast_is_enabled(self):
        receiver = object.__new__(receiver_module.SchedulerRequestReceiver)
        request = _Request(object())
        with (
            patch.object(receiver_module, "TokenizedGenerateReqInput", _Request),
            patch.object(
                receiver_module, "TokenizedEmbeddingReqInput", _EmbeddingRequest
            ),
            patch.object(
                receiver_module,
                "get_mm",
                return_value=SimpleNamespace(
                    enable_broadcast_mm_inputs_process=True,
                    mm_feature_transport="cuda_vmm",
                ),
            ),
        ):
            receiver._set_mm_process_mode([request])
        self.assertIs(request.mm_inputs_process_mode, MMInputsProcessMode.LOCAL)

    def test_mixed_work_list_uses_one_ordered_collective_per_broadcast_item(self):
        receiver = _receiver()
        first = _Request("first", MMInputsProcessMode.BROADCAST)
        text = _Request(None)
        local = _Request("local", MMInputsProcessMode.LOCAL)
        second = _EmbeddingRequest("second", MMInputsProcessMode.BROADCAST)
        requests = [first, text, _BatchRequest([local, second])]
        outputs = [
            scheduler_module.MultimodalInputs(mm_items=[]),
            scheduler_module.MultimodalInputs(mm_items=[]),
        ]
        with (
            patch.object(receiver_module, "TokenizedGenerateReqInput", _Request),
            patch.object(
                receiver_module, "TokenizedEmbeddingReqInput", _EmbeddingRequest
            ),
            patch.object(
                receiver_module, "BatchTokenizedGenerateReqInput", _BatchRequest
            ),
            patch.object(
                receiver_module,
                "BatchTokenizedEmbeddingReqInput",
                _EmbeddingBatchRequest,
            ),
            patch.object(
                receiver_module,
                "get_parallel",
                return_value=SimpleNamespace(enable_dp_attention=False),
            ),
            patch.object(
                receiver_module.MultimodalInputs,
                "from_processor_output",
                side_effect=outputs,
            ) as materialize,
            patch.object(
                receiver_module,
                "broadcast_pyobj",
                side_effect=lambda data, *args, **kwargs: data,
            ) as broadcast,
        ):
            receiver._materialize_broadcast_mm_inputs(requests)

        self.assertEqual(materialize.call_args_list, [call("first"), call("second")])
        self.assertEqual(broadcast.call_count, 2)
        self.assertIs(first.mm_inputs, outputs[0])
        self.assertIs(second.mm_inputs, outputs[1])
        self.assertEqual(local.mm_inputs, "local")

    def test_epd_waiting_success_refill_recomputes_mode_before_protocol(self):
        class WaitingReceiver:
            def __init__(self, refilled_mm_inputs):
                self.refilled_mm_inputs = refilled_mm_inputs

            def process_waiting_requests(self, requests):
                for request in requests:
                    request.mm_inputs = self.refilled_mm_inputs
                return requests, []

        for backend in ("zmq_to_scheduler", "mooncake"):
            with self.subTest(backend=backend):
                receivers = [_receiver(tp_rank=rank) for rank in range(2)]
                requests = [_Request(None), _Request(None)]
                raw_refill = object()
                for receiver in receivers:
                    receiver.ps.pp_rank = 0
                    object.__setattr__(
                        receiver, "mm_receiver", WaitingReceiver(raw_refill)
                    )
                    object.__setattr__(
                        receiver, "stream_output", lambda *args, **kwargs: None
                    )

                with (
                    patch.object(
                        receiver_module, "TokenizedGenerateReqInput", _Request
                    ),
                    patch.object(
                        receiver_module,
                        "TokenizedEmbeddingReqInput",
                        _EmbeddingRequest,
                    ),
                    patch.object(
                        receiver_module,
                        "get_disagg",
                        return_value=SimpleNamespace(
                            language_only=True,
                            encoder_transfer_backend=backend,
                        ),
                    ),
                    patch.object(
                        receiver_module,
                        "get_mm",
                        return_value=SimpleNamespace(
                            enable_broadcast_mm_inputs_process=True,
                            mm_feature_transport="cpu",
                        ),
                    ),
                ):
                    for receiver, request in zip(receivers, requests):
                        receiver._set_mm_process_mode([request])
                        self.assertIs(
                            request.mm_inputs_process_mode,
                            MMInputsProcessMode.NONE,
                        )
                        refilled = receiver._apply_mm_receiver([request])
                        receiver._set_mm_process_mode(refilled, only_if_none=True)

                self.assertTrue(
                    all(
                        request.mm_inputs_process_mode is MMInputsProcessMode.BROADCAST
                        for request in requests
                    )
                )

    def test_dp_attention_reuses_request_fanout_with_empty_peer_envelopes(self):
        output = scheduler_module.MultimodalInputs(mm_items=[])
        envelope = (output, None)
        for tp_rank, cp_rank in ((0, 0), (0, 1), (1, 0), (1, 1)):
            with self.subTest(tp_rank=tp_rank, cp_rank=cp_rank):
                receiver = _receiver(attn_tp_rank=tp_rank, attn_cp_rank=cp_rank)
                request = _Request("image", MMInputsProcessMode.BROADCAST)
                with (
                    patch.object(
                        receiver_module, "TokenizedGenerateReqInput", _Request
                    ),
                    patch.object(
                        receiver_module,
                        "get_parallel",
                        return_value=SimpleNamespace(enable_dp_attention=True),
                    ),
                    patch.object(
                        receiver_module.MultimodalInputs,
                        "from_processor_output",
                        return_value=output,
                    ) as materialize,
                    patch.object(
                        receiver_module,
                        "attn_cp_tp_broadcast_pyobj",
                        return_value=[envelope],
                    ) as broadcast,
                ):
                    receiver._materialize_broadcast_mm_inputs([request])
                is_source = tp_rank == 0 and cp_rank == 0
                broadcast.assert_called_once_with([envelope] if is_source else [])
                self.assertEqual(materialize.call_count, int(is_source))
                self.assertIs(request.mm_inputs, output)

    def test_root_error_becomes_consistent_abort_sentinel(self):
        receiver = _receiver()
        request = _Request("bad", MMInputsProcessMode.BROADCAST)
        with self.assertLogs(receiver_module.logger, level="ERROR") as logs:
            with (
                patch.object(receiver_module, "TokenizedGenerateReqInput", _Request),
                patch.object(
                    receiver_module, "TokenizedEmbeddingReqInput", _EmbeddingRequest
                ),
                patch.object(
                    receiver_module,
                    "get_parallel",
                    return_value=SimpleNamespace(enable_dp_attention=False),
                ),
                patch.object(
                    receiver_module.MultimodalInputs,
                    "from_processor_output",
                    side_effect=ValueError("bad image"),
                ),
                patch.object(
                    receiver_module,
                    "broadcast_pyobj",
                    side_effect=lambda data, *args, **kwargs: data,
                ) as broadcast,
            ):
                receiver._materialize_broadcast_mm_inputs([request])

        self.assertEqual(broadcast.call_count, 1)
        self.assertIn(
            "Failed to materialize broadcast multimodal inputs on entry rank",
            logs.output[0],
        )
        self.assertIn("ValueError: bad image", logs.output[0])
        self.assertIsInstance(request.mm_inputs, MMInputsProcessError)
        self.assertIs(request.mm_inputs_process_mode, MMInputsProcessMode.LOCAL)
        self.assertIn("ValueError: bad image", request.mm_inputs.message)
        scheduler = object.__new__(scheduler_module.Scheduler)
        with self.assertRaisesRegex(
            scheduler_module._MultimodalInputProcessingError, "ValueError: bad image"
        ):
            scheduler._get_multimodal_inputs(
                request.mm_inputs, MMInputsProcessMode.LOCAL
            )

    def test_peer_consumes_ordered_error_envelope_without_materializing(self):
        receiver = _receiver(tp_rank=1)
        request = _Request(None, MMInputsProcessMode.BROADCAST)
        envelope = (None, "Failed to process multimodal inputs on entry rank: bad")
        with self.assertNoLogs(receiver_module.logger, level="ERROR"):
            with (
                patch.object(receiver_module, "TokenizedGenerateReqInput", _Request),
                patch.object(
                    receiver_module, "TokenizedEmbeddingReqInput", _EmbeddingRequest
                ),
                patch.object(
                    receiver_module,
                    "get_parallel",
                    return_value=SimpleNamespace(enable_dp_attention=False),
                ),
                patch.object(
                    receiver_module.MultimodalInputs, "from_processor_output"
                ) as materialize,
                patch.object(receiver_module, "broadcast_pyobj", return_value=envelope),
            ):
                receiver._materialize_broadcast_mm_inputs([request])

        materialize.assert_not_called()
        self.assertIsInstance(request.mm_inputs, MMInputsProcessError)
        self.assertIs(request.mm_inputs_process_mode, MMInputsProcessMode.LOCAL)
        self.assertEqual(request.mm_inputs.message, envelope[1])

    def test_shm_unwrap_failure_is_sent_as_an_ordered_error_envelope(self):
        receiver = _receiver()
        request = _Request("shm", MMInputsProcessMode.BROADCAST)
        with (
            patch.object(receiver_module, "TokenizedGenerateReqInput", _Request),
            patch.object(
                receiver_module, "TokenizedEmbeddingReqInput", _EmbeddingRequest
            ),
            patch.object(
                receiver_module,
                "get_parallel",
                return_value=SimpleNamespace(enable_dp_attention=False),
            ),
            patch.object(
                receiver_module, "unwrap_shm_features", side_effect=OSError("shm gone")
            ),
            patch.object(receiver_module, "has_shm_features", return_value=True),
            patch.object(
                receiver_module.MultimodalInputs, "from_processor_output"
            ) as materialize,
            patch.object(
                receiver_module,
                "broadcast_pyobj",
                side_effect=lambda data, *args, **kwargs: data,
            ),
        ):
            receiver._materialize_broadcast_mm_inputs([request])

        materialize.assert_not_called()
        self.assertIsInstance(request.mm_inputs, MMInputsProcessError)
        self.assertIn("OSError: shm gone", request.mm_inputs.message)

    def test_pp_downstream_preserves_error_sentinel_without_shm_access(self):
        receiver = _receiver()
        object.__setattr__(
            receiver, "model_config", SimpleNamespace(is_multimodal=True)
        )
        sentinel = MMInputsProcessError("upstream failure")
        request = TokenizedGenerateReqInput(
            input_text="",
            input_ids=array("q", [1]),
            input_embeds=None,
            mm_inputs=sentinel,
            token_type_ids=None,
            sampling_params=SamplingParams(),
            return_logprob=False,
            logprob_start_len=0,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            mm_inputs_process_mode=MMInputsProcessMode.BROADCAST,
        )
        with patch.object(receiver_module, "broadcast_pyobj") as broadcast:
            receiver._set_mm_process_mode([request])
            receiver._finalize_shm_features([request])
            receiver._materialize_broadcast_mm_inputs([request])

        self.assertIs(request.mm_inputs, sentinel)
        self.assertIs(request.mm_inputs_process_mode, MMInputsProcessMode.LOCAL)
        broadcast.assert_not_called()

    def test_pp_downstream_preserves_prepared_mm_inputs_without_rebroadcasting(self):
        receiver = _receiver()
        prepared = scheduler_module.MultimodalInputs(mm_items=[])
        request = _Request(prepared, MMInputsProcessMode.BROADCAST)
        with (
            patch.object(receiver_module, "TokenizedGenerateReqInput", _Request),
            patch.object(
                receiver_module, "TokenizedEmbeddingReqInput", _EmbeddingRequest
            ),
            patch.object(receiver_module, "broadcast_pyobj") as broadcast,
        ):
            receiver._set_mm_process_mode([request])
            receiver._materialize_broadcast_mm_inputs([request])

        self.assertIs(request.mm_inputs_process_mode, MMInputsProcessMode.LOCAL)
        self.assertIs(request.mm_inputs, prepared)
        broadcast.assert_not_called()

    def test_text_requests_add_no_mm_collective(self):
        receiver = _receiver()
        text = _Request(None, MMInputsProcessMode.NONE)
        with patch.object(receiver_module, "broadcast_pyobj") as broadcast:
            receiver._materialize_broadcast_mm_inputs([text])
        broadcast.assert_not_called()


if __name__ == "__main__":
    unittest.main()
