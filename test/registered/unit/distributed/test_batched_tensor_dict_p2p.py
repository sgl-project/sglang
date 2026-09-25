import multiprocessing
import pickle
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.distributed.parallel_state as parallel_state
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import find_available_port

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _coordinator():
    coordinator = parallel_state.GroupCoordinator.__new__(
        parallel_state.GroupCoordinator
    )
    coordinator.world_size = 2
    coordinator.rank_in_group = 0
    coordinator.ranks = [0, 1]
    coordinator.device_group = object()
    coordinator.cpu_group = object()
    coordinator.send_object = MagicMock(return_value=[])
    coordinator.recv_object = MagicMock()
    return coordinator


class _DeferredWork:
    def __init__(self, complete_callback=None):
        self._complete_callback = complete_callback
        self._completed = False
        self.wait_count = 0

    def complete(self):
        if self._complete_callback is not None:
            self._complete_callback()
        self._completed = True

    def is_completed(self):
        return self._completed

    def wait(self, timeout=None):
        self.wait_count += 1
        if not self._completed:
            if timeout is not None:
                raise RuntimeError("Operation timed out")
            raise AssertionError("poll path blocked on incomplete work")
        return True


def _run_async_tensor_dict_gloo(rank, port, output):
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    coordinator = parallel_state.GroupCoordinator.__new__(
        parallel_state.GroupCoordinator
    )
    coordinator.world_size = 2
    coordinator.rank_in_group = rank
    coordinator.ranks = [0, 1]
    coordinator.device_group = torch.distributed.group.WORLD
    coordinator.cpu_group = torch.distributed.group.WORLD
    try:
        if rank == 0:
            time.sleep(0.1)
            works = coordinator.send_tensor_dict(
                {
                    "hidden_states": torch.arange(4, dtype=torch.float32),
                    "vpp_batch_seq": 3,
                    "vpp_stage_id": 5,
                },
                dst=1,
                async_send=True,
                batch_p2p=True,
                tag=79,
            )
            work_count = len(works)
            send_handle = parallel_state.P2PWorkGroup(works)
            deadline = time.monotonic() + 10
            while not send_handle.poll() and time.monotonic() < deadline:
                time.sleep(0.001)
            if send_handle.items:
                output.put(("timeout", rank, work_count, "send"))
                return
            output.put(("sent", work_count))
        else:
            handle = coordinator.recv_tensor_dict_async(
                src=0,
                batch_p2p=True,
                tag=79,
            )
            polls = 0
            result = None
            deadline = time.monotonic() + 10
            while result is None and time.monotonic() < deadline:
                result = handle.poll()
                polls += 1
                if result is None:
                    time.sleep(0.001)
            if result is None:
                output.put(("timeout", rank, polls, handle._state))
                return
            output.put(
                (
                    "received",
                    polls,
                    result["hidden_states"].tolist(),
                    result["vpp_batch_seq"],
                    result["vpp_stage_id"],
                )
            )
    except Exception as exc:
        output.put(("error", rank, type(exc).__name__, str(exc)))
        raise
    finally:
        torch.distributed.destroy_process_group()


class TestBatchedTensorDictP2P(unittest.TestCase):
    def test_default_send_keeps_unbatched_path(self):
        coordinator = _coordinator()
        tensor = torch.arange(4)
        send_work = MagicMock()

        with (
            patch.object(
                parallel_state.torch.distributed,
                "isend",
                return_value=send_work,
            ) as isend,
            patch.object(
                parallel_state.torch.distributed,
                "batch_isend_irecv",
            ) as batch,
        ):
            result = parallel_state.GroupCoordinator.send_tensor_dict(
                coordinator,
                {"hidden_states": tensor},
                async_send=True,
            )

        isend.assert_called_once_with(tensor, 1, group=coordinator.cpu_group)
        batch.assert_not_called()
        self.assertIs(result[0].work, send_work)
        self.assertIs(result[0].payload, tensor)

    def test_send_batches_all_tensor_operations(self):
        coordinator = _coordinator()
        tensors = {
            "hidden_states": torch.arange(4),
            "prev_pre": torch.arange(2),
            "stage": 1,
        }
        works = [MagicMock(), MagicMock()]

        with (
            patch.object(
                parallel_state.torch.distributed,
                "P2POp",
                side_effect=lambda op, tensor, peer, group: SimpleNamespace(
                    op=op,
                    tensor=tensor,
                    peer=peer,
                    group=group,
                ),
            ),
            patch.object(
                parallel_state.torch.distributed,
                "batch_isend_irecv",
                return_value=works,
            ) as batch,
            patch.object(parallel_state.torch.distributed, "isend") as isend,
        ):
            result = parallel_state.GroupCoordinator.send_tensor_dict(
                coordinator,
                tensors,
                async_send=True,
                batch_p2p=True,
            )

        batch.assert_called_once()
        isend.assert_not_called()
        self.assertEqual([work.work for work in result], works)
        self.assertEqual(
            [work.payload for work in result],
            [tensors["hidden_states"], tensors["prev_pre"]],
        )

    def test_send_materializes_non_dense_tensor_payloads(self):
        coordinator = _coordinator()
        source = torch.arange(4).expand(3, 4)
        self.assertFalse(source.is_contiguous())
        work = MagicMock()

        with (
            patch.object(
                parallel_state.torch.distributed,
                "P2POp",
                side_effect=lambda op, tensor, peer, group: SimpleNamespace(
                    op=op,
                    tensor=tensor,
                    peer=peer,
                    group=group,
                ),
            ),
            patch.object(
                parallel_state.torch.distributed,
                "batch_isend_irecv",
                return_value=[work],
            ) as batch,
        ):
            result = parallel_state.GroupCoordinator.send_tensor_dict(
                coordinator,
                {"hidden_states": source},
                async_send=True,
                batch_p2p=True,
            )

        sent = batch.call_args.args[0][0].tensor
        self.assertTrue(sent.is_contiguous())
        torch.testing.assert_close(sent, source)
        self.assertIs(result[0].payload, sent)
        coordinator.send_object.assert_called_once_with(
            [
                (
                    "hidden_states",
                    parallel_state.TensorMetadata(
                        "cpu",
                        source.dtype,
                        source.size(),
                    ),
                )
            ],
            dst=1,
            async_send=True,
            tag=0,
        )

    def test_recv_posts_all_tensor_operations_before_waiting(self):
        coordinator = _coordinator()
        coordinator.recv_object.return_value = [
            (
                "hidden_states",
                parallel_state.TensorMetadata("cpu", torch.float32, torch.Size([4])),
            ),
            (
                "prev_pre",
                parallel_state.TensorMetadata("cpu", torch.float32, torch.Size([2])),
            ),
            ("stage", 1),
        ]
        works = [MagicMock(), MagicMock()]

        def p2p_op(op, tensor, peer, group):
            return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

        def batch_p2p(ops):
            for value, op in enumerate(ops, start=1):
                op.tensor.fill_(value)
            return works

        with (
            patch.object(
                parallel_state.torch.distributed,
                "is_initialized",
                return_value=True,
            ),
            patch.object(
                parallel_state.torch.distributed,
                "P2POp",
                side_effect=p2p_op,
            ),
            patch.object(
                parallel_state.torch.distributed,
                "batch_isend_irecv",
                side_effect=batch_p2p,
            ) as batch,
            patch.object(parallel_state.torch.distributed, "irecv") as irecv,
        ):
            result = parallel_state.GroupCoordinator.recv_tensor_dict(
                coordinator,
                batch_p2p=True,
            )

        batch.assert_called_once()
        irecv.assert_not_called()
        for work in works:
            work.wait.assert_called_once_with()
        torch.testing.assert_close(result["hidden_states"], torch.ones(4))
        torch.testing.assert_close(result["prev_pre"], torch.full((2,), 2.0))
        self.assertEqual(result["stage"], 1)

    def test_send_batches_propagates_tag_to_metadata_and_payload(self):
        coordinator = _coordinator()
        tensor = torch.arange(4)
        coordinator.send_object.return_value = []

        with (
            patch.object(
                parallel_state.torch.distributed,
                "P2POp",
                side_effect=lambda op, value, peer, group, tag=0: SimpleNamespace(
                    op=op,
                    tensor=value,
                    peer=peer,
                    group=group,
                    tag=tag,
                ),
            ),
            patch.object(
                parallel_state.torch.distributed,
                "batch_isend_irecv",
                return_value=[MagicMock()],
            ) as batch,
        ):
            parallel_state.GroupCoordinator.send_tensor_dict(
                coordinator,
                {"hidden_states": tensor},
                async_send=True,
                batch_p2p=True,
                tag=37,
            )

        coordinator.send_object.assert_called_once_with(
            [
                (
                    "hidden_states",
                    parallel_state.TensorMetadata("cpu", tensor.dtype, tensor.size()),
                )
            ],
            dst=1,
            async_send=True,
            tag=37,
        )
        self.assertEqual(batch.call_args.args[0][0].tag, 37)

    def test_async_recv_polls_metadata_and_payload_without_blocking(self):
        coordinator = _coordinator()
        metadata = [
            (
                "hidden_states",
                parallel_state.TensorMetadata("cpu", torch.float32, torch.Size([4])),
            ),
            ("vpp_batch_seq", 3),
            ("vpp_stage_id", 5),
        ]
        metadata_bytes = pickle.dumps(metadata)
        recv_works = []
        payload_work = None

        def irecv(tensor, *, src, group, tag):
            nonlocal payload_work
            call_index = len(recv_works)
            if call_index == 0:

                def callback():
                    tensor.fill_(len(metadata_bytes))
            elif call_index == 1:

                def callback():
                    tensor.copy_(torch.tensor(list(metadata_bytes), dtype=torch.uint8))
            else:
                payload_work = _DeferredWork(lambda: tensor.fill_(7))
                recv_works.append((payload_work, src, group, tag))
                return payload_work
            work = _DeferredWork(callback)
            recv_works.append((work, src, group, tag))
            return work

        with (
            patch.object(
                parallel_state.torch.distributed,
                "is_initialized",
                return_value=True,
            ),
            patch.object(
                parallel_state.torch.distributed,
                "irecv",
                side_effect=irecv,
            ),
            patch.object(
                parallel_state.torch.distributed,
                "P2POp",
                side_effect=lambda op, tensor, peer, group, tag=0: SimpleNamespace(
                    op=op,
                    tensor=tensor,
                    peer=peer,
                    group=group,
                    tag=tag,
                ),
            ),
            patch.object(
                parallel_state.torch.distributed,
                "batch_isend_irecv",
            ) as batch_p2p,
        ):
            handle = parallel_state.GroupCoordinator.recv_tensor_dict_async(
                coordinator,
                batch_p2p=True,
                tag=53,
            )

            self.assertIsNone(handle.poll())
            self.assertEqual(len(recv_works), 1)
            recv_works[0][0].complete()
            self.assertIsNone(handle.poll())
            self.assertEqual(len(recv_works), 2)
            recv_works[1][0].complete()
            self.assertIsNone(handle.poll())
            self.assertIsNotNone(payload_work)
            payload_work.complete()
            result = handle.poll()
            repeated_result = handle.poll()

        batch_p2p.assert_not_called()
        self.assertEqual([item[3] for item in recv_works], [53, 53, 53])
        self.assertEqual(recv_works[0][0].wait_count, 1)
        self.assertEqual(recv_works[1][0].wait_count, 1)
        self.assertEqual(payload_work.wait_count, 1)
        torch.testing.assert_close(result["hidden_states"], torch.full((4,), 7.0))
        self.assertEqual(result["vpp_batch_seq"], 3)
        self.assertEqual(result["vpp_stage_id"], 5)
        self.assertIs(repeated_result, result)

    def test_async_recv_keeps_tp_all_gather_pollable(self):
        coordinator = _coordinator()
        tp_group = SimpleNamespace(
            world_size=2,
            rank_in_group=1,
            cpu_group=object(),
            device_group=object(),
        )
        metadata = [
            (
                "hidden_states",
                parallel_state.TensorMetadata("cpu", torch.float32, torch.Size([4])),
            )
        ]
        metadata_bytes = pickle.dumps(metadata)
        recv_works = []
        payload_work = None
        gather_work = None

        def irecv(tensor, *, src, group, tag):
            nonlocal payload_work
            call_index = len(recv_works)
            if call_index == 0:

                def callback():
                    tensor.fill_(len(metadata_bytes))
            elif call_index == 1:

                def callback():
                    tensor.copy_(torch.tensor(list(metadata_bytes), dtype=torch.uint8))
            else:
                self.assertEqual(tensor.numel(), 2)
                payload_work = _DeferredWork(lambda: tensor.fill_(5))
                recv_works.append(payload_work)
                return payload_work
            work = _DeferredWork(callback)
            recv_works.append(work)
            return work

        def all_gather(output, input_, *, group, async_op):
            nonlocal gather_work
            self.assertEqual(input_.numel(), 2)
            self.assertIs(group, tp_group.cpu_group)
            self.assertTrue(async_op)
            gather_work = _DeferredWork(
                lambda: output.copy_(torch.tensor([1.0, 2.0, 5.0, 5.0]))
            )
            return gather_work

        with (
            patch.object(
                parallel_state.torch.distributed,
                "is_initialized",
                return_value=True,
            ),
            patch.object(
                parallel_state.torch.distributed,
                "irecv",
                side_effect=irecv,
            ),
            patch.object(
                parallel_state.torch.distributed,
                "P2POp",
                side_effect=lambda op, tensor, peer, group, tag=0: SimpleNamespace(
                    op=op,
                    tensor=tensor,
                    peer=peer,
                    group=group,
                    tag=tag,
                ),
            ),
            patch.object(
                parallel_state.torch.distributed,
                "batch_isend_irecv",
            ) as batch_p2p,
            patch.object(
                parallel_state.torch.distributed,
                "all_gather_into_tensor",
                side_effect=all_gather,
            ),
        ):
            handle = parallel_state.GroupCoordinator.recv_tensor_dict_async(
                coordinator,
                all_gather_group=tp_group,
                batch_p2p=True,
                tag=61,
            )
            recv_works[0].complete()
            self.assertIsNone(handle.poll())
            recv_works[1].complete()
            self.assertIsNone(handle.poll())
            payload_work.complete()
            self.assertTrue(handle.poll_payload_ready())
            self.assertIsNone(gather_work)
            handle.start_all_gather()
            self.assertFalse(handle.poll_all_gather())
            gather_work.complete()
            self.assertTrue(handle.poll_all_gather())
            result = handle.result()

        torch.testing.assert_close(
            result["hidden_states"],
            torch.tensor([1.0, 2.0, 5.0, 5.0]),
        )
        batch_p2p.assert_not_called()
        self.assertEqual(gather_work.wait_count, 1)

    def test_async_recv_completes_without_payload_work_for_empty_tensor(self):
        coordinator = _coordinator()
        metadata = [
            (
                "empty",
                parallel_state.TensorMetadata("cpu", torch.float32, torch.Size([0])),
            ),
            ("stage", 1),
        ]
        metadata_bytes = pickle.dumps(metadata)
        recv_works = []

        def irecv(tensor, *, src, group, tag):
            callback = (
                (lambda: tensor.fill_(len(metadata_bytes)))
                if not recv_works
                else (
                    lambda: tensor.copy_(
                        torch.tensor(list(metadata_bytes), dtype=torch.uint8)
                    )
                )
            )
            work = _DeferredWork(callback)
            recv_works.append(work)
            return work

        with (
            patch.object(
                parallel_state.torch.distributed,
                "is_initialized",
                return_value=True,
            ),
            patch.object(
                parallel_state.torch.distributed,
                "irecv",
                side_effect=irecv,
            ),
            patch.object(
                parallel_state.torch.distributed,
                "batch_isend_irecv",
            ) as batch_p2p,
        ):
            handle = parallel_state.GroupCoordinator.recv_tensor_dict_async(
                coordinator,
                batch_p2p=True,
                tag=67,
            )
            recv_works[0].complete()
            self.assertIsNone(handle.poll())
            recv_works[1].complete()
            result = handle.poll()

        batch_p2p.assert_not_called()
        self.assertEqual(result["stage"], 1)
        self.assertEqual(result["empty"].numel(), 0)
        self.assertTrue(handle.is_completed())

    def test_async_recv_rejects_invalid_metadata_size(self):
        coordinator = _coordinator()
        size_work = _DeferredWork(lambda: None)

        with (
            patch.object(
                parallel_state.torch.distributed,
                "is_initialized",
                return_value=True,
            ),
            patch.object(
                parallel_state.torch.distributed,
                "irecv",
                return_value=size_work,
            ),
        ):
            handle = parallel_state.GroupCoordinator.recv_tensor_dict_async(
                coordinator,
                tag=71,
            )
            size_work.complete()
            with self.assertRaisesRegex(RuntimeError, "metadata size"):
                handle.poll()

    def test_async_recv_with_real_gloo_transport(self):
        context = multiprocessing.get_context("spawn")
        output = context.Queue()
        port = find_available_port(29690)
        processes = [
            context.Process(
                target=_run_async_tensor_dict_gloo,
                args=(rank, port, output),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()

        try:
            messages = [output.get(timeout=90) for _ in processes]
        finally:
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
        for process in processes:
            self.assertEqual(process.exitcode, 0)

        self.assertFalse(
            [message for message in messages if message[0] in ("error", "timeout")],
            messages,
        )
        received = next(message for message in messages if message[0] == "received")
        self.assertGreater(received[1], 1)
        self.assertEqual(received[2:], ([0.0, 1.0, 2.0, 3.0], 3, 5))


if __name__ == "__main__":
    unittest.main()
