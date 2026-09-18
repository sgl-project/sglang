import threading
import unittest

import numpy as np
from sglang.srt.disaggregation.layerwise_kv import (
    LayerwiseKVController,
    LayerwiseKVJob,
    LayerwiseStateJob,
    reserve_layerwise_kv_chunk,
)


class FakeProgress:
    def __init__(self):
        self.values = {}

    def completed_layers(self, generation):
        return self.values.get(generation, 0)


class FakeSubmitter:
    def __init__(self):
        self.calls = []

    def submit_layerwise_kv(self, job, layer_slots):
        self.calls.append((job.key, list(layer_slots)))
        return f"handle-{len(self.calls)}"

    def submit_layerwise_state(self, job, layer_slots):
        self.calls.append((job.key, list(layer_slots)))
        return f"handle-{len(self.calls)}"


class BlockingStateSubmitter(FakeSubmitter):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def submit_layerwise_state(self, job, layer_slots):
        self.entered.set()
        self.release.wait(timeout=2)
        return super().submit_layerwise_state(job, layer_slots)


class BlockingSubmitter(FakeSubmitter):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def submit_layerwise_kv(self, job, layer_slots):
        self.entered.set()
        self.release.wait(timeout=2)
        return super().submit_layerwise_kv(job, layer_slots)


def make_job(generation=1, chunk_id=0):
    return LayerwiseKVJob(
        room=7,
        chunk_id=chunk_id,
        agent_name="decode-0",
        generation=generation,
        page_indices=np.array([2, 5], dtype=np.int32),
        dst_page_indices=np.array([11, 12], dtype=np.int32),
    )


def make_state_job(generation=1, src_state_index=3, dst_state_index=9):
    return LayerwiseStateJob(
        room=7,
        agent_name="decode-0",
        generation=generation,
        src_state_index=src_state_index,
        dst_state_index=dst_state_index,
    )


class FakeSender:
    def __init__(self):
        self.chunk_id = 0
        self.curr_idx = 0


class TestLayerwiseKVReservation(unittest.TestCase):
    def test_reserves_ahead_of_sender_for_overlap(self):
        sender = FakeSender()
        first = reserve_layerwise_kv_chunk(sender, 0, 8, 2)
        second = reserve_layerwise_kv_chunk(sender, 8, 20, 3)
        self.assertEqual(first.chunk_id, 0)
        self.assertEqual((first.index_slice.start, first.index_slice.stop), (0, 2))
        self.assertEqual(second.chunk_id, 1)
        self.assertEqual((second.index_slice.start, second.index_slice.stop), (2, 5))

        sender.chunk_id = 2
        sender.curr_idx = 5
        third = reserve_layerwise_kv_chunk(sender, 20, 24, 1)
        self.assertEqual(third.chunk_id, 2)
        self.assertEqual((third.index_slice.start, third.index_slice.stop), (5, 6))

    def test_rejects_start_cursor_drift(self):
        sender = FakeSender()
        reserve_layerwise_kv_chunk(sender, 0, 8, 2)
        with self.assertRaises(ValueError):
            reserve_layerwise_kv_chunk(sender, 7, 12, 1)


class TestLayerwiseKVController(unittest.TestCase):
    def make_controller(
        self,
        submitter=None,
        submit_batch=2,
        layer_to_state_slot=None,
        num_state_layers=0,
        state_submit_batch=2,
    ):
        self.progress = FakeProgress()
        self.submitter = submitter or FakeSubmitter()
        return LayerwiseKVController(
            submitter=self.submitter,
            progress=self.progress,
            layer_to_kv_slot=[-1, 0, -1, 1, -1, 2],
            num_kv_layers=3,
            submit_batch=submit_batch,
            layer_to_state_slot=layer_to_state_slot,
            num_state_layers=num_state_layers,
            state_submit_batch=state_submit_batch,
            start_thread=False,
        )

    def test_rejects_non_compact_mapping(self):
        with self.assertRaises(ValueError):
            LayerwiseKVController(
                FakeSubmitter(),
                FakeProgress(),
                layer_to_kv_slot=[0, 2],
                num_kv_layers=2,
                submit_batch=1,
                start_thread=False,
            )

    def test_batches_ready_compact_slots(self):
        controller = self.make_controller(submit_batch=2)
        job = make_job()
        controller.arm([job])

        self.progress.values[1] = 2
        controller.drive_once()
        self.assertEqual(self.submitter.calls, [])

        self.progress.values[1] = 4
        controller.drive_once()
        self.assertEqual(self.submitter.calls, [(job.key, [0, 1])])

        self.progress.values[1] = 6
        controller.drive_once()
        self.assertEqual(self.submitter.calls[-1], (job.key, [2]))

        drained = controller.drain(
            *job.key, job.page_indices.copy(), job.dst_page_indices.copy()
        )
        self.assertEqual(drained.missing_slots, [])
        self.assertEqual(drained.handles, ["handle-1", "handle-2"])

    def test_retired_job_can_be_drained_after_next_arm(self):
        controller = self.make_controller()
        old_job = make_job(generation=1, chunk_id=0)
        new_job = make_job(generation=2, chunk_id=1)
        controller.arm([old_job])
        controller.arm([new_job])

        drained = controller.drain(
            *old_job.key, old_job.page_indices, old_job.dst_page_indices
        )
        self.assertEqual(drained.missing_slots, [0, 1, 2])

    def test_plan_mismatch_requires_full_fallback(self):
        controller = self.make_controller()
        job = make_job()
        controller.arm([job])
        self.progress.values[1] = 4
        controller.drive_once()

        drained = controller.drain(
            *job.key,
            np.array([99], dtype=np.int32),
            job.dst_page_indices,
        )
        self.assertIsNone(drained.missing_slots)
        self.assertEqual(drained.handles, ["handle-1"])

    def test_close_room_collects_active_and_retired_handles(self):
        controller = self.make_controller(submit_batch=1)
        first = make_job(generation=1, chunk_id=0)
        second = make_job(generation=2, chunk_id=1)
        controller.arm([first])
        self.progress.values[1] = 2
        controller.drive_once()
        controller.arm([second])
        self.progress.values[2] = 2
        controller.drive_once()

        self.assertEqual(sorted(controller.close_room(7)), ["handle-1", "handle-2"])
        controller.drive_once([first, second])
        self.assertEqual(len(self.submitter.calls), 2)

    def test_state_batches_and_tail_flush(self):
        controller = self.make_controller(
            layer_to_state_slot=[0, -1, 1, -1, 2, -1],
            num_state_layers=3,
            state_submit_batch=2,
        )
        job = make_state_job()
        controller.arm_state([job])

        self.progress.values[1] = 1
        controller.drive_once()
        self.assertEqual(self.submitter.calls, [])

        self.progress.values[1] = 3
        controller.drive_once()
        self.assertEqual(self.submitter.calls, [(job.key, [0, 1])])

        self.progress.values[1] = 6
        controller.drive_once()
        self.assertEqual(self.submitter.calls[-1], (job.key, [2]))
        drained = controller.drain_state(7, "decode-0", 3, 9)
        self.assertEqual(drained.missing_slots, [])
        self.assertEqual(drained.handles, ["handle-1", "handle-2"])

    def test_state_plan_mismatch_requires_full_fallback(self):
        controller = self.make_controller(
            layer_to_state_slot=[0, -1, 1, -1, 2, -1],
            num_state_layers=3,
            state_submit_batch=1,
        )
        job = make_state_job()
        controller.arm_state([job])
        self.progress.values[1] = 1
        controller.drive_once()

        drained = controller.drain_state(7, "decode-0", 4, 9)
        self.assertIsNone(drained.missing_slots)
        self.assertEqual(drained.handles, ["handle-1"])

    def test_state_drain_waits_for_inflight_submit_and_closes_job(self):
        submitter = BlockingStateSubmitter()
        controller = self.make_controller(
            submitter=submitter,
            layer_to_state_slot=[0, -1, 1, -1, 2, -1],
            num_state_layers=3,
            state_submit_batch=1,
        )
        job = make_state_job()
        controller.arm_state([job])
        self.progress.values[1] = 1

        drive_thread = threading.Thread(target=controller.drive_once)
        drive_thread.start()
        self.assertTrue(submitter.entered.wait(timeout=1))

        result = []
        drain_thread = threading.Thread(
            target=lambda: result.append(controller.drain_state(7, "decode-0", 3, 9))
        )
        drain_thread.start()
        self.assertTrue(drain_thread.is_alive())

        submitter.release.set()
        drive_thread.join(timeout=1)
        drain_thread.join(timeout=1)
        self.assertFalse(drive_thread.is_alive())
        self.assertFalse(drain_thread.is_alive())
        self.assertEqual(result[0].handles, ["handle-1"])
        self.assertEqual(result[0].missing_slots, [1, 2])

        self.progress.values[1] = 6
        controller.drive_once()
        self.assertEqual(len(submitter.calls), 1)

    def test_retired_state_job_can_be_drained(self):
        controller = self.make_controller(
            layer_to_state_slot=[0, -1, 1, -1, 2, -1],
            num_state_layers=3,
        )
        old_job = make_state_job(generation=1)
        controller.arm_state([old_job])
        controller.arm_state([])
        drained = controller.drain_state(7, "decode-0", 3, 9)
        self.assertEqual(drained.missing_slots, [0, 1, 2])

    def test_drain_waits_for_inflight_submit_and_closes_job(self):
        submitter = BlockingSubmitter()
        controller = self.make_controller(submitter=submitter, submit_batch=1)
        job = make_job()
        controller.arm([job])
        self.progress.values[1] = 2

        drive_thread = threading.Thread(target=controller.drive_once)
        drive_thread.start()
        self.assertTrue(submitter.entered.wait(timeout=1))

        result = []
        drain_thread = threading.Thread(
            target=lambda: result.append(
                controller.drain(*job.key, job.page_indices, job.dst_page_indices)
            )
        )
        drain_thread.start()
        self.assertTrue(drain_thread.is_alive())

        submitter.release.set()
        drive_thread.join(timeout=1)
        drain_thread.join(timeout=1)
        self.assertFalse(drive_thread.is_alive())
        self.assertFalse(drain_thread.is_alive())
        self.assertEqual(result[0].handles, ["handle-1"])
        self.assertEqual(result[0].missing_slots, [1, 2])

        self.progress.values[1] = 6
        controller.drive_once([job])
        self.assertEqual(len(submitter.calls), 1)


if __name__ == "__main__":
    unittest.main()
