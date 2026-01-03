import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.debug_utils.schedule_simulator import (
    AttentionBalancednessRecorder,
    BatchSizeBalancednessRecorder,
    FIFOScheduler,
    GPUState,
    RandomRouter,
    RequestStage,
    RoundRobinRouter,
    ScheduleDecision,
    SimRequest,
    Simulator,
    load_from_request_logger,
)
from sglang.test.test_utils import CustomTestCase


class TestSimRequest(CustomTestCase):
    def test_basic(self):
        req = SimRequest(request_id="r1", input_len=100, output_len=50)
        self.assertEqual(req.stage, RequestStage.PREFILL)
        self.assertEqual(req.decoded_tokens, 0)
        self.assertEqual(req.seq_len(), 100)
        self.assertFalse(req.is_finished())

    def test_seq_len_decode(self):
        req = SimRequest(
            request_id="r1",
            input_len=100,
            output_len=50,
            stage=RequestStage.DECODE,
            decoded_tokens=10,
        )
        self.assertEqual(req.seq_len(), 110)

    def test_is_finished(self):
        req = SimRequest(
            request_id="r1",
            input_len=100,
            output_len=50,
            stage=RequestStage.DECODE,
            decoded_tokens=50,
        )
        self.assertTrue(req.is_finished())


class TestGPUState(CustomTestCase):
    def test_batch_size(self):
        gpu = GPUState(gpu_id=0)
        self.assertEqual(gpu.batch_size(), 0)

        gpu.running_requests = [
            SimRequest(request_id="r1", input_len=100, output_len=50),
            SimRequest(request_id="r2", input_len=200, output_len=100),
        ]
        self.assertEqual(gpu.batch_size(), 2)

    def test_total_seq_len(self):
        gpu = GPUState(gpu_id=0)
        gpu.running_requests = [
            SimRequest(request_id="r1", input_len=100, output_len=50),
            SimRequest(
                request_id="r2",
                input_len=200,
                output_len=100,
                stage=RequestStage.DECODE,
                decoded_tokens=10,
            ),
        ]
        self.assertEqual(gpu.total_seq_len(), 100 + 210)


class TestRouters(CustomTestCase):
    def test_round_robin(self):
        router = RoundRobinRouter()
        gpu_states = [GPUState(gpu_id=i) for i in range(4)]
        req = SimRequest(request_id="r1", input_len=100, output_len=50)

        results = [router.route(req, gpu_states) for _ in range(8)]
        self.assertEqual(results, [0, 1, 2, 3, 0, 1, 2, 3])

    def test_random_router(self):
        router = RandomRouter()
        gpu_states = [GPUState(gpu_id=i) for i in range(4)]
        req = SimRequest(request_id="r1", input_len=100, output_len=50)

        results = [router.route(req, gpu_states) for _ in range(100)]
        self.assertTrue(all(0 <= r < 4 for r in results))


class TestFIFOScheduler(CustomTestCase):
    def test_basic(self):
        scheduler = FIFOScheduler(max_running_requests=2)
        gpu = GPUState(gpu_id=0)
        gpu.pending_requests = [
            SimRequest(request_id=f"r{i}", input_len=100, output_len=50)
            for i in range(5)
        ]

        decision = scheduler.schedule(gpu)
        self.assertEqual(len(decision.to_run), 2)
        self.assertEqual(len(decision.to_preempt), 0)

    def test_respects_running(self):
        scheduler = FIFOScheduler(max_running_requests=3)
        gpu = GPUState(gpu_id=0)
        gpu.running_requests = [
            SimRequest(request_id="running", input_len=100, output_len=50)
        ]
        gpu.pending_requests = [
            SimRequest(request_id=f"r{i}", input_len=100, output_len=50)
            for i in range(5)
        ]

        decision = scheduler.schedule(gpu)
        self.assertEqual(len(decision.to_run), 2)


class TestMetrics(CustomTestCase):
    def test_batch_size_balancedness(self):
        recorder = BatchSizeBalancednessRecorder()
        gpu_states = [GPUState(gpu_id=i) for i in range(2)]
        gpu_states[0].running_requests = [
            SimRequest(request_id="r1", input_len=100, output_len=50)
        ]
        gpu_states[1].running_requests = [
            SimRequest(request_id="r2", input_len=100, output_len=50),
            SimRequest(request_id="r3", input_len=100, output_len=50),
        ]

        recorder.on_step_end(0, gpu_states)
        summary = recorder.get_summary()
        self.assertAlmostEqual(summary["batch_size_balancedness_mean"], 0.75)

    def test_attention_balancedness(self):
        recorder = AttentionBalancednessRecorder()
        gpu_states = [GPUState(gpu_id=i) for i in range(2)]
        gpu_states[0].running_requests = [
            SimRequest(request_id="r1", input_len=100, output_len=50)
        ]
        gpu_states[1].running_requests = [
            SimRequest(request_id="r2", input_len=200, output_len=50)
        ]

        recorder.on_step_end(0, gpu_states)
        summary = recorder.get_summary()
        self.assertAlmostEqual(summary["attention_balancedness_mean"], 0.75)


class TestDataLoader(CustomTestCase):
    def test_load_from_request_logger(self):
        log_data = [
            {
                "event": "request.received",
                "rid": "r1",
                "obj": {"text": "hello"},
            },
            {
                "event": "request.finished",
                "rid": "r1",
                "out": {"meta_info": {"prompt_tokens": 100, "completion_tokens": 50}},
            },
            {
                "event": "request.finished",
                "rid": "r2",
                "out": {"meta_info": {"prompt_tokens": 200, "completion_tokens": 100}},
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            for item in log_data:
                f.write(json.dumps(item) + "\n")
            f.flush()

            requests = load_from_request_logger(f.name)

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].request_id, "r1")
        self.assertEqual(requests[0].input_len, 100)
        self.assertEqual(requests[0].output_len, 50)
        self.assertEqual(requests[1].request_id, "r2")
        self.assertEqual(requests[1].input_len, 200)
        self.assertEqual(requests[1].output_len, 100)


class TestSimulator(CustomTestCase):
    def test_basic_run(self):
        requests = [
            SimRequest(request_id=f"r{i}", input_len=10, output_len=5)
            for i in range(10)
        ]
        sim = Simulator(
            num_gpus=2,
            router=RoundRobinRouter(),
            scheduler=FIFOScheduler(max_running_requests=4),
            recorders=[
                BatchSizeBalancednessRecorder(),
                AttentionBalancednessRecorder(),
            ],
        )

        summary = sim.run(requests)
        self.assertIn("batch_size_balancedness_mean", summary)
        self.assertIn("attention_balancedness_mean", summary)

    def test_all_requests_complete(self):
        requests = [
            SimRequest(request_id=f"r{i}", input_len=10, output_len=3)
            for i in range(4)
        ]
        sim = Simulator(
            num_gpus=2,
            router=RoundRobinRouter(),
            scheduler=FIFOScheduler(max_running_requests=10),
        )

        sim.run(requests)

        for gpu in sim.gpu_states:
            self.assertEqual(len(gpu.pending_requests), 0)
            self.assertEqual(len(gpu.running_requests), 0)


if __name__ == "__main__":
    unittest.main()

