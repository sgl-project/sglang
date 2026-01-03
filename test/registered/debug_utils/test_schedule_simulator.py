import argparse
import io
import json
import subprocess
import sys
import tempfile
import unittest

import polars as pl

from sglang.srt.debug_utils.schedule_simulator import (
    AttentionBalancednessRecorder,
    BatchSizeBalancednessRecorder,
    FIFOScheduler,
    GPUState,
    RandomRouter,
    RequestStage,
    RoundRobinRouter,
    SimRequest,
    SimulationResult,
    Simulator,
    StepRecord,
    generate_random_requests,
    load_from_request_logger,
)
from sglang.srt.debug_utils.schedule_simulator.entrypoint import main
from sglang.test.test_utils import CustomTestCase


# ==================== Non-E2E Tests ====================


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

    def test_copy(self):
        req = SimRequest(
            request_id="r1",
            input_len=100,
            output_len=50,
            stage=RequestStage.DECODE,
            decoded_tokens=10,
        )
        req_copy = req.copy()
        self.assertEqual(req_copy.request_id, req.request_id)
        self.assertEqual(req_copy.input_len, req.input_len)
        req_copy.decoded_tokens = 20
        self.assertEqual(req.decoded_tokens, 10)


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
    def test_basic_token_limit(self):
        # max_total_tokens=200, each req has seq_len=100, so only 2 can run
        scheduler = FIFOScheduler(max_total_tokens=200)
        gpu = GPUState(gpu_id=0)
        gpu.pending_requests = [
            SimRequest(request_id=f"r{i}", input_len=100, output_len=50)
            for i in range(5)
        ]
        decision = scheduler.schedule(gpu)
        self.assertEqual(len(decision.to_run), 2)
        self.assertEqual(len(decision.to_preempt), 0)

    def test_respects_running_tokens(self):
        # max_total_tokens=300, running has 100 tokens, so 2 more pending can run
        scheduler = FIFOScheduler(max_total_tokens=300)
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

    def test_preemption_when_over_budget(self):
        # Simulate: running requests grow beyond max_total_tokens
        scheduler = FIFOScheduler(max_total_tokens=250)
        gpu = GPUState(gpu_id=0)
        # 3 requests with 100 tokens each = 300 tokens, over budget of 250
        gpu.running_requests = [
            SimRequest(
                request_id=f"r{i}",
                input_len=100,
                output_len=50,
                stage=RequestStage.DECODE,
                decoded_tokens=0,
            )
            for i in range(3)
        ]
        decision = scheduler.schedule(gpu)
        # Should preempt 1 request to get under 250
        self.assertEqual(len(decision.to_preempt), 1)
        self.assertEqual(decision.to_preempt[0].request_id, "r2")  # LIFO: last added


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
        self.assertAlmostEqual(
            recorder.get_summary()["batch_size_balancedness_mean"], 0.75
        )

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
        self.assertAlmostEqual(
            recorder.get_summary()["attention_balancedness_mean"], 0.75
        )

    def test_empty_history(self):
        recorder = BatchSizeBalancednessRecorder()
        self.assertEqual(recorder.get_summary()["batch_size_balancedness_mean"], 0.0)

    def test_all_zero_batch_size(self):
        recorder = BatchSizeBalancednessRecorder()
        gpu_states = [GPUState(gpu_id=i) for i in range(2)]
        recorder.on_step_end(0, gpu_states)
        self.assertAlmostEqual(
            recorder.get_summary()["batch_size_balancedness_mean"], 1.0
        )


class TestDataLoader(CustomTestCase):
    def test_load_from_request_logger(self):
        log_data = [
            {"event": "request.received", "rid": "r1", "obj": {"text": "hello"}},
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
        self.assertEqual(requests[1].input_len, 200)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("")
            f.flush()
            self.assertEqual(len(load_from_request_logger(f.name)), 0)

    def test_invalid_json_skipped(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("not json\n")
            f.write(
                '{"event": "request.finished", "rid": "r1", "out": {"meta_info": {"prompt_tokens": 100, "completion_tokens": 50}}}\n'
            )
            f.flush()
            self.assertEqual(len(load_from_request_logger(f.name)), 1)

    def test_missing_fields_skipped(self):
        log_data = [
            {"event": "request.finished", "rid": "r1", "out": {}},
            {
                "event": "request.finished",
                "rid": "r2",
                "out": {"meta_info": {"prompt_tokens": 100}},
            },
            {
                "event": "request.finished",
                "rid": "r3",
                "out": {"meta_info": {"prompt_tokens": 100, "completion_tokens": 50}},
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            for item in log_data:
                f.write(json.dumps(item) + "\n")
            f.flush()
            requests = load_from_request_logger(f.name)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].request_id, "r3")


class TestDataSynthesis(CustomTestCase):
    def test_generate_basic(self):
        requests = generate_random_requests(
            num_requests=10, input_len=100, output_len=50
        )
        self.assertEqual(len(requests), 10)
        for req in requests:
            self.assertEqual(req.input_len, 100)
            self.assertEqual(req.output_len, 50)

    def test_generate_with_range_ratio(self):
        requests = generate_random_requests(
            num_requests=100, input_len=100, output_len=50, range_ratio=0.5, seed=42
        )
        for req in requests:
            self.assertGreaterEqual(req.input_len, 50)
            self.assertLessEqual(req.input_len, 100)

    def test_generate_with_seed(self):
        r1 = generate_random_requests(
            num_requests=10, input_len=100, output_len=50, range_ratio=0.5, seed=42
        )
        r2 = generate_random_requests(
            num_requests=10, input_len=100, output_len=50, range_ratio=0.5, seed=42
        )
        for a, b in zip(r1, r2):
            self.assertEqual(a.input_len, b.input_len)


class TestSimulator(CustomTestCase):
    def test_basic_run(self):
        requests = [
            SimRequest(request_id=f"r{i}", input_len=10, output_len=5)
            for i in range(10)
        ]
        sim = Simulator(
            num_gpus=2,
            router=RoundRobinRouter(),
            scheduler=FIFOScheduler(max_total_tokens=100),
            recorders=[
                BatchSizeBalancednessRecorder(),
                AttentionBalancednessRecorder(),
            ],
        )
        result = sim.run(requests)
        self.assertIsInstance(result, SimulationResult)
        self.assertIn("batch_size_balancedness_mean", result.summary)
        self.assertGreater(len(result.step_records), 0)

    def test_all_requests_complete(self):
        requests = [
            SimRequest(request_id=f"r{i}", input_len=10, output_len=3) for i in range(4)
        ]
        sim = Simulator(
            num_gpus=2,
            router=RoundRobinRouter(),
            scheduler=FIFOScheduler(max_total_tokens=10000),
        )
        sim.run(requests)
        for gpu in sim.gpu_states:
            self.assertEqual(len(gpu.pending_requests), 0)
            self.assertEqual(len(gpu.running_requests), 0)

    def test_empty_requests(self):
        sim = Simulator(
            num_gpus=2, router=RoundRobinRouter(), scheduler=FIFOScheduler()
        )
        result = sim.run([])
        self.assertEqual(result.summary, {})
        self.assertEqual(len(result.step_records), 0)

    def test_prefill_instant(self):
        requests = [SimRequest(request_id="r0", input_len=100, output_len=2)]
        sim = Simulator(
            num_gpus=1, router=RoundRobinRouter(), scheduler=FIFOScheduler()
        )
        sim.run(requests)
        self.assertEqual(len(sim.gpu_states[0].running_requests), 0)

    def test_log_level_1(self):
        requests = [
            SimRequest(request_id=f"r{i}", input_len=10, output_len=2) for i in range(4)
        ]
        sim = Simulator(
            num_gpus=2,
            router=RoundRobinRouter(),
            scheduler=FIFOScheduler(),
            log_level=1,
        )
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            sim.run(requests)
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        self.assertIn("step=", output)
        self.assertIn("GPU0", output)
        self.assertIn("R=", output)

    def test_log_level_2(self):
        requests = [
            SimRequest(request_id=f"req{i}", input_len=10, output_len=2)
            for i in range(2)
        ]
        sim = Simulator(
            num_gpus=2,
            router=RoundRobinRouter(),
            scheduler=FIFOScheduler(),
            log_level=2,
        )
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            sim.run(requests)
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        self.assertIn("req0", output)
        self.assertIn("req1", output)

    def test_step_records(self):
        requests = [
            SimRequest(request_id=f"r{i}", input_len=10, output_len=3) for i in range(4)
        ]
        sim = Simulator(
            num_gpus=2,
            router=RoundRobinRouter(),
            scheduler=FIFOScheduler(max_total_tokens=10000),
        )
        result = sim.run(requests)
        self.assertGreater(len(result.step_records), 0)
        for record in result.step_records:
            self.assertIsInstance(record, StepRecord)
            self.assertIn(record.gpu_id, [0, 1])
        self.assertEqual(len([r for r in result.step_records if r.step == 0]), 2)

    def test_preemption_due_to_token_growth(self):
        # 2 requests on 1 GPU, each input_len=50, output_len=10
        # max_total_tokens=110, so initially both can run (100 tokens)
        # After 5 decode steps, total = 50+5 + 50+5 = 110, still ok
        # After 6 decode steps, total = 50+6 + 50+6 = 112 > 110, need preempt
        requests = [
            SimRequest(request_id="r0", input_len=50, output_len=10),
            SimRequest(request_id="r1", input_len=50, output_len=10),
        ]
        sim = Simulator(
            num_gpus=1,
            router=RoundRobinRouter(),
            scheduler=FIFOScheduler(max_total_tokens=110),
        )
        result = sim.run(requests)

        # Check that preemption happened at some point
        found_preemption = False
        for record in result.step_records:
            if record.running_count == 1 and record.pending_count == 1:
                found_preemption = True
                break
        self.assertTrue(found_preemption, "Expected preemption to occur due to token growth")


# ==================== E2E Tests ====================


class TestMain(CustomTestCase):
    def _run_and_verify(
        self,
        synth_num_requests: int,
        synth_input_len: int,
        synth_output_len: int,
        num_gpus: int,
        max_total_tokens: int,
        expected_rows: list,
    ):
        args = argparse.Namespace(
            input=None,
            synthetic=True,
            synth_num_requests=synth_num_requests,
            synth_input_len=synth_input_len,
            synth_output_len=synth_output_len,
            synth_range_ratio=1.0,
            synth_seed=42,
            num_gpus=num_gpus,
            router="round_robin",
            scheduler="fifo",
            max_total_tokens=max_total_tokens,
            output=None,
            log_level=0,
        )
        df = main(args)

        self.assertEqual(
            set(df.columns),
            {
                "step",
                "gpu_id",
                "running_count",
                "pending_count",
                "total_seq_len",
                "running_req_ids",
                "pending_req_ids",
            },
        )
        self.assertEqual(len(df), len(expected_rows))

        for expected in expected_rows:
            row = df.filter(
                (pl.col("step") == expected["step"])
                & (pl.col("gpu_id") == expected["gpu_id"])
            )
            self.assertEqual(
                len(row), 1, f"step={expected['step']}, gpu_id={expected['gpu_id']}"
            )
            self.assertEqual(row["running_count"][0], expected["running_count"])
            self.assertEqual(row["pending_count"][0], expected["pending_count"])

    def test_simple_no_queuing(self):
        # 4 requests, input_len=10, output_len=2, 2 GPUs
        # Each GPU gets 2 requests, total 20 tokens per GPU, well under 10000
        self._run_and_verify(
            synth_num_requests=4,
            synth_input_len=10,
            synth_output_len=2,
            num_gpus=2,
            max_total_tokens=10000,
            expected_rows=[
                {"step": 0, "gpu_id": 0, "running_count": 2, "pending_count": 0},
                {"step": 0, "gpu_id": 1, "running_count": 2, "pending_count": 0},
                {"step": 1, "gpu_id": 0, "running_count": 0, "pending_count": 0},
                {"step": 1, "gpu_id": 1, "running_count": 0, "pending_count": 0},
            ],
        )

    def test_queuing_due_to_token_limit(self):
        # 4 requests, input_len=100, output_len=3, 1 GPU
        # max_total_tokens=200, so only 2 requests can run at a time
        # step 0: r0,r1 running (200 tokens), r2,r3 pending
        # step 1: still running (202 tokens but they complete soon)
        # step 2: r0,r1 finish (decoded 3), r2,r3 start
        # step 3-4: r2,r3 running
        # step 5: all done
        self._run_and_verify(
            synth_num_requests=4,
            synth_input_len=100,
            synth_output_len=3,
            num_gpus=1,
            max_total_tokens=200,
            expected_rows=[
                {"step": 0, "gpu_id": 0, "running_count": 2, "pending_count": 2},
                {"step": 1, "gpu_id": 0, "running_count": 2, "pending_count": 2},
                {"step": 2, "gpu_id": 0, "running_count": 0, "pending_count": 2},
                {"step": 3, "gpu_id": 0, "running_count": 2, "pending_count": 0},
                {"step": 4, "gpu_id": 0, "running_count": 2, "pending_count": 0},
                {"step": 5, "gpu_id": 0, "running_count": 0, "pending_count": 0},
            ],
        )


class TestCLI(CustomTestCase):
    def _run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "sglang.srt.debug_utils.schedule_simulator", *args],
            capture_output=True,
            text=True,
        )

    def test_cli_basic(self):
        log_data = [
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
            input_file = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_file = f.name

        result = self._run_cli(
            "--input", input_file, "--num-gpus", "2", "--output", output_file
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("Loaded 2 requests", result.stdout)
        with open(output_file) as f:
            self.assertIn("batch_size_balancedness_mean", json.load(f))

    def test_cli_random_router(self):
        log_data = [
            {
                "event": "request.finished",
                "rid": "r1",
                "out": {"meta_info": {"prompt_tokens": 100, "completion_tokens": 50}},
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            for item in log_data:
                f.write(json.dumps(item) + "\n")
            input_file = f.name

        result = self._run_cli("--input", input_file, "--router", "random")
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("router=random", result.stdout)

    def test_cli_synthetic(self):
        result = self._run_cli(
            "--synthetic",
            "--synth-num-requests",
            "100",
            "--synth-input-len",
            "512",
            "--synth-output-len",
            "128",
            "--synth-range-ratio",
            "0.5",
            "--num-gpus",
            "4",
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("Generated 100 synthetic requests", result.stdout)

    def test_cli_log_level(self):
        result = self._run_cli(
            "--synthetic",
            "--synth-num-requests",
            "10",
            "--synth-output-len",
            "5",
            "--num-gpus",
            "2",
            "--log-level",
            "1",
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("step=", result.stdout)


if __name__ == "__main__":
    unittest.main()
