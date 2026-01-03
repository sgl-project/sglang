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
    SimulationResult,
    Simulator,
    StepRecord,
    generate_random_requests,
    load_from_request_logger,
)
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
        self.assertEqual(req_copy.stage, req.stage)
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

    def test_empty_history(self):
        recorder = BatchSizeBalancednessRecorder()
        summary = recorder.get_summary()
        self.assertEqual(summary["batch_size_balancedness_mean"], 0.0)

    def test_all_zero_batch_size(self):
        recorder = BatchSizeBalancednessRecorder()
        gpu_states = [GPUState(gpu_id=i) for i in range(2)]
        recorder.on_step_end(0, gpu_states)
        summary = recorder.get_summary()
        self.assertAlmostEqual(summary["batch_size_balancedness_mean"], 1.0)


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

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("")
            f.flush()
            requests = load_from_request_logger(f.name)
        self.assertEqual(len(requests), 0)

    def test_invalid_json_skipped(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("not json\n")
            f.write('{"event": "request.finished", "rid": "r1", "out": {"meta_info": {"prompt_tokens": 100, "completion_tokens": 50}}}\n')
            f.flush()
            requests = load_from_request_logger(f.name)
        self.assertEqual(len(requests), 1)

    def test_missing_fields_skipped(self):
        log_data = [
            {"event": "request.finished", "rid": "r1", "out": {}},
            {"event": "request.finished", "rid": "r2", "out": {"meta_info": {"prompt_tokens": 100}}},
            {"event": "request.finished", "rid": "r3", "out": {"meta_info": {"prompt_tokens": 100, "completion_tokens": 50}}},
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
            num_requests=10,
            input_len=100,
            output_len=50,
        )
        self.assertEqual(len(requests), 10)
        for req in requests:
            self.assertEqual(req.input_len, 100)
            self.assertEqual(req.output_len, 50)
            self.assertTrue(req.request_id.startswith("synthetic_"))

    def test_generate_with_range_ratio(self):
        requests = generate_random_requests(
            num_requests=100,
            input_len=100,
            output_len=50,
            range_ratio=0.5,
            seed=42,
        )
        self.assertEqual(len(requests), 100)
        for req in requests:
            self.assertGreaterEqual(req.input_len, 50)
            self.assertLessEqual(req.input_len, 100)
            self.assertGreaterEqual(req.output_len, 25)
            self.assertLessEqual(req.output_len, 50)

    def test_generate_with_seed(self):
        requests1 = generate_random_requests(
            num_requests=10,
            input_len=100,
            output_len=50,
            range_ratio=0.5,
            seed=42,
        )
        requests2 = generate_random_requests(
            num_requests=10,
            input_len=100,
            output_len=50,
            range_ratio=0.5,
            seed=42,
        )
        for r1, r2 in zip(requests1, requests2):
            self.assertEqual(r1.input_len, r2.input_len)
            self.assertEqual(r1.output_len, r2.output_len)


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

        result = sim.run(requests)
        self.assertIsInstance(result, SimulationResult)
        self.assertIn("batch_size_balancedness_mean", result.summary)
        self.assertIn("attention_balancedness_mean", result.summary)
        self.assertGreater(len(result.step_records), 0)

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

    def test_empty_requests(self):
        sim = Simulator(
            num_gpus=2,
            router=RoundRobinRouter(),
            scheduler=FIFOScheduler(),
        )
        result = sim.run([])
        self.assertEqual(result.summary, {})
        self.assertEqual(len(result.step_records), 0)

    def test_prefill_instant(self):
        # With output_len=2, request should complete in 2 steps (not 3)
        # because prefill is instant and decode starts immediately
        requests = [SimRequest(request_id="r0", input_len=100, output_len=2)]
        sim = Simulator(
            num_gpus=1,
            router=RoundRobinRouter(),
            scheduler=FIFOScheduler(),
        )
        sim.run(requests)
        # Request should be finished
        self.assertEqual(len(sim.gpu_states[0].running_requests), 0)

    def test_log_level_1(self):
        import io
        import sys

        requests = [
            SimRequest(request_id=f"r{i}", input_len=10, output_len=2)
            for i in range(4)
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
        self.assertIn("GPU1", output)
        self.assertIn("R=", output)
        self.assertIn("Q=", output)

    def test_log_level_2(self):
        import io
        import sys

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
            SimRequest(request_id=f"r{i}", input_len=10, output_len=3)
            for i in range(4)
        ]
        sim = Simulator(
            num_gpus=2,
            router=RoundRobinRouter(),
            scheduler=FIFOScheduler(max_running_requests=10),
        )
        result = sim.run(requests)

        self.assertGreater(len(result.step_records), 0)
        for record in result.step_records:
            self.assertIsInstance(record, StepRecord)
            self.assertIn(record.gpu_id, [0, 1])
            self.assertGreaterEqual(record.running_count, 0)
            self.assertGreaterEqual(record.pending_count, 0)

        step_0_records = [r for r in result.step_records if r.step == 0]
        self.assertEqual(len(step_0_records), 2)


# ==================== E2E Tests ====================


class TestMain(CustomTestCase):
    def test_main_returns_dataframe_with_expected_rows(self):
        import argparse
        import io
        import sys

        import polars as pl

        from sglang.srt.debug_utils.schedule_simulator.entrypoint import main

        # 4 requests with output_len=2, distributed across 2 GPUs via round_robin
        # r0 -> GPU0, r1 -> GPU1, r2 -> GPU0, r3 -> GPU1
        # Each request needs 2 decode steps to complete
        args = argparse.Namespace(
            input=None,
            synthetic=True,
            synth_num_requests=4,
            synth_input_len=10,
            synth_output_len=2,
            synth_range_ratio=1.0,
            synth_seed=42,
            num_gpus=2,
            router="round_robin",
            scheduler="fifo",
            max_running=256,
            output=None,
            log_level=0,
        )

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            df = main(args)
        finally:
            sys.stdout = old_stdout

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(set(df.columns), {
            "step", "gpu_id", "running_count", "pending_count",
            "total_seq_len", "running_req_ids", "pending_req_ids"
        })

        # Expected behavior:
        # step 0: GPU0 has r0,r2 running (2 reqs), GPU1 has r1,r3 running (2 reqs)
        #         After decode, each has decoded_tokens=1, not finished yet
        # step 1: All requests finish (decoded_tokens=2 == output_len)
        #         GPU0 has 0 running, GPU1 has 0 running
        expected_rows = [
            {"step": 0, "gpu_id": 0, "running_count": 2, "pending_count": 0},
            {"step": 0, "gpu_id": 1, "running_count": 2, "pending_count": 0},
            {"step": 1, "gpu_id": 0, "running_count": 0, "pending_count": 0},
            {"step": 1, "gpu_id": 1, "running_count": 0, "pending_count": 0},
        ]

        self.assertEqual(len(df), len(expected_rows))
        for expected in expected_rows:
            row = df.filter(
                (pl.col("step") == expected["step"]) &
                (pl.col("gpu_id") == expected["gpu_id"])
            )
            self.assertEqual(len(row), 1, f"Expected exactly one row for step={expected['step']}, gpu_id={expected['gpu_id']}")
            self.assertEqual(row["running_count"][0], expected["running_count"])
            self.assertEqual(row["pending_count"][0], expected["pending_count"])

    def test_main_with_queuing_and_multiple_waves(self):
        """Test a more complex scenario with request queuing and multiple completion waves."""
        import argparse
        import io
        import sys

        import polars as pl

        from sglang.srt.debug_utils.schedule_simulator.entrypoint import main

        # 8 requests with output_len=3, 2 GPUs, max_running=2
        # This creates queuing: each GPU gets 4 requests but can only run 2 at a time
        # Round-robin: r0,r2,r4,r6 -> GPU0; r1,r3,r5,r7 -> GPU1
        args = argparse.Namespace(
            input=None,
            synthetic=True,
            synth_num_requests=8,
            synth_input_len=10,
            synth_output_len=3,
            synth_range_ratio=1.0,
            synth_seed=42,
            num_gpus=2,
            router="round_robin",
            scheduler="fifo",
            max_running=2,
            output=None,
            log_level=0,
        )

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            df = main(args)
        finally:
            sys.stdout = old_stdout

        # Expected behavior with max_running=2 and output_len=3:
        #
        # Step 0: Schedule r0,r2 on GPU0 (r4,r6 pending), r1,r3 on GPU1 (r5,r7 pending)
        #         Execute: decoded_tokens=1
        #         Record: running=2, pending=2 for both GPUs
        #
        # Step 1: No change in schedule (requests not finished)
        #         Execute: decoded_tokens=2
        #         Record: running=2, pending=2 for both GPUs
        #
        # Step 2: No change in schedule
        #         Execute: decoded_tokens=3, r0,r1,r2,r3 finish and are removed
        #         Record: running=0, pending=2 for both GPUs
        #
        # Step 3: Schedule r4,r6 on GPU0, r5,r7 on GPU1 (all pending->running)
        #         Execute: decoded_tokens=1
        #         Record: running=2, pending=0 for both GPUs
        #
        # Step 4: No change in schedule
        #         Execute: decoded_tokens=2
        #         Record: running=2, pending=0 for both GPUs
        #
        # Step 5: Execute: decoded_tokens=3, all finish
        #         Record: running=0, pending=0 for both GPUs

        expected_rows = [
            {"step": 0, "gpu_id": 0, "running_count": 2, "pending_count": 2},
            {"step": 0, "gpu_id": 1, "running_count": 2, "pending_count": 2},
            {"step": 1, "gpu_id": 0, "running_count": 2, "pending_count": 2},
            {"step": 1, "gpu_id": 1, "running_count": 2, "pending_count": 2},
            {"step": 2, "gpu_id": 0, "running_count": 0, "pending_count": 2},
            {"step": 2, "gpu_id": 1, "running_count": 0, "pending_count": 2},
            {"step": 3, "gpu_id": 0, "running_count": 2, "pending_count": 0},
            {"step": 3, "gpu_id": 1, "running_count": 2, "pending_count": 0},
            {"step": 4, "gpu_id": 0, "running_count": 2, "pending_count": 0},
            {"step": 4, "gpu_id": 1, "running_count": 2, "pending_count": 0},
            {"step": 5, "gpu_id": 0, "running_count": 0, "pending_count": 0},
            {"step": 5, "gpu_id": 1, "running_count": 0, "pending_count": 0},
        ]

        self.assertEqual(len(df), len(expected_rows))
        for expected in expected_rows:
            row = df.filter(
                (pl.col("step") == expected["step"]) &
                (pl.col("gpu_id") == expected["gpu_id"])
            )
            self.assertEqual(
                len(row), 1,
                f"Expected exactly one row for step={expected['step']}, gpu_id={expected['gpu_id']}"
            )
            self.assertEqual(
                row["running_count"][0], expected["running_count"],
                f"step={expected['step']}, gpu_id={expected['gpu_id']}: running_count mismatch"
            )
            self.assertEqual(
                row["pending_count"][0], expected["pending_count"],
                f"step={expected['step']}, gpu_id={expected['gpu_id']}: pending_count mismatch"
            )


class TestCLI(CustomTestCase):
    def test_cli_basic(self):
        import subprocess
        import sys

        log_data = [
            {"event": "request.finished", "rid": "r1", "out": {"meta_info": {"prompt_tokens": 100, "completion_tokens": 50}}},
            {"event": "request.finished", "rid": "r2", "out": {"meta_info": {"prompt_tokens": 200, "completion_tokens": 100}}},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            for item in log_data:
                f.write(json.dumps(item) + "\n")
            input_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_file = f.name

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.srt.debug_utils.schedule_simulator",
                "--input", input_file,
                "--num-gpus", "2",
                "--router", "round_robin",
                "--output", output_file,
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("Loaded 2 requests", result.stdout)

        with open(output_file) as f:
            output = json.load(f)
        self.assertIn("batch_size_balancedness_mean", output)
        self.assertIn("attention_balancedness_mean", output)

    def test_cli_random_router(self):
        import subprocess
        import sys

        log_data = [
            {"event": "request.finished", "rid": "r1", "out": {"meta_info": {"prompt_tokens": 100, "completion_tokens": 50}}},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            for item in log_data:
                f.write(json.dumps(item) + "\n")
            input_file = f.name

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.srt.debug_utils.schedule_simulator",
                "--input", input_file,
                "--router", "random",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("router=random", result.stdout)

    def test_cli_synthetic(self):
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.srt.debug_utils.schedule_simulator",
                "--synthetic",
                "--synth-num-requests", "100",
                "--synth-input-len", "512",
                "--synth-output-len", "128",
                "--synth-range-ratio", "0.5",
                "--num-gpus", "4",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("Generated 100 synthetic requests", result.stdout)
        self.assertIn("synth_input_len=512", result.stdout)
        self.assertIn("synth_output_len=128", result.stdout)

    def test_cli_log_level(self):
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.srt.debug_utils.schedule_simulator",
                "--synthetic",
                "--synth-num-requests", "10",
                "--synth-output-len", "5",
                "--num-gpus", "2",
                "--log-level", "1",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("step=", result.stdout)
        self.assertIn("GPU0", result.stdout)
        self.assertIn("R=", result.stdout)


if __name__ == "__main__":
    unittest.main()

