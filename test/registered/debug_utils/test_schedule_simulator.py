import json
import subprocess
import sys
import tempfile
import unittest

from sglang.srt.debug_utils.schedule_simulator import (
    AttentionBalancednessRecorder,
    BatchSizeBalancednessRecorder,
    FIFOScheduler,
    GPUState,
    RandomRouter,
    RoundRobinRouter,
    SimRequest,
    SimulationResult,
    Simulator,
    StepRecord,
    StickyRouter,
    generate_gsp_requests,
    generate_random_requests,
    load_from_request_logger,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="default", nightly=True)


# ==================== Non-E2E Tests ====================


class TestSimRequest(CustomTestCase):
    def test_basic(self):
        req = SimRequest(request_id="r1", input_len=100, output_len=50)
        self.assertEqual(req.decoded_tokens, 0)
        self.assertEqual(req.seq_len(), 100)
        self.assertFalse(req.is_finished())

    def test_seq_len_with_decoded(self):
        req = SimRequest(
            request_id="r1", input_len=100, output_len=50, decoded_tokens=10
        )
        self.assertEqual(req.seq_len(), 110)

    def test_is_finished(self):
        req = SimRequest(
            request_id="r1", input_len=100, output_len=50, decoded_tokens=50
        )
        self.assertTrue(req.is_finished())


class TestGPUState(CustomTestCase):
    def test_batch_size(self):
        gpu = GPUState(gpu_id=0, max_total_tokens=10000)
        self.assertEqual(gpu.batch_size(), 0)
        gpu.running_requests = [
            SimRequest(request_id="r1", input_len=100, output_len=50),
            SimRequest(request_id="r2", input_len=200, output_len=100),
        ]
        self.assertEqual(gpu.batch_size(), 2)

    def test_total_seq_len(self):
        gpu = GPUState(gpu_id=0, max_total_tokens=10000)
        gpu.running_requests = [
            SimRequest(request_id="r1", input_len=100, output_len=50),
            SimRequest(
                request_id="r2", input_len=200, output_len=100, decoded_tokens=10
            ),
        ]
        self.assertEqual(gpu.total_seq_len(), 100 + 210)

    def test_total_seq_len_shared_prefix(self):
        gpu = GPUState(gpu_id=0, max_total_tokens=10000)
        gpu.running_requests = [
            SimRequest(
                request_id="r1",
                input_len=150,
                output_len=50,
                group_id="g0",
                prefix_len=100,
            ),
            SimRequest(
                request_id="r2",
                input_len=150,
                output_len=50,
                group_id="g0",
                prefix_len=100,
            ),
        ]
        self.assertEqual(gpu.total_seq_len(), 150 + 50)

    def test_total_seq_len_shared_prefix_with_decoded(self):
        gpu = GPUState(gpu_id=0, max_total_tokens=10000)
        gpu.running_requests = [
            SimRequest(
                request_id="r1",
                input_len=150,
                output_len=50,
                decoded_tokens=10,
                group_id="g0",
                prefix_len=100,
            ),
            SimRequest(
                request_id="r2",
                input_len=150,
                output_len=50,
                decoded_tokens=5,
                group_id="g0",
                prefix_len=100,
            ),
        ]
        self.assertEqual(gpu.total_seq_len(), 160 + 55)

    def test_total_seq_len_multiple_groups(self):
        gpu = GPUState(gpu_id=0, max_total_tokens=10000)
        gpu.running_requests = [
            SimRequest(
                request_id="r1",
                input_len=150,
                output_len=50,
                group_id="g0",
                prefix_len=100,
            ),
            SimRequest(
                request_id="r2",
                input_len=150,
                output_len=50,
                group_id="g0",
                prefix_len=100,
            ),
            SimRequest(
                request_id="r3",
                input_len=200,
                output_len=50,
                group_id="g1",
                prefix_len=150,
            ),
            SimRequest(request_id="r4", input_len=80, output_len=20),
        ]
        self.assertEqual(gpu.total_seq_len(), 150 + 50 + 200 + 80)


class TestRouters(CustomTestCase):
    def test_round_robin(self):
        router = RoundRobinRouter()
        gpu_states = [GPUState(gpu_id=i, max_total_tokens=10000) for i in range(4)]
        req = SimRequest(request_id="r1", input_len=100, output_len=50)
        results = [router.route(req, gpu_states) for _ in range(8)]
        self.assertEqual(results, [0, 1, 2, 3, 0, 1, 2, 3])

    def test_random_router(self):
        router = RandomRouter()
        gpu_states = [GPUState(gpu_id=i, max_total_tokens=10000) for i in range(4)]
        req = SimRequest(request_id="r1", input_len=100, output_len=50)
        results = [router.route(req, gpu_states) for _ in range(100)]
        self.assertTrue(all(0 <= r < 4 for r in results))

    def test_sticky_router_same_group_same_gpu(self):
        router = StickyRouter()
        gpu_states = [GPUState(gpu_id=i, max_total_tokens=10000) for i in range(4)]
        reqs = [
            SimRequest(request_id=f"r{i}", input_len=100, output_len=50, group_id="g0")
            for i in range(10)
        ]
        results = [router.route(req, gpu_states) for req in reqs]
        self.assertEqual(len(set(results)), 1)

    def test_sticky_router_no_group_fallback(self):
        router = StickyRouter()
        gpu_states = [GPUState(gpu_id=i, max_total_tokens=10000) for i in range(4)]
        reqs = [
            SimRequest(request_id=f"r{i}", input_len=100, output_len=50)
            for i in range(100)
        ]
        results = [router.route(req, gpu_states) for req in reqs]
        self.assertTrue(all(0 <= r < 4 for r in results))

    def test_sticky_router_multiple_groups(self):
        router = StickyRouter()
        gpu_states = [GPUState(gpu_id=i, max_total_tokens=10000) for i in range(4)]
        for group_id in ["g0", "g1", "g2"]:
            reqs = [
                SimRequest(
                    request_id=f"{group_id}_r{i}",
                    input_len=100,
                    output_len=50,
                    group_id=group_id,
                )
                for i in range(5)
            ]
            results = [router.route(req, gpu_states) for req in reqs]
            self.assertEqual(len(set(results)), 1)


class TestFIFOScheduler(CustomTestCase):
    def test_runs_pending_requests(self):
        scheduler = FIFOScheduler()
        gpu = GPUState(gpu_id=0, max_total_tokens=10000)
        gpu.pending_requests = [
            SimRequest(request_id=f"r{i}", input_len=100, output_len=50)
            for i in range(3)
        ]
        scheduler.schedule(gpu)
        self.assertEqual(len(gpu.running_requests), 3)
        self.assertEqual(len(gpu.pending_requests), 0)

    def test_respects_token_limit(self):
        scheduler = FIFOScheduler()
        gpu = GPUState(gpu_id=0, max_total_tokens=250)
        gpu.pending_requests = [
            SimRequest(request_id=f"r{i}", input_len=100, output_len=50)
            for i in range(5)
        ]
        scheduler.schedule(gpu)
        self.assertEqual(len(gpu.running_requests), 2)
        self.assertEqual(len(gpu.pending_requests), 3)

    def test_evicts_lifo_when_over_budget(self):
        scheduler = FIFOScheduler()
        gpu = GPUState(gpu_id=0, max_total_tokens=250)
        gpu.running_requests = [
            SimRequest(request_id=f"r{i}", input_len=100, output_len=50)
            for i in range(3)
        ]  # 300 tokens total
        scheduler.schedule(gpu)
        self.assertEqual(len(gpu.running_requests), 2)
        self.assertEqual(len(gpu.pending_requests), 1)
        self.assertEqual(gpu.pending_requests[0].request_id, "r2")


class TestMetrics(CustomTestCase):
    def test_batch_size_balancedness(self):
        recorder = BatchSizeBalancednessRecorder()
        gpu_states = [GPUState(gpu_id=i, max_total_tokens=10000) for i in range(2)]
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
        gpu_states = [GPUState(gpu_id=i, max_total_tokens=10000) for i in range(2)]
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
        gpu_states = [GPUState(gpu_id=i, max_total_tokens=10000) for i in range(2)]
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

    def test_generate_gsp_basic(self):
        requests = generate_gsp_requests(
            num_groups=4,
            prompts_per_group=3,
            system_prompt_len=100,
            question_len=50,
            output_len=25,
            seed=42,
        )
        self.assertEqual(len(requests), 12)
        for req in requests:
            self.assertIsNotNone(req.group_id)
            self.assertEqual(req.prefix_len, 100)
            self.assertEqual(req.input_len, 150)
            self.assertEqual(req.output_len, 25)

    def test_generate_gsp_group_assignment(self):
        requests = generate_gsp_requests(
            num_groups=3,
            prompts_per_group=2,
            system_prompt_len=100,
            question_len=50,
            output_len=25,
            seed=42,
        )
        group_counts = {}
        for req in requests:
            group_counts[req.group_id] = group_counts.get(req.group_id, 0) + 1
        self.assertEqual(len(group_counts), 3)
        for count in group_counts.values():
            self.assertEqual(count, 2)

    def test_generate_gsp_with_range_ratio(self):
        requests = generate_gsp_requests(
            num_groups=4,
            prompts_per_group=5,
            system_prompt_len=100,
            question_len=50,
            output_len=25,
            range_ratio=0.5,
            seed=42,
        )
        for req in requests:
            self.assertGreaterEqual(req.prefix_len, 50)
            self.assertLessEqual(req.prefix_len, 100)
            self.assertGreaterEqual(req.input_len - req.prefix_len, 25)
            self.assertLessEqual(req.input_len - req.prefix_len, 50)

    def test_generate_gsp_shuffled(self):
        requests = generate_gsp_requests(
            num_groups=4,
            prompts_per_group=10,
            system_prompt_len=100,
            question_len=50,
            output_len=25,
            seed=42,
        )
        group_ids = [req.group_id for req in requests]
        is_sorted = all(
            group_ids[i] <= group_ids[i + 1] for i in range(len(group_ids) - 1)
        )
        self.assertFalse(is_sorted)


class TestSimulator(CustomTestCase):
    def test_basic_run(self):
        requests = [
            SimRequest(request_id=f"r{i}", input_len=10, output_len=5)
            for i in range(10)
        ]
        sim = Simulator(
            num_gpus=2,
            router=RoundRobinRouter(),
            scheduler=FIFOScheduler(),
            recorders=[
                BatchSizeBalancednessRecorder(),
                AttentionBalancednessRecorder(),
            ],
            max_total_tokens=100,
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
            scheduler=FIFOScheduler(),
            max_total_tokens=10000,
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

    def test_step_records(self):
        requests = [
            SimRequest(request_id=f"r{i}", input_len=10, output_len=3) for i in range(4)
        ]
        sim = Simulator(
            num_gpus=2,
            router=RoundRobinRouter(),
            scheduler=FIFOScheduler(),
            max_total_tokens=10000,
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
            scheduler=FIFOScheduler(),
            max_total_tokens=110,
        )
        result = sim.run(requests)

        # Check that preemption happened at some point
        found_preemption = False
        for record in result.step_records:
            if record.running_count == 1 and record.pending_count == 1:
                found_preemption = True
                break
        self.assertTrue(
            found_preemption, "Expected preemption to occur due to token growth"
        )


# ==================== E2E Tests ====================


class TestCLI(CustomTestCase):
    def _run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "sglang.srt.debug_utils.schedule_simulator", *args],
            capture_output=True,
            text=True,
        )

    def _assert_output_contains(self, output: str, expected_lines: str):
        for line in expected_lines.strip().split("\n"):
            self.assertIn(line, output)

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

    def test_cli_sticky_router(self):
        result = self._run_cli(
            "--synth-gsp",
            "--synth-gsp-num-groups",
            "2",
            "--synth-gsp-prompts-per-group",
            "3",
            "--synth-seed",
            "42",
            "--num-gpus",
            "4",
            "--router",
            "sticky",
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("router=sticky", result.stdout)

    def test_cli_synthetic(self):
        result = self._run_cli(
            "--synthetic",
            "--synth-random-num-requests",
            "100",
            "--synth-random-input-len",
            "512",
            "--synth-random-output-len",
            "128",
            "--synth-random-range-ratio",
            "0.5",
            "--num-gpus",
            "4",
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("Generated 100 random requests", result.stdout)

    def test_cli_log_level(self):
        result = self._run_cli(
            "--synthetic",
            "--synth-random-num-requests",
            "10",
            "--synth-random-output-len",
            "5",
            "--num-gpus",
            "2",
            "--log-level",
            "1",
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("step=", result.stdout)

    def test_e2e_simple_no_queuing(self):
        # 4 requests, input_len=10, output_len=2, 2 GPUs, all fit in memory
        result = self._run_cli(
            "--synthetic",
            "--synth-random-num-requests",
            "4",
            "--synth-random-input-len",
            "10",
            "--synth-random-output-len",
            "2",
            "--synth-random-range-ratio",
            "1.0",
            "--synth-seed",
            "42",
            "--num-gpus",
            "2",
            "--max-total-tokens",
            "10000",
            "--log-level",
            "2",
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn(
            "step=0    | GPU0[R=2:syn0,syn2 Q=0:-] | GPU1[R=2:syn1,syn3 Q=0:-]",
            result.stdout,
        )
        self.assertIn(
            "step=1    | GPU0[R=0:- Q=0:-] | GPU1[R=0:- Q=0:-]", result.stdout
        )
        self.assertIn("batch_size_balancedness_mean: 1.0000", result.stdout)

    def test_e2e_queuing_due_to_token_limit(self):
        result = self._run_cli(
            "--synthetic",
            "--synth-random-num-requests",
            "4",
            "--synth-random-input-len",
            "100",
            "--synth-random-output-len",
            "3",
            "--synth-random-range-ratio",
            "1.0",
            "--synth-seed",
            "42",
            "--num-gpus",
            "1",
            "--max-total-tokens",
            "210",
            "--log-level",
            "2",
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self._assert_output_contains(
            result.stdout,
            """
step=0    | GPU0[R=2:syn0,syn1 Q=2:syn2,syn3]
step=1    | GPU0[R=2:syn0,syn1 Q=2:syn2,syn3]
step=2    | GPU0[R=0:- Q=2:syn2,syn3]
step=3    | GPU0[R=2:syn2,syn3 Q=0:-]
step=4    | GPU0[R=2:syn2,syn3 Q=0:-]
step=5    | GPU0[R=0:- Q=0:-]""",
        )

    def test_e2e_retraction_due_to_token_growth(self):
        result = self._run_cli(
            "--synthetic",
            "--synth-random-num-requests",
            "2",
            "--synth-random-input-len",
            "50",
            "--synth-random-output-len",
            "10",
            "--synth-random-range-ratio",
            "1.0",
            "--synth-seed",
            "42",
            "--num-gpus",
            "1",
            "--max-total-tokens",
            "110",
            "--log-level",
            "2",
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self._assert_output_contains(
            result.stdout,
            """
step=0    | GPU0[R=2:syn0,syn1 Q=0:-]
step=5    | GPU0[R=2:syn0,syn1 Q=0:-]
step=6    | GPU0[R=1:syn0 Q=1:syn1]
step=9    | GPU0[R=0:- Q=1:syn1]
step=10   | GPU0[R=1:syn1 Q=0:-]
step=13   | GPU0[R=0:- Q=0:-]""",
        )

    def test_cli_gsp_basic(self):
        result = self._run_cli(
            "--synth-gsp",
            "--synth-gsp-num-groups",
            "4",
            "--synth-gsp-prompts-per-group",
            "8",
            "--synth-gsp-system-prompt-len",
            "100",
            "--synth-gsp-question-len",
            "50",
            "--synth-gsp-output-len",
            "10",
            "--synth-seed",
            "42",
            "--num-gpus",
            "2",
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("Generated 32 GSP requests", result.stdout)
        self.assertIn("4 groups x 8 prompts", result.stdout)

    def test_e2e_gsp_shared_prefix_enables_batching(self):
        for has_long_prefix in [True, False]:
            prefix_len, question_len = (50, 10) if has_long_prefix else (10, 50)
            result = self._run_cli(
                "--synth-gsp",
                "--synth-gsp-num-groups",
                "1",
                "--synth-gsp-prompts-per-group",
                "2",
                "--synth-gsp-system-prompt-len",
                str(prefix_len),
                "--synth-gsp-question-len",
                str(question_len),
                "--synth-gsp-output-len",
                "2",
                "--synth-seed",
                "42",
                "--num-gpus",
                "1",
                "--max-total-tokens",
                "80",
                "--log-level",
                "2",
            )
            self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
            if has_long_prefix:
                self.assertIn("R=2:", result.stdout)
            else:
                self.assertNotIn("R=2:", result.stdout)


if __name__ == "__main__":
    unittest.main()
