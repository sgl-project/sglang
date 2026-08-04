import argparse
import asyncio
import json
import queue
import random
import threading
import time
from datetime import datetime

import numpy as np
import requests
from tqdm.asyncio import tqdm

from sglang.bench_serving import RequestFuncOutput
from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.utils import get_tokenizer
from sglang.test.kits.cache_hit_kit import (
    async_request_openai_chat_completions,
    async_request_sglang_generate,
    gen_payload,
    gen_payload_openai,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to benchmark concurrent requests to a server."
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=256,
        help="Number of concurrent clients",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=128,
        help="Maximum number of parallel requests",
    )
    parser.add_argument(
        "--request-length",
        type=int,
        default=512,
        help="Length of each new request",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=64,
        help="Length of each output",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=5,
        help="Number of rounds per client",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="poisson",
        choices=["poisson", "uniform"],
        help="Distribution type for request intervals (poisson or uniform)",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=1.0,
        help="Average number of requests per second",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server hostname or IP (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Server port (default: 30000)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="model path compatible with Hugging Face Transformers",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="local dataset to sample tokens from",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="performance_metrics.jsonl",
        help="File to log performance metrics",
    )
    parser.add_argument(
        "--disable-auto-run",
        action="store_true",
        help="If set, disable automatically testing with a range of request rates.",
    )
    parser.add_argument(
        "--disable-random-sample",
        action="store_true",
        help="If set, disable random sampling of requests from the ShareGPT dataset.",
    )
    parser.add_argument(
        "--enable-round-barrier",
        action="store_true",
        help="If set, only send i-th turn requests after all (i-1)-th turn requests finished.",
    )
    parser.add_argument(
        "--sub-question-input-length",
        type=int,
        default=0,
        help="Length of the sub question input for each request, if set 0 use request_length",
    )
    parser.add_argument(
        "--ready-queue-policy",
        type=str,
        default="random",
        help="Policy for popping requests from the ready queue (random or fifo)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Tag of a certain run in the log file",
    )
    parser.add_argument(
        "--min-rounds",
        type=int,
        default=0,
        help="Min rounds per client (0 = use --num-rounds)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="Max rounds per client (0 = use --num-rounds)",
    )
    parser.add_argument(
        "--range-ratio",
        type=float,
        default=1.0,
        help="Length variation ratio for prompts and outputs (1.0 = no variation, 0.5 = 50%% variation)",
    )
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument(
        "--lora-path",
        type=str,
        default="",
        help="String of LoRA path. Currently we only support benchmarking on a single LoRA adaptor.",
    )
    parser.add_argument(
        "--api-format",
        type=str,
        default="sglang",
        choices=["sglang", "openai"],
        help="API format to use: 'sglang' for native /generate endpoint, "
        "'openai' for OpenAI-compatible /v1/chat/completions endpoint.",
    )
    # === Long-running stability benchmark options ===
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Total duration (seconds) for the long-running stability benchmark. "
        "When > 0, the long-run mode is enabled: the benchmark stops immediately "
        "once duration is reached; whenever a client finishes its session_rounds, "
        "a new session is started (history is reset) so the load keeps going. "
        "When = 0, the original behavior (stop after total_requests) is used.",
    )
    parser.add_argument(
        "--min-session-rounds",
        type=int,
        default=5,
        help="Minimum number of rounds per session in long-run mode (inclusive).",
    )
    parser.add_argument(
        "--max-session-rounds",
        type=int,
        default=25,
        help="Maximum number of rounds per session in long-run mode (inclusive).",
    )
    return parser.parse_args()


def log_to_jsonl_file(data, file_path="performance_metrics.jsonl", tag=""):
    """Append the data with a timestamp and tag to the specified JSONL file."""
    timestamped_data = {"timestamp": datetime.now().isoformat(), "tag": tag, **data}
    try:
        with open(file_path, "a") as file:
            file.write(
                json.dumps(timestamped_data) + "\n"
            )  # Write as a single line in JSONL format
    except IOError as e:
        print(f"Error writing to JSONL file: {e}")


class ReadyQueue:
    """
    Thread-safe queue that can pop requests in different orders based on given policy.
    """

    def __init__(self, init_requests=None, policy="random"):
        self.lock = threading.Lock()
        self.requests = init_requests or []
        self.policy = policy

    def append(self, item):
        with self.lock:
            self.requests.append(item)

    def pop(self):
        with self.lock:
            if not self.requests:
                return None
            if self.policy == "random":
                index = random.randrange(len(self.requests))
                return self.requests.pop(index)
            elif self.policy == "fifo":
                return self.requests.pop(0)
            else:
                # todo, varying thinking time of clients
                raise ValueError(f"{self.policy} not implemented")


class WorkloadGenerator:
    def __init__(self, args):
        self.args = args
        self.api_format = args.api_format
        self.model_path = args.model_path

        # Construct the base URL and select request/payload functions
        if self.api_format == "openai":
            self.url = f"http://{args.host}:{args.port}/v1/chat/completions"
            self.request_func = async_request_openai_chat_completions
        else:
            self.url = f"http://{args.host}:{args.port}/generate"
            self.request_func = async_request_sglang_generate

        self.tokenizer = get_tokenizer(args.model_path)
        self.distribution = args.distribution
        self.request_rate = args.request_rate
        self.start_time = None
        self.finished_time = None
        self.lora_path = args.lora_path

        self.sent_requests = 0
        self.completed_requests = 0

        # === Long-run mode flags ===
        self.long_run = args.duration > 0
        self.duration = args.duration
        self.deadline = None  # set in run() based on start_time
        if self.long_run:
            if args.min_session_rounds < 1:
                raise ValueError(
                    f"--min-session-rounds must be >= 1, got {args.min_session_rounds}"
                )
            if args.min_session_rounds > args.max_session_rounds:
                raise ValueError(
                    f"--min-session-rounds ({args.min_session_rounds}) must be "
                    f"<= --max-session-rounds ({args.max_session_rounds})"
                )
            self.min_session_rounds = args.min_session_rounds
            self.max_session_rounds = args.max_session_rounds
            # Long-run mode is primarily designed for the OpenAI chat format.
            # The sglang format may also work but is not the main target.
            if self.api_format != "openai":
                print(
                    "[WARN] long-run mode is designed for --api-format=openai. "
                    "sglang mode may also work but is not the primary target."
                )

        # Resolve per-client round counts
        min_rounds = args.min_rounds
        max_rounds = args.max_rounds
        if min_rounds == 0 and max_rounds == 0:
            # Backward compat: all clients use --num-rounds
            min_rounds = args.num_rounds
            max_rounds = args.num_rounds
        elif min_rounds == 0:
            min_rounds = max_rounds
        elif max_rounds == 0:
            max_rounds = min_rounds
        if min_rounds < 1:
            raise ValueError(f"--min-rounds must be >= 1, got {min_rounds}")
        if min_rounds > max_rounds:
            raise ValueError(
                f"--min-rounds ({min_rounds}) must be <= --max-rounds ({max_rounds})"
            )

        self.min_rounds = min_rounds
        self.max_rounds = max_rounds

        if min_rounds == max_rounds:
            # All clients have the same round count; skip randint to preserve random state
            self.client_total_rounds = [min_rounds] * args.num_clients
        else:
            self.client_total_rounds = [
                random.randint(min_rounds, max_rounds) for _ in range(args.num_clients)
            ]

        # clients_per_round[r] = number of clients participating in round r
        self.clients_per_round = [
            sum(1 for t in self.client_total_rounds if t > r) for r in range(max_rounds)
        ]
        self.total_requests = sum(self.client_total_rounds)

        range_ratio = args.range_ratio

        # Use return_text=False to get token ids instead of text
        first_round_samples = sample_random_requests(
            input_len=args.request_length,
            output_len=args.output_length,
            num_prompts=args.num_clients,
            range_ratio=range_ratio,
            tokenizer=self.tokenizer,
            dataset_path=args.dataset_path,
            random_sample=not args.disable_random_sample,
            return_text=False,
        )
        # Store per-sample output_len for first round
        first_round_output_lens = [row.output_len for row in first_round_samples]
        # r.prompt is now List[int] when return_text=False
        self.candidate_inputs = [list(i.prompt) for i in first_round_samples]

        if args.sub_question_input_length != 0:
            sub_question_input_length = args.sub_question_input_length
        else:
            sub_question_input_length = args.request_length

        num_sub_questions = sum(max(t - 1, 0) for t in self.client_total_rounds)

        self.sub_question_inputs = sample_random_requests(
            input_len=sub_question_input_length,
            output_len=args.output_length,
            num_prompts=max(num_sub_questions, 1),
            range_ratio=range_ratio,
            tokenizer=self.tokenizer,
            dataset_path=args.dataset_path,
            random_sample=not args.disable_random_sample,
            return_text=False,
        )

        if self.api_format == "openai":
            # OpenAI mode: history is a messages list for /v1/chat/completions
            initial_messages = {
                i: [
                    {
                        "role": "user",
                        "content": self.tokenizer.decode(self.candidate_inputs[i]),
                    }
                ]
                for i in range(args.num_clients)
            }
            init_requests = [
                (
                    i,
                    gen_payload_openai(
                        initial_messages[i],
                        first_round_output_lens[i],
                        self.model_path,
                    ),
                )
                for i in range(args.num_clients)
            ]
            self.client_records = {
                i: {
                    "round": 0,
                    "history": initial_messages[i],
                    "total_rounds": self.client_total_rounds[i],
                }
                for i in range(args.num_clients)
            }
        else:
            # SGLang mode: history is List[int] (token ids)
            init_requests = [
                (
                    i,
                    gen_payload(
                        self.candidate_inputs[i],
                        first_round_output_lens[i],
                        args.lora_path,
                    ),
                )
                for i in range(args.num_clients)
            ]
            self.client_records = {
                i: {
                    "round": 0,
                    "history": list(self.candidate_inputs[i]),
                    "total_rounds": self.client_total_rounds[i],
                }
                for i in range(args.num_clients)
            }
        self.ready_queue = ReadyQueue(
            init_requests=init_requests, policy=args.ready_queue_policy
        )
        self.candidate_inputs = self.candidate_inputs[args.num_clients :]

        self.response_queue = queue.Queue()
        # In long-run mode, `total` is only a rough estimate used to render the
        # progress bar; it is NOT used as a termination condition.
        if self.long_run:
            # Estimate: assume each request takes ~ 1 / request_rate seconds.
            est_total = max(
                1,
                int(
                    self.duration
                    * max(self.request_rate, 1.0)
                ),
            )
            self.pbar = tqdm(total=est_total, desc="long-run")
        else:
            self.pbar = tqdm(total=self.total_requests)
        self.performance_metrics = {
            "ttft": [],
            "itl": [],
            "latency": [],
            "prompt_len": [],
            "cached_tokens": [],
            "generated_len": [],
        }
        self.enable_round_barrier = args.enable_round_barrier
        if self.enable_round_barrier:
            # Add round-specific metrics while preserving the original structure
            for i in range(self.max_rounds):
                self.performance_metrics[f"round_{i}"] = {
                    "ttft": [],
                    "latency": [],
                    "prompt_len": [],
                    "cached_tokens": [],
                    "generated_len": [],
                }
        self.num_clients = args.num_clients

        self.num_rounds = self.max_rounds
        self.max_parallel = args.max_parallel
        self.output_length = args.output_length

    def _stop(self):
        """Stop predicate for long-run mode: stop as soon as deadline is hit."""
        return (
            self.long_run
            and self.deadline is not None
            and time.perf_counter() >= self.deadline
        )

    def _sample_one_prompt(self):
        """Sample a new initial prompt on demand (used when starting a new
        session in long-run mode)."""
        return sample_random_requests(
            input_len=self.args.request_length,
            output_len=self.args.output_length,
            num_prompts=1,
            range_ratio=self.args.range_ratio,
            tokenizer=self.tokenizer,
            dataset_path=self.args.dataset_path,
            random_sample=not self.args.disable_random_sample,
            return_text=False,
        )[0]

    def _sample_one_sub_question(self):
        """Sample one new sub-question on demand (used in long-run mode to
        avoid exhausting the pre-generated sub-question pool)."""
        sub_q_len = (
            self.args.sub_question_input_length
            if self.args.sub_question_input_length != 0
            else self.args.request_length
        )
        return sample_random_requests(
            input_len=sub_q_len,
            output_len=self.args.output_length,
            num_prompts=1,
            range_ratio=self.args.range_ratio,
            tokenizer=self.tokenizer,
            dataset_path=self.args.dataset_path,
            random_sample=not self.args.disable_random_sample,
            return_text=False,
        )[0]

    async def handle_request(self, item):
        client_id, payload = item
        try:
            response = await self.request_func(payload, self.url, self.pbar)
            if self.long_run:
                # Long-run mode: stop immediately when deadline is reached;
                # use deadline (not pbar) as the terminating condition.
                if self.finished_time is None and self._stop():
                    self.finished_time = time.perf_counter()
            else:
                if self.pbar.n == self.pbar.total:
                    self.finished_time = time.perf_counter()
            self.response_queue.put((client_id, response))
        except Exception as e:
            print(f"Request failed for client {client_id}: {e}")
            failed_response = RequestFuncOutput()
            failed_response.success = False
            failed_response.error = str(e)
            self.response_queue.put((client_id, failed_response))

    def request_sender(self):
        async def request_loop():
            while True:
                # Long-run mode: exit immediately once deadline is reached.
                if self._stop():
                    break
                if self.sent_requests - self.completed_requests < self.max_parallel:
                    new_request = self.ready_queue.pop()
                    if new_request:
                        asyncio.create_task(self.handle_request(new_request))
                        self.sent_requests += 1
                else:
                    await asyncio.sleep(0.05)
                    continue

                if not self.long_run and self.pbar.n == self.pbar.total:
                    break

                # Calculate Poisson-distributed wait time
                if self.distribution == "poisson":
                    sleep_time = random.expovariate(self.request_rate)
                elif self.distribution == "uniform":
                    avg_interval = (
                        1.0 / self.request_rate if self.request_rate > 0 else 1.0
                    )
                    sleep_time = random.uniform(0, 2 * avg_interval)
                else:
                    raise ValueError("Invalid distribution type")
                await asyncio.sleep(sleep_time)  # Wait before sending the next request

        # Create and run the event loop for asynchronous requests
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(request_loop())
        loop.close()

    def response_handler(self):
        next_round_reqs = []
        current_barrier_round = 0
        barrier_round_completed = 0
        # Use a shorter queue timeout in long-run mode so we can exit promptly
        # once the deadline is reached.
        get_timeout = 1 if self.long_run else 10
        while True:
            # Long-run mode: exit immediately once deadline is reached;
            # we no longer wait for in-flight responses.
            if self.long_run and self._stop():
                break
            try:
                client_id, response = self.response_queue.get(
                    timeout=get_timeout
                )  # Block until response is available
                if not response.success:
                    print(f"Request failed for client {client_id}: {response.error}")
                    self.completed_requests += 1
                    continue
                # Extend history with response
                if self.api_format == "openai":
                    if response.generated_text:
                        self.client_records[client_id]["history"].append(
                            {"role": "assistant", "content": response.generated_text}
                        )
                else:
                    self.client_records[client_id]["history"].extend(
                        response.output_ids
                    )
                current_round = self.client_records[client_id]["round"]
                self.client_records[client_id]["round"] += 1
                self.performance_metrics["ttft"].append(response.ttft)
                self.performance_metrics["itl"].extend(response.itl)
                self.performance_metrics["latency"].append(response.latency)
                self.performance_metrics["prompt_len"].append(response.prompt_len)
                self.performance_metrics["cached_tokens"].append(response.cached_tokens)
                self.performance_metrics["generated_len"].append(response.generated_len)
                if self.enable_round_barrier:
                    self.performance_metrics[f"round_{current_round}"]["ttft"].append(
                        response.ttft
                    )
                    self.performance_metrics[f"round_{current_round}"][
                        "latency"
                    ].append(response.latency)
                    self.performance_metrics[f"round_{current_round}"][
                        "prompt_len"
                    ].append(response.prompt_len)
                    self.performance_metrics[f"round_{current_round}"][
                        "cached_tokens"
                    ].append(response.cached_tokens)
                    self.performance_metrics[f"round_{current_round}"][
                        "generated_len"
                    ].append(response.generated_len)
                self.completed_requests += 1

                client_total = self.client_records[client_id]["total_rounds"]
                need_next_round = (
                    self.client_records[client_id]["round"] < client_total
                )
                # Long-run mode: if the current session has reached its
                # configured number of rounds, reset history and start a new
                # session; otherwise just continue to the next round.
                if self.long_run and not need_next_round and not self._stop():
                    # === Reset session ===
                    new_prompt = self._sample_one_prompt()
                    new_prompt_ids = list(new_prompt.prompt)
                    new_total = random.randint(
                        self.min_session_rounds, self.max_session_rounds
                    )
                    if self.api_format == "openai":
                        new_history = [
                            {
                                "role": "user",
                                "content": self.tokenizer.decode(new_prompt_ids),
                            }
                        ]
                        self.client_records[client_id] = {
                            "round": 0,
                            "history": new_history,
                            "total_rounds": new_total,
                        }
                        new_req = (
                            client_id,
                            gen_payload_openai(
                                new_history,
                                new_prompt.output_len,
                                self.model_path,
                            ),
                        )
                    else:
                        self.client_records[client_id] = {
                            "round": 0,
                            "history": new_prompt_ids,
                            "total_rounds": new_total,
                        }
                        new_req = (
                            client_id,
                            gen_payload(
                                new_prompt_ids,
                                new_prompt.output_len,
                                self.lora_path,
                            ),
                        )
                    self.ready_queue.append(new_req)
                    # already enqueued as round 1 of the new session
                    need_next_round = False

                if need_next_round:
                    if self.long_run:
                        sub_q = self._sample_one_sub_question()
                    else:
                        sub_q = self.sub_question_inputs.pop()
                    if self.api_format == "openai":
                        # Append sub-question as a new user message
                        sub_q_text = self.tokenizer.decode(list(sub_q.prompt))
                        self.client_records[client_id]["history"].append(
                            {"role": "user", "content": sub_q_text}
                        )
                        new_req = (
                            client_id,
                            gen_payload_openai(
                                self.client_records[client_id]["history"],
                                sub_q.output_len,
                                self.model_path,
                            ),
                        )
                    else:
                        # Append sub-question token ids to client's history
                        sub_q_ids = list(sub_q.prompt)
                        self.client_records[client_id]["history"].extend(sub_q_ids)
                        new_req = (
                            client_id,
                            gen_payload(
                                self.client_records[client_id]["history"],
                                sub_q.output_len,
                                self.lora_path,
                            ),
                        )
                    if self.enable_round_barrier:
                        next_round_reqs.append(new_req)
                    else:
                        self.ready_queue.append(new_req)

                # Barrier logic: release next round when all clients for
                # current barrier round have completed
                if (
                    self.enable_round_barrier
                    and current_barrier_round < self.max_rounds
                ):
                    barrier_round_completed += 1
                    expected = self.clients_per_round[current_barrier_round]
                    if barrier_round_completed == expected:
                        print(
                            f"\n  Barrier: round {current_barrier_round} complete "
                            f"({expected} clients), releasing {len(next_round_reqs)} "
                            f"requests for round {current_barrier_round + 1}"
                        )
                        self._send_heartbeat(input_len=100, output_len=100)
                        time.sleep(10)
                        for req in next_round_reqs:
                            self.ready_queue.append(req)
                        next_round_reqs = []
                        current_barrier_round += 1
                        barrier_round_completed = 0
            except queue.Empty:
                if self.long_run:
                    if self._stop():
                        break
                elif self.pbar.n == self.pbar.total:
                    break
            except ValueError as e:
                print(f"Error processing response for client {client_id}: {e}")
                continue

    def _send_heartbeat(self, input_len=100, output_len=20):
        """Send a small heartbeat request to the server."""
        heartbeat_input = [1] * input_len
        payload = gen_payload(heartbeat_input, output_len, self.lora_path)
        try:
            requests.post(self.url, json=payload, timeout=30)
        except Exception as e:
            print(f"Heartbeat request failed: {e}")

    def run(self):
        request_thread = threading.Thread(target=self.request_sender, daemon=True)
        response_thread = threading.Thread(target=self.response_handler, daemon=True)

        self.start_time = time.perf_counter()
        if self.long_run:
            self.deadline = self.start_time + self.duration
        request_thread.start()
        response_thread.start()

        request_thread.join()
        response_thread.join()
        self.pbar.close()

        # Fallback: ensure finished_time is set in long-run mode in case
        # neither of the two threads hit the assignment branch.
        if self.finished_time is None:
            self.finished_time = time.perf_counter()

        duration = self.finished_time - self.start_time
        sorted_ttft = sorted(self.performance_metrics["ttft"])
        sorted_latency = sorted(self.performance_metrics["latency"])
        sorted_itl = sorted(self.performance_metrics["itl"])
        sorted_prompt_len = sorted(self.performance_metrics["prompt_len"])
        sorted_output_len = sorted(self.performance_metrics["generated_len"])

        def percentile(sorted_vals, q):
            if not sorted_vals:
                return 0.0
            idx = int(q * len(sorted_vals))
            if idx >= len(sorted_vals):
                idx = len(sorted_vals) - 1
            return sorted_vals[idx]

        def max_or_zero(sorted_vals):
            return sorted_vals[-1] if sorted_vals else 0.0

        performance_data = {
            "summary": {
                "total_requests": len(self.performance_metrics["ttft"]),
                "request_rate": self.request_rate,
                "average_prompt_len": (
                    sum(self.performance_metrics["prompt_len"])
                    / len(self.performance_metrics["prompt_len"])
                    if self.performance_metrics["prompt_len"]
                    else 0.0
                ),
                "average_output_len": (
                    sum(self.performance_metrics["generated_len"])
                    / len(self.performance_metrics["generated_len"])
                    if self.performance_metrics["generated_len"]
                    else 0.0
                ),
                "p90_prompt_len": percentile(sorted_prompt_len, 0.9),
                "p99_prompt_len": percentile(sorted_prompt_len, 0.99),
                "p90_output_len": percentile(sorted_output_len, 0.9),
                "p99_output_len": percentile(sorted_output_len, 0.99),
                "average_ttft": sum(self.performance_metrics["ttft"])
                / len(self.performance_metrics["ttft"]),
                "p90_ttft": percentile(sorted_ttft, 0.9),
                "p99_ttft": percentile(sorted_ttft, 0.99),
                "median_ttft": percentile(sorted_ttft, 0.5),
                "max_ttft": max_or_zero(sorted_ttft),
                "average_itl": (
                    sum(self.performance_metrics["itl"])
                    / len(self.performance_metrics["itl"])
                    if self.performance_metrics["itl"]
                    else 0.0
                ),
                "p90_itl": percentile(sorted_itl, 0.9),
                "p99_itl": percentile(sorted_itl, 0.99),
                "median_itl": percentile(sorted_itl, 0.5),
                "max_itl": max_or_zero(sorted_itl),
                "average_latency": sum(self.performance_metrics["latency"])
                / len(self.performance_metrics["latency"]),
                "p90_latency": percentile(sorted_latency, 0.9),
                "p99_latency": percentile(sorted_latency, 0.99),
                "median_latency": percentile(sorted_latency, 0.5),
                "max_latency": max_or_zero(sorted_latency),
                "input_token_throughput": sum(self.performance_metrics["prompt_len"])
                / duration,
                "output_token_throughput": sum(
                    self.performance_metrics["generated_len"]
                )
                / duration,
                # Use the number of successfully completed requests (which is
                # also what `total_requests` reports above) instead of
                # `pbar.total`. In long-run mode `pbar.total` is just a rough
                # estimate (duration * request_rate) used for the progress bar
                # and does NOT reflect the actual completed throughput.
                "throughput": len(self.performance_metrics["ttft"]) / duration,
                "cache_hit_rate": (
                    0
                    if sum(self.performance_metrics["prompt_len"]) == 0
                    else sum(self.performance_metrics["cached_tokens"])
                    / sum(self.performance_metrics["prompt_len"])
                ),
            },
        }
        if self.enable_round_barrier:
            performance_data["round"] = {}
            for round_num in range(self.num_rounds):
                round_key = f"round_{round_num}"
                round_metrics = self.performance_metrics[round_key]
                performance_data["round"][round_key] = {
                    "average_ttft": (
                        sum(round_metrics["ttft"]) / len(round_metrics["ttft"])
                        if round_metrics["ttft"]
                        else 0
                    ),
                    "cache_hit_rate": (
                        0
                        if sum(round_metrics["prompt_len"]) == 0
                        else sum(round_metrics["cached_tokens"])
                        / sum(round_metrics["prompt_len"])
                    ),
                    "request_count": len(round_metrics["ttft"]),
                }
        print("All requests completed")
        print("Performance metrics summary:")
        print(
            f"  Total requests: {performance_data['summary']['total_requests']} at {performance_data['summary']['request_rate']} requests per second"
        )
        print(
            f"  Average Prompt Length: {performance_data['summary']['average_prompt_len']:.2f} tokens"
        )
        print(
            f"  Average Output Length: {performance_data['summary']['average_output_len']:.2f} tokens"
        )
        print(
            f"  P90 Prompt Length: {performance_data['summary']['p90_prompt_len']:.0f} tokens"
        )
        print(
            f"  P99 Prompt Length: {performance_data['summary']['p99_prompt_len']:.0f} tokens"
        )
        print(
            f"  P90 Output Length: {performance_data['summary']['p90_output_len']:.0f} tokens"
        )
        print(
            f"  P99 Output Length: {performance_data['summary']['p99_output_len']:.0f} tokens"
        )
        print(f"  Average TTFT: {performance_data['summary']['average_ttft']:.2f}")
        print(f"  P90 TTFT: {performance_data['summary']['p90_ttft']:.2f}")
        print(f"  P99 TTFT: {performance_data['summary']['p99_ttft']:.2f}")
        print(f"  Median TTFT: {performance_data['summary']['median_ttft']:.2f}")
        print(f"  Max TTFT: {performance_data['summary']['max_ttft']:.2f}")
        print(f"  Average ITL: {performance_data['summary']['average_itl']:.4f}")
        print(f"  P90 ITL: {performance_data['summary']['p90_itl']:.4f}")
        print(f"  P99 ITL: {performance_data['summary']['p99_itl']:.4f}")
        print(f"  Median ITL: {performance_data['summary']['median_itl']:.4f}")
        print(f"  Max ITL: {performance_data['summary']['max_itl']:.4f}")
        print(
            f"  Average latency: {performance_data['summary']['average_latency']:.2f}"
        )
        print(f"  P90 latency: {performance_data['summary']['p90_latency']:.2f}")
        print(f"  P99 latency: {performance_data['summary']['p99_latency']:.2f}")
        print(f"  Median latency: {performance_data['summary']['median_latency']:.2f}")
        print(f"  Max latency: {performance_data['summary']['max_latency']:.2f}")
        print(
            f"  Input token throughput: {performance_data['summary']['input_token_throughput']:.2f} tokens per second"
        )
        print(
            f"  Output token throughput: {performance_data['summary']['output_token_throughput']:.2f} tokens per second"
        )
        print(
            f"  Request Throughput: {performance_data['summary']['throughput']:.2f} requests per second"
        )
        print(f"  Cache Hit Rate: {performance_data['summary']['cache_hit_rate']:.6f}")

        if self.enable_round_barrier:
            # Print round-basedsummary
            print("Per-round metrics:")
            if "round" in performance_data:
                for round_num in range(self.num_rounds):
                    round_key = f"round_{round_num}"
                    if round_key in performance_data["round"]:
                        round_data = performance_data["round"][round_key]
                        avg_ttft = round_data["average_ttft"]
                        cache_hit_rate = round_data["cache_hit_rate"]
                        request_count = round_data["request_count"]
                        clients_in_round = self.clients_per_round[round_num]
                        print(
                            f"  Round {round_num}: Average TTFT = {avg_ttft:.2f}s, "
                            f"Cache Hit Rate = {cache_hit_rate:.6f} "
                            f"({request_count} requests, "
                            f"{clients_in_round} clients)"
                        )
                    else:
                        print(f"  Round {round_num}: No requests completed")

        return performance_data


if __name__ == "__main__":
    args = parse_args()
    flush_cache_url = f"http://{args.host}:{args.port}/flush_cache"

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.duration > 0:
        # Long-run mode: single run with the user-specified rate; do NOT
        # call flush_cache so that the system reaches/keeps a steady state.
        print(
            f"Long-run mode: duration={args.duration}s, "
            f"clients={args.num_clients}, "
            f"session_rounds=[{args.min_session_rounds}, {args.max_session_rounds}], "
            f"rate={args.request_rate}"
        )
        request_rates = [args.request_rate]
        auto_flush = False
    elif args.disable_auto_run:
        print("Running with specified request rate...")
        request_rates = [args.request_rate]
        auto_flush = True
    else:
        print("Auto-running with different request rates...")
        request_rates = [16, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        auto_flush = True

    for rate in request_rates:
        args.request_rate = rate
        if auto_flush:
           requests.post(flush_cache_url)
           time.sleep(1)
        performance_data = WorkloadGenerator(args).run()
        log_to_jsonl_file(performance_data, args.log_file, tag=args.tag)
