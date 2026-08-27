import argparse
import asyncio
import json
import logging
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from functools import wraps

import aiohttp

from sglang.bench_serving import (
    RequestFuncOutput,
    get_tokenizer,
    remove_prefix,
    sample_random_requests,
)

# Set up logger
logger = logging.getLogger(__name__)

# Set up JSONL file for debug logging
debug_log_file = None
# Create a lock for thread-safe debug log writing
debug_log_lock = threading.Lock()


def write_debug_log(data):
    global debug_log_file

    """Write debug information to a JSONL file"""
    if debug_log_file is None:
        return

    # Acquire lock for thread-safe writing
    with debug_log_lock:
        # Write as JSONL (JSON Line format)
        debug_log_file.write(json.dumps(data) + "\n")
        debug_log_file.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to benchmark concurrent requests to a server."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/models/Qwen3-0.6B",
        help="model path compatible with Hugging Face Transformers",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/data/models/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json",
        help="local dataset to sample tokens from",
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
        "--duration",
        type=int,
        default=600,
        help="Duration to run the benchmark in seconds (default: 300 seconds)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info"],
        help="Set the logging level (default: info)",
    )
    parser.add_argument(
        "--debug-log-file",
        type=str,
        default="debug.log.jsonl",
        help="File to write debug logs in JSONL format",
    )
    return parser.parse_args()


def load_config():
    config_path = os.getenv("CONFIG_PATH")
    if not config_path:
        raise ValueError("Environment variable 'CONFIG_PATH' is not set.")

    with open(config_path, "r") as f:
        config = json.load(f)

    required_keys = [
        "num_rounds",
        "num_clients",
        "round_ratios",
        "mean_new_tokens_per_round",
        "mean_return_tokens_per_round",
        "mean_inter_round_interval",
    ]

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required configuration key: {key}")

    num_rounds = config["num_rounds"]
    assert len(config["round_ratios"]) == num_rounds
    assert len(config["mean_new_tokens_per_round"]) == num_rounds
    assert len(config["mean_return_tokens_per_round"]) == num_rounds
    assert len(config["mean_inter_round_interval"]) == num_rounds

    print(config)

    return config


@dataclass
class UserData:
    user_id: int
    current_round: int
    total_rounds: int
    prompt: str
    return_tokens: int
    start: int


def synchronized():
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.lock:
                return func(self, *args, **kwargs)

        return wrapper

    return _decorator


class UserGenerator:
    def __init__(self, config, model_path, dataset_path):
        self.tokenizer_path = model_path
        self.tokenizer = get_tokenizer(self.tokenizer_path)
        self.dataset_path = dataset_path

        self.user_id = 0
        self.lock = threading.Lock()

        self.num_rounds = config["num_rounds"]

        self.cumulative_ratios = [
            sum(config["round_ratios"][: i + 1])
            for i in range(len(config["round_ratios"]))
        ]
        self.mean_new_tokens_per_round = config["mean_new_tokens_per_round"]
        self.mean_return_tokens_per_round = config["mean_return_tokens_per_round"]
        self.mean_inter_round_interval = config["mean_inter_round_interval"]

        self.sigma = 100
        self.range_ratio = 0.8
        assert self.range_ratio <= 1

        self.candidate_inputs = [
            [
                r
                for r in sample_random_requests(
                    input_len=(
                        self.mean_new_tokens_per_round[i] * (2 - self.range_ratio)
                    ),
                    output_len=(
                        self.mean_return_tokens_per_round[i] * (2 - self.range_ratio)
                    ),
                    num_prompts=config["num_clients"],
                    range_ratio=self.range_ratio / (2 - self.range_ratio),
                    tokenizer=self.tokenizer,
                    dataset_path=self.dataset_path,
                    random_sample=False,
                )
            ]
            for i in range(self.num_rounds)
        ]

        self.multiturn_queue = []

        self.user_stats = [0 for _ in range(self.num_rounds)]
        self.input_stats = [[0, 0] for _ in range(self.num_rounds)]
        self.output_stats = [[0, 0] for _ in range(self.num_rounds)]

    def gen(self):
        user_id = self.user_id
        self.user_id += 1

        rand_ratio = random.randint(0, self.cumulative_ratios[-1])
        i = len(self.cumulative_ratios)
        for idx, cumulative_ratio in enumerate(self.cumulative_ratios):
            if rand_ratio >= cumulative_ratio:
                continue
            else:
                i = idx + 1
                break
        total_rounds = i
        current_round = 0

        candidate_input = random.sample(self.candidate_inputs[current_round], 1)[0]
        self.input_stats[0][0] += candidate_input.prompt_len
        self.input_stats[0][1] += 1
        prompt = f"{user_id} " + candidate_input.prompt
        return_tokens = int(
            random.gauss(self.mean_return_tokens_per_round[current_round], self.sigma)
        )
        if return_tokens <= 0:
            return_tokens = self.mean_return_tokens_per_round[current_round]
        start = 0

        user_data = UserData(
            user_id, current_round, total_rounds, prompt, return_tokens, start
        )

        self.user_stats[total_rounds - 1] += 1

        return user_data

    @synchronized()
    def push(self, user_data, generated_text, len_itl):
        self.output_stats[user_data.current_round][0] += len_itl + 1
        self.output_stats[user_data.current_round][1] += 1
        user_data.current_round += 1
        if user_data.current_round >= user_data.total_rounds:
            return

        candidate_input = random.sample(
            self.candidate_inputs[user_data.current_round], 1
        )[0]
        self.input_stats[user_data.current_round][0] += candidate_input.prompt_len
        self.input_stats[user_data.current_round][1] += 1
        user_data.prompt += generated_text + candidate_input.prompt
        user_data.return_tokens = int(
            random.gauss(
                self.mean_return_tokens_per_round[user_data.current_round], self.sigma
            )
        )
        if user_data.return_tokens <= 0:
            user_data.return_tokens = self.mean_return_tokens_per_round[
                user_data.current_round
            ]
        interval = random.gauss(
            self.mean_inter_round_interval[user_data.current_round], self.sigma
        )
        if interval <= 0:
            interval = self.mean_inter_round_interval[user_data.current_round]
        user_data.start = time.perf_counter() + interval

        if len(self.multiturn_queue) == 0:
            self.multiturn_queue.append(user_data)
        else:
            i = len(self.multiturn_queue)
            for idx, d in enumerate(self.multiturn_queue):
                if user_data.start < d.start:
                    i = idx
                    break
            self.multiturn_queue.insert(idx, user_data)

    @synchronized()
    def pop(self):
        if (
            len(self.multiturn_queue)
            and time.perf_counter() > self.multiturn_queue[0].start
        ):
            return self.multiturn_queue.pop(0)
        return self.gen()


def gen_payload(prompt, output_len):
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": True,
        "stream_options": {"include_usage": True},
        "lora_path": "",
        "return_logprob": False,
        "logprob_start_len": -1,
    }
    return payload


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)


async def async_request_sglang_generate(
    user_data,
    url,
    atomic_counter,
):
    """
    Sends a streaming request to the server. Gathers text token-by-token.
    """
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {}
        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        output = RequestFuncOutput()
        payload = gen_payload(user_data.prompt, user_data.return_tokens)
        write_debug_log({"timestamp": st, "user_data": user_data.__dict__})

        try:
            async with session.post(url=url, json=payload, headers=headers) as response:
                if response.status == 200:
                    prompt_tokens = 0
                    cached_tokens = 0
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            if data.get("text"):
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft
                                    prompt_tokens = (data.get("meta_info") or {}).get(
                                        "prompt_tokens", 0
                                    )
                                    cached_tokens = (data.get("meta_info") or {}).get(
                                        "cached_tokens", 0
                                    )

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text = data["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.prompt_len = prompt_tokens
                    output.cached_tokens = cached_tokens
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception as e:
            output.success = False
            output.error = str(e)
            print(f"Request failed: {e}")

    atomic_counter.increment(1)
    return output


class AtomicCounter:
    def __init__(self, initial_value=0):
        self._value = initial_value
        self.lock = threading.Lock()

    @synchronized()
    def increment(self, amount=1):
        self._value += amount

    @synchronized()
    def get(self):
        return self._value


class WorkloadGenerator:
    def __init__(self, args):
        config = load_config()
        user_generator = UserGenerator(
            config,
            args.model_path,
            args.dataset_path,
        )

        self.url = f"http://{args.host}:{args.port}/generate"

        self.tokenizer = user_generator.tokenizer
        self.start_time = None
        self.finished_time = None
        self.duration = args.duration
        self.done = False

        self.sent_requests = 0
        self.completed_requests = 0

        self.user_generator = user_generator
        self.response_queue = queue.Queue()
        self.performance_metrics = {
            "ttft": [],
            "latency": [],
            "prompt_len": [],
            "cached_tokens": [],
        }
        self.max_parallel = config["num_clients"]

        self.atomic_counter = AtomicCounter()

    async def handle_request(self, user_data):
        try:
            response = await async_request_sglang_generate(
                user_data, self.url, self.atomic_counter
            )
            self.response_queue.put((user_data, response))
        except Exception as e:
            print(f"Request failed: {e}")
            self.completed_requests += 1

    def request_sender(self):
        async def request_loop():
            while True:
                if self.sent_requests - self.completed_requests < self.max_parallel:
                    new_request = self.user_generator.pop()
                    if new_request:
                        asyncio.create_task(self.handle_request(new_request))
                        self.sent_requests += 1
                else:
                    await asyncio.sleep(0.05)
                    continue

                if time.perf_counter() - self.start_time > self.duration:
                    self.done = True
                    break

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(request_loop())
        loop.close()

    def response_handler(self):
        while True:
            try:
                user_data, response = self.response_queue.get(timeout=10)
                logger.info(
                    f"{((time.perf_counter()-self.start_time)/self.duration*100):.2f}%"
                )
                if not response.success:
                    raise ValueError(f"Request failed with error: {response.error}")

                self.user_generator.push(
                    user_data, response.generated_text, len(response.itl)
                )
                self.performance_metrics["ttft"].append(response.ttft)
                self.performance_metrics["latency"].append(response.latency)
                self.performance_metrics["prompt_len"].append(response.prompt_len)
                self.performance_metrics["cached_tokens"].append(response.cached_tokens)
                self.completed_requests += 1
                self.finished_time = time.perf_counter()

            except queue.Empty:
                if self.done:
                    break
            except ValueError as e:
                print(f"Error processing response for client {user_data}: {e}")
                continue

    def run(self):
        request_thread = threading.Thread(target=self.request_sender, daemon=True)
        response_thread = threading.Thread(target=self.response_handler, daemon=True)

        self.start_time = time.perf_counter()
        request_thread.start()
        response_thread.start()

        request_thread.join()
        response_thread.join()

        performance_data = {
            "summary": {
                "total_requests": len(self.performance_metrics["ttft"]),
                "average_ttft": sum(self.performance_metrics["ttft"])
                / len(self.performance_metrics["ttft"]),
                "p90_ttft": sorted(self.performance_metrics["ttft"])[
                    int(0.9 * len(self.performance_metrics["ttft"]))
                ],
                "median_ttft": sorted(self.performance_metrics["ttft"])[
                    len(self.performance_metrics["ttft"]) // 2
                ],
                "average_latency": sum(self.performance_metrics["latency"])
                / len(self.performance_metrics["latency"]),
                "p90_latency": sorted(self.performance_metrics["latency"])[
                    int(0.9 * len(self.performance_metrics["latency"]))
                ],
                "median_latency": sorted(self.performance_metrics["latency"])[
                    len(self.performance_metrics["latency"]) // 2
                ],
                "throughput": self.atomic_counter.get()
                / (self.finished_time - self.start_time),
                "cache_hit_rate": (
                    0
                    if sum(self.performance_metrics["prompt_len"]) == 0
                    else sum(self.performance_metrics["cached_tokens"])
                    / sum(self.performance_metrics["prompt_len"])
                ),
            },
        }
        print("All requests completed")
        print("Performance metrics summary:")
        print(f"  Total requests: {performance_data['summary']['total_requests']}")
        print(f"  Average TTFT: {performance_data['summary']['average_ttft']:.2f}")
        print(f"  P90 TTFT: {performance_data['summary']['p90_ttft']:.2f}")
        print(f"  Median TTFT: {performance_data['summary']['median_ttft']:.2f}")
        print(
            f"  Average latency: {performance_data['summary']['average_latency']:.2f}"
        )
        print(f"  P90 latency: {performance_data['summary']['p90_latency']:.2f}")
        print(f"  Median latency: {performance_data['summary']['median_latency']:.2f}")
        print(
            f"  Throughput: {performance_data['summary']['throughput']:.2f} requests per second"
        )
        print(f"  Cache Hit Rate: {performance_data['summary']['cache_hit_rate']:.6f}")

        user_stats = self.user_generator.user_stats
        input_stats = self.user_generator.input_stats
        output_stats = self.user_generator.output_stats
        print(f"round_ratios: {user_stats}")
        print(
            f"mean_new_tokens_per_round: {[int(a/b) if b > 0 else 0 for a, b in input_stats]}"
        )
        print(
            f"mean_return_tokens_per_round: {[int(a/b) if b > 0 else 0 for a, b in output_stats]}"
        )
        return performance_data


def main():
    global debug_log_file

    args = parse_args()
    if args.log_level == "debug":
        logging.basicConfig(level=logging.DEBUG)
        logger.info("use log_level debug")
        # Initialize debug log file
        debug_log_file = open(args.debug_log_file, "w")
    else:
        logging.basicConfig(level=logging.INFO)
        logger.info("use log_level info")
    performance_data = WorkloadGenerator(args).run()

    # Close debug log file if it was opened
    if debug_log_file:
        debug_log_file.close()


if __name__ == "__main__":
    main()
