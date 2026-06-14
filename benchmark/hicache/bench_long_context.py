import json
import queue
import time

import requests
from bench_multiturn import (
    ReadyQueue,
    WorkloadGenerator,
    gen_payload,
    gen_payload_openai,
    log_to_jsonl_file,
    parse_args,
)
from tqdm.asyncio import tqdm

from sglang.benchmark.utils import get_tokenizer
from sglang.test.kits.cache_hit_kit import (
    async_request_openai_chat_completions,
    async_request_sglang_generate,
)


class ContextWorkloadGenerator(WorkloadGenerator):
    def __init__(self, args):
        # Honor --api-format the same way ``WorkloadGenerator.__init__``
        # does; previously this subclass hard-coded the native
        # ``/generate`` endpoint and silently ignored ``--api-format
        # openai``, sending requests through the wrong API surface
        # regardless of the user's choice.
        self.api_format = args.api_format
        self.model_path = args.model_path
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

        self.sent_requests = 0
        self.completed_requests = 0

        self.dataset = json.load(open(args.dataset_path))
        num_requests = min(args.num_clients, len(self.dataset["queries"]))

        init_requests = []
        for i in range(num_requests):
            context_id = self.dataset["queries"][i]["context"]
            # Tokenize the context + question to get input_ids
            prompt_text = (
                self.dataset["contexts"][context_id]
                + self.dataset["queries"][i]["question"]
            )
            input_ids = self.tokenizer.encode(prompt_text)
            output_len = len(
                self.tokenizer(self.dataset["queries"][i]["reference_answer"])[
                    "input_ids"
                ]
            )
            # Match payload shape to the selected API: ``gen_payload``
            # produces an ``input_ids`` / ``sampling_params`` body for
            # ``/generate`` while ``gen_payload_openai`` produces a
            # ``messages`` body for ``/v1/chat/completions``.
            if self.api_format == "openai":
                messages = [{"role": "user", "content": prompt_text}]
                payload = gen_payload_openai(messages, output_len, self.model_path)
            else:
                payload = gen_payload(input_ids, output_len)
            init_requests.append((i, payload))
        self.ready_queue = ReadyQueue(init_requests=init_requests)

        self.response_queue = queue.Queue()
        self.pbar = tqdm(total=num_requests)
        self.performance_metrics = {
            "ttft": [],
            "latency": [],
            "itl": [],
            "prompt_len": [],
            "cached_tokens": [],
            "generated_len": [],
        }

        self.max_parallel = args.max_parallel
        self.logfile = args.log_file
        self.enable_round_barrier = False

    def response_handler(self):
        while True:
            try:
                client_id, response = self.response_queue.get(
                    timeout=10
                )  # Block until response is available
                if not response.success:
                    raise ValueError(f"Request failed with error: {response.error}")
                self.performance_metrics["ttft"].append(response.ttft)
                self.performance_metrics["itl"].extend(response.itl)
                self.performance_metrics["latency"].append(response.latency)
                self.performance_metrics["prompt_len"].append(response.prompt_len)
                self.performance_metrics["cached_tokens"].append(response.cached_tokens)
                self.performance_metrics["generated_len"].append(response.generated_len)
                self.completed_requests += 1

            except queue.Empty:
                if self.pbar.n == self.pbar.total:
                    break


if __name__ == "__main__":
    args = parse_args()
    args.num_rounds = 1
    args.max_parallel = 24
    flush_cache_url = f"http://{args.host}:{args.port}/flush_cache"

    for request_rate in [24, 16, 12, 8, 4, 2, 1]:
        args.request_rate = request_rate
        requests.post(flush_cache_url)
        time.sleep(1)
        performance_data = ContextWorkloadGenerator(args).run()
        log_to_jsonl_file(performance_data, args.log_file, args.tag)
