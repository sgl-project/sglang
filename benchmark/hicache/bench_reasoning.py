import json
import queue
import time
import requests
from tqdm.asyncio import tqdm

from sglang.bench_serving import (
    get_tokenizer,
)


from bench_multiturn import (
    parse_args,
    gen_payload,
    ReadyQueue,
    WorkloadGenerator,
)


class ReasoningWorkloadGenerator(WorkloadGenerator):
    def __init__(self, args):
        # Construct the base URL for requests
        self.baseurl = f"http://{args.host}:{args.port}/"
        self.url = self.baseurl + "generate"

        self.tokenizer = get_tokenizer(args.model_path)
        self.distribution = args.distribution
        self.request_rate = args.request_rate
        self.start_time = None
        self.finished_time = None

        self.sent_requests = 0
        self.completed_requests = 0

        self.dataset = json.load(open(args.dataset_path))

        init_requests = [
            (
                i,
                gen_payload(
                    self.dataset[i]["system"] + self.dataset[i]["inputs"][0],
                    len(
                        self.tokenizer(self.dataset[i]["reference_outputs"][0])[
                            "input_ids"
                        ]
                    ),
                ),
            )
            for i in range(args.num_clients)
        ]
        self.client_records = {
            i: {"round": 0, "history": init_requests[i][1]["text"]}
            for i in range(args.num_clients)
        }
        self.ready_queue = ReadyQueue(init_requests=init_requests)

        self.response_queue = queue.Queue()
        self.pbar = tqdm(total=args.num_clients * args.num_rounds)
        self.performance_metrics = {"ttft": [], "latency": []}

        self.max_parallel = args.max_parallel
        self.logfile = args.log_file

    def response_handler(self):
        while True:
            try:
                client_id, response = self.response_queue.get(
                    timeout=10
                )  # Block until response is available
                if not response.success:
                    raise ValueError(f"Request failed with error: {response.error}")
                self.client_records[client_id]["history"] += response.generated_text
                self.client_records[client_id]["round"] += 1
                self.performance_metrics["ttft"].append(response.ttft)
                self.performance_metrics["latency"].append(response.latency)
                self.completed_requests += 1

                if self.client_records[client_id]["round"] < args.num_rounds:
                    self.client_records[client_id]["history"] += self.dataset[
                        client_id
                    ]["inputs"][self.client_records[client_id]["round"]]
                    self.ready_queue.append(
                        (
                            client_id,
                            gen_payload(
                                self.client_records[client_id]["history"],
                                len(
                                    self.tokenizer(
                                        self.dataset[client_id]["reference_outputs"][
                                            self.client_records[client_id]["round"]
                                        ]
                                    )["input_ids"]
                                ),
                            ),
                        )
                    )
            except queue.Empty:
                if self.pbar.n == self.pbar.total:
                    break


if __name__ == "__main__":
    args = parse_args()
    args.num_clients = 256
    args.num_rounds = 4
    args.max_parallel = 64
    flush_cache_url = f"http://{args.host}:{args.port}/flush_cache"

    for request_rate in [8, 4, 2, 1]:
        args.request_rate = request_rate
        requests.post(flush_cache_url)
        time.sleep(1)
        ReasoningWorkloadGenerator(args).run()
