"""In-process benchmark runner for prefill/decode disaggregation."""

import asyncio
import heapq
import random
from dataclasses import asdict, replace
from typing import Optional, Sequence, Union

import numpy as np
from sglang_simulator.dataset import BaseDataset, SimpleDataset
from sglang_simulator.simulation.benchmark import BaseBenchmarkRunner, BenchmarkConfig
from sglang_simulator.simulation.manager import ConfigManager
from sglang_simulator.simulation.sglang.kv_waves import plan_kv_transfers
from sglang_simulator.simulation.types import RequestStats
from sglang_simulator.simulation.utils import calc_metrics
from sglang_simulator.utils.logger import get_logger

from benchmark.simulator.bench_runner import (
    SGLANG_SIMULATOR_OUTPUT_DIR,
    SGLangBenchmarkRunner,
)
from sglang.srt.server_args import ServerArgs

logger = get_logger("sglang_simulator")

# Every request needs a bootstrap endpoint even though nothing is transferred:
# a Req without a bootstrap room is not a disaggregated request and never enters
# the prefill bootstrap queue.
DEFAULT_BOOTSTRAP_PORT = 30500

_STAT_TIME_KEYS = ("created_time", "queue_start", "queue_end", "last_event_time")


def _gateway():
    try:
        from sglang_router import sglang_router_rs
    except ImportError as error:
        raise ImportError(
            "PD routing runs the gateway's own policies and needs sglang_router, "
            "built from sgl-model-gateway/bindings/python."
        ) from error
    if not hasattr(sglang_router_rs, "PyPolicy"):
        raise ImportError(
            "The installed sglang_router has no PyPolicy; rebuild it from "
            "sgl-model-gateway/bindings/python."
        )
    return sglang_router_rs


class _InstancePool:
    """One role's instances as the gateway sees them.

    The gateway counts a request against its instance from routing until the
    response completes; that count is the load the policies read.
    """

    def __init__(self, policy, urls: list, worker_type: str):
        self.policy = policy
        self.urls = urls
        self.worker_type = worker_type
        self._loads = [0] * len(urls)
        self._inflight = []
        # The gateway seeds bucket boundaries from the prefill pool only.
        if worker_type == "prefill" or policy.name() != "bucket":
            policy.init_workers(self._workers())

    def _workers(self) -> list:
        worker_info = _gateway().PyWorkerInfo
        return [
            worker_info(url, load=load, worker_type=self.worker_type)
            for url, load in zip(self.urls, self._loads)
        ]

    def route(self, arrival: float, end: Optional[float], **request) -> int:
        while self._inflight and self._inflight[0][0] <= arrival:
            self._loads[heapq.heappop(self._inflight)[1]] -= 1
        index = self.policy.select_worker(self._workers(), **request)
        if index is None:
            raise RuntimeError(
                f"Policy {self.policy.name()} selected no {self.worker_type} instance"
            )
        if end is not None and end > arrival:
            self._loads[index] += 1
            heapq.heappush(self._inflight, (end, index))
        return index


class PDRouter:
    """Pick a prefill and a decode instance for each request in arrival order.

    The policies are the gateway's own, run through ``sglang_router_rs.PyPolicy``;
    ``policy_args`` are the router's policy flags (``cache_threshold``,
    ``balance_abs_threshold``, ``assignment_mode``, ...), defaulting as
    ``launch_router`` does. ``decode_policy`` defaults to ``prefill_policy``, as
    the gateway falls back to its main policy.

    ``connection`` follows the gateway's two PD routers. Over ``http``, prefill
    and decode each have their own policy instance and see the prompt text only
    when a policy asks for it, never tokens. Over ``grpc``, one policy instance
    picks both and sees text and tokens; ``prefix_hash`` hashes tokens, so it
    selects nothing over ``http``.

    ``cache_aware``, ``power_of_two`` and ``prefix_hash`` read how many requests
    are in flight on each instance, which needs each request's completion time.
    Pass ``end_times`` (rid -> completion time on the workload's clock), e.g.
    from a measured run; without them the benchmark runner takes them from a
    previous simulation round. ``prefill_urls`` and ``decode_urls`` name the
    instances; consistent hashing places instances on its ring by URL.
    """

    POLICIES = (
        "round_robin",
        "random",
        "cache_aware",
        "power_of_two",
        "manual",
        "bucket",
        "consistent_hashing",
        "prefix_hash",
    )

    def __init__(
        self,
        prefill_policy: str = "round_robin",
        decode_policy: Optional[str] = None,
        end_times: Optional[dict] = None,
        prefill_urls: Optional[Sequence[str]] = None,
        decode_urls: Optional[Sequence[str]] = None,
        connection: str = "http",
        **policy_args,
    ):
        decode_policy = decode_policy or prefill_policy
        if connection not in ("http", "grpc"):
            raise ValueError(f"Unknown connection {connection!r}: http or grpc")
        if connection == "grpc" and decode_policy != prefill_policy:
            raise ValueError("The gRPC PD router picks both roles with one policy.")
        if decode_policy == "bucket":
            raise ValueError("The gateway supports the bucket policy for prefill only.")
        self.prefill_policy = prefill_policy
        self.decode_policy = decode_policy
        self.connection = connection
        self.end_times = end_times
        self.prefill_urls = list(prefill_urls) if prefill_urls else None
        self.decode_urls = list(decode_urls) if decode_urls else None
        self.policy_args = policy_args
        # A bad policy name or argument fails here, before the engines start.
        for policy in {prefill_policy, decode_policy}:
            _gateway().PyPolicy(policy, **policy_args)

    @property
    def load_aware(self) -> bool:
        def reads_load(policy):
            return policy in ("cache_aware", "power_of_two", "prefix_hash") or (
                policy == "manual"
                and self.policy_args.get("assignment_mode") == "min_load"
            )

        return reads_load(self.prefill_policy) or reads_load(self.decode_policy)

    def _pool(self, policy, urls, count: int, worker_type: str) -> _InstancePool:
        urls = urls or [
            f"{self.connection}://{worker_type}-{index}" for index in range(count)
        ]
        if len(urls) != count:
            raise ValueError(f"{len(urls)} {worker_type} URLs for {count} instances")
        if policy.name() == "power_of_two" and count < 2:
            raise ValueError(
                f"Power-of-two policy for {worker_type} requires at least 2 "
                f"{worker_type} workers"
            )
        return _InstancePool(policy, urls, worker_type)

    def assign(
        self,
        requests: Sequence,
        num_prefill: int,
        num_decode: int,
        end_times: Optional[dict] = None,
    ) -> list[tuple[int, int]]:
        """(prefill index, decode index) for each request, in arrival order.

        Each call starts from a freshly launched router. A request without an
        entry in ``end_times`` adds no load.
        """
        policy = _gateway().PyPolicy
        prefill_policy = policy(self.prefill_policy, **self.policy_args)
        decode_policy = (
            prefill_policy
            if self.connection == "grpc"
            else policy(self.decode_policy, **self.policy_args)
        )
        prefill = self._pool(prefill_policy, self.prefill_urls, num_prefill, "prefill")
        decode = self._pool(decode_policy, self.decode_urls, num_decode, "decode")
        ring_urls = prefill.urls + decode.urls
        grpc = self.connection == "grpc"
        send_text = grpc or any(
            pool.policy.needs_request_text() for pool in (prefill, decode)
        )
        end_times = end_times or {}
        pairs = []
        for req in requests:
            arrival = req.custom_params.get("created_time", 0)
            end = end_times.get(req.extra_args.get("rid"))
            request = dict(
                request_text=req.prompt
                if send_text and isinstance(req.prompt, str)
                else None,
                tokens=list(req.token_ids) if grpc and req.token_ids else None,
                headers=req.custom_params.get("headers"),
                ring_urls=ring_urls,
            )
            pairs.append(
                (
                    prefill.route(arrival, end, **request),
                    decode.route(arrival, end, **request),
                )
            )
        return pairs


class PDDisaggBenchmarkRunner(BaseBenchmarkRunner):
    """Run one workload across prefill instances and decode instances.

    The instances never exchange KV cache. Each request is routed to one
    prefill and one decode instance. The workload runs twice: once on the
    prefill instances, then once on the decode instances with each request's
    clock moved to the moment its own prefill finished. Transfer time is
    modelled between the two passes from prefill statistics alone, so the
    instances need no live coordination; instances of the same role run
    concurrently.

    Routing happens before both passes. When the router reads instance load
    and has no completion times, the workload runs ``routing_rounds`` times,
    each round routed with the completion times of the round before.
    """

    def __init__(
        self,
        prefill_server_args: Union[ServerArgs, Sequence[ServerArgs]],
        decode_server_args: Union[ServerArgs, Sequence[ServerArgs]],
        output_dir: Optional[str] = None,
        seed: int = 0,
        router: Optional[PDRouter] = None,
        routing_rounds: int = 2,
    ):
        if routing_rounds < 1:
            raise ValueError("routing_rounds must be >= 1")
        base_dir = output_dir or SGLANG_SIMULATOR_OUTPUT_DIR
        # Separate directories: every instance writes its own request.jsonl.
        self.prefill_runners = [
            SGLangBenchmarkRunner(args, output_dir=f"{base_dir}/prefill_{index}")
            for index, args in enumerate(_as_list(prefill_server_args))
        ]
        self.decode_runners = [
            SGLangBenchmarkRunner(args, output_dir=f"{base_dir}/decode_{index}")
            for index, args in enumerate(_as_list(decode_server_args))
        ]
        self.router = router or PDRouter()
        self.routing_rounds = routing_rounds
        self._rng = random.Random(seed)
        self.routes = {}
        self.transfers = {}
        self.prefill_stats = {}
        self.decode_stats = {}
        self.request_stats = []
        self._shutdown = False

    def assign_bootstrap_endpoints(self, dataset: BaseDataset) -> None:
        """Give every request one bootstrap room for both passes, as PD needs."""
        for req in dataset:
            req.extra_args.setdefault("bootstrap_port", DEFAULT_BOOTSTRAP_PORT)
            req.extra_args.setdefault("bootstrap_room", self._rng.randint(0, 2**63 - 1))

    def route_requests(
        self, dataset: BaseDataset, end_times: Optional[dict] = None
    ) -> None:
        """Route every request to a prefill and a decode instance by arrival."""
        arrivals = sorted(
            dataset, key=lambda req: req.custom_params.get("created_time", 0)
        )
        pairs = self.router.assign(
            arrivals, len(self.prefill_runners), len(self.decode_runners), end_times
        )
        self.routes = {
            req.extra_args["rid"]: pair for req, pair in zip(arrivals, pairs)
        }

    @staticmethod
    def _merge_gen_token_latencies(
        prefill_stat: dict, decode_stat: dict
    ) -> tuple[list, float]:
        """Merge both passes and attribute Decode transfer wait to TTFT."""
        prefill_gen = list(prefill_stat["gen_token_latencies"])
        decode_gen = list(decode_stat["gen_token_latencies"])
        merged = prefill_gen + decode_gen
        created = decode_stat["created_time"]
        queue_end = decode_stat["queue_end"]
        if not prefill_gen or min(created, queue_end) < 0:
            return merged, 0.0
        wait = max(queue_end - created, 0.0)
        if not decode_gen:
            merged[0] += wait
            return merged, wait
        first_decode = len(prefill_gen)
        shifted = min(wait, merged[first_decode])
        merged[first_decode] -= shifted
        merged[0] += shifted
        return merged, shifted

    def build_combined_request_stats(self, dataset: BaseDataset) -> list:
        """Pair both passes by rid and build client-visible request timelines."""
        combined = []
        for req in dataset:
            rid = req.extra_args.get("rid")
            p_stat = self.prefill_stats.get(rid)
            d_stat = self.decode_stats.get(rid)
            if p_stat is None or d_stat is None:
                continue
            latencies, wait = self._merge_gen_token_latencies(p_stat, d_stat)
            # The prefill pass owns arrival, queueing and cache-hit accounting.
            stat = RequestStats(**p_stat)
            stat.gen_token_latencies = latencies
            stat.last_event_time = stat.created_time + sum(latencies)
            stat.decode_first_token_wait = wait
            stat.prefill_instance, stat.decode_instance = self.routes[rid]
            transfer = self.transfers.get(rid)
            if transfer is not None:
                for key, value in transfer.as_stats().items():
                    setattr(stat, key, value)
            combined.append(stat)
        return combined

    def annotate_kv_transfers(
        self, dataset: BaseDataset, kv_layouts: Sequence[Optional[dict]]
    ) -> Optional[int]:
        """Schedule KV transfers from the Prefill pass and annotate Decode inputs.

        ``kv_layouts`` holds what each prefill instance reports in its metrics.
        Returns the number of annotated requests, or None when no
        ``kv_transfer_config`` is set. An invalid configuration raises.
        """
        kv_cfg = ConfigManager.get_kv_transfer_config()
        if not kv_cfg:
            logger.info("KV transfer model disabled: kv_transfer_config is empty")
            return None
        nominal_gb = ConfigManager.get_accelerator_info().inter_node_bandwidth_gb
        plan = plan_kv_transfers(
            self.prefill_stats,
            self.routes,
            kv_layouts,
            kv_cfg,
            accelerator_bandwidth=None if nominal_gb is None else nominal_gb * 1e9,
        )
        self.transfers = plan.transfers
        for req in dataset:
            transfer = self.transfers.get(req.extra_args.get("rid"))
            if transfer is not None:
                req.custom_params["kv_xfer_visible_s"] = transfer.visible_s
        logger.info(
            "KV waves: annotated %d/%d requests on %d transfer queue(s); estimator=%s",
            len(self.transfers),
            len(dataset),
            plan.num_queues,
            plan.estimator.describe(),
        )
        return len(self.transfers)

    def carry_prefill_end_to_decode(self, dataset: BaseDataset) -> int:
        """Move each request's clock to the end of its own prefill.

        Returns the number of requests that were re-based, so a caller can tell
        a silent pairing failure from a genuinely empty run.
        """
        rebased = 0
        for req in dataset:
            stat = self.prefill_stats.get(req.extra_args.get("rid"))
            if stat is None or "last_event_time" not in stat:
                continue
            # Decode holds the request for the unhidden transfer on top of this.
            req.custom_params["created_time"] = float(stat["last_event_time"])
            rebased += 1
        return rebased

    def _run_pass(
        self,
        runners: list,
        role_index: int,
        config: BenchmarkConfig,
        dataset: BaseDataset,
    ) -> tuple[list, dict]:
        """Run each instance on its own requests; return metrics and stats.

        An instance reports times relative to its own first arrival; they are
        shifted back so that every instance shares the workload's timeline.
        """
        subsets = [[] for _ in runners]
        for req in dataset:
            subsets[self.routes[req.extra_args["rid"]][role_index]].append(req)
        jobs = [(index, reqs) for index, reqs in enumerate(subsets) if reqs]

        async def run_all():
            return await asyncio.gather(
                *(
                    runners[index].async_benchmark(config, SimpleDataset(reqs=reqs))
                    for index, reqs in jobs
                )
            )

        if len(jobs) == 1:
            index, reqs = jobs[0]
            results = [runners[index].benchmark(config, SimpleDataset(reqs=reqs))]
        else:
            # Every engine in this process shares one event loop.
            results = runners[jobs[0][0]].engine.loop.run_until_complete(run_all())

        metrics = [None] * len(runners)
        stats = {}
        for (index, reqs), result in zip(jobs, results):
            metrics[index] = result
            origin = min(req.custom_params.get("created_time", 0) for req in reqs)
            for stat in runners[index].get_request_stats():
                if stat.get("rid") is None:
                    continue
                for key in _STAT_TIME_KEYS:
                    stat[key] += origin
                for chunk in stat.get("kv_prefill_chunks", []):
                    chunk["t"] += origin
                stats[stat["rid"]] = stat
        return metrics, stats

    def benchmark(self, benchmark_config: BenchmarkConfig, dataset: BaseDataset):
        self.assign_bootstrap_endpoints(dataset)
        for index, req in enumerate(dataset):
            # Pairs the passes: request.jsonl is in completion order, not ours.
            req.extra_args.setdefault("rid", f"pd-{index}")
        # The passes move each request's clock; give the caller its dataset back.
        original = [dict(req.custom_params) for req in dataset]
        try:
            return self._benchmark_rounds(benchmark_config, dataset)
        finally:
            for req, params in zip(dataset, original):
                req.custom_params = params

    def _benchmark_rounds(
        self, benchmark_config: BenchmarkConfig, dataset: BaseDataset
    ) -> dict:
        if benchmark_config.ignore_request_timestamp:
            # Generate arrivals once for the whole workload, as a single
            # instance would, before they are split across instances.
            arrival = 0.0
            for req in dataset:
                req.custom_params["created_time"] = arrival
                arrival += np.random.exponential(1.0 / benchmark_config.request_rate)
        # Every pass replays created_time: generated arrivals above, then each
        # request's prefill end on the decode pass.
        config = replace(benchmark_config, ignore_request_timestamp=False)
        arrivals = [req.custom_params.get("created_time", 0) for req in dataset]
        end_times = self.router.end_times
        rounds = (
            self.routing_rounds if end_times is None and self.router.load_aware else 1
        )
        for round_index in range(rounds):
            if round_index:
                end_times = {
                    stat.rid: stat.last_event_time for stat in self.request_stats
                }
                for req, arrival in zip(dataset, arrivals):
                    req.custom_params["created_time"] = arrival
                    req.custom_params.pop("kv_xfer_visible_s", None)
                self.flush_cache()
            previous = self.routes
            self.route_requests(dataset, end_times)
            if round_index:
                changed = sum(
                    previous[rid] != pair for rid, pair in self.routes.items()
                )
                logger.info(
                    "PD routing round %d/%d: %d of %d requests changed instance",
                    round_index + 1,
                    rounds,
                    changed,
                    len(self.routes),
                )
            metrics = self._simulate(config, dataset)
        return metrics

    def _simulate(self, config: BenchmarkConfig, dataset: BaseDataset) -> dict:
        """Run both passes over the routed workload."""
        logger.info("PD pass 1/2: %d prefill instance(s)", len(self.prefill_runners))
        prefill_metrics, self.prefill_stats = self._run_pass(
            self.prefill_runners, 0, config, dataset
        )

        annotated = self.annotate_kv_transfers(
            dataset, [(metrics or {}).get("kv_layout") for metrics in prefill_metrics]
        )
        if annotated is not None and annotated != len(dataset):
            logger.warning(
                "KV annotation incomplete: %d of %d requests.",
                annotated,
                len(dataset),
            )

        rebased = self.carry_prefill_end_to_decode(dataset)
        if rebased != len(dataset):
            logger.warning(
                "PD pairing incomplete: %d of %d requests carried a prefill end "
                "time into the decode pass.",
                rebased,
                len(dataset),
            )

        logger.info("PD pass 2/2: %d decode instance(s)", len(self.decode_runners))
        decode_metrics, self.decode_stats = self._run_pass(
            self.decode_runners, 1, config, dataset
        )
        self.request_stats = self.build_combined_request_stats(dataset)
        combined_metrics = calc_metrics(self.request_stats)

        return {
            **combined_metrics,
            "prefill": prefill_metrics,
            "decode": decode_metrics,
            "paired_requests": rebased,
        }

    def get_request_stats(self) -> list[dict]:
        """Combined client-visible statistics across Prefill, transfer and Decode."""
        return [asdict(stat) for stat in self.request_stats]

    def get_decode_request_stats(self) -> list[dict]:
        return list(self.decode_stats.values())

    def get_prefill_request_stats(self) -> list[dict]:
        return list(self.prefill_stats.values())

    def flush_cache(self):
        for runner in self.prefill_runners + self.decode_runners:
            runner.flush_cache()

    def shutdown(self):
        if self._shutdown:
            return
        self._shutdown = True
        errors = []
        for runner in self.prefill_runners + self.decode_runners:
            try:
                runner.shutdown()
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]


def _as_list(server_args) -> list:
    if isinstance(server_args, ServerArgs):
        return [server_args]
    server_args = list(server_args)
    if not server_args:
        raise ValueError("PD disaggregation needs at least one instance per role")
    return server_args
