"""The half of the PPU disaggregated performance harness that needs devices.

:mod:`pd_perf_eval_kit` stays hardware-free so every PD config can be validated
and every command line rendered on a laptop; this module is the part that imports
``torch``, launches the three processes a disaggregated run is made of and drives
``sglang.bench_serving`` against the router in front of them.

What a node does depends only on the rank it was handed, because a PD group is
one workflow job running one registered file on every pod:

  rank 0        a prefill server, the router, the benchmark, the report
  other ranks   their own role's server, and nothing else until rank 0 is done

The two roles do not join one process group -- each server is a rank 0 of its
own -- so there is no rendezvous address to agree on.  What they have to exchange
is an HTTP endpoint, and they exchange it the way the colocated multi-node suites
exchange a device inventory: through the shared results directory on NAS, one
small file per rank, written staged-and-renamed because a reader on NFS can
otherwise observe a partial file.  Each node publishes its endpoint as soon as its
server process exists rather than once that server is ready, so neither role can
be left waiting on the other before either has finished loading a checkpoint; the
readiness of a peer is then established over HTTP, against the endpoint it
published.

One launch, several measurements, as on the colocated line and for the same
reason: loading these checkpoints twice costs more than the measurements do.  A
measurement is graded by the same rule -- red only when it produced no numbers,
never for being slow -- and reported through the same record shape, so the
evidence collector reads a PD report without knowing one exists.
"""

import json
import os
import time
from pathlib import Path

import requests
import torch

from sglang.bench_serving import get_auth_headers, run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import get_local_ip_auto
from sglang.test.kits.environment_fingerprint import EnvironmentRecorder
from sglang.test.kits.pd_perf_eval_kit import (
    DEFAULT_PEER_WAIT_TIMEOUT_SECONDS,
    DEFAULT_ROUTER_STARTUP_TIMEOUT_SECONDS,
    PD_ROLES,
    build_pd_server_args,
    build_router_args,
    pd_endpoint_url,
    pd_provenance,
    pd_role_environment,
    pd_role_for_node_rank,
    resolve_pd_runtime,
    validate_pd_test_config,
)
from sglang.test.kits.perf_eval_kit import (
    MeasurementError,
    build_report,
    canonical_digest,
    failed_measurement_record,
    load_json,
    measurement_record,
    render_summary,
    resolve_measurement_plan,
    write_report_files,
)
from sglang.test.test_utils import (
    get_benchmark_args,
    popen_launch_pd_server,
    popen_with_error_check,
)
from sglang.utils import wait_for_http_ready

TEST_CONFIG_PATH_ENV = "SGLANG_PPU_PD_PERF_TEST_CONFIG"
RESULTS_DIR_ENV = "SGLANG_PPU_PD_PERF_RESULTS_DIR"

# Same two budgets the colocated multi-node suites hold their workers with, and
# for the same reasons: a measurement's duration is the number being measured and
# so has no bound in the config, and the margin covers the report write plus the
# interval between two nodes observing the same NAS directory.  This is a guard
# against an unbounded hold, not the run's real fence, which is the pod timeout
# the workflow sets.
MEASUREMENT_HOLD_BUDGET_SECONDS = 3600
WORKER_HOLD_MARGIN_SECONDS = 900

# How often a node re-reads the shared directory while waiting for a peer.  The
# wait is minutes long -- a peer is loading a checkpoint -- so a slower poll costs
# nothing and keeps the NAS traffic of a large group down.
PEER_POLL_SECONDS = 10


class PDPerfSuiteMixin:
    """The reviewed disaggregated performance contract, executed on real boards.

    A test file supplies the configuration and nothing else::

        class TestPPUGlm52PdPerf(PDPerfSuiteMixin, unittest.TestCase):
            default_test_config_path = (
                Path(__file__).parent / "configs" / "glm5.2" / "....json"
            )

    ``SGLANG_PPU_PD_PERF_TEST_CONFIG`` overrides that default, which is how the
    workflow keeps the file it names and the file the test reads identical.

    A mixin rather than a ``TestCase`` subclass for the reason the colocated one
    is: the test files run as ``python3 <file> -f`` and ``unittest.main()``
    collects every ``TestCase`` in the module namespace, so a shared base class
    would also run itself with no configuration selected.
    """

    default_test_config_path = None
    runtime = None
    node_rank = 0
    role = None
    process = None
    router_process = None

    # ------------------------------------------------------------------ config

    @classmethod
    def _load_test_config(cls):
        configured_path = os.environ.get(TEST_CONFIG_PATH_ENV)
        config_path = (
            Path(configured_path) if configured_path else cls.default_test_config_path
        )
        if config_path is None:
            raise RuntimeError(
                "no disaggregated performance test config selected: set "
                f"default_test_config_path on the test class or {TEST_CONFIG_PATH_ENV} "
                "in the environment"
            )
        config = load_json(config_path)
        validate_pd_test_config(config)
        return config, Path(config_path)

    @classmethod
    def _provenance(cls, accelerator):
        provenance = pd_provenance(cls.test_config, accelerator=accelerator)
        provenance["test_config_id"] = cls.test_config["test_id"]
        provenance["test_config_sha256"] = canonical_digest(cls.test_config)
        return provenance

    # -------------------------------------------------------------- inventory

    @classmethod
    def _local_accelerator(cls):
        device_count = torch.cuda.device_count()
        return {
            "visible_device_count": device_count,
            "devices": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(
                        index
                    ).total_memory,
                }
                for index in range(device_count)
            ],
        }

    @classmethod
    def _accelerator(cls):
        """What this run actually held, as one record per node.

        ``visible_device_count`` keeps meaning "what this process can see", so
        the block is shaped like the colocated one; the role of each node is the
        addition a disaggregated report needs, because a throughput number
        attributed to two boards says nothing about which of them decoded.
        """

        accelerator = cls._local_accelerator()
        accelerator["nnodes"] = cls.runtime["nnodes"]
        accelerator["node_rank"] = cls.node_rank
        accelerator["node_name"] = os.environ.get("NODE_NAME")
        accelerator["role"] = cls.role
        accelerator["topology"] = cls.runtime["topology"]
        nodes, missing = cls._read_node_inventories()
        accelerator["nodes"] = nodes
        accelerator["total_device_count"] = sum(
            node["visible_device_count"] for node in nodes
        )
        if missing:
            accelerator["node_ranks_without_inventory"] = missing
        return accelerator

    @classmethod
    def _node_inventory_path(cls, node_rank):
        return cls.rank_dir / f"rank-{node_rank}-devices.json"

    @classmethod
    def _publish_json(cls, path, payload):
        """Write one small file where the other nodes of the group can read it.

        Staged and renamed for the reason the colocated suites do it: the nodes
        reach this directory over NFS, and a reader can otherwise observe a
        half-written file and take it for the real one.
        """

        staged = path.parent / f"{path.name}.partial"
        staged.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged.replace(path)

    @classmethod
    def _write_node_inventory(cls):
        cls._publish_json(
            cls._node_inventory_path(cls.node_rank),
            {
                "node_rank": cls.node_rank,
                "node_name": os.environ.get("NODE_NAME"),
                "role": cls.role,
                **cls._local_accelerator(),
            },
        )

    @classmethod
    def _read_node_inventories(cls):
        nodes = []
        missing = []
        for node_rank in range(cls.runtime["nnodes"]):
            try:
                nodes.append(load_json(cls._node_inventory_path(node_rank)))
            except (OSError, ValueError):
                missing.append(node_rank)
        return nodes, missing

    # --------------------------------------------------------------- endpoints

    @classmethod
    def _endpoint_path(cls, node_rank):
        return cls.rank_dir / f"rank-{node_rank}-endpoint.json"

    @classmethod
    def _run_stamp(cls):
        """Which run an endpoint file belongs to.

        Stamped rather than cleaned up, which is what the sixth dispatch of this
        line cost.  Rank 0 used to delete every node's endpoint before starting,
        so that no peer could read one left behind in a reused directory -- and
        deleted the decode node's endpoint, published thirty-six seconds
        earlier, on the way past.  Rank 0 then waited its whole peer budget for
        a file it had removed itself.  A stamp keeps the protection without the
        race: a reader can tell a stale endpoint from a live one, and no node
        ever touches a file another node owns.

        Empty outside CI, where nothing supplies the two variables and nothing
        reuses the directory either; two empty stamps compare equal, so the
        check is inert there rather than an obstacle.
        """

        return (
            f"{os.environ.get('GITHUB_RUN_ID', '')}"
            f"-{os.environ.get('GITHUB_RUN_ATTEMPT', '')}"
        )

    @classmethod
    def _publish_endpoint(cls):
        """Tell the group where this node's server listens.

        The address is probed rather than taken from ``MASTER_ADDR``: the pods
        run on the host network with cluster DNS, so the injected address is a
        service name that does not resolve, which is the same reason the
        colocated multi-node line publishes a probed address through NAS.
        """

        cls._publish_json(
            cls._endpoint_path(cls.node_rank),
            {
                "node_rank": cls.node_rank,
                "node_name": os.environ.get("NODE_NAME"),
                "role": cls.role,
                "url": cls.server_url,
                "bootstrap_port": cls.test_config["disaggregation"]["bootstrap_port"],
                "run": cls._run_stamp(),
            },
        )

    @classmethod
    def _await_peer_endpoints(cls):
        """Collect every other node's endpoint, grouped by role.

        Rank 0 alone needs this, because rank 0 alone runs the router.  The wait
        is bounded, and the bound has to cover a peer loading a multi-terabyte
        checkpoint -- an endpoint is published before its server is ready, but
        the pod publishing it may still be minutes from starting Python.

        The role each peer claims is checked against the role its rank is
        assigned, so a group whose pods were handed the wrong ranks fails here
        rather than as a router pointed at two decode servers.  An endpoint
        stamped with another run counts as not yet published, and is named on
        the way out, because a directory holding one is a directory two runs
        share and that is worth saying rather than timing out silently.
        """

        deadline = time.monotonic() + cls.test_config["disaggregation"].get(
            "peer_wait_timeout_seconds", DEFAULT_PEER_WAIT_TIMEOUT_SECONDS
        )
        expected = [rank for rank in range(cls.runtime["nnodes"]) if rank != 0]
        endpoints = {0: {"role": cls.role, "url": cls.server_url}}
        pending = list(expected)
        stale = {}
        while pending:
            for node_rank in list(pending):
                try:
                    endpoint = load_json(cls._endpoint_path(node_rank))
                except (OSError, ValueError):
                    continue
                if endpoint.get("run") != cls._run_stamp():
                    stale[node_rank] = endpoint.get("run")
                    continue
                stale.pop(node_rank, None)
                expected_role = pd_role_for_node_rank(cls.test_config, node_rank)
                if endpoint.get("role") != expected_role:
                    raise RuntimeError(
                        f"node {node_rank} published the role "
                        f"{endpoint.get('role')!r} and this config serves "
                        f"{expected_role!r} there"
                    )
                if not endpoint.get("url"):
                    raise RuntimeError(f"node {node_rank} published no endpoint URL")
                endpoints[node_rank] = endpoint
                pending.remove(node_rank)
            if not pending:
                break
            exit_code = cls.process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"the {cls.role} server of rank 0 exited with code {exit_code} "
                    f"while waiting for nodes {pending}"
                )
            if time.monotonic() >= deadline:
                message = (
                    f"nodes {pending} never published an endpoint under {cls.rank_dir}"
                )
                if stale:
                    message += (
                        f"; what they left there is stamped {sorted(stale.values())} "
                        f"and this run is {cls._run_stamp()}"
                    )
                raise RuntimeError(message)
            time.sleep(PEER_POLL_SECONDS)
        return endpoints

    @classmethod
    def _urls_by_role(cls, endpoints):
        by_role = {role: [] for role in PD_ROLES}
        for node_rank in sorted(endpoints):
            by_role[endpoints[node_rank]["role"]].append(endpoints[node_rank]["url"])
        return by_role

    # ------------------------------------------------------------------ setup

    @classmethod
    def _resolve_rank_dir(cls):
        """The directory the nodes of the group address each other through.

        Insisted on being absolute, and not merely defaulted: the endpoints and
        rank 0's completion travel through it, and a relative path resolves to a
        different directory in every pod, where each node would wait out its
        whole budget for peers that had already published.
        """

        if not cls.output_dir.is_absolute():
            raise RuntimeError(
                f"a disaggregated performance config needs {RESULTS_DIR_ENV} set to "
                f"a path every node of the group shares; got {cls.output_dir}"
            )
        return cls.output_dir / "ranks"

    @classmethod
    def _sentinel_path(cls):
        return cls.rank_dir / "rank0-complete"

    @classmethod
    def _release_worker_nodes(cls):
        """Tell the other nodes that rank 0 is done with them.

        Called from the teardown and from the setup failure path both: a rank 0
        that never reached a measurement still has to release peers that are
        holding boards, or those pods sit there until their own budget expires.
        """

        if cls.node_rank != 0:
            return
        try:
            cls.rank_dir.mkdir(parents=True, exist_ok=True)
            # Stamped with this run, and checked against it by the peers, for the
            # reason the endpoints are: it is the other file one node writes and
            # another reads, and a release meant for an earlier run would send a
            # peer home while this one still needs its board.
            cls._sentinel_path().write_text(f"{cls._run_stamp()}\n", encoding="utf-8")
        except OSError as error:
            print(
                f"failed to release the worker nodes: {type(error).__name__}: {error}",
                flush=True,
            )

    @classmethod
    def _write_setup_failure(cls, stage, detail):
        """Publish the report a run that never reached a measurement still owes.

        Every planned measurement is recorded as unmeasured with the stage that
        stopped it, so the artifact of a failed run has the shape of a successful
        one and the run page still names what was supposed to be measured.
        """

        try:
            accelerator = cls._local_accelerator()
        except Exception:
            accelerator = {"visible_device_count": None}
        accelerator["node_rank"] = cls.node_rank
        accelerator["role"] = cls.role
        if cls.runtime is not None:
            accelerator["nnodes"] = cls.runtime["nnodes"]
        measurements = [
            failed_measurement_record(entry, "server_start_failed", detail)
            for entry in cls.plan
        ]
        provenance = cls._provenance(accelerator)
        provenance["setup_stage"] = stage
        provenance["setup_role"] = cls.role
        report = build_report(cls.test_config, measurements, provenance=provenance)
        write_report_files(report, cls.report_dir)
        print(render_summary(report), flush=True)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_config, cls.test_config_path = cls._load_test_config()
        cls.model_config = cls.test_config["model"]
        cls.disaggregation = cls.test_config["disaggregation"]
        cls.plan = resolve_measurement_plan(cls.test_config)
        cls.output_dir = Path(os.environ.get(RESULTS_DIR_ENV, "ppu-pd-perf-artifacts"))
        # Before the try: which node this is and how many the launcher started
        # are what the structured evidence below is filed under, and the
        # directory it is written to is what this resolves.
        cls.runtime = resolve_pd_runtime(cls.test_config)
        cls.node_rank = cls.runtime["node_rank"]
        cls.role = cls.runtime["role"]
        cls.rank_dir = cls._resolve_rank_dir()
        # Rank 0 publishes the numbers where the workflow collects them; every
        # other node writes its own evidence one level down, because all nodes
        # share this directory and a peer's report would otherwise overwrite the
        # numbers with its own view of a run it did not measure.
        cls.report_dir = (
            cls.output_dir
            if cls.node_rank == 0
            else cls.rank_dir / f"rank-{cls.node_rank}"
        )
        cls.model_path = cls.model_config["path"]
        cls.process = None
        cls.environment_recorder = EnvironmentRecorder(
            cls.report_dir, cls.test_config, node_rank=cls.node_rank
        )
        cls.host = get_local_ip_auto()
        cls.server_url = pd_endpoint_url(
            cls.host, cls.disaggregation[f"{cls.role}_port"]
        )
        # The benchmark posts to the router, which rank 0 runs; the peers never
        # use this, and it is set here so a failure report names it anyway.
        cls.base_url = pd_endpoint_url(cls.host, cls.disaggregation["router_port"])
        stage = "runner_preflight"
        try:
            cls._preflight()
            cls.rank_dir.mkdir(parents=True, exist_ok=True)
            if cls.node_rank == 0:
                # Its own two files, and no peer's.  The sentinel is rank 0's to
                # clear, and a peer that read a stale one would exit before this
                # run had started.  The peers' endpoints were cleared here too
                # until the sixth dispatch showed what that costs: the decode
                # node published at 11:13:44 and rank 0, still opening its own
                # server, deleted that file at 11:14:1x, then waited its full
                # ninety-minute peer budget for it.  A stale endpoint is now
                # told apart by its run stamp, which no peer can lose a race to.
                cls._sentinel_path().unlink(missing_ok=True)
                cls._endpoint_path(0).unlink(missing_ok=True)

            stage = "server_start"
            server_args = build_pd_server_args(cls.test_config, cls.role)
            server_env = cls._server_environment()
            cls.environment_recorder.capture(
                "before_start", server_args=server_args, server_env=server_env
            )
            cls.process = popen_launch_pd_server(
                model=cls.model_path,
                base_url=cls.server_url,
                timeout=cls.test_config[cls.role]["startup_timeout_seconds"],
                other_args=server_args,
                env=server_env,
            )
            # Published before the server is ready on purpose: a prefill server
            # and a decode server that each waited for the other's readiness
            # before announcing themselves would deadlock, and rank 0 has to
            # know a peer exists before it can wait for that peer over HTTP.
            stage = "endpoint_exchange"
            cls._publish_endpoint()
            cls._write_node_inventory()

            stage = "server_ready"
            wait_for_http_ready(
                url=f"{cls.server_url}/health",
                timeout=cls.test_config[cls.role]["startup_timeout_seconds"],
                process=cls.process,
            )
            cls.environment_recorder.capture("server_ready", server_pid=cls.process.pid)
            print(f"the {cls.role} server of node {cls.node_rank} is ready", flush=True)

            if cls.node_rank == 0:
                stage = "peer_endpoints"
                cls.endpoints = cls._await_peer_endpoints()
                cls.urls_by_role = cls._urls_by_role(cls.endpoints)
                stage = "peer_ready"
                cls._await_peer_servers()
                stage = "router_start"
                cls._launch_router()
        except Exception as exc:
            cls.environment_recorder.capture(
                "setup_failed", server_pid=getattr(cls.process, "pid", None)
            )
            try:
                cls._write_setup_failure(stage, f"{type(exc).__name__}: {exc}")
            except Exception as report_error:
                print(
                    "failed to write structured disaggregated performance setup "
                    f"evidence: {type(report_error).__name__}",
                    flush=True,
                )
            cls._release_worker_nodes()
            raise

    @classmethod
    def _preflight(cls):
        """Refuse a node that cannot serve the config it was handed.

        The checkpoint checks are the colocated ones, because a PD run measures
        the same weights; the device check is per node rather than per group,
        which is what the schema already ties this role's tensor-parallel degree
        to.
        """

        model_dir = Path(cls.model_path)
        if not model_dir.is_dir():
            raise RuntimeError(
                f"the model directory configured by {cls.test_config_path} is missing"
            )
        if model_dir.name != cls.model_config["checkpoint_name"]:
            raise RuntimeError(
                f"model.path must identify {cls.model_config['checkpoint_name']}"
            )
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            raise RuntimeError(f"{cls.model_path} does not contain config.json")
        checkpoint_config = load_json(config_path)
        accepted_model_types = cls.model_config["accepted_model_types"]
        if checkpoint_config.get("model_type") not in accepted_model_types:
            raise RuntimeError(
                f"checkpoint config model_type "
                f"{checkpoint_config.get('model_type')!r} is not one of the accepted "
                f"types {accepted_model_types}"
            )
        expected_device_count = len(cls.test_config["hardware"]["visible_devices"])
        if torch.cuda.device_count() != expected_device_count:
            raise RuntimeError(
                f"{cls.test_config['test_id']} requires {expected_device_count} "
                f"visible PPU devices per node; node {cls.node_rank} found "
                f"{torch.cuda.device_count()}"
            )

    @classmethod
    def _server_environment(cls):
        """What this role's server is launched with on top of the inherited env.

        None when the role names no variables, which leaves the launcher's
        environment untouched.  Nothing about the group is exported here: the two
        servers do not form one process group, so there is no ``NNODES`` or
        ``RANK`` for them to read, and the only thing they have to agree on --
        the bootstrap port -- is a command-line argument the schema states once.
        """

        return pd_role_environment(cls.test_config, cls.role) or None

    @classmethod
    def _await_peer_servers(cls):
        """Wait for the peers' servers over HTTP, not for their files.

        Their endpoints are published before they are ready, so this is where
        rank 0 learns the group can actually serve a request.  ``process`` is not
        passed because these processes are on other nodes: a peer that dies is
        seen as its ``/health`` never answering, and as that node's own suite
        failing in its own log.
        """

        timeout = max(
            cls.test_config[role]["startup_timeout_seconds"] for role in PD_ROLES
        )
        for node_rank, endpoint in sorted(cls.endpoints.items()):
            if node_rank == 0:
                continue
            wait_for_http_ready(url=f"{endpoint['url']}/health", timeout=timeout)
            print(
                f"the {endpoint['role']} server of node {node_rank} is ready at "
                f"{endpoint['url']}",
                flush=True,
            )

    @classmethod
    def _launch_router(cls):
        """Put the router in front of the two roles and wait for it.

        ``--mini-lb``, because that is what the ported cases ran and what this
        repository's own disaggregation fixture launches; the full router's
        scheduling would be a different thing to measure.
        """

        command = build_router_args(
            cls.test_config,
            host=cls.host,
            prefill_urls=cls.urls_by_role["prefill"],
            decode_urls=cls.urls_by_role["decode"],
        )
        print(f"starting the router: {' '.join(command)}", flush=True)
        cls.router_process = popen_with_error_check(command)
        wait_for_http_ready(
            url=f"{cls.base_url}/health",
            timeout=cls.disaggregation.get(
                "router_startup_timeout_seconds",
                DEFAULT_ROUTER_STARTUP_TIMEOUT_SECONDS,
            ),
            process=cls.router_process,
        )
        print(f"the router is ready at {cls.base_url}", flush=True)

    @classmethod
    def tearDownClass(cls):
        recorder = getattr(cls, "environment_recorder", None)
        if recorder is not None:
            recorder.capture(
                "before_stop",
                server_pid=getattr(getattr(cls, "process", None), "pid", None),
            )
        # Released before anything is killed, and not the other way round: a
        # peer treats the loss of its own server as a failure, and tearing down
        # rank 0's side is what ends the KV path, so the other order would report
        # healthy peer pods as failed ones.
        cls._release_worker_nodes()
        for process in (cls.router_process, cls.process):
            if process is not None:
                kill_process_tree(process.pid)
        super().tearDownClass()

    # ------------------------------------------------------------ measurement

    @property
    def raw_dir(self):
        """Where the raw ``bench_serving`` output of each measurement lands."""

        path = self.report_dir / "raw"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def build_benchmark_args(self, plan_entry):
        """The ``bench_serving`` arguments one measurement is run with.

        Identical to the colocated suite's, which is deliberate -- the two lines
        have to measure the same thing for their numbers to be comparable -- with
        two differences a disaggregated run forces:

        ``base_url`` is the router's, because that is what a client of a PD
        deployment posts to and what the ported cases measured.

        ``flush_cache`` is off.  The benchmark would post that admin route to the
        router, which is not asked to proxy it here, and it ignores the response
        code either way; this suite flushes the two servers itself and asserts
        each response -- see :meth:`_flush_cache`.
        """

        args = get_benchmark_args(
            base_url=self.base_url,
            backend="sglang",
            dataset_name=plan_entry["dataset"],
            tokenizer=self.model_path,
            num_prompts=plan_entry["num_prompts"],
            random_input_len=plan_entry["input_len"],
            random_output_len=plan_entry["output_len"],
            request_rate=float("inf"),
            seed=plan_entry["seed"],
            max_concurrency=plan_entry["concurrency"],
        )
        args.random_range_ratio = plan_entry["random_range_ratio"]
        args.tokenize_prompt = True
        args.model = self.model_path
        args.served_model_name = self.model_config["served_model_name"]
        args.warmup_requests = plan_entry["warmup_requests"]
        args.flush_cache = False
        args.output_details = True
        args.output_file = str(self.raw_dir / f"{plan_entry['id']}.jsonl")
        return args

    def _flush_cache(self, plan_entry):
        """Drop the KV cache of both servers between two measurements.

        Posted to each server directly rather than to the router, because this is
        an administrative route of a server and the router in front of them is
        the thing being measured, not the thing being administered.  The response
        code is asserted -- ``bench_serving``'s own flush ignores it -- because a
        flush that silently failed would turn every measurement after it into a
        cache-hit measurement wearing a prefill's name.
        """

        for node_rank, endpoint in sorted(self.endpoints.items()):
            response = requests.post(
                f"{endpoint['url']}/flush_cache",
                headers=get_auth_headers(),
                timeout=plan_entry["flush_cache_timeout_seconds"],
            )
            if response.status_code != 200:
                raise MeasurementError(
                    "cache_flush_failed",
                    f"flushing the KV cache of the {endpoint['role']} server on node "
                    f"{node_rank} after {plan_entry['id']} returned HTTP "
                    f"{response.status_code}",
                )

    def _run_warmup_passes(self, plan_entry):
        """Run and discard the measurement's warmup passes.

        Each is a full pass at the measurement's own shape, so the recorded pass
        that follows begins with this image's compilation and allocator state
        already warm; the KV cache is flushed after each, so the recorded pass
        does not also begin with the warmup's prompts cached and report a prefill
        it did not perform.  This is the "run twice, keep the second" shape the
        internal disaggregated cases were measured under, and it is what absorbs
        the per-batch-shape compilation cost the first request of each shape pays
        on these boards -- a cost `warmup_requests`, capped at 32 output tokens,
        cannot reach.  ``warmup_passes`` is zero on the colocated line, where a
        prefill measurement pays that cost once and there is nothing to discard.

        The discarded output lands in ``raw`` under a ``-warmupN`` name rather
        than the measurement's own, so it is kept as evidence the pass ran
        without overwriting the number the recorded pass reports.
        """

        for warmup_index in range(plan_entry["warmup_passes"]):
            label = f"{plan_entry['id']}-warmup{warmup_index + 1}"
            print(f"::group::warmup {label}", flush=True)
            try:
                args = self.build_benchmark_args(plan_entry)
                args.output_file = str(self.raw_dir / f"{label}.jsonl")
                run_benchmark(args)
            finally:
                print("::endgroup::", flush=True)
            self._flush_cache(plan_entry)

    def _measure(self, plan_entry):
        """Run one measurement and grade it, never raising.

        Every failure mode is recorded as an unmeasured measurement rather than
        propagated, so one workload that cannot be measured does not cost the run
        the workloads after it -- both servers are up, and re-loading them is the
        expensive thing.  The run still ends red, because ``build_report`` counts
        them.
        """

        try:
            self._run_warmup_passes(plan_entry)
        except SystemExit as exc:
            # A warmup pass that exits the process is the recorded pass's fate
            # too; recording it here keeps the rest of the plan and the report.
            return failed_measurement_record(
                plan_entry,
                "benchmark_crashed",
                f"a warmup pass exited with status {exc.code}",
            )
        except MeasurementError as exc:
            # A warmup flush that failed endangers the recorded pass the same way
            # a between-measurement flush does; its reason code is carried as is.
            return failed_measurement_record(plan_entry, exc.reason_code, str(exc))
        except Exception as exc:
            return failed_measurement_record(
                plan_entry, "benchmark_crashed", f"{type(exc).__name__}: {exc}"
            )

        print(f"::group::measure {plan_entry['id']}", flush=True)
        try:
            raw = run_benchmark(self.build_benchmark_args(plan_entry))
        except SystemExit as exc:
            # run_benchmark exits the process on a few argument and readiness
            # failures instead of raising, and a bare SystemExit here would take
            # the rest of the measurements and the report with it.
            return failed_measurement_record(
                plan_entry,
                "benchmark_crashed",
                f"the benchmark exited with status {exc.code}",
            )
        except Exception as exc:
            return failed_measurement_record(
                plan_entry, "benchmark_crashed", f"{type(exc).__name__}: {exc}"
            )
        finally:
            print("::endgroup::", flush=True)

        try:
            record = measurement_record(plan_entry, raw)
        except MeasurementError as exc:
            return failed_measurement_record(plan_entry, exc.reason_code, str(exc))
        try:
            self._flush_cache(plan_entry)
        except MeasurementError as exc:
            # The numbers this measurement produced are real, so they are kept;
            # what the failed flush endangers is the measurements after it, and
            # naming it on this record is what makes that visible.
            record["status"] = "failed"
            record["reason_code"] = exc.reason_code
            record["detail"] = str(exc)
        except Exception as exc:
            record["status"] = "failed"
            record["reason_code"] = "cache_flush_failed"
            record["detail"] = f"{type(exc).__name__}: {exc}"
        return record

    def _hold_until_rank_zero_completes(self):
        """A peer node's share of the suite is to keep serving its role.

        Only rank 0 holds the router and the numbers; this node holds one half of
        the KV path, so returning early would take the deployment down under the
        benchmark.  The pass condition is that it served until rank 0 published
        its completion, and the failure conditions are its own server dying
        first, or that completion never arriving.
        """

        sentinel = self._sentinel_path()
        budget = (
            MEASUREMENT_HOLD_BUDGET_SECONDS * len(self.plan)
            + WORKER_HOLD_MARGIN_SECONDS
        )
        deadline = time.monotonic() + budget
        print(
            f"the {self.role} node {self.node_rank} is serving and waiting for "
            f"{sentinel}",
            flush=True,
        )
        while time.monotonic() < deadline:
            # listdir rather than exists: the directory is on NFS, where a
            # readdir revalidates the entry a cached negative lookup would not.
            try:
                released = sentinel.name in os.listdir(self.rank_dir)
            except OSError:
                released = sentinel.exists()
            if released:
                # A release this run did not write does not count, and a
                # half-written one reads as no release and is read again.
                try:
                    stamp = sentinel.read_text(encoding="utf-8").strip()
                except OSError:
                    stamp = None
                released = stamp == self._run_stamp()
            if released:
                print(f"node {self.node_rank} was released by rank 0", flush=True)
                return
            exit_code = self.process.poll()
            if exit_code is not None:
                self.fail(
                    f"node {self.node_rank} lost its {self.role} server with exit "
                    f"code {exit_code} before rank 0 completed"
                )
            time.sleep(PEER_POLL_SECONDS)
        self.fail(
            f"node {self.node_rank} served for {budget}s and rank 0 never published "
            f"its completion at {sentinel}"
        )

    def test_public_pd_perf_suite(self):
        if self.node_rank != 0:
            self._hold_until_rank_zero_completes()
            return

        measurements = [self._measure(plan_entry) for plan_entry in self.plan]
        report = build_report(
            self.test_config,
            measurements,
            provenance=self._provenance(self._accelerator()),
        )
        write_report_files(report, self.report_dir)
        print(render_summary(report), flush=True)

        self.assertEqual(
            report["summary"]["failed"],
            0,
            json.dumps(
                {
                    record["id"]: record["reason_code"]
                    for record in report["measurements"]
                    if record["status"] == "failed"
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )


__all__ = [
    "MEASUREMENT_HOLD_BUDGET_SECONDS",
    "PDPerfSuiteMixin",
    "PEER_POLL_SECONDS",
    "RESULTS_DIR_ENV",
    "TEST_CONFIG_PATH_ENV",
    "WORKER_HOLD_MARGIN_SECONDS",
]
