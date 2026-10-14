"""The half of the PPU performance harness that needs a device and a server.

:mod:`perf_eval_kit` stays hardware-free so the unit suite can validate every
config on a laptop; this module is the part that imports ``torch`` through
``sglang.test.test_utils`` and drives ``sglang.bench_serving`` against a live
server, so only the PPU performance test files import it.  Those files exist one
per device topology because ``register_ppu_ci`` registers a suite per file, and
everything else that differs between models lives in the reviewed test config,
so the mixin below is the entire body of such a test.

It is a plain mixin rather than a ``TestCase`` subclass for the same reason
``AnswerSuiteMixin`` is: the test files are executed as ``python3 <file> -f``,
and ``unittest.main()`` collects every ``TestCase`` subclass it finds in the
module namespace, so a shared base class would also run itself with no
configuration selected.

One server, several measurements.  A config states every workload its
checkpoint is measured under, and they share a single launch: loading a 2.4T
checkpoint costs more than the measurements do, and re-launching per workload
would spend the nightly budget on model loading.  What keeps them from
contaminating each other is an explicit flush of the KV cache between them,
asserted rather than assumed -- see :meth:`PerfSuiteMixin._flush_cache`.
"""

import json
import os
import time
from pathlib import Path

import requests
import torch

from sglang.bench_serving import get_auth_headers, run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.kits.environment_fingerprint import EnvironmentRecorder
from sglang.test.kits.perf_eval_kit import (
    MeasurementError,
    build_perf_server_args,
    build_report,
    canonical_digest,
    failed_measurement_record,
    load_json,
    measurement_record,
    perf_provenance,
    perf_server_environment,
    render_summary,
    resolve_distributed_runtime,
    resolve_measurement_plan,
    validate_test_config,
    write_report_files,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    get_benchmark_args,
    popen_launch_server,
)

TEST_CONFIG_PATH_ENV = "SGLANG_PPU_PERF_TEST_CONFIG"
RESULTS_DIR_ENV = "SGLANG_PPU_PERF_RESULTS_DIR"

# What a worker node allows rank 0 beyond the measurement budget the config
# describes: the report write and the interval between two nodes observing the
# same NAS directory take seconds, and this covers them.  It is a guard against
# an unbounded hold, not the run's real fence, which is the pod timeout the
# workflow sets.
WORKER_HOLD_MARGIN_SECONDS = 900

# How long a single measurement may take before a worker node stops believing
# rank 0 is still working.  A measurement's own duration has no bound in the
# config -- that is the number being measured -- so the hold budget is built
# from this instead, per measurement, plus the flush the suite performs after
# each one.  Generous on purpose: a 64k-token prefill sweep at low concurrency
# is minutes of work, and the cost of overestimating here is that a worker pod
# outlives a hung rank 0 until the workflow's own timeout, while the cost of
# underestimating is a healthy run reported as three failed pods.
MEASUREMENT_HOLD_BUDGET_SECONDS = 3600


class PerfSuiteMixin:
    """The reviewed performance contract, executed against a live server.

    A test file supplies the configuration and nothing else::

        class TestPPUQwen38PerfTwoNode(PerfSuiteMixin, unittest.TestCase):
            default_test_config_path = (
                Path(__file__).parent / "configs" / "qwen3_8" / "....json"
            )

    ``SGLANG_PPU_PERF_TEST_CONFIG`` overrides that default, which is how the
    workflow keeps the file it warms and the file the test reads identical.  A
    config that does not match the file it is handed to still fails loudly:
    validation ties ``tp_size`` times ``pp_size`` to the declared devices, and
    the preflight below ties the checkpoint to the devices actually visible.

    A config that declares ``hardware.nnodes`` greater than one turns the same
    file into one node of a group: every node runs this mixin, rank 0 owns the
    HTTP API and the numbers, and the other ranks hold their devices in the
    group until rank 0 is done.  Which node this process is comes from the
    environment, not from the config; see
    :func:`perf_eval_kit.resolve_distributed_runtime`.

    Unlike the Answer suites there is no ``data_root``: a measurement's workload
    is generated from the length and count the config states, so the suite has
    no corpus asset to pin to the checkout.
    """

    default_test_config_path = None
    distributed = None
    node_rank = 0

    @classmethod
    def _load_test_config(cls):
        configured_path = os.environ.get(TEST_CONFIG_PATH_ENV)
        config_path = (
            Path(configured_path) if configured_path else cls.default_test_config_path
        )
        if config_path is None:
            raise RuntimeError(
                "no performance test config selected: set default_test_config_path "
                f"on the test class or {TEST_CONFIG_PATH_ENV} in the environment"
            )
        config = load_json(config_path)
        validate_test_config(config)
        return config, Path(config_path)

    @classmethod
    def _provenance(cls, accelerator):
        provenance = perf_provenance(cls.test_config, accelerator=accelerator)
        provenance["test_config_id"] = cls.test_config["test_id"]
        provenance["test_config_sha256"] = canonical_digest(cls.test_config)
        return provenance

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

        ``visible_device_count`` keeps meaning "what this process can see", so a
        single-node report is shaped exactly like the ones already collected;
        the multi-node keys are additive.  A node whose inventory is absent is
        named rather than silently dropped, because a throughput number
        attributed to thirty-two devices and a throughput number attributed to
        eight nodes that reported nothing are different pieces of evidence.
        """

        accelerator = cls._local_accelerator()
        if cls.distributed is None:
            return accelerator
        accelerator["nnodes"] = cls.distributed["nnodes"]
        accelerator["node_rank"] = cls.node_rank
        accelerator["node_name"] = os.environ.get("NODE_NAME")
        accelerator["dist_init_addr"] = cls.distributed["dist_init_addr"]
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
    def _write_node_inventory(cls):
        """Publish this node's devices where rank 0 can read them.

        The action streams the log of worker-0 only, so without this the report
        could attribute a multi-node throughput number to eight devices.  The
        write is staged and renamed because the nodes reach this directory over
        NFS, where a reader can otherwise observe a partial file.
        """

        inventory = {
            "node_rank": cls.node_rank,
            "node_name": os.environ.get("NODE_NAME"),
            **cls._local_accelerator(),
        }
        path = cls._node_inventory_path(cls.node_rank)
        staged = path.parent / f"{path.name}.partial"
        staged.write_text(
            json.dumps(inventory, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged.replace(path)

    @classmethod
    def _read_node_inventories(cls):
        nodes = []
        missing = []
        for node_rank in range(cls.distributed["nnodes"]):
            try:
                nodes.append(load_json(cls._node_inventory_path(node_rank)))
            except (OSError, ValueError):
                missing.append(node_rank)
        return nodes, missing

    @classmethod
    def _sentinel_path(cls):
        return cls.rank_dir / "rank0-complete"

    @classmethod
    def _release_worker_nodes(cls):
        """Tell the worker nodes that rank 0 is done with them.

        Called from the teardown and from the setup failure path both: a setup
        that never launched a server still has to release nodes that did, or
        those pods sit on their devices until their own budget expires.
        """

        if cls.distributed is None or cls.node_rank != 0:
            return
        try:
            cls.rank_dir.mkdir(parents=True, exist_ok=True)
            cls._sentinel_path().write_text(
                f"{os.environ.get('GITHUB_RUN_ID', 'local')}\n", encoding="utf-8"
            )
        except OSError as error:
            print(
                f"failed to release the worker nodes: {type(error).__name__}: {error}",
                flush=True,
            )

    @classmethod
    def _write_setup_failure(cls, stage, detail):
        """Publish the report a run that never reached a measurement still owes.

        Every planned measurement is recorded as unmeasured with the stage that
        stopped it, so the artifact of a failed run has the same shape as the
        artifact of a successful one and the run page still names what was
        supposed to be measured.
        """

        try:
            accelerator = cls._local_accelerator()
        except Exception:
            accelerator = {"visible_device_count": None}
        if cls.distributed is not None:
            accelerator["nnodes"] = cls.distributed["nnodes"]
            accelerator["node_rank"] = cls.node_rank
        measurements = [
            failed_measurement_record(entry, "server_start_failed", detail)
            for entry in cls.plan
        ]
        provenance = cls._provenance(accelerator)
        provenance["setup_stage"] = stage
        report = build_report(cls.test_config, measurements, provenance=provenance)
        write_report_files(report, cls.report_dir)
        print(render_summary(report), flush=True)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_config, cls.test_config_path = cls._load_test_config()
        cls.model_config = cls.test_config["model"]
        cls.server_config = cls.test_config["server"]["parameters"]
        cls.plan = resolve_measurement_plan(cls.test_config)
        cls.output_dir = Path(os.environ.get(RESULTS_DIR_ENV, "ppu-perf-artifacts"))
        # Before the try: a config that asks for several nodes without giving
        # them a way to reach each other is a misconfiguration of the caller,
        # and the structured evidence below is written to a directory this very
        # check is what establishes.
        cls.distributed = resolve_distributed_runtime(cls.test_config)
        cls.node_rank = 0 if cls.distributed is None else cls.distributed["node_rank"]
        cls.rank_dir = cls._resolve_rank_dir()
        # Rank 0 publishes the numbers where the workflow collects them; a
        # worker node writes its own evidence one level down, because all nodes
        # share this directory and a worker's report would otherwise overwrite
        # the numbers with its own view of a run it did not measure.
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
        stage = "runner_preflight"
        try:
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
                    f"{checkpoint_config.get('model_type')!r} is not one of the "
                    f"accepted types {accepted_model_types}"
                )
            expected_device_count = len(cls.test_config["hardware"]["visible_devices"])
            if torch.cuda.device_count() != expected_device_count:
                raise RuntimeError(
                    f"{cls.test_config['test_id']} requires {expected_device_count} "
                    f"visible PPU devices per node; node {cls.node_rank} found "
                    f"{torch.cuda.device_count()}"
                )

            if cls.distributed is not None:
                cls.rank_dir.mkdir(parents=True, exist_ok=True)
                if cls.node_rank == 0:
                    # A worker only looks for this after its own server is
                    # ready, which cannot happen before rank 0 has joined the
                    # group, so clearing it here cannot race a worker into an
                    # early exit.
                    cls._sentinel_path().unlink(missing_ok=True)

            cls.base_url = DEFAULT_URL_FOR_TEST
            stage = "server_start"
            server_args = build_perf_server_args(
                cls.test_config, distributed=cls.distributed
            )
            server_env = cls._server_environment()
            cls.environment_recorder.capture(
                "before_start", server_args=server_args, server_env=server_env
            )
            cls.process = popen_launch_server(
                model=cls.model_path,
                base_url=cls.base_url,
                timeout=cls.test_config["server"]["startup_timeout_seconds"],
                other_args=server_args,
                env=server_env,
            )
            cls.environment_recorder.capture("server_ready", server_pid=cls.process.pid)
            # A worker rank serves a dummy health endpoint once its own
            # schedulers are ready, so the launch above returns on every node
            # and this is the point where each one knows what it holds.
            if cls.distributed is not None:
                stage = "node_inventory"
                cls._write_node_inventory()
        except Exception as exc:
            cls.environment_recorder.capture(
                "setup_failed", server_pid=getattr(cls.process, "pid", None)
            )
            try:
                cls._write_setup_failure(stage, f"{type(exc).__name__}: {exc}")
            except Exception as report_error:
                print(
                    "failed to write structured performance setup evidence: "
                    f"{type(report_error).__name__}",
                    flush=True,
                )
            cls._release_worker_nodes()
            raise

    @classmethod
    def _server_environment(cls):
        """What the server is launched with on top of the inherited environment.

        None when the config names no variables and the group is a single node,
        which leaves the launcher's environment untouched.

        The reviewed variables are applied first and the group's description of
        itself second, so a config cannot rename the rendezvous; the schema does
        not admit those names anyway, and the ordering says so without relying
        on that.

        ``MASTER_ADDR`` is taken from the rendezvous actually in force, not
        copied from the injected variable of the same name: when the address
        came from ``SGLANG_PPU_PERF_DIST_INIT_ADDR`` it did so because the
        injected one does not resolve, and re-exporting that would defeat the
        override.
        """

        environment = perf_server_environment(cls.test_config)
        if cls.distributed is not None:
            master_addr, _, _ = cls.distributed["dist_init_addr"].rpartition(":")
            environment.update(
                {
                    "MASTER_ADDR": master_addr,
                    "NNODES": str(cls.distributed["nnodes"]),
                    "RANK": str(cls.distributed["node_rank"]),
                }
            )
        return environment or None

    @classmethod
    def _resolve_rank_dir(cls):
        """The directory the nodes of a group use to address each other.

        None for a single-node config.  The multi-node case insists on an
        absolute results directory because the nodes exchange their device
        inventory and rank 0's completion through it, and the default is a
        relative path that resolves to a different directory in each pod.
        """

        if cls.distributed is None:
            return None
        if not cls.output_dir.is_absolute():
            raise RuntimeError(
                f"a multi-node performance config needs {RESULTS_DIR_ENV} set to a "
                f"path every node of the group shares; got {cls.output_dir}"
            )
        return cls.output_dir / "ranks"

    @classmethod
    def tearDownClass(cls):
        recorder = getattr(cls, "environment_recorder", None)
        if recorder is not None:
            recorder.capture(
                "before_stop",
                server_pid=getattr(getattr(cls, "process", None), "pid", None),
            )
        # Released before the server is killed, and not the other way round: a
        # worker treats the loss of its own server as a failure, and killing
        # rank 0's process is what makes the workers' schedulers exit, so a
        # teardown in the other order would report healthy worker pods as
        # failed.
        cls._release_worker_nodes()
        process = getattr(cls, "process", None)
        if process is not None:
            kill_process_tree(process.pid)
        super().tearDownClass()

    def build_benchmark_args(self, plan_entry):
        """The ``bench_serving`` arguments one measurement is run with.

        ``get_benchmark_args`` is the repository's own constructor for this
        namespace, so the fields ``run_benchmark`` reads without a ``hasattr``
        guard all exist; the assignments after it are the ones it does not take
        as parameters and whose defaults are wrong for this suite:

        ``random_range_ratio`` -- its default of 0.0 draws each length from
        ``[0, requested]``, which measures a distribution rather than the length
        the case names.  The reviewed configs state 1.0, which pins it.

        ``tokenize_prompt`` -- sends the prompt to ``/generate`` as ids rather
        than as text, so the prompt length the server sees is the requested one
        exactly.  Without it the ids are decoded to text and re-encoded by the
        server, and a 64k-token prefill measurement would be attributed to a
        length nobody chose.  ``run_benchmark`` asserts this needs the
        ``sglang`` backend, which is the backend below.

        ``model`` and ``served_model_name`` -- left as None,
        ``run_benchmark`` asks ``/v1/models`` for the name and exits the process
        on failure.  Naming both here keeps the measurement independent of that
        round trip and makes the served name in the request identical to the one
        the server was launched with.

        ``tokenizer`` is the checkpoint path for the same reason: it is what
        ``get_tokenizer`` loads with ``trust_remote_code``, and these
        checkpoints define their own tokenizer classes.

        ``output_file`` -- ``benchmark`` appends its raw result to a JSONL file
        unconditionally, deriving a name in the current working directory when
        this is None.  Pointing it into the evidence directory is what makes
        that write a collected artifact instead of a stray file next to
        whatever the runner happened to cd into.

        ``output_details`` -- puts the per-request lengths, TTFTs and errors in
        that file as well, which is the raw evidence behind every number in the
        report.

        ``flush_cache`` -- ``benchmark`` flushes once after its own warmup, so
        the measured phase never reuses what the warmup left behind.  This suite
        flushes again after each measurement; see :meth:`_flush_cache`.
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
        args.flush_cache = True
        args.output_details = True
        args.output_file = str(self.raw_dir / f"{plan_entry['id']}.jsonl")
        return args

    def _flush_cache(self, plan_entry):
        """Drop the KV cache between two measurements of one server.

        The measurements of a config share a launch, so without this the second
        one would begin against a radix tree holding the first one's prompts and
        report a prefill it did not perform.  The response code is asserted
        rather than ignored -- ``bench_serving``'s own post-warmup flush ignores
        it -- because a flush that silently failed would turn every subsequent
        measurement into a cache-hit measurement wearing a prefill's name.
        """

        response = requests.post(
            f"{self.base_url}/flush_cache",
            headers=get_auth_headers(),
            timeout=plan_entry["flush_cache_timeout_seconds"],
        )
        if response.status_code != 200:
            raise MeasurementError(
                "cache_flush_failed",
                f"flushing the KV cache after {plan_entry['id']} returned HTTP "
                f"{response.status_code}",
            )

    def _run_warmup_passes(self, plan_entry):
        """Run and discard the measurement's warmup passes.

        Each is a full pass at the measurement's own shape, so the recorded pass
        that follows begins with this image's compilation and allocator state
        already warm; the KV cache is flushed after each, so the recorded pass
        does not also begin with the warmup's prompts cached and report a prefill
        it did not perform.  ``warmup_passes`` is zero on every colocated config
        -- a prefill measurement pays its compilation cost once and there is
        nothing to discard -- so this is a no-op here today, and exists so the
        two lines drive a measurement the same way should a colocated case ever
        want the disaggregated line's "run twice, keep the second" shape.

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

        Every failure mode a measurement has is recorded as an unmeasured
        measurement rather than propagated, so one workload that cannot be
        measured does not cost the run the workloads after it -- the server is
        already up and the remaining measurements are the expensive thing to
        redo.  The run still ends red, because ``build_report`` counts them.
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
        """A worker node's share of the suite is to stay in the group.

        Only rank 0 holds the tokenizer and the HTTP API; this node's schedulers
        own their slice of every request, so returning early would tear the
        group down under rank 0.  The pass condition is therefore that the node
        held its devices until rank 0 published its completion, and the failure
        conditions are its own server dying first, or that completion never
        arriving.
        """

        sentinel = self._sentinel_path()
        budget = (
            MEASUREMENT_HOLD_BUDGET_SECONDS * len(self.plan)
            + WORKER_HOLD_MARGIN_SECONDS
        )
        deadline = time.monotonic() + budget
        print(
            f"node {self.node_rank} is holding the group and waiting for {sentinel}",
            flush=True,
        )
        while time.monotonic() < deadline:
            # listdir rather than exists: the directory is on NFS, and a readdir
            # revalidates the entry that a cached negative lookup would not.
            try:
                released = sentinel.name in os.listdir(self.rank_dir)
            except OSError:
                released = sentinel.exists()
            if released:
                print(f"node {self.node_rank} was released by rank 0", flush=True)
                return
            exit_code = self.process.poll()
            if exit_code is not None:
                self.fail(
                    f"node {self.node_rank} lost its server with exit code "
                    f"{exit_code} before rank 0 completed"
                )
            time.sleep(10)
        self.fail(
            f"node {self.node_rank} held the group for {budget}s and rank 0 never "
            f"published its completion at {sentinel}"
        )

    @property
    def raw_dir(self):
        """Where the raw ``bench_serving`` output of each measurement lands."""

        path = self.report_dir / "raw"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_public_perf_suite(self):
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
