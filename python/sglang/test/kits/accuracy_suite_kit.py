"""The half of the Accuracy line that needs torch, a server, and EvalScope.

`accuracy_eval_kit` stays hardware-free so the CPU unit test can exercise the
whole contract; this module owns the live server and the `evalscope` subprocess,
so only the registered PPU Accuracy test files import it.  Those files exist one
per model and dataset because `register_ppu_ci` registers a suite per file, and
everything that differs between them lives in the reviewed test config, so the
mixin below is the entire body of such a test.

It is a plain mixin rather than a `TestCase` subclass for the same reason
`AnswerSuiteMixin` is: the test files run as `python3 <file> -f`, and
`unittest.main()` collects every `TestCase` subclass in the module namespace, so
a shared base class would also run itself with no configuration selected.

Unlike the Answer and perf lines this one is single-node only, and the schema
says so by having no `nnodes`.  Every internal `P0_daily` case this line is
ported from serves its model on one board, so the rank directory, the device
inventory exchange and the worker release sentinel would all be machinery with
no case behind it.
"""

import os
import subprocess
from pathlib import Path
from shutil import which

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.accuracy_eval_kit import (
    AccuracyEvalError,
    accuracy_provenance,
    accuracy_server_environment,
    build_accuracy_server_args,
    build_evalscope_command,
    build_report,
    failed_measurement_record,
    load_json,
    locate_evalscope_report,
    measurement_record,
    read_evalscope_report,
    render_summary,
    resolve_evaluation_plan,
    validate_test_config,
    write_report_files,
)
from sglang.test.kits.environment_fingerprint import EnvironmentRecorder
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server

TEST_CONFIG_PATH_ENV = "SGLANG_PPU_ACCURACY_TEST_CONFIG"
RESULTS_DIR_ENV = "SGLANG_PPU_ACCURACY_RESULTS_DIR"
# Where the `evalscope` entry point is.  A name rather than a path by default,
# but the PPU pods install EvalScope into a virtual environment of its own --
# see the isolation note in `accuracy_eval_kit` -- so the runner points at that
# environment's script instead of putting it on `PATH`, where it would shadow
# nothing but would also let a stray system install be picked up silently.
EVALSCOPE_BIN_ENV = "SGLANG_PPU_EVALSCOPE_BIN"
# Where the staged dataset is, when it is not where the config says.
#
# The only path in a reviewed config that this repository cannot check: a
# checkpoint path is shared with the Answer and perf lines and has been loaded
# many times, while these public splits are staged by hand on shared storage and
# the convention for where is not ours to fix.  An override rather than a
# guessing preflight, and one recorded in the provenance of every report it
# affects, so a score can never be traced back to data other than the data it
# was measured on.
DATASET_DIR_ENV = "SGLANG_PPU_ACCURACY_DATASET_DIR"

# How much of the EvalScope log to echo when a run fails.  The whole log is kept
# as an artifact; this is what a reader sees on the run page without downloading
# it, and a stack trace or an HTTP error from the last request fits.
LOG_TAIL_LINES = 60

# How long the NLTK corpora a scorer needs get to arrive.  Generous because the
# pods reach a mirror rather than a local copy, and small against the hours the
# evaluation itself takes -- the point is to bound a hang, not to be tight.
NLTK_FETCH_TIMEOUT_SECONDS = 600

# The program that resolves one corpus inside EvalScope's own environment.  A
# subprocess rather than an import because EvalScope lives in a virtual
# environment of its own, so this test process may have no `nltk` at all and
# certainly not the one that will do the scoring.
_NLTK_RESOLVE_PROGRAM = """
import sys

import nltk

download_id, lookup_path = sys.argv[1], sys.argv[2]
try:
    nltk.data.find(lookup_path)
except LookupError:
    nltk.download(download_id, quiet=True)
    nltk.data.find(lookup_path)
print(nltk.data.find(lookup_path))
"""


class AccuracySuiteMixin:
    """The reviewed Accuracy contract, executed against a live server.

    A test file supplies the configuration and nothing else::

        class TestPPUGLM52AccuracyGSM8K(AccuracySuiteMixin, unittest.TestCase):
            default_test_config_path = (
                Path(__file__).parent / "configs" / ... / "....json"
            )

    `SGLANG_PPU_ACCURACY_TEST_CONFIG` overrides that default, which is how the
    workflow keeps the file it warms and the file the test reads identical.  A
    config that does not match the file it is handed to still fails loudly:
    validation ties `tp_size` to the declared devices, and the preflight below
    ties the checkpoint to the devices actually visible.

    The dataset is not a repository asset here, unlike the Answer corpus: these
    are public splits far too large to check in, so the config names a staged
    directory on shared storage and the preflight refuses a run whose dataset is
    absent rather than letting EvalScope try to download it.
    """

    default_test_config_path = None

    @classmethod
    def _load_test_config(cls):
        configured_path = os.environ.get(TEST_CONFIG_PATH_ENV)
        config_path = (
            Path(configured_path) if configured_path else cls.default_test_config_path
        )
        if config_path is None:
            raise RuntimeError(
                "no Accuracy test config selected: set default_test_config_path on "
                f"the test class or {TEST_CONFIG_PATH_ENV} in the environment"
            )
        config = load_json(config_path)
        validate_test_config(config)
        return config, Path(config_path)

    @classmethod
    def _accelerator(cls):
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
    def _provenance(cls, accelerator):
        provenance = accuracy_provenance(cls.test_config, accelerator=accelerator)
        provenance["evalscope_command"] = cls.evalscope_command
        # The directory actually read, which is the reviewed one unless the
        # runner overrode it. Recorded either way: a score is only as traceable
        # as the data behind it, and `evaluation` above carries the config's
        # value rather than the effective one.
        provenance["dataset_dir"] = cls.plan["dataset_dir"]
        provenance["dataset_dir_overridden"] = bool(cls.dataset_dir_override)
        return provenance

    @classmethod
    def _write_setup_failure(cls, reason_code, detail):
        """Publish the same report shape a completed run publishes.

        A setup that never reached EvalScope still has to leave a machine
        readable verdict behind: the workflow reads `result.json` for the
        annotations, and a missing file is indistinguishable from a pod that
        died, which is a different thing to investigate.
        """

        try:
            accelerator = cls._accelerator()
        except Exception:
            accelerator = {"visible_device_count": None}
        record = failed_measurement_record(cls.plan, reason_code, detail)
        report = build_report(
            cls.test_config, [record], provenance=cls._provenance(accelerator)
        )
        write_report_files(report, cls.output_dir)
        print(render_summary(report), flush=True)

    @classmethod
    def _resolve_evalscope(cls):
        """Where `evalscope` is, checked before a checkpoint is loaded.

        Deliberately before the server: the whole run is pointless without the
        evaluator, and finding that out costs seconds here against the tens of
        minutes a weight load takes.  The same reasoning puts the dataset check
        next to it.
        """

        executable = os.environ.get(EVALSCOPE_BIN_ENV, "evalscope")
        if os.path.sep in executable:
            path = Path(executable)
            if not path.is_file() or not os.access(path, os.X_OK):
                raise RuntimeError(
                    f"{EVALSCOPE_BIN_ENV} points at {executable}, which is not an "
                    "executable file"
                )
            return str(path)
        resolved = which(executable)
        if resolved is None:
            raise RuntimeError(
                f"{executable!r} is not on PATH; install EvalScope and point "
                f"{EVALSCOPE_BIN_ENV} at its entry point"
            )
        return resolved

    @classmethod
    def _evalscope_interpreter(cls):
        """The Python that will do the scoring.

        Taken from beside the `evalscope` entry point rather than from
        `sys.executable`: the two are different interpreters here by design, and
        the one worth asking about a corpus is the one that will look for it.
        """

        bin_dir = Path(cls.evalscope_bin).parent
        for name in ("python3", "python"):
            candidate = bin_dir / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        raise RuntimeError(
            f"no python interpreter beside {cls.evalscope_bin}; EvalScope is "
            "expected to be installed into an environment whose entry point and "
            "interpreter share a directory"
        )

    @classmethod
    def _resolve_metric_resources(cls):
        """Make the scorer's corpora present before a single sample is scored.

        EvalScope fetches them lazily, inside the scoring of the first sample
        that needs one, and charges a failed fetch to that sample rather than to
        the run: the cost of losing the race is a smaller denominator, which only
        the sample-count check downstream makes visible. Fetching here moves the
        whole download before the first sample, so there is no race to lose.

        Best effort on purpose. This is an improvement on a run that would
        otherwise fetch mid-scoring, not a precondition for one -- EvalScope
        reaches its own mirror where this reaches NLTK's index, and refusing a
        run because *our* route failed would turn five entries that score all 541
        prompts today into red ones. So a failure here is a warning that names
        the exposure, and the sample-count check remains the thing that stops a
        short run from being read as a score.

        Deliberately before the server: a corpus this line can fetch takes a
        minute or two, against the tens of minutes a weight load takes.
        """

        resources = cls.plan["nltk_resources"]
        if not resources:
            return
        for download_id, lookup_path in resources:
            try:
                completed = subprocess.run(
                    [
                        cls._evalscope_interpreter(),
                        "-c",
                        _NLTK_RESOLVE_PROGRAM,
                        download_id,
                        lookup_path,
                    ],
                    capture_output=True,
                    text=True,
                    env=cls._evalscope_environment(),
                    timeout=NLTK_FETCH_TIMEOUT_SECONDS,
                )
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                cls._warn_unresolved_resource(
                    download_id, f"{type(exc).__name__}: {exc}"
                )
                continue
            if completed.returncode != 0:
                cls._warn_unresolved_resource(
                    download_id, completed.stderr.strip()[-800:]
                )
                continue
            print(
                f"the NLTK corpus {download_id} that scores {cls.plan['dataset']} "
                f"is at {completed.stdout.strip()}",
                flush=True,
            )

    @classmethod
    def _warn_unresolved_resource(cls, download_id, detail):
        print(
            f"warning: could not resolve the NLTK corpus {download_id!r} that "
            f"scores {cls.plan['dataset']} before evaluation, so EvalScope will "
            "fetch it while scoring and may score one sample fewer than the "
            "split holds; staging it on shared storage and exporting NLTK_DATA "
            f"would remove the fetch: {detail}",
            flush=True,
        )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_config, cls.test_config_path = cls._load_test_config()
        cls.plan = resolve_evaluation_plan(cls.test_config)
        cls.dataset_dir_override = os.environ.get(DATASET_DIR_ENV)
        if cls.dataset_dir_override:
            cls.plan["dataset_dir"] = cls.dataset_dir_override
        cls.model_config = cls.test_config["model"]
        cls.output_dir = Path(os.environ.get(RESULTS_DIR_ENV, "ppu-accuracy-artifacts"))
        # EvalScope writes its predictions, reviews and report under here, and
        # the report is read back from it, so the suite names the directory
        # rather than letting the tool timestamp one.
        cls.work_dir = cls.output_dir / "evalscope"
        cls.evalscope_log = cls.output_dir / "evalscope.log"
        cls.model_path = cls.model_config["path"]
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = None
        cls.evalscope_command = None
        cls.environment_recorder = EnvironmentRecorder(cls.output_dir, cls.test_config)
        reason_code = "server_start_failed"
        try:
            cls.evalscope_bin = cls._resolve_evalscope()
            dataset_dir = Path(cls.plan["dataset_dir"])
            if not dataset_dir.is_dir():
                source = (
                    DATASET_DIR_ENV
                    if cls.dataset_dir_override
                    else str(cls.test_config_path)
                )
                raise RuntimeError(
                    f"the {cls.plan['dataset']} directory {dataset_dir} configured "
                    f"by {source} is missing; the pods are offline, so the dataset "
                    f"has to be staged there beforehand, or {DATASET_DIR_ENV} set "
                    "to where it was staged"
                )
            if not Path(cls.model_path).is_dir():
                raise RuntimeError(
                    f"the model directory configured by {cls.test_config_path} is "
                    "missing"
                )
            model_dir = Path(cls.model_path)
            if model_dir.name != cls.model_config["checkpoint_name"]:
                raise RuntimeError(
                    f"model.path must identify {cls.model_config['checkpoint_name']}"
                )
            checkpoint_config_path = model_dir / "config.json"
            if not checkpoint_config_path.is_file():
                raise RuntimeError(f"{cls.model_path} does not contain config.json")
            checkpoint_config = load_json(checkpoint_config_path)
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
                    f"visible PPU devices; found {torch.cuda.device_count()}"
                )

            cls._resolve_metric_resources()

            cls.evalscope_command = build_evalscope_command(
                cls.test_config,
                cls.plan,
                base_url=cls.base_url,
                work_dir=cls.work_dir,
                executable=cls.evalscope_bin,
            )
            server_args = build_accuracy_server_args(cls.test_config)
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
        except Exception as exc:
            cls.environment_recorder.capture(
                "setup_failed", server_pid=getattr(cls.process, "pid", None)
            )
            try:
                cls._write_setup_failure(reason_code, f"{type(exc).__name__}: {exc}")
            except Exception as report_error:
                print(
                    "failed to write structured Accuracy setup evidence: "
                    f"{type(report_error).__name__}: {report_error}",
                    flush=True,
                )
            if cls.process is not None:
                kill_process_tree(cls.process.pid)
            raise

    @classmethod
    def _server_environment(cls):
        """What the server is launched with on top of the inherited environment.

        None when the config names no variables, which leaves the launcher's
        environment untouched -- the case for an entry that pins nothing.
        """

        environment = accuracy_server_environment(cls.test_config)
        return environment or None

    @classmethod
    def _evalscope_environment(cls):
        """The environment EvalScope runs in.

        Inherited wholesale and then narrowed, because the offline settings the
        pod already exports -- the Hugging Face cache and its offline flag -- are
        the runner's to set and this process has no better value for them.

        `NLTK_DATA` is among them wherever the dataset's scorer needs a corpus:
        the IFEval jobs export it, pointing at a corpus staged on the shared NAS,
        so `_resolve_metric_resources` finds it there and fetches nothing. Where
        it is unset the resolver falls back to a download into the interpreter's
        default search path, which still beats leaving EvalScope to fetch during
        the scoring and charge a failure to whichever sample was unlucky.

        The two cache directories are redirected under the work directory so a
        run leaves nothing in a home directory that the next run would inherit:
        a stale cache is how an evaluation silently scores a dataset other than
        the one the config names.
        """

        environment = dict(os.environ)
        environment["EVALSCOPE_CACHE"] = str(cls.work_dir / "cache")
        environment["MODELSCOPE_CACHE"] = str(cls.work_dir / "modelscope")
        return environment

    @classmethod
    def tearDownClass(cls):
        recorder = getattr(cls, "environment_recorder", None)
        if recorder is not None:
            recorder.capture(
                "before_stop",
                server_pid=getattr(getattr(cls, "process", None), "pid", None),
            )
        if cls.process is not None:
            kill_process_tree(cls.process.pid)
        super().tearDownClass()

    def _log_tail(self):
        try:
            lines = self.evalscope_log.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            return "the EvalScope log could not be read"
        return "\n".join(lines[-LOG_TAIL_LINES:])

    def _run_evalscope(self):
        """Run EvalScope to completion, or say why it did not.

        Its output goes straight to a file rather than through a pipe: the run
        lasts hours and this is the only way to watch it from inside the pod,
        and a pipe nobody drains is how a chatty subprocess deadlocks.
        """

        self.work_dir.mkdir(parents=True, exist_ok=True)
        print(
            "running EvalScope: " + " ".join(self.evalscope_command),
            flush=True,
        )
        with self.evalscope_log.open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(
                    self.evalscope_command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=self._evalscope_environment(),
                    timeout=self.plan["timeout_seconds"],
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return (
                    "evalscope_failed",
                    f"EvalScope did not finish within {self.plan['timeout_seconds']}s",
                )
            except OSError as error:
                return "evalscope_failed", f"{type(error).__name__}: {error}"
        if completed.returncode != 0:
            return (
                "evalscope_failed",
                f"EvalScope exited {completed.returncode}; last lines of its log:\n"
                f"{self._log_tail()}",
            )
        return None

    def test_public_accuracy_benchmark(self):
        """Score the served model on the dataset the config names.

        The verdict is asserted at the very end, after the report has been
        written: a failed run that leaves no artifact behind is a run nobody can
        diagnose, and the assertion is what turns the report red.
        """

        failure = self._run_evalscope()
        if failure is None:
            try:
                report_path = locate_evalscope_report(
                    self.work_dir, self.plan["dataset"]
                )
            except AccuracyEvalError as error:
                report_path = None
                failure = ("report_unreadable", str(error))
            if failure is None and report_path is None:
                failure = (
                    "report_missing",
                    f"EvalScope succeeded but wrote no report for "
                    f"{self.plan['dataset']} under {self.work_dir / 'reports'}",
                )

        if failure is None:
            try:
                reading = read_evalscope_report(report_path)
            except (AccuracyEvalError, OSError, ValueError) as error:
                failure = (
                    "report_unreadable",
                    f"{report_path}: {type(error).__name__}: {error}",
                )

        if failure is None:
            record = measurement_record(self.plan, reading)
        else:
            record = failed_measurement_record(self.plan, failure[0], failure[1])

        report = build_report(
            self.test_config, [record], provenance=self._provenance(self._accelerator())
        )
        write_report_files(report, self.output_dir)
        summary = render_summary(report)
        print(summary, flush=True)
        self.assertEqual(
            report["summary"]["verdict"],
            "passed",
            msg=f"{record['reason_code']}: {record['detail']}",
        )
