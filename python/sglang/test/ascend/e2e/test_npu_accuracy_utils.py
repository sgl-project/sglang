import glob
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    PdMixNodeMixin,
    PdSepNodeMixin,
    check_role,
    kill_process_group,
)
from sglang.test.ascend.test_ascend_utils import maybe_start_hicache_storage_service
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    dump_metric,
    popen_launch_server,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

EVALSCOPE = "evalscope"
BENCHMARK_TOOL_DEFAULT = EVALSCOPE

PYTHON_FOR_TEST_TOOL = "test_env_transformers_tool/bin/python"
if not os.path.exists(PYTHON_FOR_TEST_TOOL) or not os.access(
    PYTHON_FOR_TEST_TOOL, os.X_OK
):
    PYTHON_FOR_TEST_TOOL = "python3"
logger.info(f"PYTHON_FOR_TEST_TOOL: {PYTHON_FOR_TEST_TOOL}")

DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 3600
MAX_SERVER_KEEP_ALIVE_TIME = 3600

ACCURACY_TOLERANCE = 0.99

# Dataset total question counts and allowed fluctuation (in questions)
DATASET_QUESTION_COUNTS = {
    "aime25": 30,
    "aime26": 30,
    "gpqa_diamond": 198,
}

DATASET_FLUCTUATION = {
    "aime25": 2,
    "aime26": 2,
    "gpqa_diamond": 5,
    "gsm8k": 3,
}

MAX_RETRY_COUNT = 3

# Datasets whose evalscope adapter shells out to Docker (agent environments).
# NPU pods have no Docker daemon, so these run on a separate amd64 pod that
# provides DinD, while the NPU pods keep serving the model. See
# ``deepswe-eval-plan.md`` (方案 A).
REMOTE_EVAL_DATASETS = frozenset({"deep_swe"})

# DeepSWE eval pod: an amd64 node (kubernetes.io/arch=amd64) holding a
# ``docker:dind`` sidecar plus a python container that drives evalscope.
# Images/arch are overridable through the pod env when the cluster mirrors
# them under a different registry; an empty value keeps the default.
DEEPSWE_EVAL_IMAGE = os.environ.get("DEEPSWE_EVAL_IMAGE") or "python:3.12"
DEEPSWE_DIND_IMAGE = os.environ.get("DEEPSWE_DIND_IMAGE") or "docker:dind"
DEEPSWE_EVAL_NODE_ARCH = os.environ.get("DEEPSWE_EVAL_NODE_ARCH") or "amd64"
# Time to get the pod scheduled and the eval container up (image pulls, node
# availability) versus the wall-clock budget of the evaluation itself.
DEEPSWE_EVAL_SCHEDULE_TIMEOUT = int(
    os.environ.get("DEEPSWE_EVAL_SCHEDULE_TIMEOUT") or "1200"
)
DEEPSWE_EVAL_RUN_TIMEOUT = int(os.environ.get("DEEPSWE_EVAL_RUN_TIMEOUT") or "14400")
# Kubernetes-side safety net: if the router pod (or the whole runner) dies
# before its finally-block deletion runs, kubelet still terminates the eval
# pod once this many seconds pass, instead of leaving a dind pod holding an
# amd64 node and its emptyDir forever. Covers the schedule window plus the
# run window plus slack.
DEEPSWE_EVAL_ACTIVE_DEADLINE = int(
    os.environ.get("DEEPSWE_EVAL_ACTIVE_DEADLINE")
    or str(DEEPSWE_EVAL_SCHEDULE_TIMEOUT + DEEPSWE_EVAL_RUN_TIMEOUT + 900)
)
# The eval container prints its metrics as a single machine-readable line so
# the router pod can forward them to the runner's pod-log monitor.
DEEPSWE_EVAL_METRICS_MARKER = "[DEEPSWE_METRICS]"
# Forwarded eval-pod log lines carry this prefix so they stay distinguishable
# from the router/unittest output in the CI log.
DEEPSWE_EVAL_LOG_PREFIX = "[deepswe-eval]"

SERVER_INITIALIZATION_DELAY = 120

if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
    DEFAULT_SERVER_PORT_FOR_TEST = (
        20000 + int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")[0]) * 100
    )
else:
    DEFAULT_SERVER_PORT_FOR_TEST = (
        20000 + int(os.environ.get("ASCEND_VISIBLE_DEVICES", "0")[0]) * 100
    )
DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{DEFAULT_SERVER_PORT_FOR_TEST + 66}"


def get_accuracy_threshold(datasets, baseline_accuracy):
    """Calculate accuracy threshold based on dataset fluctuation tolerance.

    For datasets with defined fluctuation (aime*, gpqa_diamond), use absolute
    question count tolerance. For others (e.g. mmmu), use percentage tolerance.
    """
    dataset = datasets[0] if datasets else None
    if dataset in DATASET_FLUCTUATION and dataset in DATASET_QUESTION_COUNTS:
        fluctuation = DATASET_FLUCTUATION[dataset] / DATASET_QUESTION_COUNTS[dataset]
        return baseline_accuracy - fluctuation
    return baseline_accuracy * ACCURACY_TOLERANCE


def get_max_retries(datasets):
    """Return max retry count for accuracy tests.

    gpqa and aime datasets support up to MAX_RETRY_COUNT retries.
    mmmu and others use 1 attempt (no retry).
    """
    dataset = datasets[0] if datasets else None
    if dataset in DATASET_FLUCTUATION:
        return MAX_RETRY_COUNT
    return 1


# Exported by the nightly pipeline to pick the dataset of multi-node accuracy
# cases: gsm8k on weekdays (Mon-Fri) and gpqa_diamond on weekends.
ACCURACY_DATASET_ENV = "NPU_ACCURACY_DATASET"

# Sample cap kept for the overridden gsm8k run, to stay inside the step budget.
ACCURACY_DATASET_LIMITS = {"gsm8k": 200}


def apply_accuracy_dataset_override(test_cls):
    """Apply the pipeline-selected dataset on `test_cls`.

    No-op when NPU_ACCURACY_DATASET is unset or selects the dataset the case
    already declares. Cases opt in to the other dataset by declaring its
    baseline in `dataset_baselines`; uncalibrated cases keep their own dataset.
    """
    dataset = os.environ.get(ACCURACY_DATASET_ENV, "").strip()
    if not dataset or dataset in test_cls.datasets:
        return

    baselines = getattr(test_cls, "dataset_baselines", None) or {}
    if dataset not in baselines:
        logger.warning(
            f"[accuracy-dataset] {ACCURACY_DATASET_ENV}={dataset} ignored for "
            f"{test_cls.__name__}: no dataset_baselines['{dataset}'], keeping "
            f"datasets={test_cls.datasets}."
        )
        return

    test_cls.datasets = [dataset]
    test_cls.accuracy = baselines[dataset]
    test_cls.limit = ACCURACY_DATASET_LIMITS.get(dataset, test_cls.limit)
    # dataset_args is keyed by dataset name; keep the selected entry only.
    if test_cls.dataset_args:
        entry = test_cls.dataset_args.get(dataset)
        test_cls.dataset_args = {dataset: entry} if entry else None
    logger.info(
        f"[accuracy-dataset] {ACCURACY_DATASET_ENV}={dataset}: "
        f"datasets={test_cls.datasets} accuracy={test_cls.accuracy} "
        f"limit={test_cls.limit}"
    )


def extract_report_accuracy(report_data):
    """Return the primary accuracy ratio from an evalscope report.

    Older reports carry a top-level ``score``. Schema v2 reports (evalscope
    1.11+) moved it into ``metrics[]``: each entry has ``identity.name`` plus a
    ``score`` already normalised to 0..1, next to ``num``/``num_correct``.
    """
    score = report_data.get("score")
    if isinstance(score, (int, float)):
        return float(score)

    primary_identity = report_data.get("primary_metric_identity")
    primary = (
        primary_identity.get("name") if isinstance(primary_identity, dict) else None
    )
    metrics = report_data.get("metrics")
    if not isinstance(metrics, list):
        metrics = []
    # Agent-style datasets (e.g. deep_swe) report a binary metric named ``acc``
    # rather than ``accuracy``; accept both.
    for name in (primary, "accuracy", "acc"):
        if not name:
            continue
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            identity = metric.get("identity")
            if not isinstance(identity, dict) or identity.get("name") != name:
                continue
            value = metric.get("score")
            if isinstance(value, (int, float)):
                return float(value)
    return None


def run_evalscope(
    host,
    port,
    model,
    datasets,
    dataset_args=None,
    eval_batch_size=16,
    limit=100000,
    generation_config=None,
    dataset_dir=None,
    timeout=60000,
    stream=True,
    eval_type="openai_api",
):

    metrics_path = os.getenv("METRICS_DATA_FILE")
    result_path = "./evalscope_result" if not metrics_path else metrics_path
    logger.info(f"The metrics result file: {result_path}")

    api_url = f"http://{host}:{port}/v1/chat/completions"

    if generation_config is None:
        generation_config = {"max_tokens": 512}

    config_dict = {
        "model": model,
        "api_url": api_url,
        "eval_type": eval_type,
        "datasets": datasets,
        "eval_batch_size": eval_batch_size,
        "generation_config": generation_config,
        "timeout": timeout,
        "stream": stream,
        "limit": limit,
        "work_dir": result_path,
    }
    if dataset_args:
        config_dict["dataset_args"] = dataset_args
    if dataset_dir:
        config_dict["dataset_dir"] = dataset_dir

    config_json = json.dumps(config_dict, ensure_ascii=False, indent=2)
    config_json_escaped = config_json.replace("\\", "\\\\").replace("'''", "\\'\\'\\'")

    script_content = "import json\n"
    script_content += "from evalscope import TaskConfig, run_task\n\n"
    script_content += f"config = json.loads('''{config_json_escaped}''')\n"
    script_content += "task_cfg = TaskConfig(**config)\n"
    script_content += "run_task(task_cfg=task_cfg)\n"

    script_path = f"/tmp/evalscope_run_{model}_{'_'.join(datasets)}.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    logger.info(f"Generated evalscope script: {script_path}")

    # Resolve the install script relative to this file so it works from any
    # deployment path (e.g. /sgl-workspace/sglang in CI images).
    install_cmd = f"/bin/bash {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_evalscope.sh')}"
    subprocess.run(install_cmd, shell=True, check=True)

    python_bin = "test_env_evalscope/bin/python"
    cmd = f"{python_bin} {script_path}"

    logger.info(f"Command: {cmd}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=True,
        start_new_session=True,
    )

    logger.info(
        f"run_evalscope spawned: pid={process.pid} "
        f"pgid={os.getpgid(process.pid)} cmd={cmd}"
    )

    output_lines = []
    try:
        for line in iter(process.stdout.readline, ""):
            if line.strip():
                print(line, end="")
            output_lines.append(line.strip())

        process.wait()

        logger.info(
            f"run_evalscope finished: pid={process.pid} returncode={process.returncode}"
        )

        kill_process_group(process)

        if process.returncode != 0:
            logger.error(f"Command failed with return code: {process.returncode}")
            raise subprocess.CalledProcessError(process.returncode, cmd)

        logger.info("Command executed successfully")

        metrics = {}
        full_output = "\n".join(output_lines)

        report_match = re.search(r"Dump report to:\s*(\S+)", full_output)
        if report_match:
            report_path = report_match.group(1)
            logger.info(f"Found evalscope report file: {report_path}")
            try:
                with open(report_path, "r") as rf:
                    report_data = json.load(rf)
                score = extract_report_accuracy(report_data)
                if score is not None:
                    metrics["accuracy"] = score
                    logger.info(f"The Final Accuracy from report: {score}")
            except Exception as e:
                logger.warning(f"Failed to read report file {report_path}: {e}")

        if "accuracy" not in metrics:
            accuracy_patterns = [
                # evalscope 1.11+ adds a Correct column between Num and Score:
                # │ ... │ Accuracy ↑ │ default │ 20 │ 18 │ 90% │
                r"Accuracy\s*[↑↓]?\s*│\s*[^│]*│\s*\d+\s*│\s*\d+\s*│\s*([\d.]+)%?\s*│",
                # Add adaptation for evalscope 1.11+ table format
                r"Accuracy\s*[↑↓]?\s*│\s*[^│]*│\s*\d+\s*│\s*([\d.]+)%?\s*│",
                r"mean_acc\s*.*?│\s*\d+\s*│\s*([\d.]+)\s*│",
                r"│\s+([\d.]+)\s+│\s+\S+\s+│\s*$",
                r"accuracy\s*[:=]?\s*([\d.]+)",
                # Keep compatibility with legacy evalscope 1.10 table format
                r"Accuracy\s*[:=]?\s*([\d.]+)",
                r"score\s*[:=]?\s*([\d.]+)",
            ]

            for pattern in accuracy_patterns:
                matches = list(re.finditer(pattern, full_output))
                if matches:
                    final_accuracy = float(matches[-1].group(1))
                    # evalscope 1.11+ reports accuracy as a percentage (e.g. 66.67%);
                    # normalize it to a 0-1 fraction to compare against the baseline.
                    # Accuracy can never exceed 1, so a bare "90" is a percentage too.
                    if "%" in matches[-1].group(0) or final_accuracy > 1.0:
                        final_accuracy /= 100.0
                    metrics["accuracy"] = final_accuracy
                    logger.info(f"The Final Accuracy from output: {final_accuracy}")
                    break

        if "accuracy" not in metrics:
            logger.info("Can Not Find The Accuracy in evalscope output")

        return metrics

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, terminating process...")
        process.terminate()
        try:
            process.wait(timeout=5)
            logger.info("Process terminated")
        except subprocess.TimeoutExpired:
            logger.warning("Process did not terminate gracefully, killing it...")
            process.kill()
            logger.info("Process killed")
        kill_process_group(process)
        raise

    except Exception as e:
        logger.error(f"Error executing command: {e}")
        process.terminate()
        process.wait(timeout=5)
        kill_process_group(process)
        raise


def requires_remote_eval(datasets):
    """Whether ``datasets`` must be evaluated on the amd64 DinD pod."""
    return bool(datasets) and datasets[0] in REMOTE_EVAL_DATASETS


def _deepswe_eval_pod_name():
    """Derive the eval pod name from this router pod's (Volcano) name.

    Keeping the router job prefix in the name lets the workflow's
    ``Post process`` step sweep the leftover eval pod together with the rest
    of the run.
    """
    hostname = os.environ.get("HOSTNAME", "deepswe")
    suffix = "-sgl-router-0"
    job_name = hostname[: -len(suffix)] if hostname.endswith(suffix) else hostname
    job_name = re.sub(r"[^a-z0-9-]", "-", job_name.lower()).strip("-")
    # A per-call suffix keeps the name unique across retries in the same pod:
    # the previous pod may still be terminating (Foreground delete is async),
    # and recreating the same name would hit a 409 Conflict.
    # Pod names are capped at 63 chars; reserve room for the trailing suffix.
    unique = uuid.uuid4().hex[:6]
    return f"{job_name[:42].strip('-')}-deepswe-eval-{unique}"


def _resolve_share_volume(core_api, namespace):
    """Copy this pod's ``share`` volume (PVC or hostPath) and repo subPath.

    The CI env (``ci``) and the green env (``green``) back the shared disk with
    different volume sources, so the eval pod reuses whatever the router pod
    was given instead of hardcoding one of them.
    """
    pod_name = os.environ.get("HOSTNAME")
    pod = core_api.read_namespaced_pod(name=pod_name, namespace=namespace)
    share_volume = None
    for volume in pod.spec.volumes or []:
        if volume.name == "share":
            share_volume = volume
            break
    if share_volume is None:
        raise RuntimeError(
            f"Pod {pod_name} has no 'share' volume; the DeepSWE eval pod cannot "
            "mount the sglang source checkout"
        )
    sub_path = None
    for container in pod.spec.containers or []:
        for mount in container.volume_mounts or []:
            if mount.mount_path == "/root/sglang":
                sub_path = mount.sub_path
    if not sub_path:
        raise RuntimeError(
            f"Pod {pod_name} does not mount /root/sglang from the share volume; "
            "the DeepSWE eval pod cannot locate the sglang source"
        )
    return share_volume, sub_path


def _build_deepswe_eval_env(
    host, port, model, datasets, dataset_args, limit, timeout, api_key, metrics_dir
):
    env = {
        "DOCKER_HOST": "tcp://127.0.0.1:2375",
        # Base form (no /chat/completions): LiteLLM appends the route itself.
        "DEEPSWE_API_URL": f"http://{host}:{port}/v1",
        "DEEPSWE_MODEL": model,
        "DEEPSWE_DATASETS": ",".join(datasets),
        "DEEPSWE_LIMIT": str(limit),
        "DEEPSWE_TIMEOUT": str(timeout),
        "DEEPSWE_API_KEY": api_key,
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com",
        "EVALSCOPE_PIP_MIRROR": os.environ.get("EVALSCOPE_PIP_MIRROR")
        or "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
    }
    if dataset_args:
        env["DEEPSWE_DATASET_ARGS"] = json.dumps(dataset_args, ensure_ascii=False)
    if metrics_dir:
        # Same path on the shared PVC, so the eval report lands next to the
        # run's other artifacts.
        env["METRICS_DATA_FILE"] = metrics_dir
    if os.environ.get("SGLANG_USE_MODELSCOPE"):
        env["SGLANG_USE_MODELSCOPE"] = os.environ["SGLANG_USE_MODELSCOPE"]
    return env


def _build_deepswe_eval_pod(pod_name, namespace, env_vars, share_volume, share_sub_path):
    from kubernetes import client as k8s

    dind_container = k8s.V1Container(
        name="dind",
        image=DEEPSWE_DIND_IMAGE,
        security_context=k8s.V1SecurityContext(privileged=True),
        # Empty TLS cert dir: serve plain HTTP on 2375 so the eval container can
        # talk to it over the pod's shared network namespace.
        env=[k8s.V1EnvVar(name="DOCKER_TLS_CERTDIR", value="")],
        args=["--storage-driver=overlay2"],
        volume_mounts=[
            k8s.V1VolumeMount(name="dind-storage", mount_path="/var/lib/docker")
        ],
        resources=k8s.V1ResourceRequirements(
            requests={"cpu": "1", "memory": "2Gi"}, limits={"cpu": "2", "memory": "4Gi"}
        ),
    )
    eval_container = k8s.V1Container(
        name="eval",
        image=DEEPSWE_EVAL_IMAGE,
        env=[k8s.V1EnvVar(name=key, value=value) for key, value in env_vars.items()],
        command=["/bin/bash", "-c"],
        args=["bash /root/sglang/python/sglang/test/ascend/e2e/run_deepswe_eval.sh"],
        volume_mounts=[
            k8s.V1VolumeMount(name="share", mount_path="/root/.cache"),
            k8s.V1VolumeMount(
                name="share", mount_path="/root/sglang", sub_path=share_sub_path
            ),
        ],
        resources=k8s.V1ResourceRequirements(
            requests={"cpu": "8", "memory": "16Gi"},
            limits={"cpu": "16", "memory": "32Gi"},
        ),
    )
    metadata = k8s.V1ObjectMeta(
        name=pod_name,
        namespace=namespace,
        labels={"app": "sgl-deepswe-eval"},
    )
    spec = k8s.V1PodSpec(
        restart_policy="Never",
        node_selector={"kubernetes.io/arch": DEEPSWE_EVAL_NODE_ARCH},
        # Hard cap enforced by kubelet even if this waiter never runs again.
        active_deadline_seconds=DEEPSWE_EVAL_ACTIVE_DEADLINE,
        containers=[dind_container, eval_container],
        volumes=[
            k8s.V1Volume(
                name="dind-storage",
                empty_dir=k8s.V1EmptyDirVolumeSource(size_limit="50Gi"),
            ),
            share_volume,
        ],
    )
    return k8s.V1Pod(api_version="v1", kind="Pod", metadata=metadata, spec=spec)


def _deepswe_eval_state(pod):
    """Return ``(state, exit_code)`` for the eval container.

    ``state`` is one of ``"running"``, ``"terminated"``, ``"waiting"`` or
    ``None`` (not created yet). ``exit_code`` is set only when terminated.
    """
    for status in pod.status.container_statuses or []:
        if status.name != "eval" or status.state is None:
            continue
        if status.state.terminated is not None:
            return "terminated", status.state.terminated.exit_code
        if status.state.running is not None:
            return "running", None
        if status.state.waiting is not None:
            return "waiting", None
    return None, None


def _describe_pod_events(core_api, namespace, pod_name):
    try:
        events = core_api.list_namespaced_event(
            namespace=namespace,
            field_selector=f"involvedObject.name={pod_name}",
        )
        messages = [f"{event.reason}: {event.message}" for event in events.items]
        return "; ".join(messages) if messages else "no events"
    except Exception as e:
        return f"<failed to read events: {e}>"


def _stream_deepswe_eval_logs(core_api, namespace, pod_name, lines):
    """Follow the eval container's log into ``lines`` (and stdout) until EOF."""
    try:
        response = core_api.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            container="eval",
            follow=True,
            _preload_content=False,
        )
    except Exception as e:
        logger.warning(f"Failed to stream DeepSWE eval logs: {e}")
        return
    buffer = ""
    try:
        for chunk in response.stream(amt=4096, decode_content=True):
            buffer += chunk.decode("utf-8", "replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                print(f"{DEEPSWE_EVAL_LOG_PREFIX} {line}", flush=True)
                lines.append(line)
        if buffer:
            print(f"{DEEPSWE_EVAL_LOG_PREFIX} {buffer}", flush=True)
            lines.append(buffer)
    except Exception as e:
        logger.warning(f"DeepSWE eval log stream ended with an error: {e}")
    finally:
        try:
            response.release_conn()
        except Exception:
            pass


def _read_deepswe_eval_logs(core_api, namespace, pod_name):
    """Read the eval container's full log once (no printing/append)."""
    try:
        text = core_api.read_namespaced_pod_log(
            name=pod_name, namespace=namespace, container="eval"
        )
    except Exception as e:
        logger.warning(f"Failed to read DeepSWE eval logs: {e}")
        return []
    return text.splitlines()


def _parse_deepswe_eval_metrics(lines):
    metrics = {}
    for line in lines:
        if DEEPSWE_EVAL_METRICS_MARKER not in line:
            continue
        payload = line.split(DEEPSWE_EVAL_METRICS_MARKER, 1)[1].strip()
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        if isinstance(data, dict):
            metrics.update(
                {
                    key: value
                    for key, value in data.items()
                    if isinstance(value, (int, float))
                }
            )
    if "accuracy" not in metrics and "acc" in metrics:
        metrics["accuracy"] = metrics.pop("acc")
    return metrics


def run_evalscope_on_x86_pod(
    host,
    port,
    model,
    datasets,
    dataset_args=None,
    limit=100000,
    timeout=7200,
    api_key="EMPTY",
    metrics_dir=None,
):
    """Run evalscope on a dedicated amd64 pod that provides Docker (DinD).

    Docker-backed datasets (``deep_swe``) cannot run on the ARM NPU pods: the
    container images are amd64-only and the pods have no Docker daemon. This
    creates a short-lived pod on an amd64 node (``kubernetes.io/arch=amd64``)
    that runs the evaluation against the router while the NPU pods keep
    serving, streams its logs back so they reach the CI log, and returns the
    parsed metrics.

    Args:
        host (str): Router node IP (reachable from the eval pod).
        port (int): Router HTTP port.
        model (str): Model name passed to evalscope (ignored by SGLang).
        datasets (list): Datasets to evaluate (first must be a Docker dataset).
        dataset_args (dict): evalscope ``dataset_args`` (carries
            ``pier_agent_kwargs`` for ``deep_swe``).
        limit (int): Max number of tasks to evaluate.
        timeout (int): Dataset-level timeout forwarded to evalscope (seconds).
        api_key (str): API key sent to the router (``EMPTY`` for SGLang).
        metrics_dir (str): Shared-PVC directory for the eval report.

    Returns:
        dict: ``{"accuracy": <float>}`` parsed from the eval container.

    Raises:
        RuntimeError: If the pod cannot be created/scheduled, the evaluation
            fails, or it reports no accuracy.
    """
    from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
        NAMESPACE,
        get_k8s_api,
    )

    namespace = NAMESPACE
    if not namespace:
        raise RuntimeError(
            "DeepSWE evaluation requires the multi-node k8s environment "
            "(NAMESPACE is unset); it cannot run inside a single NPU pod."
        )

    core_api = get_k8s_api()
    pod_name = _deepswe_eval_pod_name()
    share_volume, share_sub_path = _resolve_share_volume(core_api, namespace)
    env_vars = _build_deepswe_eval_env(
        host, port, model, datasets, dataset_args, limit, timeout, api_key, metrics_dir
    )
    pod = _build_deepswe_eval_pod(
        pod_name, namespace, env_vars, share_volume, share_sub_path
    )

    logger.info(
        f"Creating DeepSWE eval pod {pod_name} on "
        f"kubernetes.io/arch={DEEPSWE_EVAL_NODE_ARCH}, "
        f"image={DEEPSWE_EVAL_IMAGE}, dind={DEEPSWE_DIND_IMAGE}, "
        f"datasets={datasets}, limit={limit}, timeout={timeout}s, "
        f"api_url={env_vars['DEEPSWE_API_URL']}, metrics_dir={metrics_dir}"
    )
    try:
        core_api.create_namespaced_pod(namespace=namespace, body=pod)
    except Exception as e:
        raise RuntimeError(f"Failed to create DeepSWE eval pod {pod_name}: {e}")
    logger.info(f"DeepSWE eval pod {pod_name} created; waiting for scheduling ...")

    lines = []
    stream_thread = None
    try:
        start_time = time.time()
        running_since = None
        exit_code = None
        next_report = start_time + 60
        while True:
            pod_status = core_api.read_namespaced_pod(
                name=pod_name, namespace=namespace
            )
            phase = pod_status.status.phase
            container_state, exit_code = _deepswe_eval_state(pod_status)
            if exit_code is not None:
                logger.info(
                    f"DeepSWE eval container in {pod_name} terminated "
                    f"(exit_code={exit_code}, elapsed={time.time() - start_time:.0f}s)"
                )
                break
            if phase == "Failed":
                raise RuntimeError(
                    f"DeepSWE eval pod {pod_name} failed to run; events: "
                    f"{_describe_pod_events(core_api, namespace, pod_name)}"
                )
            if phase == "Running":
                if running_since is None:
                    running_since = time.time()
                    node_name = pod_status.spec.node_name if pod_status.spec else None
                    logger.info(
                        f"DeepSWE eval pod {pod_name} is running on node {node_name}; "
                        "streaming eval logs ..."
                    )
                # Follow the log only once the eval container is actually
                # running; an early call would 400 while it is still starting.
                if stream_thread is None and container_state == "running":
                    stream_thread = threading.Thread(
                        target=_stream_deepswe_eval_logs,
                        args=(core_api, namespace, pod_name, lines),
                        daemon=True,
                    )
                    stream_thread.start()
                    logger.info(f"Attached to eval container log of {pod_name}")
                if time.time() - running_since > DEEPSWE_EVAL_RUN_TIMEOUT:
                    raise RuntimeError(
                        f"DeepSWE eval pod {pod_name} exceeded "
                        f"{DEEPSWE_EVAL_RUN_TIMEOUT}s"
                    )
            else:
                # Pending/Unknown: surface image-pull and scheduling reasons
                # periodically instead of only at the final timeout.
                if time.time() >= next_report:
                    logger.info(
                        f"DeepSWE eval pod {pod_name} is still {phase} "
                        f"({time.time() - start_time:.0f}s); events: "
                        f"{_describe_pod_events(core_api, namespace, pod_name)}"
                    )
                    next_report = time.time() + 60
                if time.time() - start_time > DEEPSWE_EVAL_SCHEDULE_TIMEOUT:
                    raise RuntimeError(
                        f"DeepSWE eval pod {pod_name} is not running after "
                        f"{DEEPSWE_EVAL_SCHEDULE_TIMEOUT}s (phase={phase}); events: "
                        f"{_describe_pod_events(core_api, namespace, pod_name)}"
                    )
            time.sleep(5)

        if stream_thread is not None:
            stream_thread.join(timeout=30)
        else:
            # The container terminated before we caught the pod Running.
            for line in _read_deepswe_eval_logs(core_api, namespace, pod_name):
                print(f"{DEEPSWE_EVAL_LOG_PREFIX} {line}", flush=True)
                lines.append(line)

        if exit_code != 0:
            tail = "\n".join(lines[-50:])
            raise RuntimeError(
                f"DeepSWE eval pod exited with code {exit_code}; log tail:\n{tail}"
            )

        metrics = _parse_deepswe_eval_metrics(lines)
        if "accuracy" not in metrics:
            # The follow stream can stop before the last buffered lines flush;
            # re-read the full log once before giving up.
            logger.info(
                "Accuracy not in the streamed log, re-reading the full eval log ..."
            )
            lines.extend(_read_deepswe_eval_logs(core_api, namespace, pod_name))
            metrics = _parse_deepswe_eval_metrics(lines)
        if "accuracy" not in metrics:
            raise RuntimeError(
                "DeepSWE evaluation finished but reported no accuracy metric; "
                "check the evalscope report in the eval pod log above"
            )
        logger.info(f"DeepSWE metrics: {metrics}")
        return metrics
    finally:
        # Persist the forwarded eval log next to the run's artifacts so a
        # post-mortem does not depend on the (large) router pod log.
        if metrics_dir and lines:
            try:
                os.makedirs(metrics_dir, exist_ok=True)
                eval_log_path = os.path.join(metrics_dir, "deepswe_eval.log")
                with open(eval_log_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                logger.info(f"Saved DeepSWE eval log to {eval_log_path}")
            except Exception as e:
                logger.warning(f"Failed to save DeepSWE eval log: {e}")
        try:
            core_api.delete_namespaced_pod(
                name=pod_name,
                namespace=namespace,
                grace_period_seconds=0,
                propagation_policy="Foreground",
            )
            logger.info(f"Deleted DeepSWE eval pod {pod_name}")
        except Exception as e:
            logger.warning(f"Failed to delete DeepSWE eval pod {pod_name}: {e}")


def assert_metrics(self, metrics):
    if not metrics:
        raise Exception("No metrics obtained from benchmark")

    # Keep weekday gsm8k and weekend gpqa_diamond in separate series.
    labels = {
        "test_case": self.__class__.__name__,
        "type": "accuracy",
        "dataset": (self.datasets or ["unknown"])[0],
    }
    if metrics.get("accuracy") is not None:
        dump_metric("accuracy", float(metrics["accuracy"]), labels=labels)
    # Agent datasets (deep_swe) may ship without a calibrated baseline yet:
    # record the measured value, but only assert once a baseline is set.
    if self.accuracy is not None:
        threshold = get_accuracy_threshold(self.datasets, self.accuracy)
        dump_metric("accuracy_baseline", float(self.accuracy), labels=labels)
        self.assertGreaterEqual(
            float(metrics.get("accuracy") or 0),
            threshold,
            f"Accuracy check failed. Expected >= {threshold}, Got: {metrics.get('accuracy')}",
        )


def run_evalscope_with_retry(
    self,
    host,
    port,
    model_name,
):
    """Shared evalscope retry loop for all accuracy test bases.

    Retries up to ``max_retries`` (class attribute override, or
    ``get_max_retries`` by dataset), keeping the best accuracy. Datasets that
    need Docker (``REMOTE_EVAL_DATASETS``) are evaluated on the amd64 DinD pod
    instead of the local NPU pod.
    """
    max_retries = (
        self.max_retries
        if self.max_retries is not None
        else get_max_retries(self.datasets)
    )
    remote_eval = requires_remote_eval(self.datasets)
    logger.info(
        f"[retry-budget] {type(self).__name__}: max_retries={max_retries} "
        f"(class override: {self.max_retries is not None}), "
        f"datasets={self.datasets}, threshold baseline={self.accuracy}, "
        f"remote_eval={remote_eval}"
    )
    best_metrics = None
    attempts_used = 0
    for attempt in range(max_retries):
        attempts_used = attempt + 1
        if remote_eval:
            metrics = run_evalscope_on_x86_pod(
                host=host,
                port=port,
                model=model_name,
                datasets=self.datasets,
                dataset_args=self.dataset_args,
                limit=self.limit,
                timeout=self.timeout,
                metrics_dir=os.environ.get("METRICS_DATA_FILE"),
            )
        else:
            metrics = run_evalscope(
                host=host,
                port=port,
                model=model_name,
                datasets=self.datasets,
                dataset_args=self.dataset_args,
                eval_batch_size=self.eval_batch_size,
                limit=self.limit,
                generation_config=self.generation_config,
                dataset_dir=self.dataset_dir,
                stream=self.stream,
                timeout=self.timeout,
                eval_type=self.eval_type,
            )
        if best_metrics is None or float(metrics.get("accuracy", 0)) > float(
            best_metrics.get("accuracy", 0)
        ):
            best_metrics = metrics
        # Without a calibrated baseline there is no threshold to retry against.
        if self.accuracy is None:
            break
        threshold = get_accuracy_threshold(self.datasets, self.accuracy)
        if float(best_metrics.get("accuracy", 0)) >= threshold:
            break
        if attempt < max_retries - 1:
            logger.info(
                f"Accuracy {best_metrics.get('accuracy')} below threshold "
                f"{threshold}, retrying ({attempt + 1}/{max_retries - 1})..."
            )
    logger.info(
        f"[retry-budget] {type(self).__name__}: finished after "
        f"{attempts_used}/{max_retries} attempt(s), "
        f"best accuracy={best_metrics.get('accuracy') if best_metrics else None}"
    )
    assert_metrics(self, best_metrics)


class TestNpuAccuracyTestCaseBase(CustomTestCase):
    model = None
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    backend = "sglang"
    datasets = ["gsm8k"]
    dataset_args = None
    eval_batch_size = 16
    limit = 100000
    generation_config = None
    dataset_dir = None
    stream = True
    timeout = 60000
    eval_type = "openai_api"
    other_args = None
    server_timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    envs = None
    accuracy = 0.1
    # Per-case evalscope accuracy retry count. None uses the dataset default
    # (get_max_retries); set to 1 to fail fast on first measurement.
    max_retries = None
    test_type = "accuracy"

    @classmethod
    def _get_tc_name(cls):
        """Derive the test case name from the test file (filename without
        extension). Mirrors the workflow's ``tc_name=${test_case##*/}`` logic
        so each case in a suite writes to its own output path."""
        try:
            tc_file = inspect.getfile(cls)
        except (TypeError, OSError):
            tc_file = getattr(sys.modules.get(cls.__module__), "__file__", "")
        return os.path.splitext(os.path.basename(tc_file))[0]

    @classmethod
    def _setup_per_case_output(cls):
        """Set up per-case output directories and env vars.

        When the workflow sets METRICS_DATA_FILE to a suite-level directory
        (e.g. .../output/{branch_label}-{create_date}-{run_id}-{run_attempt}/
        {workflow_name}/{test_type}/{suite}), each case in the suite
        writes to its own subdirectory under it, so results stay in the
        structured layout and are keyed by the case id. Falls back to the
        legacy per-case layout when the env var is not set.
        """
        cls.tc_name = cls._get_tc_name()
        suite_output = os.environ.get("METRICS_DATA_FILE")
        if suite_output:
            # Append the case id under the suite output prefix.
            cls.metrics_data_file = os.path.join(suite_output, cls.tc_name)
            # Mirror the output prefix to the plog location (drop the test_type/suite tail).
            suite_plog = suite_output.replace("/output/", "/logs/plog/", 1)
            cls.plog_base = os.path.dirname(os.path.dirname(suite_plog))
        else:
            current_date = datetime.now().strftime("%Y%m%d")
            test_type = getattr(cls, "test_type", "accuracy")
            base_output = f"/root/.cache/tests/output/{test_type}/{current_date}"
            cls.metrics_data_file = os.path.join(base_output, cls.tc_name)
            cls.plog_base = f"/root/.cache/tests/logs/plog"
        os.makedirs(cls.metrics_data_file, exist_ok=True)
        # Override env vars so evalscope/dump_metric write to per-case paths.
        os.environ["METRICS_DATA_FILE"] = cls.metrics_data_file
        os.environ["SGLANG_TEST_METRICS_OUTPUT"] = os.path.join(
            cls.metrics_data_file, "metrics"
        )
        logger.info(
            "Per-case output: tc_name=%s metrics_data_file=%s",
            cls.tc_name,
            cls.metrics_data_file,
        )

    @classmethod
    def _save_metrics_json(cls):
        """Write per-case ``metrics.json`` from ``dump_metric`` JSONL files.

        Replaces the workflow's stdout-parsing + ``dump_metrics.py`` logic so
        each case in a suite persists its own metrics snapshot.
        """
        if not getattr(cls, "metrics_data_file", None):
            return
        metrics = {}
        baselines = {}
        pattern = os.path.join(cls.metrics_data_file, "metrics.*.jsonl")
        for jsonl_path in glob.glob(pattern):
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        name = record.get("metric_name")
                        value = record.get("value")
                        if name is None:
                            continue
                        if name.endswith("_baseline"):
                            baselines[name[: -len("_baseline")]] = value
                        else:
                            metrics[name] = value
            except Exception as e:
                logger.warning("Failed to read %s: %s", jsonl_path, e)
        out_path = os.path.join(cls.metrics_data_file, "metrics.json")
        payload = {
            "test_case": cls.tc_name,
            "test_type": getattr(cls, "test_type", "accuracy"),
            "metrics": metrics,
            "baselines": baselines,
        }
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            logger.info("Saved per-case metrics to %s", out_path)
        except Exception as e:
            logger.warning("Failed to write metrics.json: %s", e)
        # Remove the intermediate JSONL records, keeping only the final metrics.json.
        for jsonl_path in glob.glob(pattern):
            try:
                os.remove(jsonl_path)
            except Exception as e:
                logger.warning("Failed to remove %s: %s", jsonl_path, e)

    @classmethod
    def _backup_plog(cls):
        """Backup Ascend plog files to a per-case path.

        Replaces the workflow's ``Backup plog`` step so each case in a suite
        gets its own plog snapshot instead of all cases sharing the suite name.
        """
        plog_path = "/root/ascend/log/debug/plog"
        if not os.path.isdir(plog_path):
            return
        tc_name = getattr(cls, "tc_name", None)
        if not tc_name:
            return
        hostname = os.getenv("HOSTNAME", "unknown")
        plog_base = getattr(cls, "plog_base", "/root/.cache/tests/logs/plog")
        target = os.path.join(plog_base, tc_name, hostname)
        os.makedirs(target, exist_ok=True)
        for name in os.listdir(plog_path):
            src = os.path.join(plog_path, name)
            if os.path.isfile(src):
                try:
                    shutil.copy2(src, os.path.join(target, name))
                except Exception as e:
                    logger.warning("Failed to copy plog %s: %s", name, e)
        logger.info("Backed up plog to %s", target)

    @classmethod
    def setUpClass(cls):
        cls._setup_per_case_output()
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = list(cls.other_args)
        cls.storage_service_process = maybe_start_hicache_storage_service(other_args)

        env = os.environ.copy()
        for key, value in env.items():
            logger.info(f"ENV_VAR_SYS {key}:{value}")
        if cls.envs:
            for key, value in cls.envs.items():
                logger.info(f"ENV_VAR_CASE {key}:{value}")
                env[key] = value

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.server_timeout,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            try:
                kill_process_tree(cls.process.pid)
            except Exception as e:
                logger.error(f"Error during tearDown: {e}")
        storage_service_process = getattr(cls, "storage_service_process", None)
        if storage_service_process:
            try:
                kill_process_tree(storage_service_process.pid)
            except Exception as e:
                logger.error(f"Error during storage service teardown: {e}")
        cls._save_metrics_json()
        cls._backup_plog()

    def run_accuracy(self):
        if self.benchmark_tool == EVALSCOPE:
            model_name = os.path.basename(self.model)
            parsed_url = urlparse(self.base_url)
            run_evalscope_with_retry(
                self, parsed_url.hostname, parsed_url.port, model_name
            )


class TestNpuAccuracyMultiNodePdMixTestCaseBase(PdMixNodeMixin, CustomTestCase):
    model_config = None
    initialization_delay = SERVER_INITIALIZATION_DELAY
    keep_alive_time = MAX_SERVER_KEEP_ALIVE_TIME
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    backend = "sglang"
    datasets = ["gsm8k"]
    dataset_args = None
    eval_batch_size = 16
    limit = 100000
    generation_config = None
    dataset_dir = None
    stream = True
    timeout = 60000
    eval_type = "openai_api"
    other_args = None
    server_timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    envs = None
    accuracy = 0.1
    # Per-case evalscope accuracy retry count. None uses the dataset default
    # (get_max_retries); set to 1 to fail fast on first measurement.
    max_retries = None

    @classmethod
    def setUpClass(cls):
        apply_accuracy_dataset_override(cls)
        super().setUpClass()

    @check_role(allowed_roles=["master", "worker"])
    def run_accuracy(self):
        if self.benchmark_tool == EVALSCOPE:
            model_name = os.path.basename(self.model_config.get("model_path"))
            run_evalscope_with_retry(self, self.host, self.port, model_name)


class TestNpuAccuracyMultiNodePdSepTestCaseBase(PdSepNodeMixin, CustomTestCase):
    model_config = None
    initialization_delay = SERVER_INITIALIZATION_DELAY
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    backend = "sglang"
    datasets = ["gsm8k"]
    dataset_args = None
    eval_batch_size = 16
    limit = 100000
    generation_config = None
    dataset_dir = None
    stream = True
    timeout = 60000
    eval_type = "openai_api"
    other_args = None
    server_timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    accuracy = 0.1
    # Per-case evalscope accuracy retry count. None uses the dataset default
    # (get_max_retries); set to 1 to fail fast on first measurement.
    max_retries = None

    @classmethod
    def setUpClass(cls):
        apply_accuracy_dataset_override(cls)
        super().setUpClass()

    @check_role(allowed_roles=["router"])
    def run_accuracy(self):
        if self.benchmark_tool == EVALSCOPE:
            model_name = os.path.basename(self.model_config.get("model_path"))
            run_evalscope_with_retry(self, self.host, self.port, model_name)
