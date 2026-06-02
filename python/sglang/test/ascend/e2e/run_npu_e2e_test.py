import argparse
import logging
import os
import random
import re
import string
import subprocess
import time
import uuid

import psutil
import yaml
from jinja2 import Template
from kubernetes import client, config
from kubernetes.client.rest import ApiException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

KUBE_CONFIG = os.environ.get("KUBECONFIG")
logger.info(f"KUBE_CONFIG: {KUBE_CONFIG}")
config.load_kube_config(KUBE_CONFIG)
core_api = client.CoreV1Api()
custom_api = client.CustomObjectsApi()
batch_api = client.BatchV1Api()
rbac_api = client.RbacAuthorizationV1Api()

LOCAL_TIMEOUT = 10800

script_path = os.path.dirname(os.path.abspath(__file__))

KUBE_JOB_SINGLE = "single"
KUBE_JOB_MULTI_PD_MIX = "multi-pd-mix"
KUBE_JOB_MULTI_PD_SEPARATION = "multi-pd-separation"
KUBE_JOB_MULTI_PD_MIX_GREEN = "multi-pd-mix-green"
KUBE_JOB_MULTI_PD_SEPARATION_GREEN = "multi-pd-separation-green"
KUBE_YAML_TEMPLATE = {
    KUBE_JOB_SINGLE: f"{script_path}/k8s_single.yaml.jinja2",
    KUBE_JOB_MULTI_PD_MIX: f"{script_path}/k8s_multi_pd_mix.yaml.jinja2",
    KUBE_JOB_MULTI_PD_MIX_GREEN: f"{script_path}/k8s_multi_pd_mix_green.yaml.jinja2",
    KUBE_JOB_MULTI_PD_SEPARATION: f"{script_path}/k8s_multi_pd_separation.yaml.jinja2",
    KUBE_JOB_MULTI_PD_SEPARATION_GREEN: f"{script_path}/k8s_multi_pd_separation_green.yaml.jinja2",
}


def get_unique_random_string(length: int = 16, add_random: bool = True) -> str:
    """Generate a random string."""
    uuid_str = str(uuid.uuid4()).replace("-", "")

    if add_random:
        if length < 8:
            raise ValueError("length can not be smaller than 8")
        random_length = length - 8
        char_pool = string.ascii_lowercase + string.digits
        random_chars = "".join([random.choice(char_pool) for _ in range(random_length)])
        result = uuid_str[:8] + random_chars
    else:
        result = uuid_str[:length]

    return result


def create_kube_yaml(kube_yaml_template, output_yaml, pod_context):
    """Create a k8s config yaml file"""
    with open(kube_yaml_template, "r") as f:
        template = Template(f.read())
    kube_pod_yaml = template.render(pod_context)
    with open(output_yaml, "w") as f:
        f.write(kube_pod_yaml)
    logger.info(f"Pod YAML written to {output_yaml}")


def create_pod(yaml_file, namespace):
    """Create a pod by k8s config yaml file"""
    with open(yaml_file, "r", encoding="utf-8") as f:
        yaml_docs = list(yaml.safe_load_all(f))

    for doc in yaml_docs:
        if not doc:
            continue

        kind = doc.get("kind")
        api_version = doc.get("apiVersion")

        try:
            if kind == "Pod" and api_version == "v1":
                core_api.create_namespaced_pod(namespace=namespace, body=doc)
                logger.info(f"Pod {doc['metadata']['name']} created")

            elif kind == "Job" and api_version == "batch/v1":
                batch_api.create_namespaced_job(namespace=namespace, body=doc)
                logger.info(f"Job {doc['metadata']['name']} is created")

            elif kind == "Job" and api_version == "batch.volcano.sh/v1alpha1":
                response = custom_api.create_namespaced_custom_object(
                    group="batch.volcano.sh",
                    version="v1alpha1",
                    namespace=namespace,
                    plural="jobs",
                    body=doc,
                )
                logger.info(f"Volcano job {doc['metadata']['name']} is created")
                logger.debug(response)

            elif kind == "ConfigMap" and api_version == "v1":
                core_api.create_namespaced_config_map(namespace=namespace, body=doc)
                logger.info(f"ConfigMap {doc['metadata']['name']} is created")

            elif kind == "Role" and api_version == "rbac.authorization.k8s.io/v1":
                rbac_api.create_namespaced_role(namespace=namespace, body=doc)
                logger.info(f"Role {doc['metadata']['name']} is created")

            elif (
                kind == "RoleBinding" and api_version == "rbac.authorization.k8s.io/v1"
            ):
                rbac_api.create_namespaced_role_binding(namespace=namespace, body=doc)
                logger.info(f"RoleBinding {doc['metadata']['name']} is created")

            elif kind == "Deployment" and api_version == "apps/v1":
                apps_api = client.AppsV1Api()
                apps_api.create_namespaced_deployment(namespace=namespace, body=doc)
                logger.info(f"Deployment {doc['metadata']['name']} is created")

            elif kind == "StatefulSet" and api_version == "apps/v1":
                apps_api = client.AppsV1Api()
                apps_api.create_namespaced_stateful_set(namespace=namespace, body=doc)
                logger.info(f"StatefulSet {doc['metadata']['name']} is created")

            elif kind == "Service" and api_version == "v1":
                core_api.create_namespaced_service(namespace=namespace, body=doc)
                logger.info(f"Service {doc['metadata']['name']} is created")

            else:
                raise f"Unrecognized kind: {kind}/{api_version}"
        except ApiException as e:
            print(f"create resource {kind} error: {e}")
            raise


def delete_pod(yaml_file, namespace):
    """Delete k8s pod by config yaml file"""
    with open(yaml_file, "r", encoding="utf-8") as f:
        yaml_docs = list(yaml.safe_load_all(f))
    for doc in yaml_docs:
        if not doc:
            continue

        kind = doc.get("kind")
        api_version = doc.get("apiVersion")
        try:
            if kind == "Job" and api_version == "batch.volcano.sh/v1alpha1":
                job_name = doc["metadata"]["name"]
                response = custom_api.delete_namespaced_custom_object(
                    group="batch.volcano.sh",
                    version="v1alpha1",
                    namespace=namespace,
                    plural="jobs",
                    name=job_name,
                    body=client.V1DeleteOptions(
                        grace_period_seconds=0, propagation_policy="Foreground"
                    ),
                )
                logger.info(f"Deleted job {job_name}")
                logger.info(f"Response status: {response.get('status')}")
            elif kind == "ConfigMap" and api_version == "v1":
                config_map_name = doc["metadata"]["name"]
                core_api.delete_namespaced_config_map(
                    name=config_map_name, namespace=namespace
                )
                print(f"ConfigMap {config_map_name} is deleted.")
            elif kind == "Deployment" and api_version == "apps/v1":
                deployment_name = doc["metadata"]["name"]
                apps_api = client.AppsV1Api()
                apps_api.delete_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=client.V1DeleteOptions(
                        grace_period_seconds=0, propagation_policy="Foreground"
                    ),
                )
                logger.info(f"Deployment {deployment_name} is deleted.")

            elif kind == "StatefulSet" and api_version == "apps/v1":
                statefulset_name = doc["metadata"]["name"]
                apps_api = client.AppsV1Api()
                apps_api.delete_namespaced_stateful_set(
                    name=statefulset_name,
                    namespace=namespace,
                    body=client.V1DeleteOptions(
                        grace_period_seconds=0, propagation_policy="Foreground"
                    ),
                )
                logger.info(f"StatefulSet {statefulset_name} is deleted.")

            elif kind == "Service" and api_version == "v1":
                service_name = doc["metadata"]["name"]
                core_api.delete_namespaced_service(
                    name=service_name,
                    namespace=namespace,
                    body=client.V1DeleteOptions(
                        grace_period_seconds=0, propagation_policy="Foreground"
                    ),
                )
                logger.info(f"Service {service_name} is deleted.")

            else:
                raise f"Unrecognized kind: {kind}/{api_version}"
        except ApiException as e:
            raise f"delete resource {kind} error: {e}"


def check_parent_process():
    """Check parent process is alive or not."""
    try:
        parent_pid = os.getppid()
        psutil.Process(parent_pid)
        return True
    except psutil.NoSuchProcess:
        return False


def check_pods_ready(namespace, pod_name_key_str, timeout=300):
    """Waiting for all k8s pods are ready"""
    logger.info("Waiting all pods to running...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        if not check_parent_process():
            raise Exception("Parent process exited.")

        pods = core_api.list_namespaced_pod(namespace=namespace)

        if len(pods.items) == 0:
            time.sleep(5)
            continue

        all_running = True
        sglang_pods_found = False
        for pod in pods.items:
            pod_name = pod.metadata.name
            if pod_name_key_str not in pod_name:
                continue

            sglang_pods_found = True
            status = pod.status
            phase = status.phase
            logger.info(f"Pod: {pod_name}, status: {phase}")
            if phase != "Running":
                all_running = False
                break

            containers_ready = True
            for condition in status.conditions:
                if condition.type == "Ready" and condition.status != "True":
                    containers_ready = False
                    break

            if not containers_ready:
                all_running = False
                break

        if not sglang_pods_found:
            logger.info("No sglang pod, waiting...")
            time.sleep(5)
            continue
        if all_running:
            logger.info("All sglang Pod is Running !")
            return True

        time.sleep(5)

    logger.info(f"timeout in {timeout}s")
    return False


def create_or_update_configmap(cm_name: str, data: dict, namespace: str):
    """Create a k8s configmap or update it if already exists"""
    cm_metadata = client.V1ObjectMeta(name=cm_name, namespace=namespace)
    configmap = client.V1ConfigMap(
        api_version="v1", kind="ConfigMap", metadata=cm_metadata, data=data
    )

    try:
        response = core_api.create_namespaced_config_map(
            namespace=namespace, body=configmap
        )
        logger.info(f"ConfigMap '{cm_name}' create successfully!")
        logger.info(f"data: {list(data.keys())}")
        return response
    except ApiException as e:
        if e.status == 409:
            logger.info(f"ConfigMap {cm_name} already exists. Updating...")
            response = core_api.replace_namespaced_config_map(
                namespace=namespace, name=cm_name, body=configmap
            )
            logger.info(f"ConfigMap {cm_name} updated successfully.")
            return response
        else:
            error_msg = f"ConfigMap create failed: {e.reason}"
            if e.body:
                error_msg += f" | details: {e.body}"
            logger.info(error_msg)
            raise


def prepare_cm_data(namespace, pod_string):
    """Prepare a configmap data: {pod_name: pod_ip} by the running pod's information."""
    pods = core_api.list_namespaced_pod(namespace=namespace)
    data = {}
    for pod in pods.items:
        pod_name = pod.metadata.name
        if pod_string in pod_name:
            pod_ip = pod.status.pod_ip
            data[pod_name] = pod_ip
    return data


def monitor_pod_logs(
    kube_job_type, kube_job_prefix_name, namespace, timeout=LOCAL_TIMEOUT
):
    """Monitor the logs of the specified pod until the special pattern is matched or reaches its timeout."""
    monitor_pod_name = {
        KUBE_JOB_SINGLE: f"{kube_job_prefix_name}-pod-0",
        KUBE_JOB_MULTI_PD_MIX: f"{kube_job_prefix_name}-sglang-node-0",
        KUBE_JOB_MULTI_PD_SEPARATION: f"{kube_job_prefix_name}-sglang-router-0",
    }
    pod_name = monitor_pod_name.get(kube_job_type)

    # Build kubectl command
    cmd = ["kubectl", "logs", "-f", "-n", namespace, pod_name]

    # Define multiline pattern to match
    pattern_lines = [
        r"^-{70,}$",
        r"^Ran \d+ tests? in [\d.]+s$",
        r"^$",
        r"^(OK|FAILED \(errors=\d+\))$",
    ]
    patterns = [re.compile(line_pattern) for line_pattern in pattern_lines]
    pattern_ok = re.compile(r"^OK$")

    process = None
    try:
        # Start kubectl logs process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        )

        logger.info(f"Starting to monitor logs for Pod: {pod_name}")
        match_state = 0
        is_success = False

        # Use two threads: one for reading logs, one for checking pod status
        import threading

        # Shared variables
        match_event = threading.Event()
        pod_error_event = threading.Event()

        def read_logs():
            """Thread function to read logs continuously"""
            nonlocal is_success, match_state

            while process.poll() is None and not match_event.is_set():
                line = process.stdout.readline()
                if line:
                    line = line.rstrip("\n")
                    print(line)
                    # Check if current line matches expected pattern
                    if match_state < len(patterns) and patterns[match_state].match(
                        line
                    ):
                        match_state += 1
                        if match_state == len(patterns):
                            if pattern_ok.match(line):
                                is_success = True
                            logger.info("Detected complete test completion pattern!")
                            match_event.set()
                    else:
                        match_state = 0
                        if patterns[0].match(line):
                            match_state = 1

            # Read remaining output after process exits
            if not match_event.is_set():
                remaining_output, stderr_output = process.communicate()
                if remaining_output:
                    print(remaining_output)
                if stderr_output:
                    logger.error(f"kubectl command error: {stderr_output}")
                    pod_error_event.set()

        def check_pods_running(namespace, pod_name_key_str):
            """check pods are running"""
            pods = core_api.list_namespaced_pod(namespace=namespace)
            if len(pods.items) == 0:
                logger.warning(f"No pods found in the namespace {namespace}")
                return False

            for pod in pods.items:
                pod_name = pod.metadata.name
                if pod_name_key_str not in pod_name:
                    continue
                status = pod.status
                phase = status.phase
                if phase != "Running":
                    logger.error(f"Pod {pod_name} is not running, status: {phase}")
                    return False

            return True

        def check_pod_status():
            """Thread function to check pod status periodically"""
            start_time = time.time()
            while not match_event.is_set() and not pod_error_event.is_set():
                if time.time() - start_time > timeout:
                    pod_error_event.set()
                    break

                if not check_parent_process():
                    logger.error(f"Parent process exited. Exiting...")
                    pod_error_event.set()
                    break

                if not check_pods_running(
                    namespace=namespace, pod_name_key_str=kube_job_prefix_name
                ):
                    logger.error(
                        f"Some pods are not running properly. Please check the sglang logs on these pods. Exiting..."
                    )
                    pod_error_event.set()
                    break

                # Sleep for a short time before next check
                time.sleep(0.5)

        # Start threads
        log_thread = threading.Thread(target=read_logs)
        status_thread = threading.Thread(target=check_pod_status)

        log_thread.daemon = True
        status_thread.daemon = True

        log_thread.start()
        status_thread.start()

        # Wait for either match event or error event
        start_time = time.time()
        while not match_event.is_set() and not pod_error_event.is_set():
            if time.time() - start_time > timeout:
                raise Exception(
                    f"Timeout exceeded, the thread is {timeout} seconds long."
                )
            time.sleep(0.1)

        # Check if pattern was successfully matched
        if not match_event.is_set():
            if process.poll() is not None:
                remaining_output, stderr_output = process.communicate()
                if remaining_output:
                    logger.info(remaining_output)
                if stderr_output:
                    raise Exception(f"kubectl command error: {stderr_output}")
                else:
                    raise Exception(
                        "Pod logs ended but target pattern was not detected"
                    )
            else:
                raise Exception("Monitoring ended but target pattern was not detected")
        elif not is_success:
            raise Exception("The test result was FAILED!")
        else:
            logger.info("The test result was OK!")
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def run_npu_e2e_test_case(
    docker_image_url: str,
    kube_name_space: str,
    kube_job_type: str,
    kube_job_name_prefix: str,
    resource_info: dict,
    sglang_source_relative_path: str,
    metrics_data_file: str,
    test_case: str,
    sglang_is_in_ci=False,
    install_sglang_from_source=False,
    env="debug",
    trouble_shotting=False,
    transformers_version="",
):
    """The method for running a npu e2e test case.
    Args:
        docker_image_url (str): the url of docker image for creating k8s pods.
        kube_name_space (str): the namespace of the k8s.
        kube_job_name_prefix (str): the prefix of the k8s job name which will be set as the prefix of the pod name.
        resource_info (dict): the number of k8s nodes used by the testcase.
            for pd-separation as: {"prefill_size": 1, "decode_size": 1, "router_size": 1};
            for pd-mix as: {"node_size": 2; single: {"npu_size": 4}
        sglang_source_relative_path (str): the relative path of the sglang source on shared-disk.
        metrics_data_file (str): the output path of the metrics data file, only for performance testing.
        test_case (str): the test case relative path in sglang source root path. like test/registered/...
        sglang_is_in_ci (bool): whether running in CI environment.
        install_sglang_from_source (bool): whether installing sglang from source or use docker image directly.
        env (str): the environment to run the test on.  Choose one in ["debug", "ci"]
    """
    random_str = get_unique_random_string(16, True)

    kube_config_map = f"sglang-configmap-{random_str}"
    final_kube_job_name = f"{kube_job_name_prefix}-{random_str}"

    kube_yaml_file_dict = {
        KUBE_JOB_SINGLE: f"k8s_single_{random_str}.yaml",
        KUBE_JOB_MULTI_PD_MIX: f"k8s_multi_pd_mix_{random_str}.yaml",
        KUBE_JOB_MULTI_PD_SEPARATION: f"k8s_multi_pd_separation_{random_str}.yaml",
    }
    kube_yaml_file = kube_yaml_file_dict.get(kube_job_type)

    try:
        logger.info(
            f"Apply k8s yaml... KUBE_NAME_SPACE:{kube_name_space}, KUBE_CONFIG_MAP:{kube_config_map}, "
            f"KUBE_JOB_TYPE:{kube_job_type}, KUBE_YAML_FILE:{kube_yaml_file}"
        )

        if kube_job_type == KUBE_JOB_SINGLE:
            k8s_context = {
                "image": docker_image_url,
                "name_space": kube_name_space,
                "kube_job_name": final_kube_job_name,
                "kube_config": KUBE_CONFIG,
                "npu_size": resource_info["npu_size"],
                "sglang_source_relative_path": sglang_source_relative_path,
                "metrics_data_file": metrics_data_file,
                "test_case": test_case,
                "sglang_is_in_ci": sglang_is_in_ci,
                "install_sglang_from_source": install_sglang_from_source,
                "env": env,
                "trouble_shotting": trouble_shotting,
                "transformers_version": transformers_version,
            }
            create_kube_yaml(
                kube_yaml_template=KUBE_YAML_TEMPLATE.get(kube_job_type),
                output_yaml=kube_yaml_file,
                pod_context=k8s_context,
            )
        elif kube_job_type == KUBE_JOB_MULTI_PD_MIX:
            k8s_context = {
                "image": docker_image_url,
                "name_space": kube_name_space,
                "kube_job_name": final_kube_job_name,
                "kube_config": KUBE_CONFIG,
                "kube_config_map": kube_config_map,
                "node_size": resource_info["node_size"],
                "sglang_source_relative_path": sglang_source_relative_path,
                "metrics_data_file": metrics_data_file,
                "test_case": test_case,
                "sglang_is_in_ci": sglang_is_in_ci,
                "install_sglang_from_source": install_sglang_from_source,
                "env": env,
                "trouble_shotting": trouble_shotting,
                "transformers_version": transformers_version,
            }
            template_key = (
                KUBE_JOB_MULTI_PD_MIX_GREEN if env == "green" else kube_job_type
            )
            create_kube_yaml(
                kube_yaml_template=KUBE_YAML_TEMPLATE.get(template_key),
                output_yaml=kube_yaml_file,
                pod_context=k8s_context,
            )
        elif kube_job_type == KUBE_JOB_MULTI_PD_SEPARATION:
            k8s_context = {
                "image": docker_image_url,
                "name_space": kube_name_space,
                "kube_job_name": final_kube_job_name,
                "kube_config": KUBE_CONFIG,
                "kube_config_map": kube_config_map,
                "prefill_size": resource_info["prefill_size"],
                "decode_size": resource_info["decode_size"],
                "router_size": resource_info["router_size"],
                "sglang_source_relative_path": sglang_source_relative_path,
                "metrics_data_file": metrics_data_file,
                "test_case": test_case,
                "sglang_is_in_ci": sglang_is_in_ci,
                "install_sglang_from_source": install_sglang_from_source,
                "env": env,
                "trouble_shotting": trouble_shotting,
                "transformers_version": transformers_version,
            }
            template_key = (
                KUBE_JOB_MULTI_PD_SEPARATION_GREEN if env == "green" else kube_job_type
            )
            create_kube_yaml(
                kube_yaml_template=KUBE_YAML_TEMPLATE.get(template_key),
                output_yaml=kube_yaml_file,
                pod_context=k8s_context,
            )
        else:
            raise Exception(f"Unknown k8s job type: {kube_job_type}")

        create_pod(yaml_file=kube_yaml_file, namespace=kube_name_space)

        if check_pods_ready(
            kube_name_space, final_kube_job_name, timeout=LOCAL_TIMEOUT
        ):
            if kube_job_type != "single":
                matching_pod_string = final_kube_job_name
                cm_data = prepare_cm_data(kube_name_space, matching_pod_string)
                if not cm_data:
                    logger.info(
                        f"No sglang pod found while matching {matching_pod_string}"
                    )

                response = create_or_update_configmap(
                    cm_name=kube_config_map, data=cm_data, namespace=kube_name_space
                )
                logger.info(response)
        else:
            logger.info("Pod not ready, maybe not enough resource")

        monitor_pod_logs(
            kube_job_type, final_kube_job_name, kube_name_space, LOCAL_TIMEOUT
        )
    finally:
        if os.path.exists(kube_yaml_file):
            # Don't delete pod when trouble_shotting is enabled
            if not trouble_shotting:
                delete_pod(yaml_file=kube_yaml_file, namespace=kube_name_space)
                os.remove(kube_yaml_file)
            else:
                logger.info(
                    f"Trouble shooting mode enabled, keeping pod {final_kube_job_name} alive"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply k8s yaml", formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Docker image to use",
    )

    parser.add_argument(
        "--prefill-size",
        type=int,
        required=False,
        default=1,
        help="Number of prefill nodes",
    )

    parser.add_argument(
        "--decode-size",
        type=int,
        required=False,
        default=1,
        help="Number of decode nodes",
    )

    parser.add_argument(
        "--router-size",
        type=int,
        required=False,
        default=1,
        help="Number of router nodes",
    )

    parser.add_argument(
        "--node-size",
        type=int,
        required=False,
        default=2,
        help="Number of nodes for multi-node-pd-mix scenario",
    )

    parser.add_argument(
        "--npu-size",
        type=int,
        required=False,
        default=0,
        help="Number of npu for single-node scenario",
    )

    parser.add_argument(
        "--sglang-source-relative-path",
        type=str,
        required=True,
        help="Sglang source code relative path on shared-disk(NFS_ROOT_PATH: /data/ascend-ci-share-pkking-sglang/)",
    )

    parser.add_argument(
        "--metrics-data-file",
        type=str,
        required=False,
        default="",
        help="Metrics data file",
    )

    parser.add_argument(
        "--test-case",
        type=str,
        required=True,
        help="Test case path",
    )

    parser.add_argument(
        "--sglang-is-in-ci",
        action="store_true",
        help="Used to set env var SGLANG_IS_IN_CI in pod",
    )

    parser.add_argument(
        "--install-sglang-from-source",
        action="store_true",
        help="Used to set env var INSTALL_SGLANG_FROM_SOURCE in pod",
    )

    parser.add_argument(
        "--kube-name-space",
        type=str,
        required=True,
        help="K8s name space",
    )

    parser.add_argument(
        "--kube-job-type",
        type=str,
        choices=[KUBE_JOB_SINGLE, KUBE_JOB_MULTI_PD_MIX, KUBE_JOB_MULTI_PD_SEPARATION],
        required=True,
        help=f"K8s job type [{KUBE_JOB_SINGLE}, {KUBE_JOB_MULTI_PD_MIX}, {KUBE_JOB_MULTI_PD_SEPARATION}]",
    )

    parser.add_argument(
        "--kube-job-name-prefix",
        type=str,
        required=True,
        help="K8s job name prefix",
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["debug", "ci", "green"],
        required=True,
        help="Environment type",
    )

    parser.add_argument(
        "--trouble-shotting",
        action="store_true",
        help="Used for troubleshotting issues, such as retaining pods",
    )

    parser.add_argument(
        "--transformers-version",
        type=str,
        required=False,
        default="",
        help="The transformers version number for running sglang. Use default version in image if keep empty.",
    )

    args = parser.parse_args()

    docker_image_url = args.image
    npu_size = int(args.npu_size)
    node_size = int(args.node_size)
    prefill_size = int(args.prefill_size)
    decode_size = int(args.decode_size)
    router_size = int(args.router_size)
    sglang_source_relative_path = args.sglang_source_relative_path
    metrics_data_file = args.metrics_data_file
    test_case = args.test_case
    sglang_is_in_ci = args.sglang_is_in_ci
    install_sglang_from_source = args.install_sglang_from_source
    env = args.env
    trouble_shotting = args.trouble_shotting
    transformers_version = args.transformers_version

    kube_name_space = args.kube_name_space
    kube_job_type = args.kube_job_type
    kube_job_name_prefix = args.kube_job_name_prefix

    resource_info_dict = {
        KUBE_JOB_SINGLE: {"npu_size": npu_size},
        KUBE_JOB_MULTI_PD_MIX: {"node_size": node_size},
        KUBE_JOB_MULTI_PD_SEPARATION: {
            "prefill_size": prefill_size,
            "decode_size": decode_size,
            "router_size": router_size,
        },
    }

    run_npu_e2e_test_case(
        docker_image_url=docker_image_url,
        kube_name_space=kube_name_space,
        kube_job_type=kube_job_type,
        kube_job_name_prefix=kube_job_name_prefix,
        resource_info=resource_info_dict.get(kube_job_type),
        sglang_source_relative_path=sglang_source_relative_path,
        metrics_data_file=metrics_data_file,
        test_case=test_case,
        sglang_is_in_ci=sglang_is_in_ci,
        install_sglang_from_source=install_sglang_from_source,
        env=env,
        trouble_shotting=trouble_shotting,
        transformers_version=transformers_version,
    )
