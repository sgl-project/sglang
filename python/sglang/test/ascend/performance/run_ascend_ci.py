import re
import signal
import subprocess
import sys
import time
import os

import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

config.load_kube_config(os.environ.get('KUBECONFIG'))
core_api = client.CoreV1Api()
custom_api = client.CustomObjectsApi()
batch_api = client.BatchV1Api()
rbac_api = client.RbacAuthorizationV1Api()

LOCAL_TIMEOUT = 10800
KUBE_NAME_SPACE = os.environ.get('NAMESPACE')
KUBE_CONFIG_MAP = os.environ.get('KUBE_CONFIG_MAP')
KUBE_JOB_TYPE = os.environ.get('KUBE_JOB_TYPE')
MONITOR_POD_NAME = "{}-pod-0".format(os.environ.get('KUBE_JOB_NAME')) if KUBE_JOB_TYPE == "single" else \
    "{}-sglang-node-0".format(os.environ.get('KUBE_JOB_NAME')) if KUBE_JOB_TYPE == "multi" else \
    "{}-sglang-router-0".format(os.environ.get('KUBE_JOB_NAME'))
KUBE_YAML_FILE = os.environ.get('KUBE_YAML_FILE')
if not KUBE_YAML_FILE:
    KUBE_YAML_FILE = "k8s_single.yaml" if KUBE_JOB_TYPE == "single" else "k8s_multi_pd_mix.yaml" if KUBE_JOB_TYPE == "multi" else "k8s_multi_pd_separation.yaml"

def create_pod(yaml_file=KUBE_YAML_FILE, namespace=KUBE_NAME_SPACE):
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
                print(f"Pod {doc['metadata']['name']} is created")

            elif kind == "Job" and api_version == "batch/v1":
                batch_api.create_namespaced_job(namespace=namespace, body=doc)
                print(f"Job {doc['metadata']['name']} is created")

            elif kind == "Job" and api_version == "batch.volcano.sh/v1alpha1":
                response = custom_api.create_namespaced_custom_object(
                    group="batch.volcano.sh",
                    version="v1alpha1",
                    namespace=namespace,
                    plural="jobs",
                    body=doc
                )
                print(f"Volcano Job {doc['metadata']['name']} is created")
                print(f"Response info: {response['metadata']['name']}")

            elif kind == "ConfigMap" and api_version == "v1":
                core_api.create_namespaced_config_map(namespace=namespace, body=doc)
                print(f"ConfigMap {doc['metadata']['name']} is created")

            elif kind == "Role" and api_version == "rbac.authorization.k8s.io/v1":
                rbac_api.create_namespaced_role(
                    namespace=namespace,
                    body=doc
                )
                print(f"Role {doc['metadata']['name']} is created")

            elif kind == "RoleBinding" and api_version == "rbac.authorization.k8s.io/v1":
                rbac_api.create_namespaced_role_binding(
                    namespace=namespace,
                    body=doc
                )
                print(f"RoleBinding {doc['metadata']['name']} is created")

            else:
                print(f"Unrecognized kind: {kind}/{api_version}")

        except ApiException as e:
            print(f"create resource {kind} error: {e}")
            raise

def check_pods_ready(timeout=300):
    print("Waiting all pods to running...")
    matching_string = "{}".format(os.environ.get('KUBE_JOB_NAME'))
    start_time = time.time()

    while time.time() - start_time < timeout:
        pods = core_api.list_namespaced_pod(namespace=KUBE_NAME_SPACE)

        if len(pods.items) == 0:
            time.sleep(5)
            continue

        all_running = True
        sglang_pods_found = False
        for pod in pods.items:
            pod_name = pod.metadata.name
            if matching_string not in pod_name:
                continue

            sglang_pods_found = True
            status = pod.status
            phase = status.phase
            print(f"Pod: {pod_name}, status: {phase}")
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
            print("No sglang pod, waiting...")
            time.sleep(5)
            continue
        if all_running:
            print("All sglang Pod is Running !")
            return True

        time.sleep(5)

    print(f"timeout in {timeout}s")
    return False

def create_or_update_configmap(cm_name: str, data: dict, namespace: str):
    cm_metadata = client.V1ObjectMeta(name=cm_name, namespace=namespace)
    configmap = client.V1ConfigMap(
        api_version="v1",
        kind="ConfigMap",
        metadata=cm_metadata,
        data=data)

    try:
        response = core_api.create_namespaced_config_map(
            namespace=namespace,
            body=configmap
        )
        print(f"ConfigMap '{cm_name}' create successfully!")
        print(f"data: {list(data.keys())}")
        return response
    except ApiException as e:
        if e.status == 409:
            print(f"ConfigMap {cm_name} already exists. Updating...")
            response = core_api.replace_namespaced_config_map(
                namespace=namespace,
                name=cm_name,
                body=configmap
            )
            print(f"ConfigMap {cm_name} updated successfully.")
            return response
        else:
            error_msg = f"ConfigMap create failed: {e.reason}"
            if e.body:
                error_msg += f" | details: {e.body}"
            print(error_msg)
            raise

def prepare_cm_data(pod_string):
    pods = core_api.list_namespaced_pod(namespace=KUBE_NAME_SPACE)
    data = {}
    for pod in pods.items:
        pod_name = pod.metadata.name
        if pod_string in pod_name:
            pod_ip = pod.status.pod_ip
            data[pod_name] = pod_ip
    return data

def monitor_pod_logs(pod_name, namespace=None, timeout=None):
    class TimeoutException(Exception):
        """Custom exception for timeout events"""

        pass

    def timeout_handler(signum, frame):
        """Signal handler for timeout events"""
        raise TimeoutException("Monitoring timeout")

    # Build kubectl command
    cmd = ["kubectl", "logs", "-f", pod_name]
    if namespace:
        cmd.extend(["-n", namespace])

    # Define multiline pattern to match
    pattern_lines = [r"^-{70,}$", r"^Ran \d+ test in [\d.]+s$", r"^$", r"^OK$"]

    # Compile regex patterns
    patterns = [re.compile(line_pattern) for line_pattern in pattern_lines]

    # Set up timeout handling
    if timeout:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

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

        print(f"Starting to monitor logs for Pod: {pod_name}")
        if namespace:
            print(f"Namespace: {namespace}")
        if timeout:
            print(f"Timeout set to: {timeout} seconds")
        match_state = 0
        matched = False

        # Process output
        while process.poll() is None and not matched:
            line = process.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            line = line.rstrip("\n")
            print(line)
            # Check if current line matches expected pattern
            if match_state < len(patterns) and patterns[match_state].match(line):
                match_state += 1
                if match_state == len(patterns):
                    matched = True
                    print("\nSuccessfully detected complete test completion pattern!")
            else:
                match_state = 0
                if patterns[0].match(line):
                    match_state = 1

        # Check if pattern was successfully matched
        if not matched:
            if process.poll() is not None:
                remaining_output, stderr_output = process.communicate()
                if remaining_output:
                    print(remaining_output)
                if stderr_output:
                    raise Exception(f"kubectl command error: {stderr_output}")
                else:
                    raise Exception(
                        "Pod logs ended but target pattern was not detected"
                    )
            else:
                raise Exception("Monitoring ended but target pattern was not detected")
        print("Monitoring completed successfully. Script exiting.")

    except TimeoutException:
        print(f"\nError: Target pattern not detected within {timeout} seconds")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nError: Monitoring interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        if timeout:
            signal.alarm(0)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

if __name__ == "__main__":
    print("Apply k8s yaml... KUBE_NAME_SPACE:{}, KUBE_CONFIG_MAP:{}, KUBE_JOB_TYPE:{}, KUBE_YAML_FILE:{}"
          .format(KUBE_NAME_SPACE, KUBE_CONFIG_MAP, KUBE_JOB_TYPE, KUBE_YAML_FILE))

    create_pod(yaml_file=KUBE_YAML_FILE, namespace=KUBE_NAME_SPACE)

    if check_pods_ready(timeout=LOCAL_TIMEOUT):
        if KUBE_JOB_TYPE != "single":
            matching_pod_string = os.environ.get('KUBE_JOB_NAME')
            cm_data = prepare_cm_data(matching_pod_string)
            if not cm_data:
                print(f"No sglang pod found while matching {matching_pod_string}")

            response = create_or_update_configmap(cm_name=KUBE_CONFIG_MAP, data=cm_data, namespace=KUBE_NAME_SPACE)
            print(response)
    else:
        print("Pod not ready, maybe not enough resource")

    monitor_pod_logs(MONITOR_POD_NAME, KUBE_NAME_SPACE, LOCAL_TIMEOUT)
