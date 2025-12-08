import re
import signal
import subprocess
import sys
import time
import os

from kubernetes import client, config
from kubernetes.client.rest import ApiException

config.load_kube_config(os.environ.get('KUBECONFIG'))
v1 = client.CoreV1Api()

LOCAL_TIMEOUT = 10800
KUBE_NAME_SPACE = os.environ.get('NAMESPACE')
KUBE_CONFIG_MAP = os.environ.get('KUBE_CONFIG_MAP')
KUBE_JOB_TYPE = os.environ.get('KUBE_JOB_TYPE')
MONITOR_POD_NAME = "{}-sglang-router-0".format(os.environ.get('KUBE_JOB_NAME')) if KUBE_JOB_TYPE != "single" else \
    "{}-pod-0".format(os.environ.get('KUBE_JOB_NAME'))
KUBE_YAML_FILE = os.environ.get('KUBE_YAML_FILE')
if not KUBE_YAML_FILE:
    KUBE_YAML_FILE = "k8s_single.yaml" if KUBE_JOB_TYPE == "single" else "deepep.yaml"

def run_command(cmd, shell=True):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"execute command error: {e}")
        return None

def check_pods_ready(timeout=300):
    print("Waiting all pods to running...")
    matching_string = "{}".format(os.environ.get('KUBE_JOB_NAME'))
    start_time = time.time()

    while time.time() - start_time < timeout:
        pods = v1.list_namespaced_pod(namespace=KUBE_NAME_SPACE)

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
        response = v1.create_namespaced_config_map(
            namespace=namespace,
            body=configmap
        )
        print(f"ConfigMap '{cm_name}' create successfully！")
        print(f"data: {list(data.keys())}")
        return response
    except ApiException as e:
        if e.status == 409:
            print(f"ConfigMap {cm_name} already exists. Updating...")
            response = v1.replace_namespaced_config_map(
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

def repare_cm_data(matching_pod_string):
    pods = v1.list_namespaced_pod(namespace=KUBE_NAME_SPACE)
    cm_data = {}

    for pod in pods.items:
        pod_name = pod.metadata.name
        if matching_pod_string in pod_name:
            pod_ip = pod.status.pod_ip
            cm_data[pod_name] = pod_ip
    return cm_data

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
    print("apply k8s yaml... KUBE_NAME_SPACE:{}, KUBE_CONFIG_MAP:{}, KUBE_JOB_TYPE:{}, KUBE_YAML_FILE:{}"
          .format(KUBE_NAME_SPACE, KUBE_CONFIG_MAP, KUBE_JOB_TYPE, KUBE_YAML_FILE))
    
    result = run_command("kubectl apply -f {}".format(KUBE_YAML_FILE))
    if result:
        print(result)

    if check_pods_ready(timeout=LOCAL_TIMEOUT):
        if KUBE_JOB_TYPE != "single":
            matching_pod_string = os.environ.get('KUBE_JOB_NAME')
            cm_data = repare_cm_data(matching_pod_string)
            if not cm_data:
                print(f"No sglang pod found while matching {matching_pod_string}")
                
            response = create_or_update_configmap(cm_name=KUBE_CONFIG_MAP, data=cm_data, namespace=KUBE_NAME_SPACE)
            print(response)
    else:
        print("Pod not ready, maybe not enough resource")

    monitor_pod_logs(MONITOR_POD_NAME, KUBE_NAME_SPACE, LOCAL_TIMEOUT)
