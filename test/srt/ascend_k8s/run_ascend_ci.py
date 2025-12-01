import json
import re
import signal
import subprocess
import sys
import time

from kubernetes import client, config

KUBE_CONFIG = "/data/.cache/kb.yaml"

config.load_kube_config(KUBE_CONFIG)
v1 = client.CoreV1Api()
LOCAL_TIMEOUT = 10800


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
    start_time = time.time()

    while time.time() - start_time < timeout:
        cmd = "kubectl get pods -A -o json"

        output = run_command(cmd)
        if not output:
            time.sleep(5)
            continue

        try:
            data = json.loads(output)
            all_running = True
            sglang_pods_found = False

            for item in data.get("items", []):
                metadata = item.get("metadata", {})
                pod_name = metadata.get("name", "")
                if "mindx-dls-test-sglang" not in pod_name:
                    continue

                sglang_pods_found = True
                status = item.get("status", {})
                phase = status.get("phase")
                print(f"Pod: {pod_name}, status: {phase}")
                if phase != "Running":
                    all_running = False
                    break

                containers_ready = True
                for condition in status.get("conditions", []):
                    if condition["type"] == "Ready" and condition["status"] != "True":
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
        except json.JSONDecodeError as e:
            print(f"prase json error: {e}")
        time.sleep(5)

    print(f"timeout in {timeout}s")
    return False


def create_configmap():
    cmd = "kubectl get po -A -o wide | grep mindx-dls-test-sglang"
    output = run_command(cmd)

    if not output:
        print("No exist sglang pod")
        return

    data_lines = []
    for line in output.strip().split("\n"):
        parts = line.split()
        pod_name = parts[1]
        ipv4_pattern = r"^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
        for part in parts:
            if bool(re.match(ipv4_pattern, part)):
                pod_ip = part
        data_lines.append(f" {pod_name}: {pod_ip}")

    if not data_lines:
        print("no sglang pod info")
        return

    # generate ConfigMap YAML
    configmap_yaml = """apiVersion: v1
kind: ConfigMap
metadata:
  name: sglang-info
  namespace: kube-system
data:
"""
    configmap_yaml += "\n".join(data_lines)

    # apply ConfigMap
    result = run_command(f"echo '{configmap_yaml}' | kubectl apply -f -")
    if result:
        print("Create ConfigMap successfully!")
        print(result)


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
    print("apply deepep.yaml...")
    result = run_command("kubectl apply -f deepep.yaml")
    if result:
        print(result)

    if check_pods_ready(timeout=LOCAL_TIMEOUT):
        create_configmap()
    else:
        print("Pod not ready, maybe not enough resource")

    monitor_pod_logs("mindx-dls-test-sglang-router-0", "kube-system", LOCAL_TIMEOUT)
