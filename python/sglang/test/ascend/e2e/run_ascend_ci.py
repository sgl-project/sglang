import re
import string
import subprocess
import sys
import time
import os
import argparse
import uuid
import random
from jinja2 import Template

import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException


KUBE_CONFIG = os.environ.get('KUBECONFIG')
config.load_kube_config(KUBE_CONFIG)
core_api = client.CoreV1Api()
custom_api = client.CustomObjectsApi()
batch_api = client.BatchV1Api()
rbac_api = client.RbacAuthorizationV1Api()

LOCAL_TIMEOUT = 10800

script_path = os.path.dirname(os.path.abspath(__file__))
KUBE_YAML_TEMPLATE = {
    "single": f"{script_path}/k8s_single.yaml.jinja2",
    "multi-pd-mix": f"{script_path}/k8s_multi_pd_mix.yaml.jinja2",
    "multi-pd-separation": f"{script_path}/k8s_multi_pd_separation.yaml.jinja2"
}

def get_unique_random_string(length: int = 16, add_random: bool = True) -> str:
    uuid_str = str(uuid.uuid4()).replace("-", "")

    if add_random:
        if length < 8:
            raise ValueError("length can not be smaller than 8")
        random_length = length - 8
        char_pool = string.ascii_lowercase + string.digits
        random_chars = ''.join([random.choice(char_pool) for _ in range(random_length)])
        result = uuid_str[:8] + random_chars
    else:
        result = uuid_str[:length]

    return result

def create_pod_yaml(kube_yaml_template, output_yaml, pod_context):
    with open(kube_yaml_template, 'r') as f:
        template = Template(f.read())
    kube_pod_yaml = template.render(pod_context)
    with open(output_yaml, 'w') as f:
        f.write(kube_pod_yaml)
    print(f"Pod YAML written to {output_yaml}")

def create_pod(yaml_file, namespace):
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
                raise f"Unrecognized kind: {kind}/{api_version}"
        except ApiException as e:
            print(f"create resource {kind} error: {e}")
            raise

def delete_pod(yaml_file, namespace):
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
                        grace_period_seconds=0,
                        propagation_policy="Foreground"
                    )
                )
                print(f"Volcano Job {job_name} is deleted.")
                print(f"Response status: {response.get('status')}")
            elif kind == "ConfigMap" and api_version == "v1":
                config_map_name = doc["metadata"]["name"]
                response = core_api.delete_namespaced_config_map(name=config_map_name, namespace=namespace)
                print(f"ConfigMap {config_map_name} is deleted.")
                print(f"Response: {response}")
            else:
                raise f"Unrecognized kind: {kind}/{api_version}"
        except ApiException as e:
            raise f"delete resource {kind} error: {e}"

def check_pods_ready(namespace, pod_name_key_str, timeout=300):
    print("Waiting all pods to running...")
    start_time = time.time()

    while time.time() - start_time < timeout:
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

def prepare_cm_data(namespace, pod_string):
    pods = core_api.list_namespaced_pod(namespace=namespace)
    data = {}
    for pod in pods.items:
        pod_name = pod.metadata.name
        if pod_string in pod_name:
            pod_ip = pod.status.pod_ip
            data[pod_name] = pod_ip
    return data

def monitor_pod_logs(pod_name, namespace, timeout=LOCAL_TIMEOUT):
    # Build kubectl command
    cmd = ["kubectl", "logs", "-f", "-n", namespace, pod_name]

    # Define multiline pattern to match
    pattern_lines = [r"^-{70,}$", r"^Ran \d+ tests? in [\d.]+s$", r"^$", r"^(OK|FAILED \(errors=\d+\))$"]
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

        print(f"Starting to monitor logs for Pod: {pod_name}")
        match_state = 0
        matched = False
        is_success = False

        start_time = time.time()
        # Process output
        while process.poll() is None and not matched:
            if time.time() - start_time > timeout:
                raise Exception("Timeout exceeded, the thread is {timeout} seconds long.}")

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
                    if pattern_ok.match(line):
                        is_success = True
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
        elif not is_success:
            raise Exception("The test result was FAILED!")
        else:
            print("The test result was OK!")
            return True

    except Exception as e:
        print(f"\nError: {e}")
        return False
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

def run_ascend_e2e_test_case(
    docker_image_url: str,
    kube_name_space: str,
    kube_job_type: str, # multi-pd-separation、multi-pd-mix、single
    kube_job_name_prefix: str, # kube job prefix-name
    resource_info: dict, # pd-separation: {"prefill_size": 1, "decode_size": 1, "router_size": 1}; pd-mix: {"node_size": 2; single: {"npu_size": 4}
    sglang_source_relative_path: str,
    metrics_data_file: str,
    test_case: str,
    sglang_is_in_ci = False,
    install_sglang_from_source = False,
    env = "debug", # ["debug", "ci"]
):
    random_str = get_unique_random_string(16, True)

    kube_config_map = f"sglang-configmap-{random_str}"
    final_kube_job_name = f"{kube_job_name_prefix}-{random_str}"

    kube_yaml_file_dict = {
        "single": f"k8s_single_{random_str}.yaml",
        "multi-pd-mix": f"k8s_multi_pd_mix_{random_str}.yaml",
        "multi-pd-separation": f"k8s_multi_pd_separation_{random_str}.yaml"
    }
    kube_yaml_file = kube_yaml_file_dict.get(kube_job_type)

    try:
        print(f"Apply k8s yaml... KUBE_NAME_SPACE:{kube_name_space}, KUBE_CONFIG_MAP:{kube_config_map}, "
              f"KUBE_JOB_TYPE:{kube_job_type}, KUBE_YAML_FILE:{kube_yaml_file}")

        if kube_job_type == "single":
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
                "env": env
            }
            create_pod_yaml(
                kube_yaml_template=KUBE_YAML_TEMPLATE.get(kube_job_type),
                output_yaml=kube_yaml_file,
                pod_context=k8s_context
            )
        elif kube_job_type == "multi-pd-mix":
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
                "env": env
            }
            create_pod_yaml(
                kube_yaml_template=KUBE_YAML_TEMPLATE.get(kube_job_type),
                output_yaml=kube_yaml_file,
                pod_context=k8s_context
            )
        elif kube_job_type == "multi-pd-separation":
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
                "env": env
            }
            create_pod_yaml(
                kube_yaml_template=KUBE_YAML_TEMPLATE.get(kube_job_type),
                output_yaml=kube_yaml_file,
                pod_context=k8s_context
            )
        else:
            raise Exception(f"Unknown k8s job type: {kube_job_type}")

        create_pod(yaml_file=kube_yaml_file, namespace=kube_name_space)

        if check_pods_ready(kube_name_space, final_kube_job_name, timeout=LOCAL_TIMEOUT):
            if kube_job_type != "single":
                matching_pod_string = final_kube_job_name
                cm_data = prepare_cm_data(kube_name_space, matching_pod_string)
                if not cm_data:
                    print(f"No sglang pod found while matching {matching_pod_string}")

                response = create_or_update_configmap(
                    cm_name=kube_config_map,
                    data=cm_data,
                    namespace=kube_name_space
                )
                print(response)
        else:
            print("Pod not ready, maybe not enough resource")

        monitor_pod_name = {
            "single": f"{final_kube_job_name}-pod-0",
            "multi-pd-mix": f"{final_kube_job_name}-sglang-node-0",
            "multi-pd-separation": f"{final_kube_job_name}-sglang-router-0",
        }
        return monitor_pod_logs(monitor_pod_name.get(kube_job_type), kube_name_space, LOCAL_TIMEOUT)

    except Exception as e:
        print(f"\nError occured while running k8s task: {e}")
        return False
    finally:
        if os.path.exists(kube_yaml_file):
            delete_pod(yaml_file=kube_yaml_file, namespace=kube_name_space)
            os.remove(kube_yaml_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Apply k8s yaml",
        formatter_class=argparse.RawTextHelpFormatter
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
        action='store_true',
        help="Used to set env var SGLANG_IS_IN_CI in pod",
    )

    parser.add_argument(
        "--install-sglang-from-source",
        action='store_true',
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
        choices=["single", "multi-pd-mix", "multi-pd-separation"],
        required=True,
        help="K8s job type [single, multi-pd-mix, multi-pd-separation]",
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
        choices=["debug", "ci"],
        required=True,
        help="Environment type",
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

    kube_name_space = args.kube_name_space
    kube_job_type = args.kube_job_type
    kube_job_name_prefix = args.kube_job_name_prefix

    resource_info_dict = {
        "single": {"npu_size": npu_size},
        "multi-pd-mix": {"node_size": node_size},
        "multi-pd-separation": {"prefill_size": prefill_size, "decode_size": decode_size, "router_size": router_size},
    }

    run_ascend_e2e_test_case(
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
    )
