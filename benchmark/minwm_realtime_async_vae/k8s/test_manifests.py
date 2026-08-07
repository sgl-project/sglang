import subprocess
import sys
from pathlib import Path

import yaml

from validate_manifests import find, load_documents, requirement_values, validate


ROOT = Path(__file__).parent


def _containers(deployment):
    return deployment["spec"]["template"]["spec"]["containers"]


def _container(deployment, name):
    return next(item for item in _containers(deployment) if item["name"] == name)


def _init_container(workload, name):
    return next(
        item
        for item in workload["spec"]["template"]["spec"].get("initContainers", [])
        if item["name"] == name
    )


def _gpu_workload(documents, name):
    kind = "StatefulSet" if name == "minwm-async-denoiser" else "Deployment"
    return find(documents, kind, name)


def test_gpu_nodepools_are_spot_only_and_bounded():
    documents = load_documents()
    validate(documents)
    for pool_name in (
        "minwm-async-denoiser-h100",
        "minwm-async-denoiser-h100-8x",
        "minwm-async-vae-l4",
    ):
        node_pool = find(documents, "NodePool", pool_name)
        assert node_pool["spec"]["template"]["spec"]["taints"] == [
            {
                "key": "nvidia.com/gpu",
                "value": "true",
                "effect": "NoSchedule",
            }
        ]


def test_kustomize_does_not_namespace_cluster_scoped_resources():
    kustomization = (ROOT / "kustomization.yaml").read_text()
    assert "\nnamespace:" not in kustomization
    for filename, name in (
        ("h100-denoiser.yaml", "minwm-async-denoiser-h100"),
        ("l4-vae.yaml", "minwm-async-vae-l4"),
    ):
        nodepool = find(load_documents((filename,)), "NodePool", name)
        assert "namespace" not in nodepool["metadata"]


def test_h100_pool_uses_one_fully_utilized_eight_gpu_node():
    documents = load_documents(("h100-denoiser.yaml",))
    single = find(documents, "NodePool", "minwm-async-denoiser-h100")
    packed = find(documents, "NodePool", "minwm-async-denoiser-h100-8x")
    deployment = find(documents, "StatefulSet", "minwm-async-denoiser")

    assert requirement_values(single, "node.kubernetes.io/instance-type") == [
        "p5.48xlarge"
    ]
    assert requirement_values(packed, "node.kubernetes.io/instance-type") == [
        "p5.48xlarge"
    ]
    assert requirement_values(packed, "topology.kubernetes.io/zone") == [
        "us-east-2a",
        "us-east-2b",
        "us-east-2c",
    ]
    assert int(single["spec"]["limits"]["cpu"]) >= 192
    assert int(packed["spec"]["limits"]["cpu"]) >= 192
    assert deployment["spec"]["replicas"] == "REPLACE_WITH_DENOISER_BASE_REPLICAS"
    selector = deployment["spec"]["template"]["spec"]["nodeSelector"]
    assert selector == {
        "karpenter.sh/nodepool": "REPLACE_WITH_DENOISER_NODEPOOL",
        "karpenter.sh/capacity-type": "spot",
    }


def test_denoiser_restarts_all_gpu_pods_as_one_parallel_batch():
    documents = load_documents(("h100-denoiser.yaml",))
    stateful_set = find(documents, "StatefulSet", "minwm-async-denoiser")
    service = find(documents, "Service", "minwm-async-denoiser-headless")

    assert stateful_set["spec"]["podManagementPolicy"] == "Parallel"
    assert stateful_set["spec"]["serviceName"] == "minwm-async-denoiser-headless"
    assert stateful_set["spec"]["updateStrategy"]["type"] == "OnDelete"
    assert service["spec"]["clusterIP"] == "None"
    assert service["spec"]["publishNotReadyAddresses"] is True


def test_l40s_alternate_is_spot_only_and_never_in_base_topology():
    documents = load_documents(("l40s-vae.yaml",))
    nodepool = find(documents, "NodePool", "minwm-async-vae-l40s")
    assert requirement_values(nodepool, "karpenter.sh/capacity-type") == ["spot"]
    assert all(
        value.startswith("g6e.")
        for value in requirement_values(nodepool, "node.kubernetes.io/instance-type")
    )
    kustomization = (Path(__file__).parent / "kustomization.yaml").read_text()
    assert "l40s-vae.yaml" not in kustomization


def test_vae_deployment_can_land_on_either_l4_or_l40s_pool():
    base_documents = load_documents()
    l40s_documents = load_documents(("l40s-vae.yaml",))
    deployment = find(base_documents, "Deployment", "minwm-async-vae")
    selector = deployment["spec"]["template"]["spec"]["nodeSelector"]
    assert selector["seedleap.ai/vae-worker"] == "true"
    assert "karpenter.sh/nodepool" not in selector

    for documents, name in (
        (base_documents, "minwm-async-vae-l4"),
        (l40s_documents, "minwm-async-vae-l40s"),
    ):
        nodepool = find(documents, "NodePool", name)
        labels = nodepool["spec"]["template"]["metadata"]["labels"]
        assert labels["seedleap.ai/vae-worker"] == "true"


def test_vae_pipeline_keeps_one_waiting_latent_and_sends_low_latency_batches():
    deployment = find(
        load_documents(("l4-vae.yaml",)), "Deployment", "minwm-async-vae"
    )
    args = _container(deployment, "vae")["args"]
    assert "--queue-depth-per-session=1" in args
    assert "--encoded-frames-per-batch=4" in args


def test_gpu_workers_publish_epoch_state_and_drain_before_termination():
    for filename, workload_name, worker_name, heartbeat_name, port in (
        (
            "h100-denoiser.yaml",
            "minwm-async-denoiser",
            "denoiser",
            "denoiser-heartbeat",
            30000,
        ),
        (
            "l4-vae.yaml",
            "minwm-async-vae",
            "vae",
            "vae-heartbeat",
            18081,
        ),
    ):
        workload = _gpu_workload(load_documents((filename,)), workload_name)
        worker = _container(workload, worker_name)
        heartbeat = _init_container(workload, heartbeat_name)
        worker_env = {item["name"]: item for item in worker.get("env", [])}
        heartbeat_env = {item["name"]: item for item in heartbeat.get("env", [])}

        assert heartbeat["restartPolicy"] == "Always"
        assert "WORKER_EPOCH" not in worker_env
        assert "WORKER_EPOCH" not in heartbeat_env
        assert worker_env["WORKER_EPOCH_FILE"]["value"] == "/var/run/minwm-worker/epoch"

        heartbeat_command = " ".join(heartbeat["args"])
        assert "--worker-epoch-file=/var/run/minwm-worker/epoch" in heartbeat_command
        assert (
            f"--state-url=http://127.0.0.1:{port}/v1/realtime_worker/state"
            in heartbeat_command
        )
        assert (
            f"--reservation-endpoint=http://$(POD_IP):{port}/v1/realtime_worker"
            in heartbeat_command
        )

        drain_command = " ".join(
            worker["lifecycle"]["preStop"]["exec"]["command"]
        )
        assert f"http://127.0.0.1:{port}/v1/realtime_worker/drain" in drain_command
        assert '\\"deadline\\"' in drain_command

        epoch_volume = next(
            volume
            for volume in workload["spec"]["template"]["spec"]["volumes"]
            if volume["name"] == "worker-epoch"
        )
        assert epoch_volume["emptyDir"] == {}
        for container in (worker, heartbeat):
            mounts = {mount["name"]: mount for mount in container["volumeMounts"]}
            assert mounts["worker-epoch"]["mountPath"] == "/var/run/minwm-worker"


def test_denoiser_enables_dynamic_remote_vae_handoff_without_a_static_worker_url():
    workload = find(
        load_documents(("h100-denoiser.yaml",)),
        "StatefulSet",
        "minwm-async-denoiser",
    )
    command = " ".join(_container(workload, "denoiser")["args"])

    assert "--realtime-remote-vae-enabled" in command
    assert "--realtime-vae-worker-url" not in command


def test_wan22_5b_uses_the_matching_taehv_checkpoint():
    for filename in ("h100-denoiser.yaml", "l4-vae.yaml"):
        manifest = (Path(__file__).parent / filename).read_text()
        assert "taew2_2.pth" in manifest
        assert "taew2_1.pth" not in manifest


def test_webui_enables_i2v_and_t2v_in_production_manifest():
    gateway = find(
        load_documents(("gateway.yaml",)), "Deployment", "minwm-realtime-gateway"
    )
    env = {
        item["name"]: item.get("value")
        for item in _container(gateway, "gateway").get("env", [])
    }

    assert '"generationModes":["i2v","t2v"]' in env["REALTIME_UI_CONFIG_JSON"]
    assert '"t2vDefaultNumFrames":121' in env["REALTIME_UI_CONFIG_JSON"]
    assert env["AWS_REGION"] == "REPLACE_WITH_AWS_REGION"
    assert env["AWS_DEFAULT_REGION"] == "REPLACE_WITH_AWS_REGION"


def test_west_model_artifact_uses_a_matching_read_only_s3_mount():
    documents = load_documents()
    volume = find(documents, "PersistentVolume", "minwm-async-west-s3-pv")
    assert volume["spec"]["csi"]["volumeAttributes"]["bucketName"] == (
        "leap-world-us-west-2"
    )
    assert volume["spec"]["accessModes"] == ["ReadOnlyMany"]

    deployment = find(documents, "StatefulSet", "minwm-async-denoiser")
    pod_spec = deployment["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    stager = pod_spec["initContainers"][0]
    model = next(
        entry["value"]
        for entry in container["env"]
        if entry["name"] == "MINWM_MODEL"
    )
    assert model.startswith("/model-cache/")
    assert any(
        mount["name"] == "model-cache" and mount["readOnly"]
        for mount in container["volumeMounts"]
    )
    assert any(
        mount["name"] == "checkpoint-archive" and mount["readOnly"]
        for mount in stager["volumeMounts"]
    )
    assert any(
        item["name"] == "checkpoint-archive"
        and item["persistentVolumeClaim"]["claimName"] == "minwm-async-west-s3"
        and item["persistentVolumeClaim"]["readOnly"]
        for item in pod_spec["volumes"]
    )


def test_runtime_uses_one_owned_read_only_s3_claim_and_preconverted_model():
    documents = load_documents()
    deployment = find(documents, "StatefulSet", "minwm-async-denoiser")
    pod_spec = deployment["spec"]["template"]["spec"]
    container = _container(deployment, "denoiser")
    env = {item["name"]: item.get("value") for item in container["env"]}
    command = " ".join(container["args"])

    assert env["MINWM_MODEL"] == (
        "/model-cache/"
        "REPLACE_WITH_MODEL_ID/REPLACE_WITH_MODEL_ARTIFACT_REVISION/model"
    )
    assert "MINWM_CHECKPOINT" not in env
    assert "MINWM_DONOR" not in env
    assert "convert_minwm_checkpoint.py" not in command
    assert "cp -a" not in command
    assert "model_index.json" in command
    assert 'test -f "${MINWM_MODEL}/_READY"' in command

    claims = {
        volume["persistentVolumeClaim"]["claimName"]
        for volume in pod_spec["volumes"]
        if "persistentVolumeClaim" in volume
    }
    assert claims == {"minwm-async-west-s3"}
    assert all(mount["name"] != "s3" for mount in container["volumeMounts"])


def test_model_is_staged_once_per_spot_node_before_workers_mmap_it():
    workload = find(
        load_documents(("h100-denoiser.yaml",)),
        "StatefulSet",
        "minwm-async-denoiser",
    )
    pod_spec = workload["spec"]["template"]["spec"]
    stager = pod_spec["initContainers"][0]
    command = " ".join(stager["args"])
    env = {item["name"]: item["value"] for item in stager["env"]}

    assert stager["name"] == "model-stager"
    assert env["SOURCE_MODEL"].startswith("/checkpoint-archive/")
    assert env["CACHED_MODEL"].startswith("/model-cache/")
    assert "flock -x 9" in command
    assert "cp -av" in command
    assert ".staging.$$" in command
    assert 'mv "${staging}" "${CACHED_MODEL}"' in command
    cache = next(volume for volume in pod_spec["volumes"] if volume["name"] == "model-cache")
    assert cache["hostPath"] == {
        "path": "/var/lib/minwm-model-cache",
        "type": "DirectoryOrCreate",
    }


def test_production_resources_are_isolated_in_a_dedicated_namespace():
    documents = load_documents()
    namespace = find(documents, "Namespace", "minwm-realtime")
    assert namespace["metadata"]["labels"]["app.kubernetes.io/part-of"] == (
        "minwm-realtime"
    )

    cluster_scoped = {
        "Namespace",
        "PersistentVolume",
        "ClusterRole",
        "ClusterRoleBinding",
        "NodePool",
    }
    for document in documents:
        if document["kind"] not in cluster_scoped:
            assert document["metadata"].get("namespace") == "minwm-realtime"

    kustomization = (ROOT / "kustomization.yaml").read_text()
    assert "\nnamespace:" not in kustomization
    assert "namespace.yaml" in kustomization


def test_cluster_managed_device_plugin_is_reused():
    kustomization = (ROOT / "kustomization.yaml").read_text()
    assert "gpu-device-plugin.yaml" not in kustomization
    assert "minwm-async-nvidia-device-plugin" not in kustomization


def test_gpu_pods_use_a_prebuilt_runtime_without_installing_at_startup():
    for filename in ("h100-denoiser.yaml", "l4-vae.yaml"):
        manifest = (ROOT / filename).read_text()
        assert "REPLACE_WITH_" in manifest
        assert "IMAGE_DIGEST" in manifest
        assert "GITHUB_TOKEN" not in manifest
        assert "git clone" not in manifest
        assert "pip install" not in manifest
        assert "curl --fail --location" not in manifest
        assert "REPLACE_WITH_GIT_SHA" not in manifest


def test_eight_denoiser_workers_fit_on_one_p5_48xlarge_ephemeral_disk():
    deployment = find(
        load_documents(("h100-denoiser.yaml",)),
        "StatefulSet",
        "minwm-async-denoiser",
    )
    resources = _container(deployment, "denoiser")["resources"]

    assert resources["requests"]["ephemeral-storage"] == "24Gi"
    assert resources["limits"]["ephemeral-storage"] == "32Gi"
    assert 8 * 32 + 100 < 359


def test_denoiser_restarts_as_one_batch_with_two_bounded_cold_load_slots():
    workload = find(
        load_documents(("h100-denoiser.yaml",)),
        "StatefulSet",
        "minwm-async-denoiser",
    )
    pod_spec = workload["spec"]["template"]["spec"]
    denoiser = _container(workload, "denoiser")
    command = " ".join(denoiser["args"])

    assert 'slot=$((ordinal % DENOISER_STARTUP_PARALLELISM))' in command
    assert 'denoiser-${slot}.lock' in command
    assert "flock -x 9" in command
    assert "flock -u 9" in command
    assert "python3 -m sglang.multimodal_gen.runtime.launch_server" in command
    env = {item["name"]: item for item in denoiser["env"]}
    assert env["DENOISER_STARTUP_PARALLELISM"]["value"] == "2"
    assert env["POD_NAME"]["valueFrom"]["fieldRef"]["fieldPath"] == (
        "metadata.name"
    )
    assert denoiser["startupProbe"]["httpGet"] == {
        "path": "/health",
        "port": "api",
    }
    assert any(
        mount["name"] == "startup-lock" for mount in denoiser["volumeMounts"]
    )
    lock_volume = next(
        volume for volume in pod_spec["volumes"] if volume["name"] == "startup-lock"
    )
    assert lock_volume["hostPath"] == {
        "path": "/var/run/minwm-startup-lock",
        "type": "DirectoryOrCreate",
    }


def test_public_nlb_selects_only_the_gateway_control_plane():
    documents = load_documents(("gateway.yaml", "gateway-service.yaml"))
    gateway = find(documents, "Deployment", "minwm-realtime-gateway")
    service = find(documents, "Service", "minwm-realtime-public")

    assert gateway["spec"]["replicas"] == 2
    assert service["spec"]["type"] == "LoadBalancer"
    assert service["spec"]["selector"] == {
        "app.kubernetes.io/name": "minwm-realtime-gateway"
    }
    assert service["spec"]["ports"] == [
        {"name": "http", "port": 80, "targetPort": "http"}
    ]


def test_coordinator_is_an_independent_durable_cpu_control_plane():
    documents = load_documents(("coordinator.yaml",))
    deployment = find(documents, "Deployment", "minwm-realtime-coordinator")
    service = find(documents, "Service", "minwm-realtime-coordinator")
    container = _container(deployment, "coordinator")
    command = " ".join(container["args"])

    assert deployment["spec"]["replicas"] == 2
    assert service["spec"]["type"] == "ClusterIP"
    assert "--backend=dynamodb" in command
    assert "--table-name=$(COORDINATOR_TABLE)" in command
    assert "--denoiser-capacity-limit=4" in command
    assert "--vae-capacity-limit=16" in command
    assert "nvidia.com/gpu" not in container["resources"]["requests"]


def test_cpu_control_planes_are_disruption_protected_and_autoscaled():
    documents = load_documents(
        (
            "cpu-control-plane.yaml",
            "gateway.yaml",
            "coordinator.yaml",
            "autoscaling.yaml",
        )
    )
    node_pool = find(documents, "NodePool", "minwm-realtime-cpu")
    requirements = {
        item["key"]: item["values"]
        for item in node_pool["spec"]["template"]["spec"]["requirements"]
    }
    assert requirements["karpenter.sh/capacity-type"] == ["on-demand"]
    assert node_pool["spec"]["template"]["spec"]["taints"] == [
        {
            "key": "seedleap.ai/task",
            "value": "minwm-realtime-cpu",
            "effect": "NoSchedule",
        }
    ]
    for name in ("minwm-realtime-gateway", "minwm-realtime-coordinator"):
        deployment = find(documents, "Deployment", name)
        pdb = find(documents, "PodDisruptionBudget", name)
        hpa = find(documents, "HorizontalPodAutoscaler", name)
        assert deployment["spec"]["template"]["spec"]["nodeSelector"] == {
            "karpenter.sh/nodepool": "minwm-realtime-cpu"
        }
        assert pdb["spec"]["minAvailable"] == 1
        assert hpa["spec"]["minReplicas"] == 2
        assert hpa["spec"]["maxReplicas"] >= 4


def test_gpu_pools_have_independent_bounded_scheduled_elasticity():
    documents = load_documents(("gpu-scheduled-scaling.yaml",))
    role = find(documents, "Role", "minwm-realtime-gpu-scaler")
    assert role["rules"] == [
        {
            "apiGroups": ["apps"],
            "resources": ["deployments/scale"],
            "resourceNames": ["minwm-async-vae"],
            "verbs": ["get", "patch", "update"],
        },
        {
            "apiGroups": ["apps"],
            "resources": ["statefulsets/scale"],
            "resourceNames": ["minwm-async-denoiser"],
            "verbs": ["get", "patch", "update"],
        }
    ]

    scale_up = find(documents, "CronJob", "minwm-realtime-gpu-scale-up")
    scale_down = find(documents, "CronJob", "minwm-realtime-gpu-scale-down")
    assert scale_up["spec"]["suspend"] == "REPLACE_WITH_GPU_SCALE_UP_SUSPEND"
    assert scale_down["spec"]["suspend"] == "REPLACE_WITH_GPU_SCALE_DOWN_SUSPEND"
    assert scale_up["spec"]["timeZone"] == "REPLACE_WITH_GPU_SCALE_TIME_ZONE"
    assert scale_down["spec"]["timeZone"] == "REPLACE_WITH_GPU_SCALE_TIME_ZONE"

    up_command = " ".join(
        _container(scale_up["spec"]["jobTemplate"], "scaler")["args"]
    )
    down_command = " ".join(
        _container(scale_down["spec"]["jobTemplate"], "scaler")["args"]
    )
    assert "--denoiser-replicas=REPLACE_WITH_DENOISER_PEAK_REPLICAS" in up_command
    assert "--vae-replicas=REPLACE_WITH_VAE_PEAK_REPLICAS" in up_command
    assert "--denoiser-replicas=0" in down_command
    assert "--vae-replicas=0" in down_command

    deploy_script = (ROOT.parent / "deploy_production.sh").read_text()
    for variable in (
        "GPU_SCALE_UP_SCHEDULE",
        "GPU_SCALE_DOWN_SCHEDULE",
        "GPU_SCALE_UP_SUSPEND",
        "GPU_SCALE_DOWN_SUSPEND",
        "DENOISER_PEAK_REPLICAS",
        "VAE_PEAK_REPLICAS",
    ):
        assert variable in deploy_script
    assert "KUBECTL_IMAGE_DIGEST" not in deploy_script
    for job in (scale_up, scale_down):
        container = _container(job["spec"]["jobTemplate"], "scaler")
        assert container["image"] == "REPLACE_WITH_GATEWAY_IMAGE_DIGEST"
        assert "realtime_gpu_scaler" in " ".join(container["args"])


def test_capacity_scaler_uses_the_shared_coordinator_snapshot():
    documents = load_documents(("gpu-capacity-scaler.yaml",))
    deployment = find(
        documents, "Deployment", "minwm-realtime-gpu-capacity-scaler"
    )
    assert deployment["spec"]["replicas"] == "REPLACE_WITH_GPU_EVENT_SCALER_REPLICAS"
    container = _container(deployment, "scaler")
    args = " ".join(container["args"])
    assert "realtime_gpu_scaler" in args
    assert "--coordinator-url=http://minwm-realtime-coordinator:18081" in args
    assert "--denoiser-min-replicas=REPLACE_WITH_DENOISER_BASE_REPLICAS" in args
    assert "--vae-min-replicas=REPLACE_WITH_VAE_BASE_REPLICAS" in args
    assert "--denoiser-max-replicas=REPLACE_WITH_DENOISER_PEAK_REPLICAS" in args
    assert "--vae-max-replicas=REPLACE_WITH_VAE_PEAK_REPLICAS" in args
    assert "--idle-observations-before-scale-down=24" in args
    assert container["image"] == "REPLACE_WITH_GATEWAY_IMAGE_DIGEST"

    deploy_script = (ROOT.parent / "deploy_production.sh").read_text()
    assert "GPU_EVENT_SCALER_SUSPEND" in deploy_script
    assert "GPU_EVENT_SCALER_REPLICAS" in deploy_script


def test_all_runtime_images_are_role_specific_and_immutable():
    files_and_roles = (
        ("gateway.yaml", "Deployment", "minwm-realtime-gateway", "gateway"),
        ("coordinator.yaml", "Deployment", "minwm-realtime-coordinator", "coordinator"),
        ("h100-denoiser.yaml", "StatefulSet", "minwm-async-denoiser", "denoiser"),
        ("l4-vae.yaml", "Deployment", "minwm-async-vae", "vae"),
    )
    for filename, kind, deployment_name, container_name in files_and_roles:
        deployment = find(load_documents((filename,)), kind, deployment_name)
        image = _container(deployment, container_name)["image"]
        assert image == f"REPLACE_WITH_{container_name.upper()}_IMAGE_DIGEST"

    dockerfile = (ROOT.parent / "docker" / "Dockerfile").read_text()
    for target in ("gateway", "coordinator", "denoiser", "vae"):
        assert f" AS {target}" in dockerfile
    assert "taew2_2.pth" in dockerfile


def test_cpu_runtime_dependencies_are_fully_locked_at_image_build_time():
    docker_root = ROOT.parent / "docker"
    dockerfile = (docker_root / "Dockerfile").read_text()
    lock = (docker_root / "requirements-cpu.lock").read_text()

    assert "--require-hashes" in dockerfile
    assert "requirements-cpu.lock" in dockerfile
    for dependency in (
        "boto3==",
        "fastapi==",
        "httpx==",
        "msgspec==",
        "opentelemetry-exporter-otlp-proto-http==",
        "opentelemetry-sdk==",
        "uvicorn==",
        "websockets==",
    ):
        assert dependency in lock


def test_role_image_build_requires_digest_pinned_bases_and_precreated_ecr():
    docker_root = ROOT.parent / "docker"
    dockerfile = (docker_root / "Dockerfile").read_text()
    build_script = (docker_root / "build_and_push.sh").read_text()

    assert "ARG PYTHON_IMAGE=python:" not in dockerfile
    assert "ARG GPU_IMAGE=" not in dockerfile
    assert "PYTHON_IMAGE_DIGEST" in build_script
    assert "GPU_IMAGE_DIGEST" in build_script
    assert "@sha256:" in build_script
    assert '"PYTHON_IMAGE=${PYTHON_IMAGE_DIGEST}"' in build_script
    assert '"GPU_IMAGE=${GPU_IMAGE_DIGEST}"' in build_script
    assert "create-repository" not in build_script


def test_gpu_code_overlay_reuses_the_prebuilt_dependency_runtime():
    overlay = (
        ROOT.parent / "docker" / "Dockerfile.gpu-code-overlay"
    ).read_text()

    assert "ARG GPU_RUNTIME_IMAGE" in overlay
    assert "FROM ${GPU_RUNTIME_IMAGE}" in overlay
    assert "COPY python/sglang /opt/sglang/python/sglang" in overlay
    assert "pip install" not in overlay


def test_cpu_code_overlay_reuses_the_prebuilt_dependency_runtime():
    overlay = (
        ROOT.parent / "docker" / "Dockerfile.cpu-code-overlay"
    ).read_text()

    assert "ARG CPU_RUNTIME_IMAGE" in overlay
    assert "FROM ${CPU_RUNTIME_IMAGE}" in overlay
    assert "COPY python/sglang /opt/sglang/python/sglang" in overlay
    assert "pip install" not in overlay


def test_model_artifact_is_published_immutably_before_runtime_deploy():
    publisher = (ROOT.parent / "publish_model_artifact.py").read_text()
    publish_script = (ROOT.parent / "publish_model_artifact.sh").read_text()
    deploy_script = (ROOT.parent / "deploy_production.sh").read_text()
    publisher_job = (ROOT / "model-artifact-publisher.yaml").read_text()

    assert "convert_minwm_checkpoint.py" in publisher_job
    assert "--link-donor" not in publisher
    assert "sha256" in publisher
    assert "artifact-manifest.json" in publisher
    assert "_READY" in publisher
    assert "put_object" in publisher
    assert "MODEL_ARTIFACT_PUBLISHER_ROLE_ARN" in publish_script
    assert "REPLACE_WITH_MODEL_ID" in deploy_script
    assert "REPLACE_WITH_MODEL_ARTIFACT_REVISION" in deploy_script
    assert "head-object" in deploy_script
    assert "_READY" in deploy_script


def test_model_publisher_reads_the_versioned_east_checkpoint_and_west_donor():
    east_documents = load_documents(("east-s3-source-volume.yaml",))
    east_volume = find(
        east_documents, "PersistentVolume", "minwm-async-east-s3-source-pv"
    )
    assert east_volume["spec"]["csi"]["volumeAttributes"]["bucketName"] == (
        "leap-world-us-east-2"
    )

    publisher = find(
        load_documents(("model-artifact-publisher.yaml",)),
        "Job",
        "minwm-model-artifact-publisher",
    )
    pod_spec = publisher["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    args = container["args"]
    assert "--source-checkpoint=REPLACE_WITH_SOURCE_CHECKPOINT_PATH" in args
    assert "--source-uri=REPLACE_WITH_SOURCE_CHECKPOINT_URI" in args
    assert "--source-version-id=REPLACE_WITH_SOURCE_CHECKPOINT_VERSION_ID" in args
    assert (
        "--output-prefix=world-model/minwm/serving-artifacts/REPLACE_WITH_MODEL_ID"
        in args
    )
    claims = {
        volume["name"]: volume["persistentVolumeClaim"]["claimName"]
        for volume in pod_spec["volumes"]
    }
    assert claims == {
        "checkpoint-east": "minwm-async-east-s3-source",
        "checkpoint-west": "minwm-async-west-s3",
    }


def test_model_publisher_has_a_dedicated_sized_ephemeral_node():
    documents = load_documents(("model-artifact-publisher.yaml",))
    node_class = find(
        documents, "EC2NodeClass", "minwm-model-artifact-publisher"
    )
    node_pool = find(documents, "NodePool", "minwm-model-artifact-publisher")
    mapping = node_class["spec"]["blockDeviceMappings"][0]["ebs"]
    assert mapping["volumeSize"] == "300Gi"
    assert mapping["encrypted"] is True
    assert node_pool["spec"]["template"]["spec"]["nodeClassRef"]["name"] == (
        "minwm-model-artifact-publisher"
    )
    assert set(
        requirement_values(node_pool, "node.kubernetes.io/instance-type")
    ) == {
        "r5.8xlarge",
        "r5a.8xlarge",
        "r6a.8xlarge",
        "r6i.8xlarge",
        "r7a.8xlarge",
        "r7i.8xlarge",
    }


def test_gpu_workers_register_only_internal_pod_endpoints():
    documents = load_documents(("h100-denoiser.yaml", "l4-vae.yaml"))
    expected = {
        "minwm-async-denoiser": (
            "denoiser-heartbeat",
            "denoiser",
            "ws://$(POD_IP):30000/v1/realtime_video/generate",
        ),
        "minwm-async-vae": (
            "vae-heartbeat",
            "vae",
            "ws://$(POD_IP):18081/v1/realtime_vae/decode",
        ),
    }
    for deployment_name, (sidecar_name, role, endpoint) in expected.items():
        deployment = _gpu_workload(documents, deployment_name)
        sidecar = _init_container(deployment, sidecar_name)
        args = " ".join(sidecar["args"])
        assert f"--role={role}" in args
        assert f"--endpoint={endpoint}" in args
        assert "--health-url=" in args
        assert "--node-name=$(NODE_NAME)" in args
        assert "--coordinator-url=http://minwm-realtime-coordinator:18081" in args

    rbac = load_documents(("worker-discovery.yaml",))
    role = find(rbac, "ClusterRole", "minwm-realtime-worker-discovery")
    assert role["rules"] == [
        {"apiGroups": [""], "resources": ["nodes"], "verbs": ["get"]}
    ]


def test_internal_worker_ports_are_restricted_by_network_policy():
    documents = load_documents(("network-policy.yaml",))
    expected = {
        "minwm-realtime-coordinator": 18081,
        "minwm-async-denoiser": 30000,
        "minwm-async-vae": 18081,
        "minwm-realtime-adot": 4317,
    }
    for name, port in expected.items():
        policy = find(documents, "NetworkPolicy", name)
        ingress = policy["spec"]["ingress"]
        ports = [
            item["port"]
            for rule in ingress
            for item in rule.get("ports", [])
        ]
        assert port in ports

    denoiser = find(documents, "NetworkPolicy", "minwm-async-denoiser")
    assert denoiser["spec"]["ingress"][0]["from"][0]["podSelector"] == {
        "matchLabels": {"app.kubernetes.io/name": "minwm-realtime-gateway"}
    }
    vae = find(documents, "NetworkPolicy", "minwm-async-vae")
    assert vae["spec"]["ingress"][0]["from"][0]["podSelector"] == {
        "matchLabels": {"app.kubernetes.io/name": "minwm-async-denoiser"}
    }


def test_gateway_output_queue_absorbs_one_complete_frame_burst():
    gateway = (ROOT / "gateway.yaml").read_text()

    assert "--output-queue-depth=128" in gateway
    assert "--output-drain-timeout-s=90" in gateway


def test_coordinator_candidate_window_covers_the_full_gpu_session_pool():
    documents = load_documents(("coordinator.yaml",))
    coordinator = find(
        documents, "Deployment", "minwm-realtime-coordinator"
    )
    command = " ".join(_container(coordinator, "coordinator")["args"])

    assert "--candidate-limit=64" in command
    assert "--denoiser-capacity-limit=4" in command


def test_trace_uses_otlp_and_cloudwatch_with_five_day_retention():
    documents = load_documents(
        ("gateway.yaml", "coordinator.yaml", "h100-denoiser.yaml", "l4-vae.yaml", "observability.yaml")
    )
    collector = find(documents, "Deployment", "minwm-realtime-adot")
    assert collector["spec"]["replicas"] == 2

    for kind, deployment_name in (
        ("Deployment", "minwm-realtime-gateway"),
        ("Deployment", "minwm-realtime-coordinator"),
        ("StatefulSet", "minwm-async-denoiser"),
        ("Deployment", "minwm-async-vae"),
    ):
        deployment = find(documents, kind, deployment_name)
        main = _containers(deployment)[0]
        env = {item["name"]: item.get("value") for item in main.get("env", [])}
        assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://minwm-realtime-adot:4317"

    deploy_script = (ROOT.parent / "deploy_production.sh").read_text()
    stack = (ROOT.parent / "aws" / "stack.yaml").read_text()
    assert "RetentionInDays: 5" in stack
    assert "put-retention-policy" not in deploy_script
    assert "create-log-group" not in deploy_script
    assert "retentionInDays" in deploy_script


def test_production_deploy_waits_for_every_rollout_and_restores_exact_snapshot():
    deploy_script = (ROOT.parent / "deploy_production.sh").read_text()

    assert "restart_statefulset_in_batches" in deploy_script
    assert 'DENOISER_RESTART_BATCH_SIZE="${DENOISER_RESTART_BATCH_SIZE:-2}"' in deploy_script
    assert "kubectl delete --namespace" in deploy_script
    assert "--wait=true" in deploy_script
    assert "--wait=false" not in deploy_script

    assert "kubectl apply --server-side --force-conflicts --dry-run=server" in deploy_script
    assert "snapshot_workload" in deploy_script
    assert "restore_release_snapshot" in deploy_script
    assert "trap restore_release_snapshot ERR" in deploy_script
    for workload in (
        "deployment/minwm-realtime-adot",
        "deployment/minwm-realtime-coordinator",
        "deployment/minwm-realtime-gateway",
        "deployment/minwm-realtime-gpu-capacity-scaler",
        "statefulset/minwm-async-denoiser",
        "deployment/minwm-async-vae",
    ):
        assert f'wait_for_rollout "{workload}"' in deploy_script
    assert "LEGACY_DENOISER_REPLICAS" in deploy_script
    assert "kubectl scale deployment/minwm-async-denoiser" in deploy_script
    assert "kubectl delete deployment/minwm-async-denoiser" in deploy_script
    assert "prepare_kubernetes_snapshot.py" in deploy_script
    assert "kubectl apply --server-side --force-conflicts" in deploy_script
    assert "kubectl replace --force" not in deploy_script
    assert "failed to snapshot" in deploy_script
    assert "restore_release_snapshot 130" in deploy_script
    assert "restore_release_snapshot 143" in deploy_script
    release_apply = (
        "kubectl apply --server-side --force-conflicts \\\n"
        '  --field-manager=minwm-production -f "${RENDERED}"'
    )
    assert release_apply in deploy_script
    assert deploy_script.index("RELEASE_APPLIED=1") < deploy_script.index(release_apply)
    assert 'NAMESPACE="minwm-realtime"' in deploy_script
    assert '${NAMESPACE:-' not in deploy_script


def test_production_preflight_validates_dynamodb_keys_index_and_ttl():
    validator = ROOT.parent / "validate_coordinator_table.py"
    table = {
        "Table": {
            "TableStatus": "ACTIVE",
            "AttributeDefinitions": [
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
                {"AttributeName": "allocation_key", "AttributeType": "S"},
                {"AttributeName": "allocation_sort", "AttributeType": "S"},
            ],
            "KeySchema": [
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            "GlobalSecondaryIndexes": [
                {
                    "IndexName": "allocation-index",
                    "IndexStatus": "ACTIVE",
                    "Projection": {"ProjectionType": "ALL"},
                    "KeySchema": [
                        {"AttributeName": "allocation_key", "KeyType": "HASH"},
                        {"AttributeName": "allocation_sort", "KeyType": "RANGE"},
                    ],
                }
            ],
        },
        "TimeToLiveDescription": {
            "TimeToLiveStatus": "ENABLED",
            "AttributeName": "ttl",
        },
    }
    valid = subprocess.run(
        [sys.executable, str(validator)],
        input=yaml.safe_dump(table),
        text=True,
        capture_output=True,
    )
    assert valid.returncode == 0, valid.stderr

    table["Table"]["KeySchema"][0]["AttributeName"] = "wrong_pk"
    invalid = subprocess.run(
        [sys.executable, str(validator)],
        input=yaml.safe_dump(table),
        text=True,
        capture_output=True,
    )
    assert invalid.returncode != 0
    assert "primary key schema" in invalid.stderr

    table["Table"]["KeySchema"][0]["AttributeName"] = "pk"
    table["Table"]["AttributeDefinitions"][0]["AttributeType"] = "N"
    invalid_type = subprocess.run(
        [sys.executable, str(validator)],
        input=yaml.safe_dump(table),
        text=True,
        capture_output=True,
    )
    assert invalid_type.returncode != 0
    assert "attribute definitions" in invalid_type.stderr

    table["Table"]["AttributeDefinitions"][0]["AttributeType"] = "S"
    table["Table"]["GlobalSecondaryIndexes"][0]["Projection"] = {
        "ProjectionType": "KEYS_ONLY"
    }
    invalid_projection = subprocess.run(
        [sys.executable, str(validator)],
        input=yaml.safe_dump(table),
        text=True,
        capture_output=True,
    )
    assert invalid_projection.returncode != 0
    assert "projection" in invalid_projection.stderr

    deploy_script = (ROOT.parent / "deploy_production.sh").read_text()
    assert "describe-time-to-live" in deploy_script
    assert "validate_coordinator_table.py" in deploy_script


def test_production_renderer_removes_disposable_test_labels():
    renderer = ROOT.parent / "render_production.py"
    source = """
apiVersion: v1
kind: Namespace
metadata:
  name: minwm-realtime
  labels:
    seedleap.ai/test-run: minwm-async-vae-benchmark
    seedleap.ai/ttl-after-test: required
    app.kubernetes.io/name: minwm-realtime
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: example
spec:
  template:
    metadata:
      labels:
        seedleap.ai/test-run: minwm-async-vae-benchmark
        app: example
"""
    completed = subprocess.run(
        [sys.executable, str(renderer)],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    )
    documents = list(yaml.safe_load_all(completed.stdout))

    assert documents[0]["metadata"]["labels"] == {
        "app.kubernetes.io/name": "minwm-realtime",
        "seedleap.ai/environment": "production",
    }
    assert documents[1]["metadata"]["labels"] == {
        "seedleap.ai/environment": "production"
    }
    assert documents[1]["spec"]["template"]["metadata"]["labels"] == {
        "app": "example",
        "seedleap.ai/environment": "production",
    }

    deploy_script = (ROOT.parent / "deploy_production.sh").read_text()
    assert "render_production.py" in deploy_script


def test_trace_is_exported_over_otlp_without_using_video_websocket():
    documents = load_documents(("observability.yaml",))
    assert not any(document["kind"] == "DaemonSet" for document in documents)
    config = find(documents, "ConfigMap", "minwm-realtime-adot-config")
    collector = config["data"]["collector.yaml"]
    assert "awscloudwatchlogs" in collector
    assert "raw_log: true" in collector
    assert "receivers: [otlp]" in collector
    assert "exporters: [awscloudwatchlogs]" in collector

    gateway_source = (
        ROOT.parents[2]
        / "python/sglang/multimodal_gen/runtime/entrypoints/realtime_gateway_server.py"
    ).read_text()
    assert '"/v1/realtime_video/traces/{trace_id}"' in gateway_source
    assert 'encode_message("client_trace"' not in gateway_source
    assert 'encode_message("trace_event"' not in gateway_source

    deploy_script = (ROOT.parent / "deploy_production.sh").read_text()
    assert "FLUENT_BIT_IMAGE_DIGEST" not in deploy_script


def test_kustomization_renders_the_complete_production_chain():
    resources = (ROOT / "kustomization.yaml").read_text()
    for filename in (
        "gateway.yaml",
        "coordinator.yaml",
        "autoscaling.yaml",
        "gpu-scheduled-scaling.yaml",
        "gpu-capacity-scaler.yaml",
        "observability.yaml",
        "worker-discovery.yaml",
        "network-policy.yaml",
        "h100-denoiser.yaml",
        "l4-vae.yaml",
        "gateway-service.yaml",
    ):
        assert filename in resources
