from pathlib import Path

from validate_manifests import find, load_documents, requirement_values, validate


ROOT = Path(__file__).parent


def _containers(deployment):
    return deployment["spec"]["template"]["spec"]["containers"]


def _container(deployment, name):
    return next(item for item in _containers(deployment) if item["name"] == name)


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
        "p5.4xlarge"
    ]
    assert requirement_values(packed, "node.kubernetes.io/instance-type") == [
        "p5.48xlarge"
    ]
    assert requirement_values(packed, "topology.kubernetes.io/zone") == [
        "us-east-2a"
    ]
    assert deployment["spec"]["replicas"] == "REPLACE_WITH_DENOISER_BASE_REPLICAS"
    selector = deployment["spec"]["template"]["spec"]["nodeSelector"]
    assert selector == {
        "karpenter.sh/nodepool": "REPLACE_WITH_DENOISER_NODEPOOL",
        "karpenter.sh/capacity-type": "spot",
    }


def test_denoiser_uses_ordered_startup_to_avoid_a_cold_start_storm():
    documents = load_documents(("h100-denoiser.yaml",))
    stateful_set = find(documents, "StatefulSet", "minwm-async-denoiser")
    service = find(documents, "Service", "minwm-async-denoiser-headless")

    assert stateful_set["spec"]["podManagementPolicy"] == "OrderedReady"
    assert stateful_set["spec"]["serviceName"] == "minwm-async-denoiser-headless"
    assert stateful_set["spec"]["updateStrategy"]["type"] == "RollingUpdate"
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


def test_vae_pipeline_keeps_one_waiting_latent_and_streams_single_frames():
    deployment = find(
        load_documents(("l4-vae.yaml",)), "Deployment", "minwm-async-vae"
    )
    args = _container(deployment, "vae")["args"]
    assert "--queue-depth-per-session=1" in args
    assert "--encoded-frames-per-batch=1" in args


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
    manifest = (Path(__file__).parent / "h100-denoiser.yaml").read_text()
    assert '"generationModes":["i2v","t2v"]' in manifest
    assert '"t2vDefaultNumFrames":121' in manifest


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
        "wan22-5b-stage3-dmd-30-gs1800/REPLACE_WITH_MODEL_ARTIFACT_REVISION/model"
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


def test_denoiser_serializes_cold_starts_even_after_a_node_restart():
    workload = find(
        load_documents(("h100-denoiser.yaml",)),
        "StatefulSet",
        "minwm-async-denoiser",
    )
    pod_spec = workload["spec"]["template"]["spec"]
    denoiser = _container(workload, "denoiser")
    command = " ".join(denoiser["args"])

    assert "flock -x 9" in command
    assert "flock -u 9" in command
    assert "exec python3 -m sglang.multimodal_gen.runtime.launch_server" not in command
    assert "python3 -m sglang.multimodal_gen.runtime.launch_server" in command
    assert "http://127.0.0.1:30000/health" in command
    assert any(
        mount["name"] == "startup-lock"
        and mount["mountPath"] == "/var/run/minwm-startup-lock"
        for mount in denoiser["volumeMounts"]
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
    assert "REPLACE_WITH_MODEL_ARTIFACT_REVISION" in deploy_script
    assert "head-object" in deploy_script
    assert "_READY" in deploy_script


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
        sidecar = _container(deployment, sidecar_name)
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

    assert "--output-queue-depth=32" in gateway


def test_coordinator_candidate_window_covers_the_full_gpu_session_pool():
    documents = load_documents(("coordinator.yaml",))
    coordinator = find(
        documents, "Deployment", "minwm-realtime-coordinator"
    )
    command = " ".join(_container(coordinator, "coordinator")["args"])

    assert "--candidate-limit=64" in command


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
        "observability.yaml",
        "worker-discovery.yaml",
        "network-policy.yaml",
        "h100-denoiser.yaml",
        "l4-vae.yaml",
        "gateway-service.yaml",
    ):
        assert filename in resources
