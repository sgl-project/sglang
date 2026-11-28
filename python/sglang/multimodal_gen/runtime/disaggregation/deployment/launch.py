# SPDX-License-Identifier: Apache-2.0
"""Start the coordinator or the stage workers of a declarative deployment."""

import multiprocessing as mp
import time

from sglang.multimodal_gen.registry import get_model_info, get_pipeline_class
from sglang.multimodal_gen.runtime.disaggregation.deployment.coordinator import (
    DeploymentServer,
)
from sglang.multimodal_gen.runtime.disaggregation.deployment.plan import (
    Deployment,
    ExecutionPlan,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.managers.worker_bootstrap import (
    ServerArgsPayload,
    bootstrap_stage_role,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.common import parse_tcp_host_port


def launch_stage_deployment(server_args: ServerArgs):
    # the launcher owns process startup and shutdown; reuse it rather than fork it
    from sglang.multimodal_gen.runtime import launch_server as launcher

    if server_args.disagg_timeout <= 0 or server_args.disagg_max_slots_per_instance < 1:
        raise ValueError("Deployment timeout and capacity must be positive")
    if server_args.batching_max_size != 1:
        raise ValueError(
            "Stage workers currently execute one request at a time; use role replicas for concurrency"
        )
    if server_args.disagg_role != RoleType.MONOLITHIC or server_args.dp_size != 1:
        raise ValueError(
            "Stage deployment uses named role replicas, not --disagg-role or --dp-size"
        )
    pipeline_cls = (
        get_pipeline_class(server_args.pipeline_class_name)
        if server_args.pipeline_class_name
        else get_model_info(
            server_args.model_path,
            backend=server_args.backend,
            model_id=server_args.model_id,
        ).pipeline_cls
    )
    deployment = Deployment.load(server_args.stage_deployment)
    if not deployment.stages and any(
        "model_path" in role.server_args for role in deployment.roles.values()
    ):
        raise ValueError("Per-role models require explicit deployment stage contracts")
    plan = ExecutionPlan.compile(deployment, pipeline_cls.stage_specs)
    if server_args.stage_node:
        if server_args.stage_role:
            raise ValueError("Use --stage-node or --stage-role, not both")
        context = mp.get_context("spawn")
        processes = []
        reserved = {
            parse_tcp_host_port(endpoint, "stage endpoint")[1]
            for role in deployment.roles.values()
            for endpoint in role.endpoints
        }
        try:
            for name, role in deployment.roles.items():
                if role.node != server_args.stage_node:
                    continue
                forbidden = {
                    key
                    for key in role.server_args
                    if key.startswith("_")
                    or key
                    in {
                        "stage_deployment",
                        "stage_node",
                        "stage_role",
                        "stage_endpoint",
                        "disagg_role",
                    }
                }
                if forbidden:
                    raise ValueError(f"Role {name} cannot override {sorted(forbidden)}")
                for index, endpoint in enumerate(role.endpoints):
                    overrides = dict(role.server_args)
                    for port in ("scheduler_port", "master_port"):
                        overrides[port] = launcher._find_available_port(avoid=reserved)
                        reserved.add(overrides[port])
                    overrides["base_gpu_id"] = int(
                        overrides.get("base_gpu_id", server_args.base_gpu_id)
                    ) + index * int(
                        overrides.get(
                            "num_gpus",
                            1 if "model_path" in overrides else server_args.num_gpus,
                        )
                    )
                    args = server_args.for_stage_role(
                        name,
                        **overrides,
                        stage_endpoint=endpoint,
                    )
                    process = context.Process(
                        target=bootstrap_stage_role,
                        args=(ServerArgsPayload.capture(args),),
                        name=f"diffusion-{name}-{index}",
                    )
                    process.start()
                    processes.append(process)
            if not processes:
                raise ValueError(
                    f"No workers assigned to node {server_args.stage_node}"
                )
            while all(process.is_alive() for process in processes):
                time.sleep(0.2)
            raise RuntimeError(
                "A stage worker exited; stopping the node's remaining workers"
            )
        finally:
            launcher.shutdown_scheduler_processes(
                None, processes, request_shutdown=False
            )
    if server_args.stage_role:
        role = plan.roles[server_args.stage_role]
        if server_args.stage_endpoint not in role.endpoints:
            raise ValueError(
                "--stage-endpoint must identify a worker listed for --stage-role"
            )
        server_args.warmup_mode = "off"
        processes = launcher.launch_server(server_args, launch_http_server=False)
        try:
            for process in processes:
                process.join()
        finally:
            launcher.shutdown_scheduler_processes(
                None, processes, request_shutdown=False
            )
        return
    launcher.configure_logger(server_args)
    server = DeploymentServer(
        plan,
        server_args.disagg_timeout,
        server_args.disagg_max_slots_per_instance,
        dispatch_policy=server_args.disagg_dispatch_policy,
    )
    server.start(server_args.scheduler_endpoint_for(0))
    try:
        launcher.launch_http_server_only(server_args)
    finally:
        server.stop()
