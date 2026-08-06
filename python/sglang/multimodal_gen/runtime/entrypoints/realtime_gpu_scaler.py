# SPDX-License-Identifier: Apache-2.0

"""Bounded Kubernetes GPU workload scaler used by the production CronJobs."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


SERVICE_ACCOUNT_ROOT = Path("/var/run/secrets/kubernetes.io/serviceaccount")
WORKLOAD_RESOURCES = {
    "minwm-async-denoiser": "statefulsets",
    "minwm-async-vae": "deployments",
}
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CapacityRoleSnapshot:
    waiting_sessions: int
    active_sessions: int
    queued_sessions: int
    free_slots: int
    draining_workers: int


@dataclass(frozen=True, slots=True)
class ScaleDecision:
    target_replicas: int
    reason: str


class ScalingPolicy:
    """Conservative role-local policy for bounded event-driven GPU scaling."""

    def __init__(
        self,
        *,
        minimum_replicas: int,
        maximum_replicas: int,
        sessions_per_replica: int,
        idle_observations_before_scale_down: int,
    ) -> None:
        if not 0 <= minimum_replicas <= maximum_replicas <= 8:
            raise ValueError("GPU replica bounds must satisfy 0 <= min <= max <= 8")
        if sessions_per_replica < 1:
            raise ValueError("sessions_per_replica must be positive")
        if idle_observations_before_scale_down < 1:
            raise ValueError("idle scale-down window must be positive")
        self.minimum_replicas = minimum_replicas
        self.maximum_replicas = maximum_replicas
        self.sessions_per_replica = sessions_per_replica
        self.idle_observations_before_scale_down = idle_observations_before_scale_down
        self._idle_observations: dict[str, int] = {}

    def recommend(
        self,
        role: str,
        current_replicas: int,
        snapshot: CapacityRoleSnapshot,
    ) -> ScaleDecision:
        if not 0 <= current_replicas <= self.maximum_replicas:
            raise ValueError("current replicas are outside the configured bounds")

        if snapshot.waiting_sessions > 0:
            self._idle_observations[role] = 0
            demand = (
                snapshot.active_sessions
                + snapshot.queued_sessions
                + snapshot.waiting_sessions
            )
            required = math.ceil(demand / self.sessions_per_replica)
            target = min(
                self.maximum_replicas,
                max(current_replicas + 1, required, self.minimum_replicas),
            )
            return ScaleDecision(target, "waiting-capacity")

        if snapshot.active_sessions > 0 or snapshot.queued_sessions > 0:
            self._idle_observations[role] = 0
            return ScaleDecision(current_replicas, "sessions-active")

        if snapshot.draining_workers > 0:
            self._idle_observations[role] = 0
            return ScaleDecision(current_replicas, "worker-draining")

        observations = self._idle_observations.get(role, 0) + 1
        self._idle_observations[role] = observations
        if (
            current_replicas > self.minimum_replicas
            and observations >= self.idle_observations_before_scale_down
        ):
            self._idle_observations[role] = 0
            return ScaleDecision(current_replicas - 1, "sustained-idle")
        return ScaleDecision(current_replicas, "idle-observation-window")


class CapacityScaler:
    ROLE_WORKLOADS = {
        "denoiser": "minwm-async-denoiser",
        "vae": "minwm-async-vae",
    }

    def __init__(self, *, kubernetes, policy_by_role: dict[str, ScalingPolicy]):
        self.kubernetes = kubernetes
        self.policy_by_role = policy_by_role

    def reconcile(
        self, snapshots: dict[str, CapacityRoleSnapshot]
    ) -> dict[str, ScaleDecision]:
        decisions: dict[str, ScaleDecision] = {}
        for role, workload in self.ROLE_WORKLOADS.items():
            snapshot = snapshots[role]
            current = self.kubernetes.get_scale(workload)
            decision = self.policy_by_role[role].recommend(role, current, snapshot)
            decisions[role] = decision
            if decision.target_replicas != current:
                self.kubernetes.scale(workload, decision.target_replicas)
        return decisions


class CoordinatorCapacityClient:
    def __init__(self, base_url: str, *, opener=urlopen) -> None:
        self.url = f"{base_url.rstrip('/')}/v1/capacity"
        self.opener = opener

    def fetch(self) -> dict[str, CapacityRoleSnapshot]:
        request = Request(self.url, method="GET")
        with self.opener(request, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Coordinator capacity request returned HTTP {response.status}"
                )
            payload = json.loads(response.read())
        roles = payload.get("roles")
        if not isinstance(roles, dict):
            raise RuntimeError("Coordinator capacity response is missing roles")
        return {
            role: CapacityRoleSnapshot(
                waiting_sessions=int(values["waiting_sessions"]),
                active_sessions=int(values["active_sessions"]),
                queued_sessions=int(values["queued_sessions"]),
                free_slots=int(values["free_slots"]),
                draining_workers=int(values["draining_workers"]),
            )
            for role, values in roles.items()
        }


class KubernetesScaleClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        namespace: str,
        token_path: Path = SERVICE_ACCOUNT_ROOT / "token",
        ca_path: Path = SERVICE_ACCOUNT_ROOT / "ca.crt",
        opener=urlopen,
        ssl_context=None,
    ) -> None:
        self.base_url = f"https://{host}:{port}"
        self.namespace = namespace
        self.token = token_path.read_text(encoding="utf-8").strip()
        self.opener = opener
        self.ssl_context = ssl_context or ssl.create_default_context(cafile=str(ca_path))

    def _scale_url(self, workload: str) -> str:
        resource = WORKLOAD_RESOURCES.get(workload)
        if resource is None:
            raise ValueError("workload is outside the production GPU worker allowlist")
        return (
            f"{self.base_url}/apis/apps/v1/namespaces/{self.namespace}/"
            f"{resource}/{workload}/scale"
        )

    def get_scale(self, workload: str) -> int:
        request = Request(
            self._scale_url(workload),
            method="GET",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        with self.opener(request, context=self.ssl_context, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Kubernetes scale request returned HTTP {response.status}"
                )
            payload = json.loads(response.read())
        return int(payload["spec"]["replicas"])

    def scale(self, workload: str, replicas: int) -> None:
        if replicas < 0 or replicas > 8:
            raise ValueError("GPU replicas must be between 0 and 8")
        request = Request(
            self._scale_url(workload),
            data=json.dumps({"spec": {"replicas": replicas}}).encode("utf-8"),
            method="PATCH",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/merge-patch+json",
            },
        )
        with self.opener(request, context=self.ssl_context, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Kubernetes scale request returned HTTP {response.status}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--denoiser-replicas", type=int)
    parser.add_argument("--vae-replicas", type=int)
    parser.add_argument("--coordinator-url")
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    parser.add_argument("--denoiser-min-replicas", type=int, default=1)
    parser.add_argument("--denoiser-max-replicas", type=int, default=8)
    parser.add_argument("--denoiser-sessions-per-replica", type=int, default=4)
    parser.add_argument("--vae-min-replicas", type=int, default=1)
    parser.add_argument("--vae-max-replicas", type=int, default=8)
    parser.add_argument("--vae-sessions-per-replica", type=int, default=16)
    parser.add_argument("--idle-observations-before-scale-down", type=int, default=24)
    return parser.parse_args()


def run_capacity_controller(
    args: argparse.Namespace,
    kubernetes: KubernetesScaleClient,
) -> None:
    capacity = CoordinatorCapacityClient(args.coordinator_url)
    scaler = CapacityScaler(
        kubernetes=kubernetes,
        policy_by_role={
            "denoiser": ScalingPolicy(
                minimum_replicas=args.denoiser_min_replicas,
                maximum_replicas=args.denoiser_max_replicas,
                sessions_per_replica=args.denoiser_sessions_per_replica,
                idle_observations_before_scale_down=(
                    args.idle_observations_before_scale_down
                ),
            ),
            "vae": ScalingPolicy(
                minimum_replicas=args.vae_min_replicas,
                maximum_replicas=args.vae_max_replicas,
                sessions_per_replica=args.vae_sessions_per_replica,
                idle_observations_before_scale_down=(
                    args.idle_observations_before_scale_down
                ),
            ),
        },
    )
    while True:
        try:
            decisions = scaler.reconcile(capacity.fetch())
            logger.info(
                "GPU capacity reconciliation: %s",
                {
                    role: {
                        "target_replicas": decision.target_replicas,
                        "reason": decision.reason,
                    }
                    for role, decision in decisions.items()
                },
            )
        except Exception:
            logger.exception("GPU capacity reconciliation failed")
        time.sleep(args.poll_interval_s)


def main() -> None:
    args = parse_args()
    client = KubernetesScaleClient(
        host=os.environ["KUBERNETES_SERVICE_HOST"],
        port=int(os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")),
        namespace=os.environ.get("POD_NAMESPACE", "minwm-realtime"),
    )
    if args.coordinator_url:
        if args.poll_interval_s <= 0:
            raise ValueError("poll interval must be positive")
        run_capacity_controller(args, client)
        return
    if args.denoiser_replicas is None or args.vae_replicas is None:
        raise ValueError(
            "static scaling requires --denoiser-replicas and --vae-replicas"
        )
    client.scale("minwm-async-vae", args.vae_replicas)
    client.scale("minwm-async-denoiser", args.denoiser_replicas)


if __name__ == "__main__":
    main()
