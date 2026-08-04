# SPDX-License-Identifier: Apache-2.0

"""Health-gated worker registration for the realtime Coordinator."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlsplit

import httpx


logger = logging.getLogger(__name__)
WorkerRole = Literal["denoiser", "vae"]


async def discover_kubernetes_node_az(
    client: Any,
    *,
    api_url: str,
    node_name: str,
    token: str,
) -> str:
    response = await client.get(
        f"{api_url.rstrip('/')}/api/v1/nodes/{quote(node_name, safe='')}",
        headers={"Authorization": f"Bearer {token}"},
    )
    response.raise_for_status()
    try:
        zone = response.json()["metadata"]["labels"]["topology.kubernetes.io/zone"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Kubernetes Node is missing its availability-zone label") from exc
    if not isinstance(zone, str) or not zone:
        raise RuntimeError("Kubernetes Node availability zone is invalid")
    return zone


class WorkerHeartbeatReporter:
    def __init__(
        self,
        client: Any,
        *,
        coordinator_url: str,
        health_url: str,
        worker_id: str,
        role: WorkerRole,
        endpoint: str,
        az: str,
        capacity: int,
        model_revision: str,
        vae_fingerprint: str,
    ) -> None:
        endpoint_parts = urlsplit(endpoint)
        if endpoint_parts.scheme not in ("ws", "wss") or not endpoint_parts.netloc:
            raise ValueError("worker endpoint must be a WebSocket endpoint")
        if role not in ("denoiser", "vae"):
            raise ValueError("role must be denoiser or vae")
        if not worker_id or not az or capacity < 1:
            raise ValueError("worker identity, AZ, and positive capacity are required")
        self.client = client
        self.coordinator_url = coordinator_url.rstrip("/")
        self.health_url = health_url
        self.payload = {
            "worker_id": worker_id,
            "role": role,
            "endpoint": endpoint,
            "az": az,
            "capacity": capacity,
            "model_revision": model_revision,
            "vae_fingerprint": vae_fingerprint,
        }

    async def heartbeat_once(self) -> bool:
        health = await self.client.get(self.health_url)
        if not health.is_success:
            return False
        response = await self.client.post(
            f"{self.coordinator_url}/v1/workers/heartbeat",
            json=self.payload,
        )
        response.raise_for_status()
        return True

    async def run(self, *, interval_s: float) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        while True:
            try:
                registered = await self.heartbeat_once()
                if not registered:
                    logger.warning("worker health check is not ready")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("worker heartbeat failed")
            await asyncio.sleep(interval_s)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--health-url", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--role", choices=("denoiser", "vae"), required=True)
    parser.add_argument("--endpoint", required=True)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--az")
    location.add_argument("--node-name")
    parser.add_argument(
        "--kubernetes-api-url", default="https://kubernetes.default.svc"
    )
    parser.add_argument(
        "--service-account-token-file",
        default="/var/run/secrets/kubernetes.io/serviceaccount/token",
    )
    parser.add_argument(
        "--service-account-ca-file",
        default="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
    )
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--vae-fingerprint", required=True)
    parser.add_argument("--interval-s", type=float, default=5.0)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    timeout = httpx.Timeout(3.0, connect=1.0)
    verify = args.service_account_ca_file if args.node_name else True
    async with httpx.AsyncClient(timeout=timeout, verify=verify) as client:
        az = args.az
        if args.node_name:
            token = Path(args.service_account_token_file).read_text().strip()
            az = await discover_kubernetes_node_az(
                client,
                api_url=args.kubernetes_api_url,
                node_name=args.node_name,
                token=token,
            )
        reporter = WorkerHeartbeatReporter(
            client,
            coordinator_url=args.coordinator_url,
            health_url=args.health_url,
            worker_id=args.worker_id,
            role=args.role,
            endpoint=args.endpoint,
            az=az,
            capacity=args.capacity,
            model_revision=args.model_revision,
            vae_fingerprint=args.vae_fingerprint,
        )
        await reporter.run(interval_s=args.interval_s)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
