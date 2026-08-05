# SPDX-License-Identifier: Apache-2.0

"""Bounded Kubernetes GPU workload scaler used by the production CronJobs."""

from __future__ import annotations

import argparse
import json
import os
import ssl
from pathlib import Path
from urllib.request import Request, urlopen


SERVICE_ACCOUNT_ROOT = Path("/var/run/secrets/kubernetes.io/serviceaccount")
WORKLOAD_RESOURCES = {
    "minwm-async-denoiser": "statefulsets",
    "minwm-async-vae": "deployments",
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

    def scale(self, workload: str, replicas: int) -> None:
        resource = WORKLOAD_RESOURCES.get(workload)
        if resource is None:
            raise ValueError("workload is outside the production GPU worker allowlist")
        if replicas < 0 or replicas > 8:
            raise ValueError("GPU replicas must be between 0 and 8")
        url = (
            f"{self.base_url}/apis/apps/v1/namespaces/{self.namespace}/"
            f"{resource}/{workload}/scale"
        )
        request = Request(
            url,
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
    parser.add_argument("--denoiser-replicas", type=int, required=True)
    parser.add_argument("--vae-replicas", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = KubernetesScaleClient(
        host=os.environ["KUBERNETES_SERVICE_HOST"],
        port=int(os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")),
        namespace=os.environ.get("POD_NAMESPACE", "minwm-realtime"),
    )
    client.scale("minwm-async-vae", args.vae_replicas)
    client.scale("minwm-async-denoiser", args.denoiser_replicas)


if __name__ == "__main__":
    main()
