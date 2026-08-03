#!/usr/bin/env python3
"""Policy checks for the disposable MinWM async-VAE benchmark topology."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
BASE_MANIFESTS = (
    "h100-denoiser.yaml",
    "l4-vae.yaml",
    "gateway-service.yaml",
)


def load_documents(paths: tuple[str, ...] = BASE_MANIFESTS) -> list[dict]:
    documents: list[dict] = []
    for relative_path in paths:
        path = ROOT / relative_path
        with path.open() as stream:
            documents.extend(
                document
                for document in yaml.safe_load_all(stream)
                if isinstance(document, dict)
            )
    return documents


def find(documents: list[dict], kind: str, name: str) -> dict:
    for document in documents:
        if document.get("kind") == kind and document.get("metadata", {}).get("name") == name:
            return document
    raise AssertionError(f"missing {kind}/{name}")


def requirement_values(nodepool: dict, key: str) -> list[str]:
    requirements = nodepool["spec"]["template"]["spec"]["requirements"]
    for requirement in requirements:
        if requirement.get("key") == key:
            return list(requirement.get("values") or [])
    raise AssertionError(f"missing NodePool requirement {key}")


def validate(documents: list[dict]) -> None:
    denoiser = find(documents, "NodePool", "minwm-async-denoiser-h100")
    vae = find(documents, "NodePool", "minwm-async-vae-l4")
    assert requirement_values(denoiser, "karpenter.sh/capacity-type") == ["spot"]
    assert requirement_values(vae, "karpenter.sh/capacity-type") == ["spot"]
    assert requirement_values(denoiser, "node.kubernetes.io/instance-type") == [
        "p5.4xlarge"
    ]
    assert all(value.startswith("g6.") for value in requirement_values(
        vae, "node.kubernetes.io/instance-type"
    ))
    assert denoiser["spec"]["limits"]["nvidia.com/gpu"] == "1"
    assert vae["spec"]["limits"]["nvidia.com/gpu"] == "1"

    for deployment_name in ("minwm-async-denoiser", "minwm-async-vae"):
        deployment = find(documents, "Deployment", deployment_name)
        labels = deployment["metadata"]["labels"]
        assert labels["seedleap.ai/test-run"] == "minwm-async-vae-benchmark"
        assert labels["seedleap.ai/ttl-after-test"] == "required"
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        resources = container["resources"]
        assert resources.get("requests")
        assert resources.get("limits")
        assert resources["requests"]["nvidia.com/gpu"] == "1"
        assert resources["limits"]["nvidia.com/gpu"] == "1"

    denoiser_deployment = find(documents, "Deployment", "minwm-async-denoiser")
    env = {
        item["name"]: item.get("value")
        for item in denoiser_deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert env["REALTIME_VAE_WORKER_URL"].startswith("ws://minwm-async-vae")
    assert "REALTIME_SESSION_LEASE_TABLE" not in env


def main() -> None:
    validate(load_documents())
    print("MinWM async-VAE manifests satisfy Spot, quota, and cleanup policies.")


if __name__ == "__main__":
    main()
