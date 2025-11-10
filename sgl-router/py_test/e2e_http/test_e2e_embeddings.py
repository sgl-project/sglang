import time

import pytest
import requests


def _wait_for_workers(
    base_url: str, expected_count: int, timeout: float = 60.0, headers: dict = None
) -> None:
    """Poll /workers endpoint until expected number of workers are registered."""
    start = time.perf_counter()
    with requests.Session() as session:
        while time.perf_counter() - start < timeout:
            try:
                r = session.get(f"{base_url}/workers", headers=headers, timeout=5)
                if r.status_code == 200:
                    workers = r.json().get("workers", [])
                    if len(workers) >= expected_count:
                        return
            except requests.RequestException:
                pass
            time.sleep(0.5)
    raise TimeoutError(
        f"Expected {expected_count} workers at {base_url}, timed out after {timeout}s"
    )


@pytest.mark.e2e
def test_embeddings_basic(
    e2e_router_only_rr, e2e_primary_embedding_worker, e2e_embedding_model
):
    base = e2e_router_only_rr.url
    worker_url = e2e_primary_embedding_worker.url

    # Attach embedding worker to router-only instance
    r = requests.post(f"{base}/workers", json={"url": worker_url}, timeout=180)
    assert r.status_code == 202, f"Expected 202 ACCEPTED, got {r.status_code}: {r.text}"

    # Wait for worker to be registered
    _wait_for_workers(base, expected_count=1, timeout=60.0)

    # Simple embedding request with two inputs
    payload = {
        "model": e2e_embedding_model,
        "input": [
            "the quick brown fox",
            "jumps over the lazy dog",
        ],
    }
    r = requests.post(f"{base}/v1/embeddings", json=payload, timeout=120)

    assert r.status_code == 200, f"unexpected status: {r.status_code} {r.text}"

    data = r.json()
    assert "data" in data and isinstance(data["data"], list)
    assert len(data["data"]) == 2

    # Validate shape of embedding objects
    for item in data["data"]:
        assert "embedding" in item and isinstance(item["embedding"], list)
        # Ensure non-empty vectors
        assert len(item["embedding"]) > 0
