from types import SimpleNamespace

import pytest
import requests


@pytest.mark.e2e
def test_embeddings_basic(
    e2e_router_only_rr, e2e_primary_embedding_worker, e2e_embedding_model
):
    base = e2e_router_only_rr.url
    worker_url = e2e_primary_embedding_worker.url

    # Attach embedding worker to router-only instance
    r = requests.post(f"{base}/add_worker", params={"url": worker_url}, timeout=180)
    r.raise_for_status()

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
