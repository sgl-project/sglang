import pytest
import requests

from .conftest import _spawn_mock_worker


@pytest.mark.integration
def test_discovery_shim_add_remove(router_manager):
    # Start router without workers
    rh = router_manager.start_router(worker_urls=[], policy="round_robin")

    # Initially empty
    urls = router_manager.list_workers(rh.url)
    assert urls == []

    # Add a worker (simulate discovery event)
    proc, wurl, wid = _spawn_mock_worker([])
    try:
        router_manager.add_worker(rh.url, wurl)
        urls = router_manager.list_workers(rh.url)
        assert wurl in urls

        # Can serve a request
        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": "hi",
                "max_tokens": 1,
                "stream": False,
            },
        )
        assert r.status_code == 200

        # Remove worker (simulate pod deletion)
        router_manager.remove_worker(rh.url, wurl)
        urls = router_manager.list_workers(rh.url)
        assert wurl not in urls
    finally:
        import subprocess

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
