import pytest
import requests


@pytest.mark.integration
def test_discovery_shim_add_remove(router_manager, mock_workers):
    # Start router without workers
    rh = router_manager.start_router(worker_urls=[], policy="round_robin")

    # Initially empty
    urls = router_manager.list_workers(rh.url)
    assert urls == []

    # Add a worker (simulate discovery event)
    _, [wurl], [wid] = mock_workers(n=1)
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
    # mock_workers fixture handles cleanup
