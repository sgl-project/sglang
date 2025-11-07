import pytest
import requests


@pytest.mark.integration
def test_add_and_remove_worker(mock_worker, router_manager, mock_workers):
    # Start with a single worker
    proc1, url1, id1 = mock_worker
    rh = router_manager.start_router(worker_urls=[url1], policy="round_robin")

    # Add a second worker

    procs2, urls2, ids2 = mock_workers(n=1)
    url2 = urls2[0]
    id2 = ids2[0]
    router_manager.add_worker(rh.url, url2)

    # Send some requests and ensure both workers are seen
    seen = set()
    with requests.Session() as s:
        for i in range(20):
            r = s.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"x{i}",
                    "max_tokens": 1,
                    "stream": False,
                },
            )
            assert r.status_code == 200
            wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
            seen.add(wid)
            if len(seen) == 2:
                break

    assert id1 in seen and id2 in seen

    # Now remove the second worker
    router_manager.remove_worker(rh.url, url2)

    # After removal, subsequent requests should only come from first worker
    with requests.Session() as s:
        for i in range(10):
            r = s.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"y{i}",
                    "max_tokens": 1,
                    "stream": False,
                },
            )
            assert r.status_code == 200
            wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
            assert wid == id1
    # mock_workers fixture handles cleanup
