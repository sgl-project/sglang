import asyncio

import pytest
from sglang_router import mini_lb


@pytest.mark.parametrize(
    ("result", "restored"),
    [
        (
            {
                "success": False,
                "message": "instance is not idle",
                "safe_to_restore": True,
            },
            True,
        ),
        (
            {
                "success": False,
                "message": "instance unhealthy, restart required",
                "safe_to_restore": False,
            },
            False,
        ),
        ({"success": False, "message": "connection lost"}, False),
    ],
)
def test_failed_role_switch_restores_only_healthy_worker(monkeypatch, result, restored):
    worker_url = "http://prefill:8000"
    load_balancer = mini_lb.MiniLoadBalancer.__new__(mini_lb.MiniLoadBalancer)
    load_balancer.timeout = 1
    load_balancer.prefill_urls = [worker_url]
    load_balancer.prefill_bootstrap_ports = [8998]
    load_balancer.decode_urls = ["http://decode:8000"]
    monkeypatch.setattr(mini_lb, "lb", load_balancer)

    async def post_role_switch(*_args, **_kwargs):
        return 400, result

    monkeypatch.setattr(mini_lb, "_post_role_switch", post_role_switch)
    response = asyncio.run(
        mini_lb.pd_role_switch(
            {
                "worker_url": worker_url,
                "new_role": "decode",
                "drain_timeout_secs": 0,
            }
        )
    )

    assert response.status_code == 400
    assert (worker_url in load_balancer.prefill_urls) is restored
