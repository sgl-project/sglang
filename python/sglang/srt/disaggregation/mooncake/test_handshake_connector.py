import asyncio
import json
import os
import sys
import threading
import time
from typing import Dict, Union

from aiohttp import web

# from .conn import MooncakeKVReceiver, AsyncHttpConnector
# from ..base import KVPoll

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, "../../../../.."))
sys.path.insert(0, os.path.join(root_path, "python"))

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll

# 改为绝对导入
from sglang.srt.disaggregation.mooncake.conn import (
    AsyncHttpConnector,
    MooncakeKVReceiver,
)


class MockKVManager:
    def __init__(self):
        self.request_status: Dict[int, int] = {}
        self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
        self.session_counter = 0

        self.kv_args = KVArgs()
        self.kv_args.engine_rank = 0

        self.async_http_connector = AsyncHttpConnector(self)

    def get_session_id(self):
        self.session_counter += 1
        return self.session_counter

    def update_status(self, room: int, status: int):
        self.request_status[room] = status

    def check_status(self, bootstrap_room: int):
        return self.request_status[bootstrap_room]


def run_mock_server():
    async def handle(request):
        await asyncio.sleep(1)  # simulate slow response
        engine_rank = request.rel_url.query.get("engine_rank", "0")
        return web.Response(
            text=json.dumps(
                {
                    "rank_ip": "127.0.0.1",
                    "rank_port": 6000 + int(engine_rank),
                }
            ),
            content_type="application/json",
        )

    app = web.Application()
    app.router.add_get("/route", handle)
    runner = web.AppRunner(app)

    async def start():
        await runner.setup()
        site = web.TCPSite(runner, "localhost", 8083)
        await site.start()
        while True:
            await asyncio.sleep(3600)

    threading.Thread(target=lambda: asyncio.run(start()), daemon=True).start()


def test_long_connection():
    run_mock_server()
    time.sleep(1)  # wait for server to be ready

    mgr = MockKVManager()

    start_time = time.time()
    receivers = [
        MooncakeKVReceiver(mgr, "localhost:8083", bootstrap_room=i) for i in range(5)
    ]

    for _ in range(20):
        time.sleep(0.5)
        ready = [r for r in receivers if r.poll() == KVPoll.WaitingForInput]
        if len(ready) == len(receivers):
            break

    duration = time.time() - start_time
    print(f"All {len(receivers)} connections ready in {duration:.2f}s")
    for i, r in enumerate(receivers):
        print(f"Receiver {i}: Poll={r.poll()}, Bootstrap Info={r.bootstrap_info}")


if __name__ == "__main__":
    test_long_connection()
