# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

from sglang.srt.managers.tokenizer_manager import TokenizerManager


def lifecycle_manager() -> TokenizerManager:
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.init_running_status()
    return manager


def test_prefill_completion_wakes_waiter_and_is_idempotent():
    async def run():
        manager = lifecycle_manager()
        waiter = asyncio.create_task(manager.wait_for_prefill_completed("request-1"))
        await asyncio.sleep(0)

        first = manager.mark_prefill_completed("request-1")
        assert await waiter == first
        assert manager.mark_prefill_completed("request-1") == first
        assert await manager.wait_for_prefill_completed("request-1") == first

    asyncio.run(run())


def test_cancelled_waiter_is_removed():
    async def run():
        manager = lifecycle_manager()
        waiter = asyncio.create_task(manager.wait_for_prefill_completed("request-2"))
        await asyncio.sleep(0)
        waiter.cancel()

        try:
            await waiter
        except asyncio.CancelledError:
            pass

        assert "request-2" not in manager._prefill_completion_waiters

    asyncio.run(run())
