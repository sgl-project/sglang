import asyncio

from sglang.multimodal_gen.runtime.entrypoints.openai import mesh_api


def test_mesh_job_registry_holds_task_until_completion():
    async def run_test():
        started = asyncio.Event()
        finish = asyncio.Event()

        async def job():
            started.set()
            await finish.wait()

        task = mesh_api._start_mesh_job("job-id", job())
        await started.wait()
        assert mesh_api._MESH_JOB_TASKS["job-id"] is task

        finish.set()
        await task
        await asyncio.sleep(0)
        assert "job-id" not in mesh_api._MESH_JOB_TASKS

    asyncio.run(run_test())


def test_shutdown_mesh_jobs_cancels_and_awaits_cleanup():
    async def run_test():
        started = asyncio.Event()
        cleaned_up = asyncio.Event()

        async def job():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned_up.set()

        task = mesh_api._start_mesh_job("job-id", job())
        await started.wait()
        await mesh_api.shutdown_mesh_jobs()

        assert task.cancelled()
        assert cleaned_up.is_set()
        assert "job-id" not in mesh_api._MESH_JOB_TASKS

    asyncio.run(run_test())
