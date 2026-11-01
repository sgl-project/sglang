"""Cross-process behavior of the optional CPU preprocess artifact store."""

import asyncio
import errno
import multiprocessing
import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

try:
    from sglang.srt.multimodal.cache.shared_preprocess_cache import (
        SharedCacheRequest,
        SharedPreprocessCache,
    )
except ModuleNotFoundError as error:
    if error.name != "sglang.srt.multimodal.cache.shared_preprocess_cache":
        raise
    SharedCacheRequest = SharedPreprocessCache = None


@dataclass(frozen=True)
class ImageArtifact:
    feature: torch.Tensor
    grid: tuple[int, int, int]
    metadata: dict


def artifact(number):
    return ImageArtifact(
        feature=torch.tensor([[number, 2, 3]], dtype=torch.float32),
        grid=(1, 1, 1),
        metadata={"shape": torch.tensor([1, 3]), "label": "image"},
    )


def concurrent_reader(directory, barrier, output):
    async def run():
        cache = SharedPreprocessCache(directory, max_size_bytes=65536)
        barrier.wait(timeout=30)

        async def compute(indices):
            with open(Path(directory) / "computations", "ab", buffering=0) as stream:
                stream.write(b"computed\n")
            await asyncio.sleep(0.2)
            return [artifact(7) for _ in indices]

        result = await cache.resolve_many(
            [SharedCacheRequest("same-key", b"image bytes", b"configuration")],
            compute,
            artifact_type=ImageArtifact,
        )
        output.put(result[0].feature.tolist())

    asyncio.run(run())


def interrupted_writer(directory, entered):
    async def run():
        cache = SharedPreprocessCache(directory, max_size_bytes=65536)

        async def compute(indices):
            entered.set()
            await asyncio.sleep(60)
            return [artifact(7)]

        await cache.resolve_many(
            [SharedCacheRequest("same-key", b"image bytes", b"configuration")],
            compute,
            artifact_type=ImageArtifact,
        )

    asyncio.run(run())


def many_async_readers(directory, entered):
    async def run():
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=2)
        )
        cache = SharedPreprocessCache(directory, max_size_bytes=65536)
        request = SharedCacheRequest("same-key", b"image bytes", b"configuration")

        async def compute(indices):
            entered.set()
            await asyncio.sleep(0.2)
            return [artifact(7)]

        result = await asyncio.gather(
            *[
                cache.resolve_many([request], compute, artifact_type=ImageArtifact)
                for _ in range(8)
            ]
        )
        assert all(item[0].feature.tolist() == [[7, 2, 3]] for item in result)

    asyncio.run(run())


def interrupted_serializer(directory, entered):
    def write_partial(path, *args, **kwargs):
        Path(path).write_bytes(b"partial tensor")
        entered.set()
        time.sleep(60)

    async def run():
        cache = SharedPreprocessCache(directory, max_size_bytes=65536)

        async def compute(indices):
            return [artifact(7)]

        with patch(
            "sglang.srt.multimodal.cache.shared_preprocess_cache.np.save",
            side_effect=write_partial,
        ):
            await cache.resolve_many(
                [SharedCacheRequest("same-key", b"image bytes", b"configuration")],
                compute,
                artifact_type=ImageArtifact,
            )

    asyncio.run(run())


@unittest.skipUnless(os.name == "posix", "Shared file mappings require POSIX")
class TestSharedPreprocessCache(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.assertIsNotNone(
            SharedPreprocessCache,
            "The cross-worker preprocess cache backend is not implemented",
        )
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = self.temporary.name
        self.cache = SharedPreprocessCache(self.directory, max_size_bytes=65536)
        self.calls = []

    async def resolve(self, requests, *, cache=None, number=1):
        async def compute(indices):
            self.calls.extend(indices)
            return [artifact(number) for _ in indices]

        return await (cache or self.cache).resolve_many(
            requests, compute, artifact_type=ImageArtifact
        )

    async def test_clear_while_waiting_for_lock_does_not_repopulate(self):
        import threading

        request = SharedCacheRequest("queued", b"image", b"configuration")
        lookup_done = threading.Event()
        lookup = self.cache._lookup

        def observed_lookup(*args):
            result = lookup(*args)
            lookup_done.set()
            return result

        locks = self.cache._shard_locks([request])
        locks.__enter__()
        try:
            with patch.object(self.cache, "_lookup", side_effect=observed_lookup):
                task = asyncio.create_task(self.resolve([request]))
                self.assertTrue(await asyncio.to_thread(lookup_done.wait, 5))
                self.cache.clear()
        finally:
            locks.__exit__(None, None, None)
        result = await asyncio.wait_for(task, 5)
        self.assertEqual(result[0].feature.tolist(), [[1, 2, 3]])
        self.assertEqual(self.cache.cache_info()["entries"], 0)

    async def test_damaged_tensor_is_recomputed(self):
        request = SharedCacheRequest("key", b"image", b"configuration")
        await self.resolve([request])
        tensor_file = next(Path(self.directory).glob("object-*/tensor-0.npy"))
        tensor_file.write_bytes(b"interrupted or damaged file")
        result = await self.resolve([request], number=9)
        self.assertEqual(result[0].feature.tolist(), [[9, 2, 3]])
        self.assertEqual(self.cache.cache_info()["entries"], 1)

    async def test_truncated_tensor_is_recomputed(self):
        request = SharedCacheRequest("key", b"image", b"configuration")
        await self.resolve([request])
        next(Path(self.directory).glob("object-*/tensor-0.npy")).write_bytes(b"")
        result = await self.resolve([request], number=9)
        self.assertEqual(result[0].feature.tolist(), [[9, 2, 3]])
        self.assertEqual(self.cache.cache_info()["entries"], 1)

    async def test_scalar_empty_and_bfloat16_tensors_roundtrip(self):
        request = SharedCacheRequest("special", b"image", b"configuration")
        value = ImageArtifact(
            torch.empty((0, 3), dtype=torch.bfloat16),
            (1, 1, 1),
            {"scalar": torch.tensor(7, dtype=torch.bfloat16)},
        )

        async def compute(indices):
            return [value for _ in indices]

        await self.cache.resolve_many([request], compute, artifact_type=ImageArtifact)

        async def unexpected(indices):
            self.fail("Expected a cached empty tensor")

        result = (
            await self.cache.resolve_many(
                [request], unexpected, artifact_type=ImageArtifact
            )
        )[0]
        self.assertTrue(torch.equal(result.feature, value.feature))
        self.assertTrue(
            torch.equal(result.metadata["scalar"], value.metadata["scalar"])
        )

    async def test_another_instance_reuses_the_same_artifact(self):
        request = SharedCacheRequest("key", b"encoded-image", b"resize=32")
        first = (await self.resolve([request]))[0]
        other = SharedPreprocessCache(self.directory, max_size_bytes=65536)
        second = (await self.resolve([request], cache=other, number=9))[0]
        self.assertEqual(self.calls, [0])
        self.assertTrue(torch.equal(first.feature, second.feature))
        self.assertTrue(torch.equal(first.metadata["shape"], second.metadata["shape"]))
        self.assertEqual(second.grid, (1, 1, 1))

    async def test_colliding_keys_compare_source_bytes_and_configuration(self):
        first = SharedCacheRequest("forced-collision", b"one", b"resize=32")
        second = SharedCacheRequest("forced-collision", b"two", b"resize=32")
        third = SharedCacheRequest("forced-collision", b"one", b"resize=64")
        await self.resolve([first], number=1)
        await self.resolve([second], number=2)
        await self.resolve([third], number=3)
        for request, want in [(first, 1), (second, 2), (third, 3)]:
            result = (await self.resolve([request], number=99))[0]
            self.assertEqual(result.feature.tolist(), [[want, 2, 3]])
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.cache.cache_info()["entries"], 3)

    async def test_duplicates_and_partial_hits_preserve_input_order(self):
        a = SharedCacheRequest("a", b"a", b"config")
        b = SharedCacheRequest("b", b"b", b"config")
        await self.resolve([a], number=1)
        result = await self.resolve([b, a, b], number=2)
        self.assertEqual([item.feature[0, 0].item() for item in result], [2, 1, 2])
        self.assertEqual(self.calls, [0, 0])

    async def test_returned_tensors_cannot_modify_another_reader(self):
        request = SharedCacheRequest("key", b"image", b"config")
        await self.resolve([request])
        result = (await self.resolve([request]))[0]
        result.feature[0, 0] = 99
        result.metadata["shape"][0] = 99
        again = (await self.resolve([request]))[0]
        self.assertEqual(again.feature.tolist(), [[1, 2, 3]])
        self.assertEqual(again.metadata["shape"].tolist(), [1, 3])

    async def test_lru_is_global_across_instances(self):
        directory = str(Path(self.directory) / "lru")
        self.cache = SharedPreprocessCache(
            directory, max_size_bytes=65536, max_entries=2
        )
        other = SharedPreprocessCache(directory, max_size_bytes=65536, max_entries=2)
        a, b, c = [SharedCacheRequest(key, key.encode(), b"config") for key in "abc"]
        await self.resolve([a])
        await self.resolve([b])
        await self.resolve([a], cache=other)
        await self.resolve([c], cache=other)
        await self.resolve([a])
        self.assertEqual(len(self.calls), 3)
        await self.resolve([b])
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(other.cache_info()["entries"], 2)

    async def test_oversized_artifact_is_returned_without_retention(self):
        cache = SharedPreprocessCache(
            str(Path(self.directory) / "small"), max_size_bytes=8
        )
        request = SharedCacheRequest("key", b"image", b"config")
        for _ in range(2):
            result = (await self.resolve([request], cache=cache))[0]
            self.assertEqual(result.feature.tolist(), [[1, 2, 3]])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(cache.cache_info()["entries"], 0)

    async def test_retained_bytes_are_bounded_across_instances(self):
        directory = str(Path(self.directory) / "byte-budget")
        cache = SharedPreprocessCache(directory, max_size_bytes=20000)
        other = SharedPreprocessCache(directory, max_size_bytes=20000)
        calls = []

        async def compute(indices):
            calls.extend(indices)
            return [
                ImageArtifact(torch.ones(2048, dtype=torch.float32), (1, 1, 1), {})
                for _ in indices
            ]

        requests = [SharedCacheRequest(key, key.encode(), b"config") for key in "abc"]
        for request in requests:
            await cache.resolve_many([request], compute, artifact_type=ImageArtifact)
        self.assertEqual(other.cache_info()["entries"], 2)
        self.assertLessEqual(other.cache_info()["size_bytes"], 20000)
        await other.resolve_many([requests[1]], compute, artifact_type=ImageArtifact)
        self.assertEqual(len(calls), 3)
        await other.resolve_many([requests[0]], compute, artifact_type=ImageArtifact)
        self.assertEqual(len(calls), 4)

    async def test_tensor_view_does_not_store_the_entire_batch(self):
        cache = SharedPreprocessCache(
            str(Path(self.directory) / "views"), max_size_bytes=4096
        )
        request = SharedCacheRequest("key", b"image", b"config")

        async def compute(indices):
            self.calls.extend(indices)
            storage = torch.arange(1_000_000, dtype=torch.float32)
            return [ImageArtifact(storage[1:4], (1, 1, 1), {})]

        for _ in range(2):
            result = await cache.resolve_many(
                [request], compute, artifact_type=ImageArtifact
            )
            self.assertEqual(result[0].feature.tolist(), [1, 2, 3])
        self.assertEqual(self.calls, [0])
        self.assertEqual(cache.cache_info()["entries"], 1)
        self.assertLessEqual(cache.cache_info()["size_bytes"], 4096)

    async def test_clear_during_compute_does_not_repopulate(self):
        request = SharedCacheRequest("key", b"image", b"config")
        entered, resume = asyncio.Event(), asyncio.Event()

        async def compute(indices):
            entered.set()
            await resume.wait()
            return [artifact(7)]

        task = asyncio.create_task(
            self.cache.resolve_many([request], compute, artifact_type=ImageArtifact)
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        self.cache.clear()
        resume.set()
        self.assertEqual((await task)[0].feature.tolist(), [[7, 2, 3]])
        self.assertEqual(self.cache.cache_info()["entries"], 0)
        await self.resolve([request])
        self.assertEqual(self.calls, [0])

    async def test_eviction_keeps_an_existing_mapping_usable(self):
        request = SharedCacheRequest("key", b"image", b"config")
        await self.resolve([request])
        result = (await self.resolve([request]))[0]
        self.cache.clear()
        self.assertEqual(result.feature.tolist(), [[1, 2, 3]])
        self.assertEqual(self.cache.cache_info()["entries"], 0)

    async def test_failed_compute_releases_the_key(self):
        request = SharedCacheRequest("key", b"image", b"config")

        async def compute(indices):
            raise ValueError("invalid image")

        with self.assertRaisesRegex(ValueError, "invalid image"):
            await self.cache.resolve_many(
                [request], compute, artifact_type=ImageArtifact
            )
        result = (await self.resolve([request]))[0]
        self.assertEqual(result.feature.tolist(), [[1, 2, 3]])

    async def test_full_filesystem_does_not_fail_the_request(self):
        request = SharedCacheRequest("key", b"image", b"config")
        with patch(
            "sglang.srt.multimodal.cache.shared_preprocess_cache.np.save",
            side_effect=OSError(errno.ENOSPC, "No space left on device"),
        ):
            result = (await self.resolve([request]))[0]
        self.assertEqual(result.feature.tolist(), [[1, 2, 3]])
        self.assertEqual(self.cache.cache_info()["entries"], 0)
        self.assertEqual(list(Path(self.directory).glob(".writing-*")), [])
        result = (await self.resolve([request], number=2))[0]
        self.assertEqual(result.feature.tolist(), [[2, 2, 3]])

    def test_two_processes_compute_once(self):
        context = multiprocessing.get_context("spawn")
        barrier, output = context.Barrier(2), context.Queue()
        processes = [
            context.Process(
                target=concurrent_reader, args=(self.directory, barrier, output)
            )
            for _ in range(2)
        ]
        try:
            for process in processes:
                process.start()
            self.assertEqual(output.get(timeout=60), [[7, 2, 3]])
            self.assertEqual(output.get(timeout=60), [[7, 2, 3]])
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(
                (Path(self.directory) / "computations").read_bytes(), b"computed\n"
            )
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            output.close()

    async def test_a_terminated_writer_does_not_leave_a_permanent_lock(self):
        context = multiprocessing.get_context("spawn")
        entered = context.Event()
        process = context.Process(
            target=interrupted_writer, args=(self.directory, entered)
        )
        try:
            process.start()
            self.assertTrue(await asyncio.to_thread(entered.wait, 60))
            process.terminate()
            await asyncio.to_thread(process.join, 5)
            result = await asyncio.wait_for(
                self.resolve(
                    [SharedCacheRequest("same-key", b"image bytes", b"configuration")]
                ),
                timeout=5,
            )
            self.assertEqual(result[0].feature.tolist(), [[1, 2, 3]])
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    async def test_lock_waiters_do_not_starve_the_executor_needed_to_publish(self):
        context = multiprocessing.get_context("spawn")
        entered = context.Event()
        process = context.Process(
            target=many_async_readers, args=(self.directory, entered)
        )
        try:
            process.start()
            self.assertTrue(await asyncio.to_thread(entered.wait, 60))
            await asyncio.to_thread(process.join, 5)
            self.assertFalse(
                process.is_alive(),
                "Lock waiters exhausted the executor needed to publish the result",
            )
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    async def test_restart_removes_files_left_by_an_interrupted_publication(self):
        context = multiprocessing.get_context("spawn")
        entered = context.Event()
        process = context.Process(
            target=interrupted_serializer, args=(self.directory, entered)
        )
        try:
            process.start()
            self.assertTrue(await asyncio.to_thread(entered.wait, 60))
            process.terminate()
            await asyncio.to_thread(process.join, 5)
            cache = SharedPreprocessCache(self.directory, max_size_bytes=65536)
            self.assertEqual(list(Path(self.directory).glob(".writing-*")), [])
            request = SharedCacheRequest("same-key", b"image bytes", b"configuration")
            result = (await self.resolve([request], cache=cache))[0]
            self.assertEqual(result.feature.tolist(), [[1, 2, 3]])
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
