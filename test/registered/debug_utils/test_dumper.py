import os
import tempfile
import time
import unittest
from pathlib import Path

import requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="nightly-2-gpu", nightly=True)
register_amd_ci(est_time=60, suite="nightly-amd", nightly=True)


class TestDumperPureFunctions(CustomTestCase):
    def test_get_truncated_value(self):
        from sglang.srt.debug_utils.dumper import get_truncated_value

        self.assertIsNone(get_truncated_value(None))
        self.assertEqual(get_truncated_value(42), 42)
        self.assertEqual(
            len(get_truncated_value((torch.randn(10), torch.randn(20)))), 2
        )
        self.assertEqual(get_truncated_value(torch.randn(10, 10)).shape, (10, 10))
        self.assertEqual(get_truncated_value(torch.randn(100, 100)).shape, (5, 5))

    def test_obj_to_dict(self):
        from sglang.srt.debug_utils.dumper import _obj_to_dict

        self.assertEqual(_obj_to_dict({"a": 1}), {"a": 1})

        class Obj:
            x, y = 10, 20

            def method(self):
                pass

        result = _obj_to_dict(Obj())
        self.assertEqual(result["x"], 10)
        self.assertNotIn("method", result)

    def test_get_tensor_info(self):
        from sglang.srt.debug_utils.dumper import get_tensor_info

        info = get_tensor_info(torch.randn(10, 10))
        for key in ["shape=", "dtype=", "min=", "max=", "mean="]:
            self.assertIn(key, info)

        self.assertIn("value=42", get_tensor_info(42))
        self.assertIn("min=None", get_tensor_info(torch.tensor([])))


class TestDumperDistributed(CustomTestCase):
    def test_basic(self):
        with tempfile.TemporaryDirectory(prefix="test_dumper_") as tmpdir:
            _run_distributed_test(_test_basic_func, tmpdir=tmpdir)

    def test_http_enable(self):
        _run_distributed_test(_test_http_func)

    def test_filter(self):
        with tempfile.TemporaryDirectory(prefix="test_dumper_") as tmpdir:
            _run_distributed_test(_test_filter_func, tmpdir=tmpdir)

    def test_write_disabled(self):
        with tempfile.TemporaryDirectory(prefix="test_dumper_") as tmpdir:
            _run_distributed_test(_test_write_disabled_func, tmpdir=tmpdir)


def _test_basic_func(rank, tmpdir):
    os.environ["SGLANG_DUMPER_DIR"] = tmpdir
    from sglang.srt.debug_utils.dumper import dumper

    tensor = torch.randn(10, 10, device=f"cuda:{rank}")

    dumper.on_forward_pass_start()
    dumper.dump("tensor_a", tensor, arg=100)

    dumper.on_forward_pass_start()
    dumper.set_ctx(ctx_arg=200)
    dumper.dump("tensor_b", tensor)
    dumper.set_ctx(ctx_arg=None)

    dumper.on_forward_pass_start()
    dumper.override_enable(False)
    dumper.dump("tensor_skip", tensor)
    dumper.override_enable(True)

    dumper.on_forward_pass_start()
    dumper.dump_dict("obj", {"a": torch.randn(3, device=f"cuda:{rank}"), "b": 42})

    dist.barrier()
    filenames = _get_filenames(tmpdir)

    _assert_files(
        filenames,
        exist=["tensor_a", "tensor_b", "arg=100", "ctx_arg=200", "obj_a", "obj_b"],
        not_exist=["tensor_skip"],
    )


def _test_http_func(rank):
    os.environ["SGLANG_DUMPER_ENABLE"] = "0"
    from sglang.srt.debug_utils.dumper import dumper

    assert not dumper._enable
    dumper.on_forward_pass_start()

    for enable in [True, False]:
        dist.barrier()
        if rank == 0:
            time.sleep(0.1)
            requests.post(
                "http://localhost:40000/dumper", json={"enable": enable}
            ).raise_for_status()
        dist.barrier()
        assert dumper._enable == enable


def _test_filter_func(rank, tmpdir):
    os.environ["SGLANG_DUMPER_DIR"] = tmpdir
    os.environ["SGLANG_DUMPER_FILTER"] = "keep"
    from sglang.srt.debug_utils.dumper import dumper

    dumper.on_forward_pass_start()
    dumper.dump("keep_this", torch.randn(5, device=f"cuda:{rank}"))
    dumper.dump("skip_this", torch.randn(5, device=f"cuda:{rank}"))

    dist.barrier()
    filenames = _get_filenames(tmpdir)
    _assert_files(filenames, exist=["keep_this"], not_exist=["skip_this"])


def _test_write_disabled_func(rank, tmpdir):
    os.environ["SGLANG_DUMPER_DIR"] = tmpdir
    os.environ["SGLANG_DUMPER_WRITE_FILE"] = "0"
    from sglang.srt.debug_utils.dumper import dumper

    dumper.on_forward_pass_start()
    dumper.dump("no_write", torch.randn(5, device=f"cuda:{rank}"))

    dist.barrier()
    assert len(_get_filenames(tmpdir)) == 0


def _get_filenames(tmpdir):
    return {f.name for f in Path(tmpdir).glob("sglang_dump_*/*.pt")}


def _assert_files(filenames, *, exist=(), not_exist=()):
    for p in exist:
        assert any(p in f for f in filenames), f"{p} not found in {filenames}"
    for p in not_exist:
        assert not any(
            p in f for f in filenames
        ), f"{p} should not exist in {filenames}"


def _run_distributed_test(func, world_size=2, **kwargs):
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for rank in range(world_size):
        p = ctx.Process(
            target=_run_worker, args=(rank, world_size, func, result_queue, kwargs)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    errors = [result_queue.get() for _ in range(world_size)]
    errors = [e for e in errors if e]
    if errors:
        raise AssertionError("\n".join(errors))


def _run_worker(rank, world_size, func, result_queue, kwargs):
    os.environ.update(
        MASTER_ADDR="localhost",
        MASTER_PORT="29500",
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    try:
        func(rank, **kwargs)
        result_queue.put(None)
    except Exception as e:
        import traceback

        result_queue.put(f"Rank {rank}: {e}\n{traceback.format_exc()}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
