import os
import tempfile
import time
import unittest
from pathlib import Path

import requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.test_utils import CustomTestCase


class TestDumper(CustomTestCase):
    def test_basic(self):
        tmpdir = tempfile.mkdtemp(prefix="test_dumper_")
        _run_distributed_test(_test_basic_func, tmpdir=tmpdir)

    def test_disable_at_startup_use_http_to_enable(self):
        _run_distributed_test(_test_http_func)


def _test_basic_func(rank, tmpdir):
    os.environ["SGLANG_DUMPER_DIR"] = tmpdir

    from sglang.srt.debug_utils.dumper import dumper

    tensor = torch.randn(10, 10, device=f"cuda:{rank}")

    dumper.on_forward_pass_start()
    dumper.dump("tensor_a", tensor, my_dump_arg=100)

    dumper.on_forward_pass_start()
    dumper.set_ctx(my_ctx_arg=200)
    dumper.dump("tensor_b", tensor)
    dumper.set_ctx(my_ctx_arg=None)

    dumper.on_forward_pass_start()
    dumper.override_enable(False)
    dumper.dump("tensor_should_not_exist", tensor)
    dumper.override_enable(True)

    dist.barrier()

    files = list(Path(tmpdir).glob("sglang_dump_*/*.pt"))
    filenames = {f.name for f in files}

    expected_segments = [
        {"forward_pass_id=1", "name=tensor_a", "my_dump_arg=100", f"rank={rank}"},
        {"forward_pass_id=2", "name=tensor_b", "my_ctx_arg=200", f"rank={rank}"},
    ]
    for segments in expected_segments:
        assert any(all(seg in f for seg in segments) for f in filenames), \
            f"No file matches {segments}, got {filenames}"

    assert not any("tensor_should_not_exist" in f for f in filenames)


def _test_http_func(rank):
    os.environ["SGLANG_DUMPER_ENABLE"] = "0"

    from sglang.srt.debug_utils.dumper import dumper

    assert not dumper._enable
    dumper.on_forward_pass_start()

    for enable in [True, False]:
        dist.barrier()
        if rank == 0:
            time.sleep(0.1)
            requests.post("http://localhost:40000/dumper", json={"enable": enable}).raise_for_status()
        dist.barrier()
        assert dumper._enable == enable, f"Rank {rank}: expected _enable={enable}"


# TODO extract to utility
def _run_distributed_test(func, world_size=2, **kwargs):
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for rank in range(world_size):
        p = ctx.Process(target=_run_distributed, args=(rank, world_size, func, result_queue, kwargs))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    errors = []
    for _ in range(world_size):
        rank, error = result_queue.get()
        if error:
            errors.append(f"Rank {rank}: {error}")

    if errors:
        raise AssertionError("\n".join(errors))


def _run_distributed(rank, world_size, func, result_queue, kwargs):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    try:
        func(rank, **kwargs)
        result_queue.put((rank, None))
    except Exception as e:
        import traceback
        result_queue.put((rank, f"{e}\n{traceback.format_exc()}"))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
