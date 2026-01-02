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


class TestDumperUtils(CustomTestCase):
    def test_get_truncated_value_none(self):
        from sglang.srt.debug_utils.dumper import get_truncated_value

        assert get_truncated_value(None) is None

    def test_get_truncated_value_tuple(self):
        from sglang.srt.debug_utils.dumper import get_truncated_value

        t1 = torch.randn(10)
        t2 = torch.randn(20)
        result = get_truncated_value((t1, t2))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_get_truncated_value_small_tensor(self):
        from sglang.srt.debug_utils.dumper import get_truncated_value

        tensor = torch.randn(10, 10)
        result = get_truncated_value(tensor)
        assert result.shape == tensor.shape

    def test_get_truncated_value_large_tensor(self):
        from sglang.srt.debug_utils.dumper import get_truncated_value

        tensor = torch.randn(100, 100)
        result = get_truncated_value(tensor)
        assert result.shape == (5, 5)

    def test_get_truncated_value_non_tensor(self):
        from sglang.srt.debug_utils.dumper import get_truncated_value

        assert get_truncated_value(42) == 42
        assert get_truncated_value("hello") == "hello"

    def test_obj_to_dict_with_dict(self):
        from sglang.srt.debug_utils.dumper import _obj_to_dict

        d = {"a": 1, "b": 2}
        result = _obj_to_dict(d)
        assert result == d

    def test_obj_to_dict_with_object(self):
        from sglang.srt.debug_utils.dumper import _obj_to_dict

        class MyObj:
            def __init__(self):
                self.x = 10
                self.y = 20

            def method(self):
                pass

        obj = MyObj()
        result = _obj_to_dict(obj)
        assert "x" in result and result["x"] == 10
        assert "y" in result and result["y"] == 20
        assert "method" not in result

    def test_get_tensor_info_tensor(self):
        from sglang.srt.debug_utils.dumper import get_tensor_info

        tensor = torch.randn(10, 10)
        info = get_tensor_info(tensor)
        assert "shape=" in info
        assert "dtype=" in info
        assert "min=" in info
        assert "max=" in info
        assert "mean=" in info

    def test_get_tensor_info_non_tensor(self):
        from sglang.srt.debug_utils.dumper import get_tensor_info

        info = get_tensor_info(42)
        assert "type=" in info
        assert "value=42" in info

    def test_get_tensor_info_empty_tensor(self):
        from sglang.srt.debug_utils.dumper import get_tensor_info

        tensor = torch.tensor([])
        info = get_tensor_info(tensor)
        assert "shape=" in info
        assert "min=None" in info


class TestDumperDistributed(CustomTestCase):
    def test_basic(self):
        tmpdir = tempfile.mkdtemp(prefix="test_dumper_")
        _run_distributed_test(_test_basic_func, tmpdir=tmpdir)

    def test_disable_at_startup_use_http_to_enable(self):
        _run_distributed_test(_test_http_func)

    def test_dump_dict(self):
        tmpdir = tempfile.mkdtemp(prefix="test_dumper_dump_dict_")
        _run_distributed_test(_test_dump_dict_func, tmpdir=tmpdir)

    def test_filter_env_var(self):
        tmpdir = tempfile.mkdtemp(prefix="test_dumper_filter_")
        _run_distributed_test(_test_filter_func, tmpdir=tmpdir)

    def test_write_file_disabled(self):
        tmpdir = tempfile.mkdtemp(prefix="test_dumper_no_write_")
        _run_distributed_test(_test_write_file_disabled_func, tmpdir=tmpdir)


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
        assert any(
            all(seg in f for seg in segments) for f in filenames
        ), f"No file matches {segments}, got {filenames}"

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
            requests.post(
                "http://localhost:40000/dumper", json={"enable": enable}
            ).raise_for_status()
        dist.barrier()
        assert dumper._enable == enable, f"Rank {rank}: expected _enable={enable}"


def _test_dump_dict_func(rank, tmpdir):
    os.environ["SGLANG_DUMPER_DIR"] = tmpdir

    from sglang.srt.debug_utils.dumper import dumper

    class MyObject:
        def __init__(self):
            self.tensor_field = torch.randn(5, 5, device=f"cuda:{rank}")
            self.scalar_field = 42

        def some_method(self):
            pass

    obj = MyObject()
    dumper.on_forward_pass_start()
    dumper.dump_dict("my_obj", obj)

    dict_data = {"key1": torch.randn(3, 3, device=f"cuda:{rank}"), "key2": 123}
    dumper.dump_dict("my_dict", dict_data)

    dist.barrier()

    files = list(Path(tmpdir).glob("sglang_dump_*/*.pt"))
    filenames = {f.name for f in files}

    expected_segments = [
        {"name=my_obj_tensor_field", f"rank={rank}"},
        {"name=my_obj_scalar_field", f"rank={rank}"},
        {"name=my_dict_key1", f"rank={rank}"},
        {"name=my_dict_key2", f"rank={rank}"},
    ]
    for segments in expected_segments:
        assert any(
            all(seg in f for seg in segments) for f in filenames
        ), f"No file matches {segments}, got {filenames}"


def _test_filter_func(rank, tmpdir):
    os.environ["SGLANG_DUMPER_DIR"] = tmpdir
    os.environ["SGLANG_DUMPER_FILTER"] = "tensor_a|tensor_c"

    from sglang.srt.debug_utils.dumper import _Dumper

    dumper = _Dumper()
    tensor = torch.randn(10, 10, device=f"cuda:{rank}")

    dumper.on_forward_pass_start()
    dumper.dump("tensor_a", tensor)
    dumper.dump("tensor_b", tensor)
    dumper.dump("tensor_c", tensor)

    dist.barrier()

    files = list(Path(tmpdir).glob("sglang_dump_*/*.pt"))
    filenames = {f.name for f in files}

    assert any("tensor_a" in f for f in filenames), f"tensor_a should exist, got {filenames}"
    assert not any("tensor_b" in f for f in filenames), f"tensor_b should NOT exist, got {filenames}"
    assert any("tensor_c" in f for f in filenames), f"tensor_c should exist, got {filenames}"


def _test_write_file_disabled_func(rank, tmpdir):
    os.environ["SGLANG_DUMPER_DIR"] = tmpdir
    os.environ["SGLANG_DUMPER_WRITE_FILE"] = "0"

    from sglang.srt.debug_utils.dumper import _Dumper

    dumper = _Dumper()
    tensor = torch.randn(10, 10, device=f"cuda:{rank}")

    dumper.on_forward_pass_start()
    dumper.dump("tensor_no_write", tensor)

    dist.barrier()

    files = list(Path(tmpdir).glob("sglang_dump_*/*.pt"))
    assert len(files) == 0, f"No files should be written when SGLANG_DUMPER_WRITE_FILE=0, got {files}"


# TODO extract to utility
def _run_distributed_test(func, world_size=2, **kwargs):
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for rank in range(world_size):
        p = ctx.Process(
            target=_run_distributed, args=(rank, world_size, func, result_queue, kwargs)
        )
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
