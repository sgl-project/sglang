import os
import sys
import time
from pathlib import Path

import pytest
import requests
import torch
import torch.distributed as dist

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import run_distributed_test

register_cuda_ci(est_time=30, suite="nightly-2-gpu", nightly=True)
register_amd_ci(est_time=60, suite="nightly-amd", nightly=True)


class TestDumperPureFunctions:
    def test_get_truncated_value(self):
        from sglang.srt.debug_utils.dumper import get_truncated_value

        assert get_truncated_value(None) is None
        assert get_truncated_value(42) == 42
        assert len(get_truncated_value((torch.randn(10), torch.randn(20)))) == 2
        assert get_truncated_value(torch.randn(10, 10)).shape == (10, 10)
        assert get_truncated_value(torch.randn(100, 100)).shape == (5, 5)

    def test_obj_to_dict(self):
        from sglang.srt.debug_utils.dumper import _obj_to_dict

        assert _obj_to_dict({"a": 1}) == {"a": 1}

        class Obj:
            x, y = 10, 20

            def method(self):
                pass

        result = _obj_to_dict(Obj())
        assert result["x"] == 10
        assert "method" not in result

    def test_get_tensor_info(self):
        from sglang.srt.debug_utils.dumper import get_tensor_info

        info = get_tensor_info(torch.randn(10, 10))
        for key in ["shape=", "dtype=", "min=", "max=", "mean="]:
            assert key in info

        assert "value=42" in get_tensor_info(42)
        assert "min=None" in get_tensor_info(torch.tensor([]))


class TestTorchSave:
    def test_normal(self, tmp_path):
        from sglang.srt.debug_utils.dumper import _torch_save

        path = str(tmp_path / "a.pt")
        tensor = torch.randn(3, 3)

        _torch_save(tensor, path)

        assert torch.equal(torch.load(path, weights_only=True), tensor)

    def test_parameter_fallback(self, tmp_path):
        from sglang.srt.debug_utils.dumper import _torch_save

        class BadParam(torch.nn.Parameter):
            def __reduce_ex__(self, protocol):
                raise RuntimeError("not pickleable")

        path = str(tmp_path / "b.pt")
        param = BadParam(torch.randn(4))

        _torch_save(param, path)

        assert torch.equal(torch.load(path, weights_only=True), param.data)

    def test_silent_skip(self, tmp_path, capsys):
        from sglang.srt.debug_utils.dumper import _torch_save

        path = str(tmp_path / "c.pt")

        _torch_save({"fn": lambda: None}, path)

        captured = capsys.readouterr()
        assert "[Dumper] Observe error=" in captured.out
        assert "skip the tensor" in captured.out


class TestDumperDistributed:
    def test_basic(self, tmp_path):
        run_distributed_test(self._test_basic_func, tmpdir=str(tmp_path))

    @staticmethod
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

    def test_http_enable(self):
        run_distributed_test(self._test_http_func)

    @staticmethod
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

    def test_file_content_correctness(self, tmp_path):
        run_distributed_test(self._test_file_content_func, tmpdir=str(tmp_path))

    @staticmethod
    def _test_file_content_func(rank, tmpdir):
        os.environ["SGLANG_DUMPER_DIR"] = tmpdir
        from sglang.srt.debug_utils.dumper import dumper

        tensor = torch.arange(12, device=f"cuda:{rank}").reshape(3, 4).float()

        dumper.on_forward_pass_start()
        dumper.dump("content_check", tensor)

        dist.barrier()
        path = _find_dump_file(tmpdir, rank=rank, name="content_check")
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert torch.equal(loaded, tensor.cpu())


class TestDumperFileWriteControl:
    def test_filter(self, tmp_path):
        run_distributed_test(self._test_filter_func, tmpdir=str(tmp_path))

    @staticmethod
    def _test_filter_func(rank, tmpdir):
        os.environ["SGLANG_DUMPER_DIR"] = tmpdir
        os.environ["SGLANG_DUMPER_FILTER"] = "^keep"
        from sglang.srt.debug_utils.dumper import dumper

        dumper.on_forward_pass_start()
        dumper.dump("keep_this", torch.randn(5, device=f"cuda:{rank}"))
        dumper.dump("skip_this", torch.randn(5, device=f"cuda:{rank}"))
        dumper.dump("not_keep_this", torch.randn(5, device=f"cuda:{rank}"))

        dist.barrier()
        filenames = _get_filenames(tmpdir)
        _assert_files(
            filenames,
            exist=["keep_this"],
            not_exist=["skip_this", "not_keep_this"],
        )

    def test_write_disabled(self, tmp_path):
        run_distributed_test(self._test_write_disabled_func, tmpdir=str(tmp_path))

    @staticmethod
    def _test_write_disabled_func(rank, tmpdir):
        os.environ["SGLANG_DUMPER_DIR"] = tmpdir
        os.environ["SGLANG_DUMPER_WRITE_FILE"] = "0"
        from sglang.srt.debug_utils.dumper import dumper

        dumper.on_forward_pass_start()
        dumper.dump("no_write", torch.randn(5, device=f"cuda:{rank}"))

        dist.barrier()
        assert len(_get_filenames(tmpdir)) == 0

    def test_save_false(self, tmp_path):
        run_distributed_test(self._test_save_false_func, tmpdir=str(tmp_path))

    @staticmethod
    def _test_save_false_func(rank, tmpdir):
        os.environ["SGLANG_DUMPER_DIR"] = tmpdir
        from sglang.srt.debug_utils.dumper import dumper

        dumper.on_forward_pass_start()
        dumper.dump("no_save_tensor", torch.randn(5, device=f"cuda:{rank}"), save=False)

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


def _find_dump_file(tmpdir, *, rank: int, name: str) -> Path:
    matches = [
        f
        for f in Path(tmpdir).glob("sglang_dump_*/*.pt")
        if f"rank={rank}" in f.name and name in f.name
    ]
    assert (
        len(matches) == 1
    ), f"Expected 1 file matching rank={rank} name={name}, got {matches}"
    return matches[0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
