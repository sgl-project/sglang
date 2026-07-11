import io

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")

from sglang.srt.distributed.device_communicators import cuda_wrapper


def test_find_loaded_library_prefers_real_cudart_over_tilelang_stub(monkeypatch):
    maps = """\
7f000000-7f010000 r-xp 00000000 00:00 0 /site-packages/tilelang/lib/libcudart_stub.so
7f020000-7f030000 r-xp 00000000 00:00 0 /cuda/lib64/libcudart.so.13
"""

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(maps))

    assert (
        cuda_wrapper.find_loaded_library("libcudart") == "/cuda/lib64/libcudart.so.13"
    )
