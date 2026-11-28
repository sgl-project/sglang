# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the pinned host-buffer cache in the minimax_h3 decode stage.

`_to_pinned_cpu` swaps the pageable ``.cpu()`` DtoH of decoded image frames
for a cached pinned buffer. The cache logic is exercised with a fake CUDA
tensor and a mocked ``torch.empty`` so the tests need no GPU.
"""

from unittest import mock

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.decoding import (
    _PINNED_BUF_CACHE_MAX,
    _pinned_buf_cache,
    _to_pinned_cpu,
)


class _FakeCudaTensor:
    def __init__(self, shape, dtype, index=0):
        self._shape = tuple(shape)
        self.dtype = dtype
        self.device = mock.Mock(index=index)

    @property
    def shape(self):
        return self._shape

    @property
    def is_cuda(self):
        return True


def _fake_buf():
    buf = mock.Mock()
    buf.is_pinned.return_value = True
    return buf


def test_cpu_tensor_passes_through_untouched():
    _pinned_buf_cache.clear()
    t = torch.zeros(2, 3)
    assert _to_pinned_cpu(t) is t
    assert _pinned_buf_cache == {}


def test_first_call_creates_pinned_buffer_and_copies():
    _pinned_buf_cache.clear()
    t = _FakeCudaTensor((2, 3), torch.float32)
    with mock.patch("torch.empty", return_value=_fake_buf()) as m:
        out = _to_pinned_cpu(t)
    m.assert_called_once_with(
        (2, 3), dtype=torch.float32, device="cpu", pin_memory=True
    )
    out.copy_.assert_called_once_with(t)
    assert len(_pinned_buf_cache) == 1


def test_same_key_hits_the_cache():
    _pinned_buf_cache.clear()
    t1 = _FakeCudaTensor((2, 3), torch.float32)
    t2 = _FakeCudaTensor((2, 3), torch.float32)
    with mock.patch("torch.empty", return_value=_fake_buf()) as m:
        b1 = _to_pinned_cpu(t1)
        b2 = _to_pinned_cpu(t2)
    assert b1 is b2
    assert m.call_count == 1


def test_distinct_keys_get_distinct_buffers():
    _pinned_buf_cache.clear()
    with mock.patch("torch.empty", side_effect=lambda *a, **k: _fake_buf()):
        b1 = _to_pinned_cpu(_FakeCudaTensor((2, 3), torch.float32))
        b2 = _to_pinned_cpu(_FakeCudaTensor((2, 3), torch.float64))
        b3 = _to_pinned_cpu(_FakeCudaTensor((2, 3), torch.float32, index=1))
    assert len({id(b1), id(b2), id(b3)}) == 3
    assert len(_pinned_buf_cache) == 3


def test_lru_cache_is_capped():
    _pinned_buf_cache.clear()
    with mock.patch("torch.empty", side_effect=lambda *a, **k: _fake_buf()):
        for i in range(_PINNED_BUF_CACHE_MAX):
            _to_pinned_cpu(_FakeCudaTensor((i,), torch.float32))
        assert len(_pinned_buf_cache) == _PINNED_BUF_CACHE_MAX
        _to_pinned_cpu(_FakeCudaTensor((99,), torch.float32))
        assert len(_pinned_buf_cache) == _PINNED_BUF_CACHE_MAX
        assert ((99,), torch.float32, 0) in _pinned_buf_cache
        assert ((0,), torch.float32, 0) not in _pinned_buf_cache
        assert ((1,), torch.float32, 0) in _pinned_buf_cache
