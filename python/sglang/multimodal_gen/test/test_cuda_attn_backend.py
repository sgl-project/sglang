import importlib
import sys
import types

import pytest
import torch

from sglang.multimodal_gen.runtime.platforms.cuda import CudaPlatformBase
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum

BACKEND_PATHS = {
    AttentionBackendEnum.SLIDING_TILE_ATTN: (
        "sglang.multimodal_gen.runtime.layers.attention.backends.sliding_tile_attn."
        "SlidingTileAttentionBackend"
    ),
    AttentionBackendEnum.SAGE_ATTN: (
        "sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn."
        "SageAttentionBackend"
    ),
    AttentionBackendEnum.SAGE_ATTN_3: (
        "sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn3."
        "SageAttention3Backend"
    ),
    AttentionBackendEnum.VIDEO_SPARSE_ATTN: (
        "sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn."
        "VideoSparseAttentionBackend"
    ),
    AttentionBackendEnum.VMOBA_ATTN: (
        "sglang.multimodal_gen.runtime.layers.attention.backends.vmoba."
        "VMOBAAttentionBackend"
    ),
    AttentionBackendEnum.AITER: (
        "sglang.multimodal_gen.runtime.layers.attention.backends.aiter.AITerBackend"
    ),
    AttentionBackendEnum.TORCH_SDPA: (
        "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
    ),
    AttentionBackendEnum.FA: (
        "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn."
        "FlashAttentionBackend"
    ),
}


class DummyCudaPlatform(CudaPlatformBase):
    _is_sm120 = False
    _is_blackwell = False
    _has_capability = True

    @classmethod
    def is_sm120(cls):
        return cls._is_sm120

    @classmethod
    def is_blackwell(cls):
        return cls._is_blackwell

    @classmethod
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        return cls._has_capability


def _install_module(monkeypatch, name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _ensure_package(monkeypatch, name: str) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)


def _install_nested_module(monkeypatch, name: str, **attrs: object) -> types.ModuleType:
    parts = name.split(".")
    for i in range(1, len(parts)):
        pkg_name = ".".join(parts[:i])
        _ensure_package(monkeypatch, pkg_name)
    return _install_module(monkeypatch, name, **attrs)


@pytest.fixture
def platform():
    DummyCudaPlatform._is_sm120 = False
    DummyCudaPlatform._is_blackwell = False
    DummyCudaPlatform._has_capability = True
    return DummyCudaPlatform


@pytest.mark.parametrize(
    "backend,expected_path",
    [
        (AttentionBackendEnum.AITER, BACKEND_PATHS[AttentionBackendEnum.AITER]),
        (
            AttentionBackendEnum.TORCH_SDPA,
            BACKEND_PATHS[AttentionBackendEnum.TORCH_SDPA],
        ),
    ],
)
def test_attn_backend_explicit_simple(platform, backend, expected_path):
    result = platform.get_attn_backend_cls_str(
        backend,
        head_size=64,
        dtype=torch.float16,
    )
    assert result == expected_path


@pytest.mark.parametrize(
    "backend,imports",
    [
        (
            AttentionBackendEnum.SLIDING_TILE_ATTN,
            [
                ("st_attn", "sliding_tile_attention"),
                (
                    "sglang.multimodal_gen.runtime.layers.attention.backends."
                    "sliding_tile_attn",
                    "SlidingTileAttentionBackend",
                ),
            ],
        ),
        (
            AttentionBackendEnum.VIDEO_SPARSE_ATTN,
            [
                ("vsa", "block_sparse_attn"),
                (
                    "sglang.multimodal_gen.runtime.layers.attention.backends."
                    "video_sparse_attn",
                    "VideoSparseAttentionBackend",
                ),
            ],
        ),
        (
            AttentionBackendEnum.VMOBA_ATTN,
            [
                ("kernel.attn.vmoba_attn.vmoba", "moba_attn_varlen"),
                (
                    "sglang.multimodal_gen.runtime.layers.attention.backends.vmoba",
                    "VMOBAAttentionBackend",
                ),
            ],
        ),
    ],
)
def test_attn_backend_required_imports(
    platform,
    monkeypatch,
    backend,
    imports,
):
    for module_path, symbol in imports:
        _install_nested_module(monkeypatch, module_path, **{symbol: object()})

    result = platform.get_attn_backend_cls_str(
        backend,
        head_size=64,
        dtype=torch.float16,
    )
    assert result == BACKEND_PATHS[backend]


def test_attn_backend_sage_attn_success(platform, monkeypatch):
    _install_nested_module(monkeypatch, "sageattention", sageattn=object())
    _install_nested_module(
        monkeypatch,
        "sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn",
        SageAttentionBackend=object(),
    )

    result = platform.get_attn_backend_cls_str(
        AttentionBackendEnum.SAGE_ATTN,
        head_size=64,
        dtype=torch.float16,
    )
    assert result == BACKEND_PATHS[AttentionBackendEnum.SAGE_ATTN]


def test_attn_backend_sage_attn_missing_falls_back_to_sdpa(
    platform,
    monkeypatch,
):
    original_import = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "sageattention":
            raise ImportError("sageattention missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    _install_nested_module(
        monkeypatch,
        "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn",
    )

    result = platform.get_attn_backend_cls_str(
        AttentionBackendEnum.SAGE_ATTN,
        head_size=64,
        dtype=torch.float16,
    )
    assert result == BACKEND_PATHS[AttentionBackendEnum.TORCH_SDPA]


def test_attn_backend_sage_attn3_missing_falls_back_to_sdpa(
    platform,
    monkeypatch,
):
    original_import = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn3":
            raise ImportError("sage_attn3 missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    result = platform.get_attn_backend_cls_str(
        AttentionBackendEnum.SAGE_ATTN_3,
        head_size=64,
        dtype=torch.float16,
    )
    assert result == BACKEND_PATHS[AttentionBackendEnum.TORCH_SDPA]


def test_attn_backend_fa_head_size_fallback(platform, monkeypatch):
    class FlashBackend:
        @staticmethod
        def get_supported_head_sizes():
            return [64]

    _install_nested_module(
        monkeypatch,
        "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn",
        FlashAttentionBackend=FlashBackend,
    )

    result = platform.get_attn_backend_cls_str(
        AttentionBackendEnum.FA,
        head_size=128,
        dtype=torch.float16,
    )
    assert result == BACKEND_PATHS[AttentionBackendEnum.TORCH_SDPA]


def test_attn_backend_fa_blackwell_sets_version(platform, monkeypatch):
    platform._is_blackwell = True
    called = []

    class FlashBackend:
        @staticmethod
        def get_supported_head_sizes():
            return [64]

    def set_fa_ver(version):
        called.append(version)

    _install_nested_module(
        monkeypatch,
        "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn",
        FlashAttentionBackend=FlashBackend,
        set_fa_ver=set_fa_ver,
    )

    result = platform.get_attn_backend_cls_str(
        AttentionBackendEnum.FA,
        head_size=64,
        dtype=torch.float16,
    )
    assert result == BACKEND_PATHS[AttentionBackendEnum.FA]
    assert called == [4]


def test_attn_backend_default_sm120_falls_back_to_sdpa(platform, monkeypatch):
    platform._is_sm120 = True
    original_import = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "sageattention":
            raise ImportError("sageattention missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    result = platform.get_attn_backend_cls_str(
        None,
        head_size=64,
        dtype=torch.float16,
    )
    assert result == BACKEND_PATHS[AttentionBackendEnum.TORCH_SDPA]
