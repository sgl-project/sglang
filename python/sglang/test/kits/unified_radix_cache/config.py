"""Cache configs and per-config test requirements for the UnifiedRadixCache unit tests."""

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.mem_cache.unified_cache.components.base import (
    ComponentType,
)
from sglang.test.test_utils import CustomTestCase


@dataclass(frozen=True)
class CacheConfig:
    page_size: int = 1
    components: tuple[ComponentType, ...] = (ComponentType.FULL,)

    # Layer split (only matters for SWA/Mamba)
    num_layers: int = 24
    full_attention_layer_ids: tuple[int, ...] = (3, 7, 11, 15, 19, 23)

    sliding_window_size: Optional[int] = None
    draft_swa_window: int = 0
    decoder_swa_bounded_replay: bool = False

    enable_mamba_extra_buffer: bool = False
    mamba_cache_size: int = 20
    mamba_intermediate_size: int = 256
    mamba_n_groups: int = 1
    mamba_num_heads: int = 2
    mamba_head_dim: int = 16
    mamba_state_size: int = 16
    mamba_conv_kernel: int = 4
    enable_int8_mamba_checkpoint: bool = False

    kv_size: int = 256
    max_num_reqs: int = 10
    max_context_len: int = 512
    head_num: int = 2
    head_dim: int = 64
    dtype: torch.dtype = torch.bfloat16
    eviction_policy: str = "lru"
    is_eagle: bool = False

    @property
    def has_mamba(self) -> bool:
        return ComponentType.MAMBA in self.components

    @property
    def has_swa(self) -> bool:
        return ComponentType.SWA in self.components

    @property
    def non_full_layer_ids(self) -> list[int]:
        full = set(self.full_attention_layer_ids)
        return [i for i in range(self.num_layers) if i not in full]

    @property
    def label(self) -> str:
        comp = "_".join(c.name for c in self.components)
        parts = [f"{comp}_ps{self.page_size}"]
        if self.sliding_window_size is not None:
            parts.append(f"sw{self.sliding_window_size}")
        defaults = self.__dataclass_fields__
        if (
            self.head_num != defaults["head_num"].default
            or self.num_layers != defaults["num_layers"].default
        ):
            parts.append(f"h{self.head_num}l{self.num_layers}")
        if self.is_eagle:
            parts.append("eagle")
        return "_".join(parts)


CONFIGS: list[CacheConfig] = [
    CacheConfig(page_size=1, components=(ComponentType.FULL,)),
    CacheConfig(page_size=4, components=(ComponentType.FULL,)),
    CacheConfig(page_size=16, components=(ComponentType.FULL,)),
    CacheConfig(
        page_size=64,
        components=(ComponentType.FULL,),
        kv_size=1024,
        max_context_len=1024,
    ),
    CacheConfig(
        page_size=128,
        components=(ComponentType.FULL,),
        kv_size=2048,
        max_context_len=2048,
    ),
    CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.MAMBA),
    ),
    CacheConfig(
        page_size=4,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,  # Mamba page_size > 1 requires enable_mamba_extra_buffer=True
        mamba_cache_size=60,
    ),
    CacheConfig(
        page_size=64,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        mamba_cache_size=60,
        kv_size=1024,
        max_context_len=1024,
    ),
    CacheConfig(
        page_size=128,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        mamba_cache_size=60,
        kv_size=2048,
        max_context_len=2048,
    ),
    CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=4,
    ),
    CacheConfig(
        page_size=4,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=4,
    ),
    CacheConfig(
        page_size=4,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=2,  # window < page_size edge case
    ),
    CacheConfig(
        page_size=16,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=16,
        kv_size=512,
    ),
    CacheConfig(
        page_size=64,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=64,
        kv_size=4096,
        max_context_len=4096,
    ),
    CacheConfig(
        page_size=128,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=128,
        kv_size=8192,
        max_context_len=8192,
    ),
    CacheConfig(
        page_size=128,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=4,
        kv_size=8192,
        max_context_len=8192,
    ),
    CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=128,
        kv_size=1024,
        max_context_len=1024,
    ),
    CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA),
        sliding_window_size=128,
        head_num=8,
        num_layers=32,
        full_attention_layer_ids=(7, 15, 23, 31),
        kv_size=1024,
        max_context_len=1024,
    ),
    CacheConfig(
        page_size=64,
        components=(ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA),
        sliding_window_size=64,
        enable_mamba_extra_buffer=True,
        mamba_cache_size=60,
        kv_size=4096,
        max_context_len=4096,
    ),
]


def requires(predicate, reason):
    """Keep a test only under the configs where ``predicate`` holds.

    Elsewhere the test is removed, not skipped. Stacked requirements are checked
    top to bottom, so a lower one may assume the ones above it hold.
    """

    def decorate(test):
        requirements = getattr(test, "_cfg_requirements", ())
        test._cfg_requirements = ((predicate, reason), *requirements)
        return test

    return decorate


def supports_hicache(cfg):
    return not (cfg.has_swa and cfg.has_mamba)


requires_hicache = requires(
    supports_hicache, "HiCache unit fixture does not support SWA + Mamba stacks"
)


requires_swa = requires(lambda cfg: cfg.has_swa, "requires SWA")


requires_mamba = requires(lambda cfg: cfg.has_mamba, "requires Mamba")


requires_swa_only = requires(
    lambda cfg: cfg.has_swa and not cfg.has_mamba, "requires SWA without Mamba"
)


requires_swa_and_mamba = requires(
    lambda cfg: cfg.has_swa and cfg.has_mamba, "requires SWA and Mamba"
)


requires_paged = requires(lambda cfg: cfg.page_size != 1, "requires page_size > 1")


requires_ps1_window4 = requires(
    lambda cfg: cfg.page_size == 1 and cfg.sliding_window_size == 4,
    "requires page_size=1, sliding_window_size=4",
)


requires_full_swa_ps1 = requires(
    lambda cfg: cfg.has_swa and not cfg.has_mamba and cfg.page_size == 1,
    "requires page_size=1 Full+SWA",
)


requires_full_swa_ps1_window4 = requires(
    lambda cfg: (
        cfg.has_swa
        and not cfg.has_mamba
        and cfg.page_size == 1
        and cfg.sliding_window_size == 4
    ),
    "requires page_size=1 Full+SWA with window_size=4",
)


requires_full_mamba_ps1 = requires(
    lambda cfg: cfg.has_mamba and not cfg.has_swa and cfg.page_size == 1,
    "requires page_size=1 Full+Mamba",
)


def config_attrs(suite, cfg):
    """Bind the test mixin ``suite`` to ``cfg``, hiding the tests whose requirements it fails."""
    attrs = {"cfg": cfg}
    for name, test in vars(suite).items():
        requirements = getattr(test, "_cfg_requirements", ())
        if not all(predicate(cfg) for predicate, _ in requirements):
            attrs[name] = None
    return attrs


def define_config_test_classes(suite, namespace):
    """Define ``Test_<label>`` in ``namespace`` for each config, binding the test mixin ``suite``."""
    for cfg in CONFIGS:
        name = f"Test_{cfg.label}"
        test_class = type(name, (suite, CustomTestCase), config_attrs(suite, cfg))
        test_class.__module__ = namespace["__name__"]
        namespace[name] = test_class
