"""Internal admission and kernel profiles for SharedEP."""

from __future__ import annotations

import msgspec
import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.shared_ep import SharedEpQuantization

_DEFAULT_KERNEL_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 8,
    "num_stages": 3,
}
_DSV4_SMALL_KERNEL_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 3,
}
_DSV4_LARGE_W13_KERNEL_CONFIG = {
    **_DEFAULT_KERNEL_CONFIG,
    "num_warps": 4,
    "num_stages": 4,
}
_GLM_SMALL_W13_KERNEL_CONFIG = _DSV4_SMALL_KERNEL_CONFIG
_GLM_LARGE_W13_KERNEL_CONFIG = {
    **_DEFAULT_KERNEL_CONFIG,
    "num_stages": 4,
}
_GLM_W2_KERNEL_CONFIG = {
    **_DEFAULT_KERNEL_CONFIG,
    "num_warps": 4,
}
_ROUTE_KERNEL_CONFIG = {"num_threads": 1024}
_HIP_KERNEL_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
}
_HIP_SMALL_KERNEL_CONFIG = {
    **_HIP_KERNEL_CONFIG,
    "num_warps": 4,
    "num_stages": 1,
}
_HIP_ROUTE_KERNEL_CONFIG = {"num_threads": 256}
_GFX950_MXFP4_KERNEL_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 4,
    "num_warps": 8,
    "num_stages": 2,
    "waves_per_eu": 0,
    "matrix_instr_nonkdim": 16,
    "kpack": 1,
}
RELEASE_MAX_TOKENS_PER_RANK = 32


class SharedEpProfile(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    capability: tuple[int, int]
    hidden_size: int
    intermediate_size: int
    top_k: int
    num_experts: int
    num_local_experts: int
    ep_size: int
    max_tokens_per_rank: int
    quantization: SharedEpQuantization
    block_shape: tuple[int, int]
    default_kernel_config: dict[str, int]
    small_kernel_config: dict[str, int] | None
    small_kernel_max_tokens: int
    route_kernel_config: dict[str, int]
    default_w13_kernel_config: dict[str, int] | None = None
    default_w2_kernel_config: dict[str, int] | None = None
    small_w13_kernel_config: dict[str, int] | None = None
    small_w2_kernel_config: dict[str, int] | None = None
    platform: str = "cuda"

    @property
    def block_size_m(self) -> int:
        return self.default_kernel_config["BLOCK_SIZE_M"]

    def kernel_config(self, num_tokens: int) -> dict[str, int]:
        self._validate_num_tokens(num_tokens)
        if (
            self.small_kernel_config is not None
            and num_tokens <= self.small_kernel_max_tokens
        ):
            return self.small_kernel_config
        return self.default_kernel_config

    def w13_kernel_config(self, num_tokens: int) -> dict[str, int]:
        return self._stage_kernel_config(
            num_tokens,
            default=self.default_w13_kernel_config,
            small=self.small_w13_kernel_config,
        )

    def w2_kernel_config(self, num_tokens: int) -> dict[str, int]:
        return self._stage_kernel_config(
            num_tokens,
            default=self.default_w2_kernel_config,
            small=self.small_w2_kernel_config,
        )

    def _stage_kernel_config(
        self,
        num_tokens: int,
        *,
        default: dict[str, int] | None,
        small: dict[str, int] | None,
    ) -> dict[str, int]:
        self._validate_num_tokens(num_tokens)
        if num_tokens <= self.small_kernel_max_tokens:
            if small is not None:
                return small
            if self.small_kernel_config is not None:
                return self.small_kernel_config
        if default is not None:
            return default
        return self.kernel_config(num_tokens)

    def _validate_num_tokens(self, num_tokens: int) -> None:
        if not 0 <= num_tokens <= self.max_tokens_per_rank:
            raise ValueError(
                f"local tokens {num_tokens} exceed profile capacity "
                f"{self.max_tokens_per_rank}"
            )

    def admission_tuple(self) -> tuple:
        return (
            self.platform,
            self.capability,
            self.hidden_size,
            self.intermediate_size,
            self.top_k,
            self.num_experts,
            self.num_local_experts,
            self.ep_size,
            self.max_tokens_per_rank,
            self.quantization,
            self.block_shape,
        )


GLM52 = SharedEpProfile(
    name="glm52",
    capability=(9, 0),
    hidden_size=6144,
    intermediate_size=2048,
    top_k=8,
    num_experts=256,
    num_local_experts=32,
    ep_size=8,
    max_tokens_per_rank=RELEASE_MAX_TOKENS_PER_RANK,
    quantization=SharedEpQuantization.BLOCK_FP8,
    block_shape=(128, 128),
    default_kernel_config=_DEFAULT_KERNEL_CONFIG,
    small_kernel_config=None,
    small_kernel_max_tokens=4,
    route_kernel_config=_ROUTE_KERNEL_CONFIG,
    default_w13_kernel_config=_GLM_LARGE_W13_KERNEL_CONFIG,
    default_w2_kernel_config=_GLM_W2_KERNEL_CONFIG,
    small_w13_kernel_config=_GLM_SMALL_W13_KERNEL_CONFIG,
    small_w2_kernel_config=_GLM_W2_KERNEL_CONFIG,
)

DSV4_FLASH = SharedEpProfile(
    name="dsv4_flash",
    capability=(9, 0),
    hidden_size=4096,
    intermediate_size=2048,
    top_k=6,
    num_experts=256,
    num_local_experts=32,
    ep_size=8,
    max_tokens_per_rank=RELEASE_MAX_TOKENS_PER_RANK,
    quantization=SharedEpQuantization.BLOCK_FP8,
    block_shape=(128, 128),
    default_kernel_config=_DEFAULT_KERNEL_CONFIG,
    small_kernel_config=_DSV4_SMALL_KERNEL_CONFIG,
    small_kernel_max_tokens=4,
    route_kernel_config=_ROUTE_KERNEL_CONFIG,
    default_w13_kernel_config=_DSV4_LARGE_W13_KERNEL_CONFIG,
    default_w2_kernel_config=_DSV4_SMALL_KERNEL_CONFIG,
)

GLM52_GFX950 = SharedEpProfile(
    name="glm52_gfx950",
    capability=(9, 5),
    hidden_size=6144,
    intermediate_size=2048,
    top_k=8,
    num_experts=256,
    num_local_experts=32,
    ep_size=8,
    max_tokens_per_rank=RELEASE_MAX_TOKENS_PER_RANK,
    quantization=SharedEpQuantization.BLOCK_FP8,
    block_shape=(128, 128),
    default_kernel_config=_HIP_KERNEL_CONFIG,
    small_kernel_config=_HIP_SMALL_KERNEL_CONFIG,
    small_kernel_max_tokens=4,
    route_kernel_config=_HIP_ROUTE_KERNEL_CONFIG,
    default_w13_kernel_config=_HIP_KERNEL_CONFIG,
    default_w2_kernel_config=_HIP_KERNEL_CONFIG,
    small_w13_kernel_config=_HIP_SMALL_KERNEL_CONFIG,
    small_w2_kernel_config=_HIP_SMALL_KERNEL_CONFIG,
    platform="rocm",
)

DSV4_FLASH_GFX950 = SharedEpProfile(
    name="dsv4_flash_gfx950",
    capability=(9, 5),
    hidden_size=4096,
    intermediate_size=2048,
    top_k=6,
    num_experts=256,
    num_local_experts=32,
    ep_size=8,
    max_tokens_per_rank=RELEASE_MAX_TOKENS_PER_RANK,
    quantization=SharedEpQuantization.BLOCK_FP8,
    block_shape=(128, 128),
    default_kernel_config=_HIP_KERNEL_CONFIG,
    small_kernel_config=_HIP_SMALL_KERNEL_CONFIG,
    small_kernel_max_tokens=4,
    route_kernel_config=_HIP_ROUTE_KERNEL_CONFIG,
    default_w13_kernel_config=_HIP_KERNEL_CONFIG,
    default_w2_kernel_config=_HIP_KERNEL_CONFIG,
    small_w13_kernel_config=_HIP_SMALL_KERNEL_CONFIG,
    small_w2_kernel_config=_HIP_SMALL_KERNEL_CONFIG,
    platform="rocm",
)

DSV4_PRO_MXFP4_GFX950 = SharedEpProfile(
    name="dsv4_pro_mxfp4_gfx950",
    capability=(9, 5),
    hidden_size=7168,
    intermediate_size=3072,
    top_k=6,
    num_experts=384,
    num_local_experts=48,
    ep_size=8,
    max_tokens_per_rank=RELEASE_MAX_TOKENS_PER_RANK,
    quantization=SharedEpQuantization.MXFP4,
    # MXFP4 has one E8M0 scale per output row and 32 logical K values.
    block_shape=(1, 32),
    default_kernel_config=_GFX950_MXFP4_KERNEL_CONFIG,
    small_kernel_config=None,
    small_kernel_max_tokens=RELEASE_MAX_TOKENS_PER_RANK,
    route_kernel_config=_HIP_ROUTE_KERNEL_CONFIG,
    default_w13_kernel_config=_GFX950_MXFP4_KERNEL_CONFIG,
    default_w2_kernel_config=_GFX950_MXFP4_KERNEL_CONFIG,
    platform="rocm",
)

_PROFILES = (
    GLM52,
    DSV4_FLASH,
    GLM52_GFX950,
    DSV4_FLASH_GFX950,
    DSV4_PRO_MXFP4_GFX950,
)


def _runtime_platform_key() -> str:
    return "rocm" if torch.version.hip is not None else "cuda"


def select_profile(
    config: MoeRunnerConfig,
    *,
    capability: tuple[int, int],
    ep_size: int,
    block_shape: tuple[int, int],
    max_tokens_per_rank: int,
    platform: str | None = None,
    quantization: SharedEpQuantization | str = SharedEpQuantization.BLOCK_FP8,
) -> SharedEpProfile:
    if platform is None:
        platform = _runtime_platform_key()
    if platform not in ("cuda", "rocm"):
        raise ValueError(
            f"SharedEP platform must be 'cuda' or 'rocm', got {platform!r}"
        )
    try:
        quantization = SharedEpQuantization(quantization)
    except ValueError as error:
        raise ValueError(
            f"Unsupported SharedEP expert quantization {quantization!r}"
        ) from error
    observed = (
        platform,
        capability,
        config.hidden_size,
        config.intermediate_size_per_partition,
        config.top_k,
        config.num_experts,
        config.num_local_experts,
        ep_size,
        max_tokens_per_rank,
        quantization,
        tuple(block_shape),
    )
    for profile in _PROFILES:
        if observed == profile.admission_tuple():
            return profile
    supported = [profile.admission_tuple() for profile in _PROFILES]
    raise ValueError(
        "SharedEP supports only registered release profiles: "
        f"observed={observed}, supported={supported}"
    )
