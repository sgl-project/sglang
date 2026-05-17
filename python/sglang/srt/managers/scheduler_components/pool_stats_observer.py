from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple


class SchedulerStats: ...  # type: ignore[no-redef]


@dataclasses.dataclass
class PoolStats:
    # For full pools (required)
    full_num_used: int
    full_token_usage: float
    full_available_size: int
    full_evictable_size: int

    is_hybrid_swa: bool = False
    is_hybrid_ssm: bool = False
    is_hisparse: bool = False

    # For hybrid-swa pools
    swa_num_used: Optional[int] = None
    swa_token_usage: Optional[float] = None
    swa_available_size: Optional[int] = None
    swa_evictable_size: Optional[int] = None

    # For mamba pools
    mamba_num_used: Optional[int] = None
    mamba_usage: Optional[float] = None
    mamba_available_size: Optional[int] = None
    mamba_evictable_size: Optional[int] = None

    # HiSparse device/host breakdown for decode logs (plain KV pool only)
    hisparse_device_tokens: Optional[int] = None
    hisparse_device_token_usage: Optional[float] = None
    hisparse_host_tokens: Optional[int] = None
    hisparse_host_token_usage: Optional[float] = None

    def get_kv_token_stats(self) -> Tuple[int, float]:
        # NOTE: mamba pool is not included in the "token usage" calculation.
        if self.is_hybrid_swa:
            num_used = max(self.full_num_used, self.swa_num_used)
            token_usage = max(self.full_token_usage, self.swa_token_usage)
        else:
            num_used = self.full_num_used
            token_usage = self.full_token_usage

        return num_used, token_usage

    def get_max_pool_usage(self) -> float:
        usage = self.full_token_usage
        if self.is_hybrid_swa:
            usage = max(usage, self.swa_token_usage)
        if self.is_hybrid_ssm:
            usage = max(usage, self.mamba_usage)
        assert usage is not None and usage >= 0, f"{usage=} is not valid"
        return usage

    def get_prefill_usage_msg_parts(self) -> List[str]:
        parts = []
        if self.is_hybrid_swa:
            parts += [
                f"full token usage: {self.full_token_usage:.2f}",
                f"swa token usage: {self.swa_token_usage:.2f}",
            ]
        if self.is_hybrid_ssm:
            if not self.is_hybrid_swa:
                parts.append(f"full token usage: {self.full_token_usage:.2f}")
            parts.append(f"mamba usage: {self.mamba_usage:.2f}")
        if not parts:
            parts.append(f"token usage: {self.full_token_usage:.2f}")
        return parts

    def get_decode_usage_msg_parts(self) -> List[str]:
        parts = []
        if self.is_hybrid_swa:
            parts += [
                f"#full token: {self.full_num_used}",
                f"full token usage: {self.full_token_usage:.2f}",
                f"#swa token: {self.swa_num_used}",
                f"swa token usage: {self.swa_token_usage:.2f}",
            ]
        if self.is_hybrid_ssm:
            if not self.is_hybrid_swa:
                parts += [
                    f"#full token: {self.full_num_used}",
                    f"full token usage: {self.full_token_usage:.2f}",
                ]
            parts += [
                f"mamba num: {self.mamba_num_used}",
                f"mamba usage: {self.mamba_usage:.2f}",
            ]
        if self.is_hisparse:
            parts += [
                f"#gpu token: {self.hisparse_device_tokens}",
                f"gpu token usage: {self.hisparse_device_token_usage:.2f}",
                f"#cpu token: {self.hisparse_host_tokens}",
                f"cpu token usage: {self.hisparse_host_token_usage:.2f}",
            ]
        if not parts:
            parts.append(
                f"#token: {self.full_num_used}, token usage: {self.full_token_usage:.2f}"
            )
        return parts

    def update_scheduler_stats(self, stats: SchedulerStats) -> None:
        """Update pool-related fields on SchedulerStats."""
        num_used, _ = self.get_kv_token_stats()
        stats.num_used_tokens = num_used
        stats.token_usage = round(self.get_max_pool_usage(), 2)
        stats.full_token_usage = self.full_token_usage
        if self.is_hybrid_swa:
            stats.swa_token_usage = self.swa_token_usage
            stats.swa_available_tokens = self.swa_available_size
            stats.swa_evictable_tokens = self.swa_evictable_size
            stats.swa_used_tokens = self.swa_num_used
        if self.is_hybrid_ssm:
            stats.mamba_usage = self.mamba_usage
            stats.mamba_available_tokens = self.mamba_available_size
            stats.mamba_evictable_tokens = self.mamba_evictable_size
            stats.mamba_used_tokens = self.mamba_num_used
        stats.kv_available_tokens = self.full_available_size
        stats.kv_evictable_tokens = self.full_evictable_size
        stats.kv_used_tokens = self.full_num_used
