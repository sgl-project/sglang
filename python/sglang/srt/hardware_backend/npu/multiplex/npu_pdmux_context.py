import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import yaml


@dataclass
class PDMuxConfig:
    """Configuration for NPU resource multiplexing."""

    sm_group_num: int = 8
    manual_divisions: List[List[int]] = field(default_factory=list)
    split_forward_token_budget: int = 65536
    decode_bs_divisor: int = 36
    total_cubes: Optional[int] = None


class _NPUMuxManagerImpl:
    def __init__(self, npu_id: int, config: PDMuxConfig):
        self.npu_id = npu_id
        self.config = config
        self.stream_groups: List[Tuple[torch.npu.Stream, torch.npu.Stream]] = []
        self.cube_counts: List[Tuple[int, int]] = []  # (prefill_cubes, decode_cubes)
        self.current_idx: int = 0

        self._initialize()
        self._initialized = True

    def _get_total_cubes(self) -> int:
        if self.config.total_cubes is not None:
            return self.config.total_cubes

        props = torch.npu.get_device_properties(self.npu_id)
        return props.cube_core_num

    @staticmethod
    def _divide_cubes(total_cubes: int, groups: int) -> List[Tuple[int, int]]:
        min_decode = 2
        if total_cubes - min_decode < min_decode:
            raise ValueError(f"Total cubes {total_cubes} too small for partitioning.")

        possible = []
        for prefill in range(total_cubes - min_decode, total_cubes // 2, -2):
            decode = total_cubes - prefill
            if decode >= min_decode and prefill >= decode:
                possible.append((prefill, decode))

        if not possible:
            raise ValueError(f"No valid partition for total_cubes={total_cubes}")

        if len(possible) >= groups:
            step = max(1, len(possible) // groups)
            selected = possible[::step][:groups]
        else:
            selected = possible

        selected.sort(reverse=True)
        return selected

    def _create_limited_stream(self, cube_num: int) -> torch.npu.Stream:
        stream = torch.npu.Stream(device=self.npu_id)
        if cube_num > 0:
            torch.npu.set_stream_limit(
                stream, cube_num=cube_num, vector_num=cube_num * 2
            )
        return stream

    def _initialize(self):
        total_cubes = self._get_total_cubes()

        if self.config.manual_divisions:
            divisions = [
                (prefill, decode) for prefill, decode, _ in self.config.manual_divisions
            ]

            for prefill, decode in divisions:
                if prefill + decode != total_cubes:
                    raise ValueError(
                        f"Manual division ({prefill}, {decode}) sum != total_cubes {total_cubes}"
                    )
        else:
            divisions = self._divide_cubes(total_cubes, self.config.sm_group_num - 2)

        self.cube_counts.append((total_cubes, 0))
        self.stream_groups.append(
            (torch.npu.Stream(device=self.npu_id), torch.npu.Stream(device=self.npu_id))
        )

        for prefill_cubes, decode_cubes in divisions:
            self.cube_counts.append((prefill_cubes, decode_cubes))
            prefill_stream = self._create_limited_stream(prefill_cubes)
            decode_stream = self._create_limited_stream(decode_cubes)
            self.stream_groups.append((prefill_stream, decode_stream))

        self.cube_counts.append((0, total_cubes))
        self.stream_groups.append(
            (torch.npu.Stream(device=self.npu_id), torch.npu.Stream(device=self.npu_id))
        )

        self.current_idx = 0

    @property
    def current_stream_group(self) -> Tuple[torch.npu.Stream, torch.npu.Stream]:
        return self.stream_groups[self.current_idx]

    @property
    def current_cube_counts(self) -> Tuple[int, int]:
        return self.cube_counts[self.current_idx]

    def set_current_stream_idx(self, idx: int):
        if idx < 0 or idx >= len(self.stream_groups):
            raise ValueError(f"Invalid stream index: {idx}")
        self.current_idx = idx

    def get_stream_groups(self) -> List[Tuple[torch.npu.Stream, torch.npu.Stream]]:
        return self.stream_groups

    def get_cube_counts(self) -> List[Tuple[int, int]]:
        return self.cube_counts

    def get_current_stream_idx(self) -> int:
        return self.current_idx


_manager: Optional[_NPUMuxManagerImpl] = None
_lock = threading.Lock()


def get_npu_pdmux_manager(
    npu_id: Optional[int] = None,
    config: Optional["PDMuxConfig"] = None,
) -> _NPUMuxManagerImpl:
    global _manager

    if _manager is not None:
        return _manager

    with _lock:
        if _manager is None:
            if npu_id is None or config is None:
                raise ValueError(
                    "First call to get_npu_pdmux_manager must provide npu_id and config."
                )
            _manager = _NPUMuxManagerImpl(npu_id, config)
    return _manager


def load_npu_pdmux_config(config_path: str) -> PDMuxConfig:
    if not config_path:
        return PDMuxConfig()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if "sm_group_num" not in raw:
        raise ValueError("Missing required field: sm_group_num")
    if raw["sm_group_num"] < 3:
        raise ValueError("sm_group_num must be at least 3")

    manual_divisions = raw.get("manual_divisions", [])
    expected = raw["sm_group_num"] - 2
    if manual_divisions and len(manual_divisions) != expected:
        raise ValueError(
            f"manual_divisions must have {expected} entries, "
            f"but got {len(manual_divisions)}"
        )

    return PDMuxConfig(
        sm_group_num=raw["sm_group_num"],
        manual_divisions=manual_divisions,
        split_forward_token_budget=raw.get("split_forward_token_budget", 65536),
        decode_bs_divisor=raw.get("decode_bs_divisor", 36),
        total_cubes=raw.get("total_cubes"),
    )
