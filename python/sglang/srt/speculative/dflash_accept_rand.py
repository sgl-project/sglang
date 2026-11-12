# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_DFLASH_ACCEPT_RAND_HOIST: the fused selector accept's two uniform draws, drawn before the target verify.

The accept draws torch.rand (bs, block - 1) then (bs,) after the verify, on the main stream, where the two kernels sit on
the round's critical path. The values depend only on the default CUDA generator's seed and offset at the host call, so
drawing them on a side stream before the verify launch, after the draft graph's replay has advanced the offset, gives
the same bits as long as nothing consumes the generator in between (the spec verify does not sample). take() checks
that the offset has not moved since the draw and raises otherwise.
"""

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class AcceptRandHoist:
    def __init__(self, device):
        device = torch.device(device)
        index = device.index if device.index is not None else torch.cuda.current_device()
        self.device = torch.device("cuda", index)
        self.stream = torch.cuda.Stream(device=self.device)
        self.generator = torch.cuda.default_generators[index]
        self.taken = 0
        self._pending = None

    def _offset(self) -> int:
        return self.generator.get_offset()

    def pending(self) -> bool:
        return self._pending is not None

    def draw(self, bs: int, block: int) -> None:
        """Draws the accept's uniforms on the side stream; the main stream waits for them from here on."""
        if self._pending is not None:
            raise RuntimeError("DFLASH accept rand hoist: the previous round's draw was never taken")
        main = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self.stream):
            # The order and shapes of _selector_fused_accept's draws.
            uniform_samples = torch.rand((bs, block - 1), dtype=torch.float32, device=self.device)
            uniform_samples_final = torch.rand((bs,), dtype=torch.float32, device=self.device)
        uniform_samples.record_stream(main)
        uniform_samples_final.record_stream(main)
        main.wait_stream(self.stream)
        self._pending = (bs, block, uniform_samples, uniform_samples_final, self._offset())

    def take(self, bs: int, block: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        pending, self._pending = self._pending, None
        if pending is None:
            return None
        drawn_bs, drawn_block, uniform_samples, uniform_samples_final, offset = pending
        if (drawn_bs, drawn_block) != (bs, block):
            raise RuntimeError(
                f"DFLASH accept rand hoist: drew ({drawn_bs}, {drawn_block}), the accept needs ({bs}, {block})"
            )
        if self._offset() != offset:
            raise RuntimeError(
                "DFLASH accept rand hoist: the CUDA generator was consumed between the hoisted draw and the accept"
            )
        if self.taken == 0:
            logger.info("DFLASH accept rand hoist engaged (uniforms drawn before the verify)")
        self.taken += 1
        return uniform_samples, uniform_samples_final
