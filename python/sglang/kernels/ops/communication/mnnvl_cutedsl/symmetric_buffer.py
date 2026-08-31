# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Typed ownership for one rendezvoused symmetric Tensor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Upstream reads this helper through a package-relative import; every
# FlashInfer this tree supports ships it, so only the spelling differs here.
from flashinfer.comm.torch_symmetric_memory import _enable_symm_mem_for_group


@dataclass(frozen=True, slots=True)
class SymmetricBuffer:
    """A symmetric Tensor and the mapping resources derived at rendezvous."""

    tensor: torch.Tensor
    # Keep the rendezvous mapping alive without exposing the backend handle as
    # part of a Kernel State's public surface.
    _handle: object = field(repr=False)
    multicast_address: int | None = field(default=None, repr=False)
    peer_addresses: torch.Tensor | None = field(default=None, repr=False)

    @classmethod
    def allocate(
        cls,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        group: dist.ProcessGroup,
        *,
        require_multicast: bool = False,
        materialize_peer_addresses: bool = False,
    ) -> SymmetricBuffer:
        """Allocate with the current SymmMem backend and verify requested mappings."""
        if symm_mem.get_backend(device) is None:
            raise RuntimeError(
                "PyTorch Symmetric Memory has no backend for the current device"
            )
        _enable_symm_mem_for_group(group.group_name)
        return cls.rendezvous(
            symm_mem.empty(shape, dtype=dtype, device=device),
            group,
            require_multicast=require_multicast,
            materialize_peer_addresses=materialize_peer_addresses,
        )

    @classmethod
    def rendezvous(
        cls,
        tensor: torch.Tensor,
        group: dist.ProcessGroup,
        *,
        require_multicast: bool = False,
        materialize_peer_addresses: bool = False,
    ) -> SymmetricBuffer:
        _enable_symm_mem_for_group(group.group_name)
        handle = symm_mem.rendezvous(tensor, group)
        multicast_address = None
        if require_multicast:
            multicast_address = int(handle.multicast_ptr or 0)
            if not multicast_address:
                raise RuntimeError("NVLink multicast mapping is unavailable")

        peer_addresses = None
        if materialize_peer_addresses:
            # Preserve the rendezvous offset for SymmMem Pool suballocations.
            addresses = [
                handle.get_remote_tensor(
                    peer,
                    tensor.shape,
                    tensor.dtype,
                ).data_ptr()
                for peer in range(dist.get_world_size(group))
            ]
            if any(not address for address in addresses):
                raise RuntimeError("Symmetric peer mapping is unavailable")
            peer_addresses = torch.tensor(
                addresses,
                dtype=torch.int64,
                device=tensor.device,
            )

        return cls(
            tensor=tensor,
            _handle=handle,
            multicast_address=multicast_address,
            peer_addresses=peer_addresses,
        )
