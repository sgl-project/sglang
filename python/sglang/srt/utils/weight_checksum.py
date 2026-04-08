import hashlib

import torch
from torch.distributed.tensor import DTensor


def compute_weights_checksum(named_params):
    """Compute a single SHA-256 hash over all weights, sorted by name for determinism."""
    hasher = hashlib.sha256()
    for name, tensor in sorted(named_params, key=lambda x: x[0]):
        hasher.update(name.encode())
        t = tensor.detach()
        # DTensor doesn't support .numpy(); extract the local tensor.
        if isinstance(t, DTensor):
            t = t._local_tensor
        hasher.update(t.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().data)
    return hasher.hexdigest()
