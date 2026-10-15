"""Raw row storage; model hashing, quantization and collectives stay in SGLang."""

import contextlib
import logging

import torch

logger = logging.getLogger(__name__)


class ByteRows:
    """Publish a CPU table after weight loading, then gather on the caller's stream.

    Each instance owns an anonymous local snapshot, including across TP ranks.
    Initialize during eager warmup before capturing a CUDA graph. The source
    parameters remain available to the normal checkpoint loader.
    """

    def __init__(self):
        self.store = None

    def initialize(self, rows: torch.Tensor):
        from mooncake.engram import EngramStore

        from sglang.srt.runtime_context import get_memory

        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Warm up Mooncake Engram tables before CUDA graph capture"
            )
        if rows.device.type != "cpu" or rows.dtype != torch.uint8 or rows.ndim != 2:
            raise ValueError(
                "Mooncake Engram initialization requires a CPU byte matrix"
            )
        self.row_bytes = rows.shape[1]
        self.store = EngramStore(
            tables={"rows": {"num_rows": rows.shape[0], "row_bytes": self.row_bytes}},
            namespace="sglang-v1",
            local_dir=get_memory().engram_local_dir,
            persistent=False,
        )
        self.store.populate("rows", rows.contiguous().numpy())
        logger.info("Mooncake Engram: published %s rows of %s bytes", *rows.shape)

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        indices = ids.reshape(-1, 1).contiguous()
        output = torch.zeros(
            (indices.shape[0], 1, self.row_bytes), dtype=torch.uint8, device=ids.device
        )
        if ids.is_cuda:
            self.store.lookup(
                ["rows"],
                indices,
                output,
                stream=torch.cuda.current_stream(ids.device).cuda_stream,
            )
        else:
            self.store.lookup(["rows"], indices.numpy(), output.numpy())
        return output.view(*ids.shape, self.row_bytes)

    def close(self):
        if self.store is not None:
            self.store.close()
            self.store = None

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()


def init_deepseek(module, block_size: int):
    module.weight = torch.nn.Parameter(
        torch.empty(module.rows, module.dim, dtype=torch.float8_e4m3fn, device="cpu"),
        requires_grad=False,
    )
    module.scale = torch.nn.Parameter(
        torch.empty(
            module.rows,
            module.dim // block_size,
            dtype=torch.float8_e8m0fnu,
            device="cpu",
        ),
        requires_grad=False,
    )
    module._mooncake = ByteRows()


def lookup_deepseek(module, indices: torch.Tensor, block_size: int):
    table = module._mooncake
    if table.store is None:
        table.initialize(
            torch.cat(
                [
                    module.weight.detach().view(torch.uint8),
                    module.scale.detach().view(torch.uint8),
                ],
                dim=1,
            )
        )
    local = indices - module.row_start
    owned = (local >= 0) & (local < module.rows)
    raw = table.lookup(local.masked_fill(~owned, -1))
    weight = raw[..., : module.dim].contiguous().view(torch.float8_e4m3fn)
    scale = raw[..., module.dim :].contiguous().view(torch.float8_e8m0fnu)
    values = weight.float().unflatten(-1, (-1, block_size)) * scale.float().unsqueeze(
        -1
    )
    return values.flatten(-2).to(torch.bfloat16).masked_fill(~owned.unsqueeze(-1), 0)
