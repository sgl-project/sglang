"""Fixed request-window backing for the static packed-page adapter."""

import torch

from sglang.srt.mem_cache.dsv41_request_window import (
    PackedWindowView,
    RequestWindow,
    RequestWindowSpec,
)
from sglang.srt.mem_cache.kv.decl import TensorPartDecl
from sglang.srt.mem_cache.kv.memory.fixed import FixedMemory


class RequestWindowMemory:
    def __init__(
        self, spec: RequestWindowSpec, *, dtype, device, enable_memory_saver=False
    ):
        self.spec = spec
        page_bytes = spec.kv_layout.page_bytes(spec.page_size)
        layers = tuple(range(spec.layers))
        options = dict(device=device, enable_memory_saver=enable_memory_saver)
        self.state = FixedMemory(
            (
                TensorPartDecl(
                    name="pages",
                    shape=(page_bytes,),
                    dtype=torch.uint8,
                    layer_ids=layers,
                ),
            ),
            rows=spec.state_pages,
            **options,
        )
        self.tags = FixedMemory(
            (
                TensorPartDecl(
                    name="tags",
                    shape=(),
                    dtype=torch.int64,
                    layer_ids=layers,
                    reset_value=-1,
                ),
            ),
            rows=spec.tag_rows,
            **options,
        )
        self.workspace = FixedMemory(
            (TensorPartDecl(name="workspace", shape=(page_bytes,), dtype=torch.uint8),),
            rows=spec.workspace_pages,
            **options,
        )
        self.view = RequestWindow(
            spec,
            state=PackedWindowView(
                size=spec.num_slots * spec.capacity + spec.page_size,
                page_size=spec.page_size,
                kv_layout=spec.kv_layout,
                dtype=dtype,
                kv_buffer=tuple(self.state.tensors["pages"].unbind(0)),
            ),
            tags=self.tags.tensors["tags"],
            workspace=PackedWindowView(
                size=spec.workspace_rows,
                page_size=spec.page_size,
                kv_layout=spec.kv_layout,
                dtype=dtype,
                kv_buffer=(self.workspace.tensors["workspace"],),
            ),
        )
        assert self.nbytes == spec.nbytes

    @property
    def nbytes(self):
        return self.state.nbytes + self.tags.nbytes + self.workspace.nbytes

    def reset_rows(self, rows):
        indices = torch.as_tensor(rows, dtype=torch.int64, device=self.view.tags.device)
        self.view.reset(indices)
