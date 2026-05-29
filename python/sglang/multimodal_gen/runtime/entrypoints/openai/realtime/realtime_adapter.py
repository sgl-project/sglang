# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoInitRequest,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_generate_session import (
        RealtimeGenerateSession,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
        Req,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


class RealtimeModelAdapter(Protocol):
    name: str

    def create_state(self) -> Any: ...

    async def on_init(
        self,
        session: RealtimeGenerateSession,
        request: RealtimeVideoInitRequest,
    ) -> None: ...

    def ingest_event(
        self,
        session: RealtimeGenerateSession,
        event: RealtimeEvent,
    ) -> str: ...

    def build_sampling_params(self, session: RealtimeGenerateSession): ...

    def prepare_request(self, session: RealtimeGenerateSession, batch: Req) -> Req: ...

    def on_chunk_complete(
        self, session: RealtimeGenerateSession, result: OutputBatch
    ) -> None: ...

    def dispose(self, session: RealtimeGenerateSession) -> None: ...


_REALTIME_ADAPTER_REGISTRY: dict[type, type[RealtimeModelAdapter]] = {}


def register_realtime_model_adapter(
    pipeline_config_cls: type,
    adapter_cls: type[RealtimeModelAdapter],
) -> None:
    _REALTIME_ADAPTER_REGISTRY[pipeline_config_cls] = adapter_cls


def get_realtime_model_adapter(
    server_args: ServerArgs,
) -> RealtimeModelAdapter:
    pipeline_config = server_args.pipeline_config
    for config_cls in type(pipeline_config).__mro__:
        adapter_cls = _REALTIME_ADAPTER_REGISTRY.get(config_cls)
        if adapter_cls is not None:
            return adapter_cls()

    raise ValueError(
        "Realtime video is not supported for pipeline config "
        f"{type(pipeline_config).__name__}; no realtime adapter is registered."
    )
