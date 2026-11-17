"""The output store a ``/rollout/generate`` request puts its tensors in when it asks to."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from radix_transfer_plugins import OutputStore, StoreRef
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

logger = init_logger(__name__)


class RolloutOutputStore:
    def __init__(self) -> None:
        self._store: OutputStore | None = None

    @property
    def enabled(self) -> bool:
        return self._store is not None

    def initialize(self, server_args: ServerArgs) -> None:
        if server_args.output_store_backend == "none":
            return
        from radix_transfer_plugins import OutputStore, create_backend

        self._store = OutputStore(
            backend=create_backend(
                name=server_args.output_store_backend,
                extra_config=server_args.output_store_backend_extra_config,
            )
        )

    def put(self, tensors: dict[str, Any]) -> StoreRef:
        from radix_transfer_plugins.stashes.tensor_stash import TensorStash

        (ref,) = self._store.put_many([TensorStash(tensors)])
        return ref

    def delete(self, ref: StoreRef) -> None:
        try:
            self._store.delete_many([ref])
        except Exception:
            logger.error(
                "Failed to delete an output store object; ref=%s",
                json.dumps(ref.to_builtins()),
                exc_info=True,
            )

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None


rollout_output_store = RolloutOutputStore()
