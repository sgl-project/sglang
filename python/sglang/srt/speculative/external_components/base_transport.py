"""Transport interface for external speculative decoding."""

from abc import ABC, abstractmethod

from sglang.srt.speculative.external_components.contract import (
    DraftContext,
    PrefillContext,
)


class ExternalDraftTransport(ABC):
    """Abstract transport between the ExternalWorker and an external drafter."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def configure_draft_engine(
        self, *, num_draft_tokens: int, protocol_version: int = 1
    ) -> None: ...

    @abstractmethod
    def prefill_cache(self, *, prefill_contexts: list[PrefillContext]) -> None: ...

    @abstractmethod
    def request_drafts(self, *, draft_contexts: list[DraftContext]) -> None: ...

    @abstractmethod
    def receive_drafts(
        self, *, request_ids: list[int], num_draft_tokens: int
    ) -> dict[int, list[int] | None]: ...

    @abstractmethod
    def cancel_inference(self, *, request_ids: list[int]) -> None:
        """Fire and forget; receive_drafts must not wait on a cancelled request
        and must drop a late reply for one."""
