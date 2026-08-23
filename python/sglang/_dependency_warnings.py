"""Filters for known, non-actionable dependency warnings."""

import logging


class _MessageFilter(logging.Filter):
    def __init__(self, *fragments: str):
        super().__init__()
        self.fragments = fragments

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not all(fragment in message for fragment in self.fragments)


def _install_message_filter(logger_name: str, *fragments: str) -> None:
    logger = logging.getLogger(logger_name)
    if any(
        isinstance(filter_, _MessageFilter) and filter_.fragments == fragments
        for filter_ in logger.filters
    ):
        return
    logger.addFilter(_MessageFilter(*fragments))


def suppress_known_dependency_warnings() -> None:
    """Suppress dependency warnings that do not require an SGLang action."""
    _install_message_filter(
        "torch.utils._pytree",
        "is an Enum subclass and is now natively supported by torch.compile",
        "Calling register_constant() on Enum subclasses is deprecated",
    )
    _install_message_filter(
        "diffusers.quantizers.torchao.torchao_quantizer",
        "Unable to import `torchao` Tensor objects",
        "loading checkpoints serialized with `torchao`",
    )
