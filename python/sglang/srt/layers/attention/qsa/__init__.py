"""Simple QSA operators for Qwen4-Exp.

The package intentionally avoids eager imports so reference tensor helpers can
be used without constructing the full SGLang runtime.
"""

__all__ = [
    "QSAIndexer",
    "QSAIndexerMetadata",
    "QSAProfile",
    "build_qsa_indexer",
    "get_qsa_indexer_metadata",
    "is_qwen_qsa",
    "parse_qsa_profile",
]


def __getattr__(name):
    if name == "QSAIndexer":
        from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

        return QSAIndexer
    if name == "QSAIndexerMetadata":
        from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata

        return QSAIndexerMetadata
    if name in {"QSAProfile", "is_qwen_qsa", "parse_qsa_profile"}:
        from sglang.srt.layers.attention.qsa import config as qsa_config

        return getattr(qsa_config, name)
    if name in {"build_qsa_indexer", "get_qsa_indexer_metadata"}:
        from sglang.srt.layers.attention.qsa import glue as qsa_glue

        return getattr(qsa_glue, name)
    raise AttributeError(name)
