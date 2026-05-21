"""Hand-computed Python re-implementation of the real-kv-source fold, kept independent from
``verify_ref._splitmix64_fold_bytes_scalar`` so a ref / kernel co-regression cannot silently fix the
diff comparison."""

from __future__ import annotations

from sglang.jit_kernel.kv_canary.consts import splitmix64


def _hand_fold_partial(raw_bytes: bytes) -> int:
    """PARTIAL-mode fold: first min(16, len) bytes, little-endian word-pack + splitmix64, same as ALL."""
    _u64 = (1 << 64) - 1

    truncated = raw_bytes[: min(16, len(raw_bytes))]
    pad = (8 - len(truncated) % 8) % 8
    padded = bytes(truncated) + bytes(pad)
    num_words = len(padded) // 8
    acc = 0
    for w in range(num_words):
        chunk = padded[w * 8 : (w + 1) * 8]
        word = sum(b << (8 * k) for k, b in enumerate(chunk))
        acc = splitmix64(acc ^ word)
    source_hash = acc
    return splitmix64(0 ^ source_hash)


def _hand_fold_all(raw_bytes: bytes) -> int:
    """ALL-mode fold: pack bytes little-endian into 8-byte words, fold each via splitmix64, then mix into acc=0."""
    _u64 = (1 << 64) - 1

    pad = (8 - len(raw_bytes) % 8) % 8
    padded = raw_bytes + bytes(pad)
    num_words = len(padded) // 8
    acc = 0
    for w in range(num_words):
        chunk = padded[w * 8 : (w + 1) * 8]
        word = sum(b << (8 * k) for k, b in enumerate(chunk))
        acc = splitmix64(acc ^ word)
    source_hash = acc
    return splitmix64(0 ^ source_hash)
