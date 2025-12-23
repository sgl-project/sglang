"""
Hash utility functions for prefix caching and multimodal features.

This module provides hashing functions for prefix caching (SHA-256) and
multimodal feature hashing (supports SHA-256 and xxHash).
"""

import hashlib
import logging

logger = logging.getLogger(__name__)

# Try to import xxhash, but make it optional (used only for multimodal features)
try:
    import xxhash

    _XXHASH_AVAILABLE = True
    # Check if xxh3_64 with intdigest is available
    _XXHASH_V3_64_AVAILABLE = hasattr(xxhash, "xxh3_64") and hasattr(
        xxhash.xxh3_64(), "intdigest"
    )
except ImportError:
    _XXHASH_AVAILABLE = False
    _XXHASH_V3_64_AVAILABLE = False
    xxhash = None  # type: ignore


class HashAlgorithm:
    """Hash algorithm options for multimodal features."""

    SHA256 = "sha256"
    XXHASH = "xxhash"

    @classmethod
    def choices(cls):
        """Return available hash algorithm choices."""
        choices = [cls.SHA256]
        if _XXHASH_AVAILABLE:
            choices.append(cls.XXHASH)
        return choices

    @classmethod
    def is_available(cls, algo: str) -> bool:
        """Check if a hash algorithm is available."""
        if algo == cls.SHA256:
            return True
        elif algo == cls.XXHASH:
            return _XXHASH_AVAILABLE
        return False


def hash_bytes_to_int64(data: bytes, algorithm: str = HashAlgorithm.SHA256) -> int:
    """Hash bytes data and return as 64-bit unsigned integer.

    Args:
        data: Bytes data to hash
        algorithm: Hash algorithm to use ('sha256' or 'xxhash')

    Returns:
        64-bit unsigned integer from first 8 bytes of hash digest
    """
    if algorithm == HashAlgorithm.SHA256:
        hash_bytes = hashlib.sha256(data).digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big", signed=False)
    elif algorithm == HashAlgorithm.XXHASH:
        if not _XXHASH_AVAILABLE:
            raise ValueError(
                "xxhash is not available. Please install it with: pip install xxhash"
            )
        if not _XXHASH_V3_64_AVAILABLE:
            raise ValueError(
                "xxhash version is too old. Please upgrade to xxhash>=3.0.0"
            )
        # Use xxh3_64 for 64-bit output, which is perfect for int64
        hasher = xxhash.xxh3_64()
        hasher.update(data)
        # xxh3_64.intdigest() returns 64-bit unsigned integer directly
        return hasher.intdigest()
    else:
        raise ValueError(
            f"Unknown hash algorithm: {algorithm}. "
            f"Available options: {HashAlgorithm.choices()}"
        )
