"""
Hash utility functions for prefix caching and multimodal features.

This module provides hashing functions for prefix caching (SHA-256) and
multimodal feature hashing (supports SHA-256 and xxHash).
"""

import hashlib
import logging
from typing import List, Optional

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


def get_hash_str(token_ids: List[int], prior_hash: Optional[str] = None) -> str:
    """Compute SHA-256 hash string for token IDs (used for prefix caching).

    Args:
        token_ids: List of token IDs to hash
        prior_hash: Optional prior hash value for chaining (hex string)

    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        if isinstance(t, tuple):
            # EAGLE bigram mode: hash both elements to uniquely identify the bigram
            for elem in t:
                hasher.update(elem.to_bytes(4, byteorder="little", signed=False))
        else:
            # Regular mode: single integer token
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


def hash_str_to_int64(hash_str: str) -> int:
    """Convert hash hex string to signed 64-bit integer for events.

    Takes first 16 hex characters (64 bits) and converts to signed int64 range.
    """
    # Take first 16 hex chars to get 64-bit value
    uint64_val = int(hash_str[:16], 16)
    # Convert to signed int64 range [-2^63, 2^63-1]
    if uint64_val >= 2**63:
        return uint64_val - 2**64
    return uint64_val


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
