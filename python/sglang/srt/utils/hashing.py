"""
Hash utility functions for prefix caching.

This module provides configurable hashing algorithms for prefix caching.
It supports SHA-256 (cryptographic, default) and xxHash (non-cryptographic, faster).
"""

import hashlib
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Try to import xxhash, but make it optional
try:
    import xxhash

    _XXHASH_AVAILABLE = True
    # Check if xxh3_128_digest is available (for compatibility with older versions)
    _XXHASH_V3_AVAILABLE = hasattr(xxhash, "xxh3_128") and hasattr(
        xxhash.xxh3_128(), "digest"
    )
    # Check if xxh3_64 with intdigest is available
    _XXHASH_V3_64_AVAILABLE = hasattr(xxhash, "xxh3_64") and hasattr(
        xxhash.xxh3_64(), "intdigest"
    )
except ImportError:
    _XXHASH_AVAILABLE = False
    _XXHASH_V3_AVAILABLE = False
    _XXHASH_V3_64_AVAILABLE = False
    xxhash = None  # type: ignore


class HashAlgorithm:
    """Hash algorithm options for prefix caching."""

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


def get_hash_str(
    token_ids: List[int],
    prior_hash: Optional[str] = None,
    algorithm: str = HashAlgorithm.SHA256,
) -> str:
    """Compute hash string for token IDs.

    Args:
        token_ids: List of token IDs to hash
        prior_hash: Optional prior hash value for chaining (hex string)
        algorithm: Hash algorithm to use ('sha256' or 'xxhash')

    Returns:
        Hexadecimal hash string

    Raises:
        ValueError: If the requested algorithm is not available
    """
    if algorithm == HashAlgorithm.SHA256:
        return _get_hash_str_sha256(token_ids, prior_hash)
    elif algorithm == HashAlgorithm.XXHASH:
        if not _XXHASH_AVAILABLE:
            raise ValueError(
                "xxhash is not available. Please install it with: pip install xxhash"
            )
        if not _XXHASH_V3_AVAILABLE:
            raise ValueError(
                "xxhash version is too old. Please upgrade to xxhash>=3.0.0"
            )
        return _get_hash_str_xxhash(token_ids, prior_hash)
    else:
        raise ValueError(
            f"Unknown hash algorithm: {algorithm}. "
            f"Available options: {HashAlgorithm.choices()}"
        )


def _get_hash_str_sha256(token_ids: List[int], prior_hash: Optional[str] = None) -> str:
    """Compute SHA-256 hash string for token IDs."""
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


def _get_hash_str_xxhash(token_ids: List[int], prior_hash: Optional[str] = None) -> str:
    """Compute xxHash hash string for token IDs.

    Uses xxh3_128 for 128-bit output to match SHA-256 output length.
    """
    if prior_hash:
        # For chaining, we need to incorporate the prior hash.
        # Convert prior hash hex string to bytes and use as seed
        prior_bytes = bytes.fromhex(prior_hash)
        # Use first 8 bytes as seed (xxhash uses 64-bit seed)
        seed = int.from_bytes(prior_bytes[:8], byteorder="little")
        hasher = xxhash.xxh3_128(seed=seed)
        # Update with remaining bytes if any
        if len(prior_bytes) > 8:
            hasher.update(prior_bytes[8:])
    else:
        hasher = xxhash.xxh3_128()

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
    Works with both SHA-256 and xxHash outputs.
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
