"""End-to-end MLX model runner for Apple Silicon.

Runs the entire model within MLX, bypassing PyTorch MPS entirely.
"""

import logging
import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import load as mlx_lm_load
from mlx_lm.models.cache import (
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
    make_prompt_cache,
)

logger = logging.getLogger(__name__)


@dataclass
class MlxRequestState:
    """Per-request state for MLX inference."""

    token_ids: list[int]
    cache: list  # List of KVCache per layer
    generated_tokens: int = 0


def _merge_kv_caches(
    caches_list: list[list],
) -> list:
    """Merge multiple per-request caches into batched caches."""
    if not caches_list:
        return []

    num_layers = len(caches_list[0])
    merged = []

    for layer_idx in range(num_layers):
        layer_caches = [caches[layer_idx] for caches in caches_list]
        if isinstance(layer_caches[0], KVCache):
            batch_cache = BatchKVCache.merge(layer_caches)
        elif isinstance(layer_caches[0], RotatingKVCache):
            batch_cache = BatchRotatingKVCache.merge(layer_caches)
        else:
            raise TypeError(f"Unsupported cache type: {type(layer_caches[0]).__name__}")
        merged.append(batch_cache)

    return merged


def _extract_kv_cache(batch_caches: list, idx: int) -> list:
    """Extract a single request's cache from batched caches.

    Works with both BatchKVCache (has .extract) and plain KVCache
    populated with batched data of shape (B, H, L, D).
    """
    extracted = []
    for cache in batch_caches:
        if hasattr(cache, "extract"):
            extracted.append(cache.extract(idx))
        else:
            # Plain KVCache with batched data — slice along batch dim
            new_cache = KVCache()
            new_cache.keys = mx.contiguous(cache.keys[idx : idx + 1])
            new_cache.values = mx.contiguous(cache.values[idx : idx + 1])
            new_cache.offset = cache.offset
            extracted.append(new_cache)
    return extracted


class MlxModelRunner:
    """Model runner that executes the entire model in MLX.

    This avoids the MPS<->MLX tensor bridge overhead by keeping all
    computation within MLX.
    """

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = False,
    ):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.model = None
        self._request_states: dict[str, MlxRequestState] = {}

        self._load_model()

    @staticmethod
    def _extract_logits(model_output):
        """Extract logits from model output, handling both tuple and direct returns."""
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def _load_model(self):
        """Load model using mlx_lm."""
        logger.info(f"Loading MLX model: {self.model_path}")
        start_time = time.time()

        self.model, _ = mlx_lm_load(
            self.model_path,
            tokenizer_config={"trust_remote_code": self.trust_remote_code},
        )

        load_time = time.time() - start_time
        logger.info(f"MLX model loaded in {load_time:.2f}s")

    def prefill(
        self,
        req_id: str,
        token_ids: list[int],
    ) -> int:
        """Run prefill for a single request.

        If a request with the same req_id already has state (e.g. from a
        previous partial prefill), the existing KV cache is reused and only
        the new tokens are fed through the model.

        Args:
            req_id: Request identifier
            token_ids: Input token IDs (full sequence, including any
                previously prefilled tokens)

        Returns:
            Next token ID (greedy sampled)
        """
        existing_state = self._request_states.get(req_id)
        if existing_state is not None:
            # Continuation: reuse existing cache, feed only new tokens
            cached_input_len = (
                len(existing_state.token_ids) - existing_state.generated_tokens
            )
            new_tokens = token_ids[cached_input_len:]
            cache = existing_state.cache
        else:
            new_tokens = token_ids
            cache = make_prompt_cache(self.model)

        input_ids = mx.array([new_tokens], dtype=mx.int32)
        model_output = self.model(input_ids, cache=cache)

        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        next_token_mlx = mx.argmax(last_logits, axis=-1)

        # Evaluate everything together
        mx.eval(next_token_mlx, *[c.state for c in cache])
        next_token = int(next_token_mlx.item())

        # Store state for future decoding
        self._request_states[req_id] = MlxRequestState(
            token_ids=list(token_ids) + [next_token],
            cache=cache,
            generated_tokens=1,
        )

        return next_token

    def prefill_batch(
        self,
        req_ids: list[str],
        token_ids_list: list[list[int]],
    ) -> list[int]:
        """Run batched prefill for multiple requests in a single forward pass.

        When all sequences have the same length, they are stacked into a single
        batch tensor for one forward pass.  For variable-length sequences the
        method falls back to serial prefill.

        Args:
            req_ids: List of request identifiers
            token_ids_list: List of token ID sequences, one per request

        Returns:
            List of next token IDs (greedy sampled)
        """
        if len(req_ids) == 1:
            return [self.prefill(req_ids[0], token_ids_list[0])]

        # Check if all sequences have the same length (enables true batching)
        lengths = [len(tids) for tids in token_ids_list]
        if len(set(lengths)) != 1:
            # Variable lengths – fall back to serial prefill
            return [
                self.prefill(rid, tids) for rid, tids in zip(req_ids, token_ids_list)
            ]

        # All same length – use a single set of fresh caches;
        # they'll be populated with shape (batch_size, ...) on the first forward pass
        batch_cache = make_prompt_cache(self.model)

        # Stack into (batch_size, seq_len)
        batched_input = mx.array(
            [list(tids) for tids in token_ids_list], dtype=mx.int32
        )

        # Single forward pass
        model_output = self.model(batched_input, cache=batch_cache)
        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        next_tokens_mlx = mx.argmax(last_logits, axis=-1)

        # Evaluate everything together
        mx.eval(next_tokens_mlx, *[c.state for c in batch_cache])
        next_tokens = next_tokens_mlx.tolist()

        # Extract individual caches and store per-request state
        for i, req_id in enumerate(req_ids):
            individual_cache = _extract_kv_cache(batch_cache, i)
            self._request_states[req_id] = MlxRequestState(
                token_ids=list(token_ids_list[i]) + [next_tokens[i]],
                cache=individual_cache,
                generated_tokens=1,
            )

        return next_tokens

    def decode_batch(
        self,
        req_ids: list[str],
    ) -> list[int]:
        """Run batched decode for multiple requests.

        Args:
            req_ids: List of request IDs to decode

        Returns:
            List of next token IDs
        """
        if len(req_ids) == 1:
            return [self._decode_single(req_ids[0])]

        decode_reqs = []
        for req_id in req_ids:
            state = self._request_states[req_id]
            decode_reqs.append((req_id, state))

        return self._batched_decode(decode_reqs)

    def _decode_single(self, req_id: str) -> int:
        """Decode a single token for one request."""
        state = self._request_states[req_id]
        last_token = state.token_ids[-1]

        input_ids = mx.array([[last_token]], dtype=mx.int32)
        model_output = self.model(input_ids, cache=state.cache)

        logits = self._extract_logits(model_output)

        last_logits = logits[:, -1, :]
        next_token_mlx = mx.argmax(last_logits, axis=-1)

        mx.eval(next_token_mlx, *[c.state for c in state.cache])
        next_token = int(next_token_mlx.item())

        state.token_ids.append(next_token)
        state.generated_tokens += 1

        return next_token

    def _batched_decode(
        self, decode_reqs: list[tuple[str, MlxRequestState]]
    ) -> list[int]:
        """Run a single batched forward pass for multiple decode requests."""
        last_tokens = [state.token_ids[-1] for _, state in decode_reqs]

        # Merge individual KV caches into batched cache
        caches_list = [state.cache for _, state in decode_reqs]
        batch_cache = _merge_kv_caches(caches_list)

        # Create batched input: shape (batch_size, 1)
        batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]

        # Single forward pass
        model_output = self.model(batched_input, cache=batch_cache)
        logits = self._extract_logits(model_output)

        next_token_logits = logits[:, -1, :]
        next_tokens_mlx = mx.argmax(next_token_logits, axis=-1)

        mx.eval(next_tokens_mlx, *[c.state for c in batch_cache])
        next_tokens = next_tokens_mlx.tolist()

        # Extract updated caches back to individual requests
        for i, (_, state) in enumerate(decode_reqs):
            state.cache = _extract_kv_cache(batch_cache, i)
            state.token_ids.append(next_tokens[i])
            state.generated_tokens += 1

        return next_tokens

    def remove_request(self, req_id: str):
        """Clean up state for a completed request."""
        self._request_states.pop(req_id, None)

    def clear(self):
        """Clear all request states."""
        self._request_states.clear()
