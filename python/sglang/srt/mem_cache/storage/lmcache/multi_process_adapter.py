# SPDX-License-Identifier: Apache-2.0
"""
LMCache Multi-Process Mode Adapter for SGLang

This module provides adapters that communicate with an external LMCache 
server process via ZMQ for KV cache management, similar to vLLM's implementation.

The architecture:
- LMCache server runs as a separate process
- SGLang workers connect to it via ZMQ
- KV cache tensors are shared via CUDA IPC
- Operations: REGISTER, UNREGISTER, LOOKUP, STORE, RETRIEVE

IMPORTANT: For consistent caching across process restarts, set PYTHONHASHSEED=0
before starting both LMCache server and SGLang.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
import logging

import torch
import zmq


def _check_pythonhashseed():
    """
    Check if PYTHONHASHSEED is set for consistent hashing.
    
    LMCache uses Python's builtin hash() function for computing chunk keys.
    Without PYTHONHASHSEED=0, hash values will differ across processes,
    causing cache misses after restarts.
    """
    if os.environ.get("PYTHONHASHSEED") is None:
        logger.warning(
            "PYTHONHASHSEED is not set. For consistent caching across process "
            "restarts, set PYTHONHASHSEED=0 before starting both LMCache server "
            "and SGLang. Example: export PYTHONHASHSEED=0"
        )


# Flag to check PYTHONHASHSEED only once
_pythonhashseed_checked = False

# LMCache imports
from lmcache.v1.multiprocess.custom_types import (
    CudaIPCWrapper,
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


def wrap_kv_caches(kv_caches: List[torch.Tensor]) -> KVCache:
    """
    Wrap KV cache tensors as CudaIPCWrapper for IPC communication.
    
    Args:
        kv_caches: List of KV cache tensors (k_pool + v_pool)
        
    Returns:
        List of CudaIPCWrapper objects
    """
    logger.info(f"Wrapping {len(kv_caches)} KV cache tensors for CUDA IPC")
    return [CudaIPCWrapper(tensor) for tensor in kv_caches]


def send_lmcache_request(
    mq_client: MessageQueueClient,
    request_type: RequestType,
    payloads: List[Any],
) -> MessagingFuture[Any]:
    """Helper function to send requests to LMCache server."""
    future = mq_client.submit_request(
        request_type, payloads, get_response_class(request_type)
    )
    return future


def get_lmcache_chunk_size(mq_client: MessageQueueClient) -> int:
    """Get the chunk size from LMCache server."""
    future = send_lmcache_request(mq_client, RequestType.GET_CHUNK_SIZE, [])
    chunk_size = future.result()
    return chunk_size


def striding_block_hashes(
    block_hashes: List[bytes],
    blocks_in_chunk: int,
) -> List[bytes]:
    """
    Stride the block hashes to get the block hashes for each chunk.
    For example, if blocks_in_chunk is 16, we get the 16th, 32nd, 48th, ... hashes.
    """
    return list(islice(block_hashes, blocks_in_chunk - 1, None, blocks_in_chunk))


@dataclass
class LoadStoreOp:
    """Represents a load or store operation with block hashes and IDs."""
    block_hashes: List[bytes]
    block_ids: List[int]

    def __len__(self) -> int:
        return len(self.block_hashes)

    def __post_init__(self):
        assert len(self.block_hashes) == len(self.block_ids), (
            f"Block hashes ({len(self.block_hashes)}) and block IDs "
            f"({len(self.block_ids)}) must have same length"
        )


# Type aliases for clarity
StoreResult = bool
RetrieveResult = List[bool]
LookupResult = List[bool]


class LMCacheMPAdapter:
    """
    SGLang adapter for LMCache Multi-Process mode.
    
    This is the main connector that handles:
    - KV cache registration with CUDA IPC
    - Lookup operations to check cache hits
    - Store operations to save KV cache to LMCache
    - Retrieve operations to load KV cache from LMCache
    """
    
    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        world_size: int,
        worker_id: int,
        sglang_block_size: int,
    ):
        """
        Initialize the LMCache MP adapter.
        
        Args:
            server_url: The LMCache server URL (e.g., "tcp://localhost:5555")
            context: ZMQ context
            model_name: Model name for cache key generation
            world_size: Tensor parallel world size
            worker_id: Worker ID (TP rank)
            sglang_block_size: SGLang's KV cache block size
        """
        self.mq_client = MessageQueueClient(server_url, context)
        
        # Instance ID for this GPU worker (unique per process)
        self.instance_id = os.getpid()
        
        # Model info for cache keys
        self.model_name = model_name
        self.world_size = world_size
        self.worker_id = worker_id
        
        # Registered KV caches
        self.kv_caches: List[torch.Tensor] = []
        
        # Get chunk size from LMCache server
        self.chunk_size = get_lmcache_chunk_size(self.mq_client)
        assert self.chunk_size % sglang_block_size == 0, (
            f"LMCache chunk size ({self.chunk_size}) must be a multiple of "
            f"SGLang block size ({sglang_block_size})"
        )
        self.blocks_in_chunk = self.chunk_size // sglang_block_size
        self.sglang_block_size = sglang_block_size
        
        # Request futures tracking
        self.store_futures: dict[str, Tuple[MessagingFuture[StoreResult], List[str]]] = {}
        self.retrieve_futures: dict[str, Tuple[MessagingFuture[RetrieveResult], List[str]]] = {}
        self.lookup_futures: dict[str, MessagingFuture[LookupResult]] = {}
        
        self.finished_stores: set[str] = set()
        self.previously_finished: set[str] = set()
        
        logger.info(
            f"LMCache MP Adapter initialized: server={server_url}, "
            f"instance_id={self.instance_id}, chunk_size={self.chunk_size}, "
            f"blocks_in_chunk={self.blocks_in_chunk}"
        )
    
    def register_kv_caches(self, kv_caches: List[torch.Tensor]) -> None:
        """
        Register KV cache tensors with the LMCache server via CUDA IPC.
        
        Args:
            kv_caches: List of KV cache tensors ordered as [K0, K1, ..., V0, V1, ...]
        """
        self.kv_caches = kv_caches
        logger.info(f"Registering {len(kv_caches)} KV cache tensors")
        
        # Pass "sglang" as backend_type to tell the server to use unilateral kernel
        future = send_lmcache_request(
            self.mq_client,
            RequestType.REGISTER_KV_CACHE,
            [self.instance_id, wrap_kv_caches(kv_caches), "sglang"],
        )
        future.result()  # Wait for registration to complete
        
        logger.info(f"KV caches registered successfully with backend_type=sglang")
    
    def unregister_kv_caches(self) -> None:
        """Unregister KV cache tensors from the LMCache server."""
        logger.info("Unregistering KV caches")
        
        future = send_lmcache_request(
            self.mq_client,
            RequestType.UNREGISTER_KV_CACHE,
            [self.instance_id],
        )
        future.result()
    
    def _create_key(self, block_hash: bytes) -> IPCCacheEngineKey:
        """Create an IPCCacheEngineKey from a block hash."""
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            chunk_hash=block_hash,
        )
    
    def _block_hashes_to_keys(
        self, block_hashes: List[bytes]
    ) -> List[IPCCacheEngineKey]:
        """Convert block hashes to IPC cache engine keys."""
        strided = striding_block_hashes(block_hashes, self.blocks_in_chunk)
        return [self._create_key(h) for h in strided]
    
    def _token_ids_to_chunk_hashes(self, token_ids: List[int]) -> List[bytes]:
        """
        Convert token IDs to chunk hashes.
        Each chunk of token_ids forms one hash.
        
        IMPORTANT: This uses Python's builtin hash() function to match
        LMCache server's hashing behavior. For consistent caching across
        process restarts, PYTHONHASHSEED=0 must be set before starting
        both the LMCache server and SGLang.
        
        Note: We use 0 as the initial prefix_hash to match LMCache's NONE_HASH
        constant, which defaults to 0 when vLLM is not available.
        """
        global _pythonhashseed_checked
        if not _pythonhashseed_checked:
            _check_pythonhashseed()
            _pythonhashseed_checked = True
        
        logger.info(f"_token_ids_to_chunk_hashes: input len={len(token_ids)}, chunk_size={self.chunk_size}")
        
        hashes = []
        # Use 0 as initial prefix hash (matches LMCache's NONE_HASH fallback)
        prefix_hash = 0
        for i in range(0, len(token_ids), self.chunk_size):
            chunk = token_ids[i:i + self.chunk_size]
            logger.info(f"  i={i}: chunk_len={len(chunk)}, is_full={len(chunk) == self.chunk_size}")
            if len(chunk) == self.chunk_size:
                # Use Python's builtin hash() to match LMCache server
                # LMCache uses: hash((prefix_hash, tokens_tuple, extra_keys))
                chunk_hash = hash((prefix_hash, tuple(chunk), None))
                prefix_hash = chunk_hash  # Update prefix for next chunk
                hashes.append(chunk_hash.to_bytes(8, byteorder='big', signed=True))
        
        logger.info(f"_token_ids_to_chunk_hashes: output {len(hashes)} hashes")
        return hashes
    
    def _token_ids_to_keys(self, token_ids: List[int]) -> List[IPCCacheEngineKey]:
        """Convert token IDs directly to cache keys."""
        chunk_hashes = self._token_ids_to_chunk_hashes(token_ids)
        return [self._create_key(h) for h in chunk_hashes]
    
    # ==================== Lookup Operations ====================
    
    def submit_lookup_request(
        self, request_id: str, token_ids: List[int]
    ) -> None:
        """
        Submit an async lookup request.
        
        Args:
            request_id: Unique request identifier
            token_ids: Token IDs to lookup
        """
        if request_id in self.lookup_futures:
            return  # Already submitted
        
        keys = self._token_ids_to_keys(token_ids)
        if not keys:
            return
        
        future = send_lmcache_request(
            self.mq_client,
            RequestType.LOOKUP,
            [keys, True],  # lock=True to prevent eviction before retrieve
        )
        self.lookup_futures[request_id] = future
    
    def check_lookup_result(self, request_id: str) -> Optional[int]:
        """
        Check if lookup result is ready and return hit count.
        
        Returns:
            Number of cached tokens if ready, None if still pending
        """
        if request_id not in self.lookup_futures:
            return 0
        
        future = self.lookup_futures[request_id]
        if not future.query():
            return None
        
        result = future.result()
        
        # Count consecutive True values (prefix matching)
        hit_count = 0
        for hit in result:
            if hit:
                hit_count += 1
            else:
                break
        
        return hit_count * self.chunk_size
    
    def lookup_sync(self, token_ids: List[int]) -> int:
        """
        Synchronous lookup for cached tokens.
        
        Args:
            token_ids: Token IDs to lookup
            
        Returns:
            Number of cached tokens (prefix-aligned to chunk_size)
        """
        keys = self._token_ids_to_keys(token_ids)
        if not keys:
            return 0
        
        future = send_lmcache_request(
            self.mq_client,
            RequestType.LOOKUP,
            [keys, True],
        )
        result = future.result()
        
        # Count consecutive True values
        hit_count = 0
        for hit in result:
            if hit:
                hit_count += 1
            else:
                break
        
        return hit_count * self.chunk_size
    
    def cleanup_lookup_result(self, request_id: str) -> None:
        """Clean up lookup future for a finished request."""
        self.lookup_futures.pop(request_id, None)
    
    # ==================== Store Operations ====================
    
    def submit_store_request(
        self,
        request_id: str,
        token_ids: List[int],
        block_ids: List[int],
        event: torch.cuda.Event,
    ) -> None:
        """
        Submit an async store request.
        
        Args:
            request_id: Unique request identifier
            token_ids: Token IDs to store
            block_ids: Corresponding GPU block IDs
            event: CUDA event for synchronization
        """
        keys = self._token_ids_to_keys(token_ids)
        if not keys:
            return
        
        # Calculate which block IDs correspond to complete chunks
        num_chunks = len(keys)
        chunk_block_ids = []
        for i in range(num_chunks):
            start_idx = i * self.blocks_in_chunk
            if start_idx < len(block_ids):
                chunk_block_ids.extend(
                    block_ids[start_idx:start_idx + self.blocks_in_chunk]
                )
        
        future = send_lmcache_request(
            self.mq_client,
            RequestType.STORE,
            [keys, self.instance_id, chunk_block_ids, event.ipc_handle()],
        )
        self.store_futures[request_id] = (future.to_cuda_future(), [])
    
    def submit_store_request_with_op(
        self,
        request_id: str,
        op: LoadStoreOp,
        event: torch.cuda.Event,
    ) -> None:
        """Submit store request using LoadStoreOp."""
        keys = self._block_hashes_to_keys(op.block_hashes)
        if not keys:
            return
        
        future = send_lmcache_request(
            self.mq_client,
            RequestType.STORE,
            [keys, self.instance_id, op.block_ids, event.ipc_handle()],
        )
        self.store_futures[request_id] = (future.to_cuda_future(), [])
    
    # ==================== Retrieve Operations ====================
    
    def submit_retrieve_request(
        self,
        request_id: str,
        token_ids: List[int],
        block_ids: List[int],
        event: torch.cuda.Event,
    ) -> None:
        """
        Submit an async retrieve request.
        
        Args:
            request_id: Unique request identifier
            token_ids: Token IDs to retrieve
            block_ids: Target GPU block IDs
            event: CUDA event for synchronization
        """
        keys = self._token_ids_to_keys(token_ids)
        if not keys:
            return
        
        # Calculate which block IDs correspond to complete chunks
        num_chunks = len(keys)
        chunk_block_ids = []
        for i in range(num_chunks):
            start_idx = i * self.blocks_in_chunk
            if start_idx < len(block_ids):
                chunk_block_ids.extend(
                    block_ids[start_idx:start_idx + self.blocks_in_chunk]
                )
        
        future = send_lmcache_request(
            self.mq_client,
            RequestType.RETRIEVE,
            [keys, self.instance_id, chunk_block_ids, event.ipc_handle()],
        )
        self.retrieve_futures[request_id] = (future.to_cuda_future(), [])
    
    def submit_retrieve_request_with_op(
        self,
        request_id: str,
        op: LoadStoreOp,
        event: torch.cuda.Event,
    ) -> None:
        """Submit retrieve request using LoadStoreOp."""
        keys = self._block_hashes_to_keys(op.block_hashes)
        if not keys:
            return
        
        future = send_lmcache_request(
            self.mq_client,
            RequestType.RETRIEVE,
            [keys, self.instance_id, op.block_ids, event.ipc_handle()],
        )
        self.retrieve_futures[request_id] = (future.to_cuda_future(), [])
    
    # ==================== Completion Tracking ====================
    
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> Tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Check for completed store/retrieve operations.
        
        Returns:
            Tuple of (finished_store_ids, finished_retrieve_ids)
        """
        finished_stores = set()
        finished_retrieves = set()
        
        # Check store futures
        for request_id, (future, other_reqs) in list(self.store_futures.items()):
            if not future.query():
                continue
            
            result = future.result()
            finished_stores.add(request_id)
            finished_stores.update(other_reqs)
            
            if not result:
                logger.error(f"Store failed for request {request_id}")
        
        # Check retrieve futures
        for request_id, (future, other_reqs) in list(self.retrieve_futures.items()):
            if not future.query():
                continue
            
            result = future.result()
            finished_retrieves.add(request_id)
            finished_retrieves.update(other_reqs)
            
            if not all(result):
                logger.error(f"Retrieve partial failure for request {request_id}")
        
        # Cleanup finished futures
        for request_id in finished_stores:
            self.store_futures.pop(request_id, None)
        for request_id in finished_retrieves:
            self.retrieve_futures.pop(request_id, None)
        
        # Update tracking
        self.finished_stores.update(finished_stores)
        
        ret_stores = set()
        for req_id in finished_req_ids:
            if req_id in self.finished_stores or req_id in self.store_futures:
                self.previously_finished.add(req_id)
            else:
                ret_stores.add(req_id)
        
        ret_stores.update(self._update_and_get_finished_store())
        
        return ret_stores, finished_retrieves
    
    def _update_and_get_finished_store(self) -> set[str]:
        """Get safely finished store request IDs."""
        safe_finished = self.finished_stores.intersection(self.previously_finished)
        self.finished_stores.difference_update(self.previously_finished)
        self.previously_finished.difference_update(safe_finished)
        return safe_finished
    
    # ==================== Properties ====================
    
    def get_chunk_size(self) -> int:
        """Return the LMCache chunk size."""
        return self.chunk_size
    
    def num_blocks_per_chunk(self) -> int:
        """Return the number of SGLang blocks per LMCache chunk."""
        return self.blocks_in_chunk
    
    # ==================== Lifecycle ====================
    
    def clear(self) -> None:
        """Clear all cached data on the server."""
        future = send_lmcache_request(
            self.mq_client,
            RequestType.CLEAR,
            [],
        )
        future.result()
        logger.info("LMCache cleared")
    
    def shutdown(self) -> None:
        """Shutdown the adapter and cleanup resources."""
        try:
            self.unregister_kv_caches()
        except Exception as e:
            logger.warning(f"Error unregistering KV caches: {e}")
        
        try:
            self.mq_client.close()
        except Exception as e:
            logger.warning(f"Error closing MQ client: {e}")
        
        logger.info("LMCache MP Adapter shutdown complete")


class LMCacheMPConnector:
    """
    High-level connector for SGLang's LMCRadixCache.
    
    This provides a simpler interface that matches the existing
    LMCacheLayerwiseConnector API for easier integration.
    """
    
    def __init__(
        self,
        model_config: "ModelConfig",
        tp_size: int,
        rank: int,
        k_pool: List[torch.Tensor],
        v_pool: List[torch.Tensor],
        mp_host: str = "localhost",
        mp_port: int = 5555,
        tp_group: Optional[Any] = None,
    ):
        """
        Initialize the MP connector.
        
        Args:
            model_config: SGLang model configuration
            tp_size: Tensor parallel size
            rank: Worker rank (TP rank)
            k_pool: List of K cache tensors per layer
            v_pool: List of V cache tensors per layer
            mp_host: LMCache server host
            mp_port: LMCache server port
            tp_group: Tensor parallel group (unused in MP mode)
        """
        self.model_config = model_config
        self.tp_size = tp_size
        self.rank = rank
        
        # Build server URL
        server_url = f"tcp://{mp_host}:{mp_port}"
        
        # Get block size from KV cache shape
        # SGLang KV cache shape: [num_blocks, block_size, num_heads, head_dim]
        # or [num_blocks, block_size, hidden_dim] for MLA
        if k_pool and len(k_pool) > 0:
            # Assume standard shape [num_blocks, block_size, ...]
            self.block_size = k_pool[0].shape[1] if k_pool[0].dim() >= 2 else 1
        else:
            self.block_size = 1
        
        # ZMQ context
        self._zmq_context = zmq.Context.instance()
        
        # Create the adapter
        self._adapter = LMCacheMPAdapter(
            server_url=server_url,
            context=self._zmq_context,
            model_name=model_config.model_path,
            world_size=tp_size,
            worker_id=rank,
            sglang_block_size=self.block_size,
        )
        
        # Register KV caches for unilateral kernel
        # The kernel expects: [K0, K1, ..., K_n, V0, V1, ..., V_n]
        # NOT interleaved like [K0, V0, K1, V1, ...]
        kv_caches = []
        # First all K tensors
        for k in k_pool:
            kv_caches.append(k)
        # Then all V tensors
        for v in v_pool:
            kv_caches.append(v)
        self._adapter.register_kv_caches(kv_caches)
        
        # Store references
        self.k_pool = k_pool
        self.v_pool = v_pool
        
        # For layerwise compatibility
        self.layerwise_retrievers: List[Any] = []
        self.layer_load_layer: List[int] = []
        self.lookup_id_list: List[str] = []
        
        logger.info(
            f"LMCache MP Connector initialized: host={mp_host}, port={mp_port}, "
            f"rank={rank}, block_size={self.block_size}"
        )
    
    def chunk_size(self) -> int:
        """Return the LMCache chunk size."""
        return self._adapter.get_chunk_size()
    
    def lookup(self, token_ids: List[int]) -> int:
        """
        Lookup cached tokens.
        
        Args:
            token_ids: Token IDs to lookup
            
        Returns:
            Number of cached tokens
        """
        return self._adapter.lookup_sync(token_ids)
    
    def start_load_kv(
        self,
        token_ids: List[int],
        slot_mapping: torch.Tensor,
        offset: int,
    ) -> int:
        """
        Start loading KV cache from LMCache.
        
        Args:
            token_ids: Token IDs to load
            slot_mapping: Target slot mapping
            offset: Offset for already loaded tokens
            
        Returns:
            Number of tokens retrieved
        """
        # Calculate how many tokens to retrieve
        num_tokens = len(token_ids)
        aligned_end = (num_tokens // self._adapter.chunk_size) * self._adapter.chunk_size
        
        logger.info(f"start_load_kv: num_tokens={num_tokens}, aligned_end={aligned_end}, offset={offset}")
        
        if aligned_end <= offset:
            return 0
        
        # Get keys for the tokens to retrieve
        retrieve_tokens = token_ids[offset:aligned_end]
        keys = self._adapter._token_ids_to_keys(retrieve_tokens)
        
        logger.info(f"start_load_kv: num_keys={len(keys)}, retrieve_tokens_len={len(retrieve_tokens)}")
        
        # Log key hashes for debugging
        for i, key in enumerate(keys):
            logger.info(f"start_load_kv: key[{i}].chunk_hash={key.chunk_hash.hex()}")
        
        if not keys:
            return 0
        
        # Calculate block IDs from slot mapping
        # We need unique block IDs (one per block, not per slot)
        # The server expects: len(block_ids) * block_size = len(keys) * chunk_size
        retrieve_slot_mapping = slot_mapping[offset:aligned_end]
        block_ids = []
        chunk_size = self._adapter.chunk_size
        block_size = self.block_size
        
        for i in range(len(keys)):
            chunk_start = i * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(retrieve_slot_mapping))
            chunk_slots = retrieve_slot_mapping[chunk_start:chunk_end]
            # Get unique block IDs for this chunk (one per block, not per slot)
            for slot_idx in range(0, len(chunk_slots), block_size):
                block_id = int(chunk_slots[slot_idx].item() // block_size)
                block_ids.append(block_id)
        
        logger.info(f"start_load_kv: num_block_ids={len(block_ids)}, expected={len(keys) * chunk_size // block_size}")
        
        # Create CUDA event for synchronization
        event = torch.cuda.Event(interprocess=True)
        event.record()
        
        # Send retrieve request
        future = send_lmcache_request(
            self._adapter.mq_client,
            RequestType.RETRIEVE,
            [keys, self._adapter.instance_id, block_ids, event.ipc_handle()],
        )
        
        # Wait for completion
        result = future.to_cuda_future().result()
        
        logger.info(f"start_load_kv: retrieve result={result}")
        
        if all(result):
            return len(keys) * chunk_size
        else:
            # Count successful retrievals
            return sum(result) * chunk_size
    
    def store_kv(
        self,
        token_ids: List[int],
        kv_indices: torch.Tensor,
        offset: int,
    ) -> None:
        """
        Store KV cache to LMCache.
        
        Args:
            token_ids: Token IDs to store
            kv_indices: KV cache indices
            offset: Offset for already stored tokens
        """
        # Align to chunk boundaries
        chunk_size = self._adapter.chunk_size
        store_start = (offset // chunk_size) * chunk_size
        store_end = (len(token_ids) // chunk_size) * chunk_size
        
        logger.info(f"store_kv: num_tokens={len(token_ids)}, store_start={store_start}, store_end={store_end}")
        
        if store_end <= store_start:
            return
        
        # Get keys for the tokens to store
        store_tokens = token_ids[store_start:store_end]
        keys = self._adapter._token_ids_to_keys(store_tokens)
        
        logger.info(f"store_kv: num_keys={len(keys)}, store_tokens_len={len(store_tokens)}")
        
        # Log key hashes for debugging
        for i, key in enumerate(keys):
            logger.info(f"store_kv: key[{i}] model={key.model_name}, world_size={key.world_size}, worker_id={key.worker_id}, chunk_hash={key.chunk_hash.hex()}")
        
        if not keys:
            logger.warning("store_kv: No keys to store!")
            return
        
        # Calculate block IDs from kv_indices
        # We need unique block IDs (one per block, not per slot)
        # The server expects: len(block_ids) * block_size = len(keys) * chunk_size
        store_indices = kv_indices[store_start:store_end]
        block_ids = []
        block_size = self.block_size
        for i in range(len(keys)):
            chunk_start = i * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(store_indices))
            chunk_indices = store_indices[chunk_start:chunk_end]
            # Get unique block IDs for this chunk (one per block, not per slot)
            for slot_idx in range(0, len(chunk_indices), block_size):
                block_id = int(chunk_indices[slot_idx].item() // block_size)
                block_ids.append(block_id)
        
        logger.info(f"store_kv: num_block_ids={len(block_ids)}, expected={len(keys) * chunk_size // block_size}")
        
        # Create CUDA event for synchronization
        event = torch.cuda.Event(interprocess=True)
        event.record()
        
        # Send store request
        future = send_lmcache_request(
            self._adapter.mq_client,
            RequestType.STORE,
            [keys, self._adapter.instance_id, block_ids, event.ipc_handle()],
        )
        
        # Wait for completion
        future.to_cuda_future().result()
    
    # ==================== Layerwise compatibility stubs ====================
    # These are provided for compatibility with LMCRadixCache's layerwise interface
    
    def load_kv_layerwise(self, layer_id: int) -> None:
        """Layerwise loading (stub for MP mode)."""
        # MP mode doesn't support layerwise loading
        # All data is transferred at once
        pass
    
    # ==================== Lifecycle ====================
    
    def reset(self) -> None:
        """Reset/clear the cache."""
        self._adapter.clear()
    
    def close(self) -> None:
        """Shutdown the connector."""
        self._adapter.shutdown()
