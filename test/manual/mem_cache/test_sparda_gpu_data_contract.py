"""Manual GPU plumbing check for the SparDA/LMCache sparse contract.

This is a lifecycle/data-path check, not a model-quality or throughput
benchmark. It writes deterministic KV pages through the real LMCache MP
connector, retrieves a full hit and a partial hit into different pages, and
checks the values after the device event has completed.
"""

from __future__ import annotations

import atexit
import os
import time
from dataclasses import replace
from types import SimpleNamespace

import torch
from lmcache.integration.sglang.multi_process_adapter import LMCacheMPConnector
from lmcache.integration.sglang.sglang_adapter import StoreMetadata


def _future_result(future, timeout: float = 30.0):
    return future.result(timeout=timeout)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this manual contract check")

    torch.cuda.set_device(0)
    page_size = 1
    chunk_size = 256
    capacity = 1536
    num_layers = 2
    num_heads = 1
    head_dim = 4
    dtype = torch.float32
    device = torch.device("cuda")
    lmcache_port = int(os.environ.get("SPARDA_LMCACHE_PORT", "5556"))
    token_base = int(os.environ.get("SPARDA_CONTRACT_TOKEN_BASE", "100000"))
    num_chunks = 2
    token_count = chunk_size * num_chunks

    k_pool = [
        torch.empty((capacity, num_heads, head_dim), dtype=dtype, device=device)
        for _ in range(num_layers)
    ]
    v_pool = [
        torch.empty((capacity, num_heads, head_dim), dtype=dtype, device=device)
        for _ in range(num_layers)
    ]
    source_slots = torch.arange(token_count, dtype=torch.int64, device=device)
    source_k = []
    source_v = []
    for layer_id in range(num_layers):
        layer_k = (
            torch.arange(token_count * head_dim, dtype=dtype, device=device).reshape(
                token_count, num_heads, head_dim
            )
            + layer_id * 100000
        )
        layer_v = layer_k + 0.5
        k_pool[layer_id][source_slots] = layer_k
        v_pool[layer_id][source_slots] = layer_v
        source_k.append(layer_k.clone())
        source_v.append(layer_v.clone())
    torch.cuda.synchronize()

    connector = LMCacheMPConnector(
        sgl_config=SimpleNamespace(model_path="sparda-gpu-contract"),
        tp_size=1,
        rank=0,
        page_size=page_size,
        host="127.0.0.1",
        port=lmcache_port,
        k_pool=k_pool,
        v_pool=v_pool,
    )
    atexit.register(connector.close)

    first_token_ids = list(range(token_base, token_base + chunk_size))
    second_token_ids = list(range(token_base + chunk_size, token_base + token_count))
    store_suffix = str(token_base)
    first_store_result = _future_result(
        connector.store_kv_async(
            StoreMetadata(
                last_node=None,
                token_ids=first_token_ids,
                kv_indices=source_slots[:chunk_size],
                offset=0,
                request_id=f"sparda-gpu-contract-store-first-{store_suffix}",
            )
        )
    )
    second_store_result = _future_result(
        connector.store_kv_async(
            StoreMetadata(
                last_node=None,
                token_ids=second_token_ids,
                kv_indices=source_slots[chunk_size:],
                offset=0,
                request_id=f"sparda-gpu-contract-store-second-{store_suffix}",
            )
        )
    )
    if not first_store_result or not second_store_result:
        raise RuntimeError("deterministic KV store did not complete")
    # The MP store future covers the device event.  The server publishes the
    # L1 write completion through its host callback, so give that callback a
    # short scheduling window before the first logical lookup.
    time.sleep(0.5)

    key = connector.create_sparse_object_keys(first_token_ids, [0])[0]
    second_key = connector.create_sparse_object_keys(second_token_ids, [0])[0]
    first_source_k = source_k[0][:chunk_size]
    first_source_v = source_v[0][:chunk_size]
    second_source_k = source_k[0][chunk_size:]
    second_source_v = source_v[0][chunk_size:]
    missing_key = replace(key, cache_salt="deliberate-miss")

    # Full hit: the destination is a separate page range and should match the
    # source values after sparse_retrieve's device event has been consumed.
    full_request = "sparda-gpu-contract-full"
    _future_result(connector.sparse_prefetch(full_request, 0, 0, [key]))
    full_destination = list(range(512, 512 + chunk_size))
    k_pool[0][full_destination] = -1
    v_pool[0][full_destination] = -1
    full_result = _future_result(
        connector.sparse_retrieve(
            full_request,
            0,
            0,
            [key],
            [full_destination],
        )
    )
    full_equal = (
        full_result == (True, [0])
        and torch.equal(k_pool[0][full_destination], first_source_k)
        and torch.equal(v_pool[0][full_destination], first_source_v)
    )
    if not full_equal:
        print(
            {
                "full_result": full_result,
                "full_k_equal": torch.equal(
                    k_pool[0][full_destination], first_source_k
                ),
                "full_v_equal": torch.equal(
                    v_pool[0][full_destination], first_source_v
                ),
                "full_first_value": float(k_pool[0][full_destination][0, 0, 0]),
                "source_first_value": float(source_k[0][0, 0, 0]),
            }
        )
        _future_result(connector.sparse_cancel_prefetch(full_request, 0, 0))
        raise AssertionError("full-hit sparse retrieve changed deterministic KV")
    release_full = _future_result(connector.sparse_release_prefetch(full_request, 0, 0))

    # Partial hit: the second logical object is intentionally absent. The
    # first destination range must still be correct; the missing range must
    # remain untouched.
    partial_request = "sparda-gpu-contract-partial"
    _future_result(connector.sparse_prefetch(partial_request, 1, 0, [key, missing_key]))
    partial_destination = list(range(768, 768 + 2 * chunk_size))
    k_pool[0][partial_destination] = -7
    v_pool[0][partial_destination] = -7
    partial_result = _future_result(
        connector.sparse_retrieve(
            partial_request,
            1,
            0,
            [key, missing_key],
            [partial_destination],
        )
    )
    partial_equal = (
        partial_result == (True, [0])
        and torch.equal(k_pool[0][partial_destination[:chunk_size]], first_source_k)
        and torch.equal(v_pool[0][partial_destination[:chunk_size]], first_source_v)
        and bool(torch.all(k_pool[0][partial_destination[chunk_size:]] == -7))
        and bool(torch.all(v_pool[0][partial_destination[chunk_size:]] == -7))
    )
    if not partial_equal:
        raise AssertionError("partial sparse retrieve did not preserve hit/miss pages")
    release_partial = _future_result(
        connector.sparse_release_prefetch(partial_request, 1, 0)
    )

    # Cancel before retrieve: the destination remains at its sentinel value.
    cancel_request = "sparda-gpu-contract-cancel"
    _future_result(connector.sparse_prefetch(cancel_request, 2, 0, [key]))
    cancel_destination = list(range(900, 900 + chunk_size))
    k_pool[0][cancel_destination] = -11
    cancel_result = _future_result(
        connector.sparse_cancel_prefetch(cancel_request, 2, 0)
    )
    cancel_safe = cancel_result is True and bool(
        torch.all(k_pool[0][cancel_destination] == -11)
    )

    # Reuse the same request identity with two generations and different
    # destination ranges; the newer generation must not inherit stale state.
    reuse_request = "sparda-gpu-contract-reuse"
    _future_result(connector.sparse_prefetch(reuse_request, 3, 0, [key]))
    reuse_destination = list(range(600, 600 + chunk_size))
    k_pool[0][reuse_destination] = -13
    reuse_result = _future_result(
        connector.sparse_retrieve(
            reuse_request,
            3,
            0,
            [key],
            [reuse_destination],
        )
    )
    reuse_equal = reuse_result == (True, [0]) and torch.equal(
        k_pool[0][reuse_destination], first_source_k
    )
    if not reuse_equal:
        raise AssertionError("request reuse did not retrieve deterministic KV")
    release_reuse = _future_result(
        connector.sparse_release_prefetch(reuse_request, 3, 0)
    )
    reuse_generation_two = _future_result(
        connector.sparse_prefetch(reuse_request, 4, 0, [key])
    )
    reuse_generation_two_destination = list(range(700, 700 + chunk_size))
    k_pool[0][reuse_generation_two_destination] = -17
    reuse_generation_two_result = _future_result(
        connector.sparse_retrieve(
            reuse_request,
            4,
            0,
            [key],
            [reuse_generation_two_destination],
        )
    )
    reuse_generation_two_equal = (
        reuse_generation_two
        and reuse_generation_two_result == (True, [0])
        and torch.equal(k_pool[0][reuse_generation_two_destination], first_source_k)
    )
    release_reuse_generation_two = _future_result(
        connector.sparse_release_prefetch(reuse_request, 4, 0)
    )

    # Release retry: a failed acknowledgement must not discard the only local
    # cleanup record. The second call reaches the real remote release.
    retry_request = "sparda-gpu-contract-release-retry"
    _future_result(connector.sparse_prefetch(retry_request, 5, 0, [second_key]))
    retry_destination = list(range(1100, 1100 + chunk_size))
    k_pool[0][retry_destination] = -19
    retry_retrieve = _future_result(
        connector.sparse_retrieve(
            retry_request,
            5,
            0,
            [second_key],
            [retry_destination],
        )
    )
    real_release = connector.sparse_release_prefetch
    release_attempts = 0

    def flaky_release(request_id, generation, layer_id):
        nonlocal release_attempts
        release_attempts += 1
        if release_attempts == 1:
            return connector._completed_sparse_future(False)
        return real_release(request_id, generation, layer_id)

    connector.sparse_release_prefetch = flaky_release
    retry_release_failed = _future_result(
        connector.sparse_release_prefetch(retry_request, 5, 0)
    )
    retry_handle_retained = (
        retry_request,
        5,
        0,
    ) in connector._sparse_handles
    retry_release_succeeded = _future_result(
        connector.sparse_release_prefetch(retry_request, 5, 0)
    )
    connector.sparse_release_prefetch = real_release
    retry_clean = (
        not (
            retry_request,
            5,
            0,
        )
        in connector._sparse_handles
    )

    # Race a real device-aware retrieve with cancellation. Depending on which
    # side wins, the destination is either fully copied or remains untouched;
    # neither outcome may leave a live logical handle behind.
    late_request = "sparda-gpu-contract-late"
    _future_result(connector.sparse_prefetch(late_request, 6, 0, [key, second_key]))
    late_destination = list(range(1000, 1000 + 2 * chunk_size))
    k_pool[0][late_destination] = -23
    late_retrieve_future = connector.sparse_retrieve(
        late_request,
        6,
        0,
        [key, second_key],
        [late_destination],
    )
    late_cancel_result = _future_result(
        connector.sparse_cancel_prefetch(late_request, 6, 0)
    )
    try:
        late_retrieve_result = _future_result(late_retrieve_future)
        late_retrieve_error = None
    except Exception as exc:
        late_retrieve_result = None
        late_retrieve_error = type(exc).__name__
    late_release_result = _future_result(
        connector.sparse_release_prefetch(late_request, 6, 0)
    )
    late_destination_safe = (
        late_retrieve_error is not None
        or (
            late_retrieve_result == (True, [0, 1])
            and torch.equal(k_pool[0][late_destination[:chunk_size]], first_source_k)
            and torch.equal(k_pool[0][late_destination[chunk_size:]], second_source_k)
            and torch.equal(v_pool[0][late_destination[:chunk_size]], first_source_v)
            and torch.equal(v_pool[0][late_destination[chunk_size:]], second_source_v)
        )
        or bool(torch.all(k_pool[0][late_destination] == -23))
    )

    # An incomplete destination mapping forces the server-side retrieve to
    # fail after it has accepted the request. Cleanup must still be retryable
    # and the destination must remain untouched. The targeted fault-injection
    # tests cover a copy that throws after an earlier H2D has been submitted;
    # this live check deliberately avoids poisoning the CUDA context with an
    # out-of-bounds device index.
    failure_request = "sparda-gpu-contract-mid-failure"
    _future_result(connector.sparse_prefetch(failure_request, 7, 0, [key, second_key]))
    failure_destination = list(range(1280, 1280 + chunk_size))
    k_pool[0][failure_destination] = -29
    try:
        failure_result = _future_result(
            connector.sparse_retrieve(
                failure_request,
                7,
                0,
                [key, second_key],
                [failure_destination],
            )
        )
        failure_error = None
    except Exception as exc:
        failure_result = None
        failure_error = type(exc).__name__
    failure_cancel = _future_result(
        connector.sparse_cancel_prefetch(failure_request, 7, 0)
    )
    failure_release = _future_result(
        connector.sparse_release_prefetch(failure_request, 7, 0)
    )
    failure_clean = (
        (
            failure_error is not None
            or failure_result is not None
            and failure_result[0] is False
        )
        and bool(failure_cancel or failure_release)
        and bool(torch.all(k_pool[0][failure_destination] == -29))
    )

    # The consumer reads the same destination range as a small attention-like
    # contraction. This is still a plumbing oracle, not MiniCPM model quality.
    query = torch.ones((1, head_dim), dtype=dtype, device=device)
    consumer_destination = reuse_generation_two_destination
    consumer_score = query @ k_pool[0][consumer_destination, 0].transpose(0, 1)
    expected_score = query @ first_source_k[:, 0].transpose(0, 1)
    consumer_equal = torch.equal(consumer_score, expected_score)
    if not consumer_equal:
        raise AssertionError("consumer did not observe the staged KV pages")

    print(
        {
            "store": bool(first_store_result and second_store_result),
            "full_hit": bool(full_equal and release_full),
            "partial_hit": bool(partial_equal and release_partial),
            "cancel": bool(cancel_safe),
            "request_reuse": bool(
                reuse_equal
                and release_reuse
                and reuse_generation_two_equal
                and release_reuse_generation_two
            ),
            "release_retry": bool(
                retry_retrieve == (True, [0])
                and retry_release_failed is False
                and retry_handle_retained
                and retry_release_succeeded
                and retry_clean
            ),
            "late_completion_cleanup": bool(
                (late_cancel_result or late_release_result) and late_destination_safe
            ),
            "retrieve_mapping_failure_cleanup": bool(failure_clean),
            "consumer_read": bool(consumer_equal),
            "active_sparse_handles": len(connector._sparse_handles),
        }
    )


if __name__ == "__main__":
    main()
