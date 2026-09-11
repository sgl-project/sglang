"""Cross-process CUDA-IPC integration test for the stacked-embedding transport.

Exercises the real serialization path added for the zero-copy score-API transport:
a pool-backed ``TransportProxyTensor`` is pickled in one process and unpickled in a
separate process, which reconstructs a zero-copy view over the same CUDA buffer via
the reused IPC handle. Verifies:

1.  ``__reduce_ex__`` emits handle-only (the pickle is tiny — the whole pool storage
    is NOT serialized by value), and the consumer reads the producer's values.
2.  Writes issued into the pool *after* the handle was exported are visible to the
    consumer once the producer synchronizes before handoff (the producer-sync
    contract the reused handle relies on).

Requires a single CUDA device. Not covered here (would need more infra): TP fan-out
(``tp_size>1``) and cache eviction across many handles.
"""

import multiprocessing as mp
import pickle
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

_HIDDEN = 8
# Backing pool large enough (~4 MB) that a by-value serialization of its storage would
# blow past the handle-only pickle-size assertion in _roundtrip.
_POOL_ROWS = 128 * 1024


def _consumer(payload_bytes: bytes, expected: list, result_q) -> None:
    """Run in a spawned child: unpickle the proxy tensor and read it back."""
    try:
        tensor = pickle.loads(payload_bytes)
        got = tensor.detach().to("cpu", dtype=torch.float32)
        exp = torch.tensor(expected, dtype=torch.float32)
        match = bool(got.shape == exp.shape and torch.allclose(got, exp))
        result_q.put(("ok", match, got.tolist()))
    except Exception as e:  # noqa: BLE001 (child boundary: forward type + trace to parent)
        import traceback

        # Preserve the concrete exception type so the parent can tell an expected
        # failure from an unexpected one instead of masking it.
        result_q.put(("err", type(e).__name__, f"{e}\n{traceback.format_exc()}"))


class TestStackedIpcTransport(CustomTestCase):
    def _roundtrip(self, pool: torch.Tensor, view: torch.Tensor, handle=None):
        """Pickle a pool-backed proxy view, unpickle it in a child, return (status).

        ``handle`` lets a caller pass a handle exported *before* a later write, to
        exercise the reused-pre-write-handle contract; when None it is derived here.
        """
        from sglang.srt.managers.mm_utils import TransportProxyTensor

        if handle is None:
            handle = pool.untyped_storage()._share_cuda_()
        proxy = TransportProxyTensor(view, transport_mode="cuda_ipc", ipc_handle=handle)
        payload = pickle.dumps(proxy)
        # Handle-only: the pickle must be a tiny fraction of the pool storage, so a
        # by-value serialization of the ~4 MB pool would fail this assertion.
        pool_nbytes = pool.numel() * pool.element_size()
        self.assertLess(
            len(payload),
            pool_nbytes // 100,
            msg=f"pickle {len(payload)} B not << pool {pool_nbytes} B — storage serialized by value?",
        )

        ctx = mp.get_context("spawn")
        result_q = ctx.Queue()
        expected = view.detach().to("cpu", dtype=torch.float32).tolist()
        proc = ctx.Process(target=_consumer, args=(payload, expected, result_q))
        proc.start()
        try:
            status = result_q.get(timeout=180)
        finally:
            proc.join(timeout=180)
        return status

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_handle_only_roundtrip_reads_producer_values(self):
        pool = torch.arange(
            _POOL_ROWS * _HIDDEN, dtype=torch.float32, device="cuda"
        ).reshape(_POOL_ROWS, _HIDDEN)
        view = pool[1000:1002]  # a small slice into the middle of the pool
        torch.cuda.current_stream().synchronize()

        status = self._roundtrip(pool, view)

        self.assertEqual(status[0], "ok", msg=f"consumer failed: {status}")
        self.assertTrue(status[1], msg=f"consumer read wrong values: {status}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_write_after_export_visible_after_producer_sync(self):
        pool = torch.zeros(_POOL_ROWS, _HIDDEN, dtype=torch.float32, device="cuda")
        # Export the handle BEFORE the write, so the roundtrip reuses the pre-write
        # handle (as a persistent pool does across requests).
        handle = pool.untyped_storage()._share_cuda_()
        view = pool[0:2]
        view.fill_(7.0)  # write AFTER the handle was exported
        torch.cuda.current_stream().synchronize()  # producer sync before handoff

        status = self._roundtrip(pool, view, handle=handle)

        self.assertEqual(status[0], "ok", msg=f"consumer failed: {status}")
        self.assertTrue(status[1], msg=f"post-export write not visible: {status}")


if __name__ == "__main__":
    unittest.main()
