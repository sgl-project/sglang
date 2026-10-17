"""Exercise a transfer thread in a fresh process, before device 0 has a context."""

import hashlib
import subprocess
import sys
import threading
import unittest
from pathlib import Path

import numpy as np
import torch

from sglang.srt.disaggregation.common.dcp_pack import try_pack_dcp_src
from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="2-gpu-large")


def _pack_in_fresh_thread(device_index):
    def active_contexts():
        return [
            i
            for i in range(torch.cuda.device_count())
            if torch._C._cuda_hasPrimaryContext(i)
        ]

    assert active_contexts() == [], "Run this check in a fresh process"
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    # Independent random bytes distinguish rows, layers, and source padding.
    rng = np.random.default_rng(71015)
    widths = [576, 1152, 513]
    strides = [576, 1168, 544]
    cpu_sources = [rng.integers(0, 256, (67, s), np.uint8) for s in strides]
    sources = [torch.from_numpy(x).to(device) for x in cpu_sources]
    indices = np.array([66, 0, 9, 32, 1, 63, 9], dtype=np.int64)
    required = len(indices) * sum(widths)
    offset = 13
    buffer = StagingBuffer(offset + required + 19, str(device), device_index)
    buffer.buffer.fill_(165)
    torch.cuda.synchronize(device)
    before = active_contexts()
    assert before == [device_index], before
    results = {}
    errors = []

    def worker():
        try:
            kwargs = dict(
                pack_buffer=buffer,
                kv_data_ptrs=[x.data_ptr() for x in sources],
                token_item_lens=widths,
                src_token_item_lens=strides,
                pack_offset_bytes=offset,
            )
            # Empty/overflow fallbacks must not initialize any other device.
            empty = try_pack_dcp_src(
                src_token_indices=np.array([], dtype=np.int64), **kwargs
            )
            assert empty[0] == [] and empty[1].size == 0
            assert (
                try_pack_dcp_src(
                    src_token_indices=indices,
                    pack_capacity_bytes=required - 1,
                    **kwargs,
                )
                is None
            )
            results["fallback_contexts"] = active_contexts()
            for _ in range(2):
                # Exercise lazy stream construction and reuse on the same thread.
                ptrs, dense_indices = try_pack_dcp_src(
                    src_token_indices=indices, **kwargs
                )
                np.testing.assert_array_equal(dense_indices, np.arange(len(indices)))
                expected_ptrs = []
                cursor = buffer.get_ptr() + offset
                for width in widths:
                    expected_ptrs.append(cursor)
                    cursor += len(indices) * width
                assert ptrs == expected_ptrs, (ptrs, expected_ptrs)
            results["thread_device_after"] = torch.cuda.current_device()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(90)
    assert not thread.is_alive(), "Transfer thread timed out"
    if errors:
        raise errors[0]
    expected = np.full(buffer.get_size(), 165, dtype=np.uint8)
    expected[offset : offset + required] = np.concatenate(
        [x[indices, :width].reshape(-1) for x, width in zip(cpu_sources, widths)]
    )
    actual = buffer.buffer.cpu().numpy()
    np.testing.assert_array_equal(actual, expected)
    return {
        **results,
        "device": device_index,
        "contexts_before": before,
        "contexts_after": active_contexts(),
        "packed_bytes": required,
        "buffer_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "exact_bytes_and_guards": True,
    }


@unittest.skipUnless(
    torch.version.cuda is not None and torch.cuda.device_count() >= 2,
    "Requires two CUDA devices",
)
class TestPdDcpPackDevice(unittest.TestCase):
    def test_transfer_thread_only_uses_owning_device(self):
        # CUDA contexts persist for the process lifetime; another GPU test must
        # not be able to hide a stray context by initializing device 0 first.
        for device_index in (0, 1):
            with self.subTest(device=device_index):
                code = (
                    "import json, runpy; "
                    f"m = runpy.run_path({str(Path(__file__).resolve())!r}); "
                    f"r = m['_pack_in_fresh_thread']({device_index}); "
                    "print(json.dumps(r), flush=True); "
                    f"assert r['fallback_contexts'] == [{device_index}], r; "
                    f"assert r['contexts_after'] == [{device_index}], r; "
                    f"assert r['thread_device_after'] == {device_index}, r"
                )
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(
                    result.returncode, 0, result.stdout + "\n" + result.stderr
                )


if __name__ == "__main__":
    unittest.main()
