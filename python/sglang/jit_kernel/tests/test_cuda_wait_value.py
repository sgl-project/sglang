import threading
import time

import torch

from sglang.jit_kernel.cuda_wait_value import Event


def test_wait_before_record(event: Event | torch.cuda.Event):
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    with torch.cuda.stream(stream_a):
        event.wait()

    stream_a.synchronize()

    with torch.cuda.stream(stream_b):
        event.record()


def test_custom_event_blocks():
    """Test that custom Event blocks the stream as expected."""
    block_thread = threading.Thread(
        target=test_wait_before_record, args=(Event(),), daemon=True
    )
    block_thread.start()

    non_block_thread = threading.Thread(
        target=test_wait_before_record, args=(torch.cuda.Event(),)
    )
    non_block_thread.start()

    print("Checking if custom Event blocks the stream...", flush=True)
    for _ in range(5):
        print(f"{block_thread.is_alive()=}, {non_block_thread.is_alive()=}", flush=True)
        time.sleep(1)

    assert block_thread.is_alive(), "Custom Event did not block as expected"
    assert not non_block_thread.is_alive(), "torch.cuda.Event should not block"
    print("=" * 40)
    print("Test completed successfully.")


if __name__ == "__main__":
    test_custom_event_blocks()
