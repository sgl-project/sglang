"""
Manual test for Forward Pass Metrics (FPM) ZMQ PUB/SUB path.

Tests:
1. Schema encode/decode roundtrip
2. _FpmPublisherThread ZMQ PUB -> ZMQ SUB end-to-end
3. Heartbeat emission on idle
"""

import sys
import time

import zmq


def test_schema_roundtrip():
    from sglang.srt.observability.forward_pass_metrics import (
        ForwardPassMetrics,
        QueuedRequestMetrics,
        ScheduledRequestMetrics,
        WelfordAccumulator,
        decode,
        encode,
    )

    # WelfordAccumulator
    acc = WelfordAccumulator()
    for v in [10, 20, 30]:
        acc.add(v)
    assert acc.count == 3
    assert acc.total == 60
    var = acc.variance()
    assert abs(var - 66.667) < 0.01, f"Expected ~66.667, got {var}"

    # Encode/decode roundtrip
    fpm = ForwardPassMetrics(
        worker_id="test-worker",
        dp_rank=1,
        wall_time=0.042,
        scheduled_requests=ScheduledRequestMetrics(
            num_prefill_requests=5,
            sum_prefill_tokens=1024,
            var_prefill_length=33.3,
            sum_prefill_kv_tokens=512,
            num_decode_requests=32,
            sum_decode_kv_tokens=8192,
            var_decode_kv_tokens=100.0,
        ),
        queued_requests=QueuedRequestMetrics(
            num_prefill_requests=3,
            sum_prefill_tokens=768,
            var_prefill_length=25.0,
            num_decode_requests=1,
            sum_decode_kv_tokens=128,
            var_decode_kv_tokens=0.0,
        ),
    )

    data = encode(fpm)
    fpm2 = decode(data)

    assert fpm2.worker_id == "test-worker"
    assert fpm2.dp_rank == 1
    assert fpm2.wall_time == 0.042
    assert fpm2.scheduled_requests.num_prefill_requests == 5
    assert fpm2.scheduled_requests.sum_prefill_tokens == 1024
    assert fpm2.scheduled_requests.num_decode_requests == 32
    assert fpm2.queued_requests.num_prefill_requests == 3

    print("PASS: schema roundtrip")


def test_zmq_pub_sub():
    """Test _FpmPublisherThread -> ZMQ SUB end-to-end."""
    from sglang.srt.observability.forward_pass_metrics import (
        ForwardPassMetrics,
        ScheduledRequestMetrics,
        _FpmPublisherThread,
        decode,
    )

    port = 29999
    endpoint = f"tcp://127.0.0.1:{port}"

    # Start publisher
    pub = _FpmPublisherThread(
        f"tcp://*:{port}",
        worker_id="test-pub",
        dp_rank=0,
    )

    # Connect subscriber
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 5000)  # 5s timeout

    # ZMQ PUB/SUB needs time to connect
    time.sleep(0.5)

    # Publish a metric
    fpm = ForwardPassMetrics(
        worker_id="test-pub",
        dp_rank=0,
        wall_time=0.05,
        scheduled_requests=ScheduledRequestMetrics(
            num_prefill_requests=10,
            sum_prefill_tokens=2048,
            num_decode_requests=64,
            sum_decode_kv_tokens=16384,
        ),
    )
    pub.publish(fpm)

    # Receive
    frames = sub.recv_multipart()
    assert len(frames) == 3, f"Expected 3 frames, got {len(frames)}"

    topic, seq_bytes, payload = frames
    assert topic == b""
    seq = int.from_bytes(seq_bytes, "big")
    assert seq == 0

    received = decode(payload)
    assert received.worker_id == "test-pub"
    assert received.scheduled_requests.num_prefill_requests == 10
    assert received.scheduled_requests.sum_decode_kv_tokens == 16384
    print(f"PASS: ZMQ PUB/SUB (seq={seq}, {len(payload)} bytes)")

    # Publish a second message -- seq should increment
    pub.publish(fpm)
    frames2 = sub.recv_multipart()
    seq2 = int.from_bytes(frames2[1], "big")
    assert seq2 == 1, f"Expected seq=1, got {seq2}"
    print(f"PASS: sequence incremented (seq={seq2})")

    # Cleanup
    pub.shutdown()
    sub.close()
    ctx.term()


def test_heartbeat():
    """Test that heartbeat messages are emitted when idle."""
    from sglang.srt.observability.forward_pass_metrics import (
        _FpmPublisherThread,
        decode,
    )

    port = 29998
    endpoint = f"tcp://127.0.0.1:{port}"

    pub = _FpmPublisherThread(
        f"tcp://*:{port}",
        worker_id="heartbeat-test",
        dp_rank=0,
    )
    # Override heartbeat interval for faster test
    pub.HEARTBEAT_INTERVAL = 0.3

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 3000)

    time.sleep(0.5)

    # Don't publish anything -- wait for heartbeat
    try:
        frames = sub.recv_multipart()
        heartbeat = decode(frames[2])
        assert heartbeat.worker_id == "heartbeat-test"
        assert heartbeat.wall_time == 0.0  # idle heartbeat
        assert heartbeat.scheduled_requests.num_prefill_requests == 0
        print("PASS: heartbeat received")
    except zmq.Again:
        print("FAIL: no heartbeat received within timeout")
        sys.exit(1)

    pub.shutdown()
    sub.close()
    ctx.term()


if __name__ == "__main__":
    test_schema_roundtrip()
    test_zmq_pub_sub()
    test_heartbeat()
    print("\nAll tests passed!")
