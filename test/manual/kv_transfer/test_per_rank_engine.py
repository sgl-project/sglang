"""Smoke test for per-rank mooncake transfer engine caching.

Verifies that init_mooncake_transfer_engine / get_mooncake_transfer_engine now
cache per gpu_id (not a single shared singleton), so each rank can bind its own
IB device. Does NOT require mooncake or GPUs — MooncakeTransferEngine is mocked.
"""
import sys
from unittest.mock import patch


def make_fake_engine():
    class FakeEngine:
        def __init__(self, hostname, gpu_id, ib_device):
            self.hostname = hostname
            self.gpu_id = gpu_id
            self.ib_device = ib_device
            self.session_id = f"{hostname}:{gpu_id}:{ib_device}"
    return FakeEngine


def main():
    import sglang.srt.distributed.device_communicators.mooncake_transfer_engine as m

    FakeEngine = make_fake_engine()

    with patch.object(m, "MooncakeTransferEngine", FakeEngine):
        m._mooncake_transfer_engines.clear()

        # init two distinct ranks
        e0 = m.init_mooncake_transfer_engine("h0", gpu_id=0, ib_device="mlx5_ib0")
        e1 = m.init_mooncake_transfer_engine("h0", gpu_id=1, ib_device="mlx5_ib1")
        e7 = m.init_mooncake_transfer_engine("h0", gpu_id=7, ib_device="mlx5_ib7")

        # each rank gets its OWN engine with its OWN ib_device
        assert e0.ib_device == "mlx5_ib0", e0.ib_device
        assert e1.ib_device == "mlx5_ib1", e1.ib_device
        assert e7.ib_device == "mlx5_ib7", e7.ib_device
        assert e0 is not e1 is not e7, "engines must be distinct per rank"

        # re-init rank 0 returns cached (idempotent)
        e0b = m.init_mooncake_transfer_engine("h0", gpu_id=0, ib_device="mlx5_ib0")
        assert e0b is e0, "re-init same rank must return cached engine"

        # per-rank lookup
        assert m.get_mooncake_transfer_engine(0) is e0
        assert m.get_mooncake_transfer_engine(1) is e1
        assert m.get_mooncake_transfer_engine(7) is e7

        # backward-compat: no-arg returns rank-0 engine
        assert m.get_mooncake_transfer_engine() is e0

        # unknown rank -> None
        assert m.get_mooncake_transfer_engine(99) is None

        print("PASS: per-rank engine caching works; backward-compat rank-0 fallback OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
