import os
import pickle
import socket
import struct
import sys
import types
import unittest
from pathlib import Path


def install_namespace_packages() -> None:
    root = (
        Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"] / "python/sglang"
    )
    for name, path in (
        ("sglang", root),
        ("sglang.srt", root / "srt"),
        ("sglang.srt.utils", root / "srt" / "utils"),
        ("sglang.srt.weight_cache", root / "srt" / "weight_cache"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


sys.modules.pop("torch", None)
install_namespace_packages()

from sglang.srt.weight_cache.protocol import (  # noqa: E402
    CacheConfig,
    compute_global_rank,
    hash_quant_config,
    recv_msg,
    send_msg,
)


def make_cache_config() -> CacheConfig:
    return CacheConfig(
        model_path="/models/demo",
        model_arch="LlamaForCausalLM",
        tp_size=2,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        ep_size=1,
        moe_dp_size=1,
        moe_dp_rank=0,
        moe_ep_rank=0,
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        attn_cp_size=1,
        moe_dense_tp_size=None,
        moe_a2a_backend="none",
        quant_method="",
        quant_config_hash="",
        dtype="float16",
        revision="",
        device_capability="",
        torch_version="",
    )


class WeightCacheProtocolTest(unittest.TestCase):
    def test_import_is_torch_free(self) -> None:
        self.assertNotIn("torch", sys.modules)

    def test_socket_framing_round_trip(self) -> None:
        left, right = socket.socketpair()
        try:
            send_msg(left, {"rank": 3, "handles": [1, 2]})
            self.assertEqual(recv_msg(right), {"rank": 3, "handles": [1, 2]})
        finally:
            left.close()
            right.close()

    def test_restricted_unpickler_rejects_code_execution(self) -> None:
        class Payload:
            def __reduce__(self):
                return (os.system, ("false",))

        data = pickle.dumps(Payload())
        left, right = socket.socketpair()
        try:
            left.sendall(struct.pack("!I", len(data)) + data)
            with self.assertRaisesRegex(RuntimeError, "Blocked unsafe class"):
                recv_msg(right)
        finally:
            left.close()
            right.close()

    def test_cache_config_and_rank_helpers(self) -> None:
        config = make_cache_config()
        self.assertEqual(CacheConfig.from_dict(config.to_dict()), config)
        self.assertEqual(compute_global_rank(tp_size=4, pp_rank=2, tp_rank=1), 9)
        self.assertEqual(
            hash_quant_config({"bits": 8, "group_size": 128}),
            hash_quant_config({"group_size": 128, "bits": 8}),
        )


if __name__ == "__main__":
    unittest.main()
