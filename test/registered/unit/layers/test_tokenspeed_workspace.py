import unittest

from sglang.srt.layers.attention.tokenspeed_workspace import (
    tokenspeed_workspace_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestTokenspeedWorkspaceSizing(unittest.TestCase):
    def test_small_head_verify_rows_scale_with_raw_q_len(self):
        num_sms = 120
        num_heads = 16
        kv_lora_rank = 512

        q8 = tokenspeed_workspace_bytes(num_sms, num_heads, kv_lora_rank, 8)

        self.assertEqual(q8, num_sms * 128 * (kv_lora_rank + 1) * 4)
        self.assertEqual(
            tokenspeed_workspace_bytes(num_sms, num_heads, kv_lora_rank, 16),
            2 * q8,
        )
        self.assertEqual(
            tokenspeed_workspace_bytes(num_sms, num_heads, kv_lora_rank, 24),
            3 * q8,
        )
        self.assertEqual(
            tokenspeed_workspace_bytes(num_sms, num_heads, kv_lora_rank, 32),
            4 * q8,
        )

    def test_standard_decode_keeps_one_m_tile_floor(self):
        self.assertEqual(
            tokenspeed_workspace_bytes(
                num_sms=120, num_heads=128, kv_lora_rank=512, q_len=1
            ),
            120 * 128 * (512 + 1) * 4,
        )

    def test_low_head_verify_rows_scale_by_query_chunks(self):
        num_sms = 120
        num_heads = 8
        kv_lora_rank = 512

        q8 = tokenspeed_workspace_bytes(num_sms, num_heads, kv_lora_rank, 8)

        self.assertEqual(q8, num_sms * 128 * (kv_lora_rank + 1) * 4)
        self.assertEqual(
            tokenspeed_workspace_bytes(num_sms, num_heads, kv_lora_rank, 9),
            2 * q8,
        )
        self.assertEqual(
            tokenspeed_workspace_bytes(num_sms, num_heads, kv_lora_rank, 16),
            2 * q8,
        )
        self.assertEqual(
            tokenspeed_workspace_bytes(num_sms, num_heads, kv_lora_rank, 17),
            3 * q8,
        )


if __name__ == "__main__":
    unittest.main()
