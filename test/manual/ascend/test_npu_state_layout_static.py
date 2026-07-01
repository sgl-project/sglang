import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


class TestNpuGdnStateLayoutStatic(unittest.TestCase):
    def test_speculative_intermediate_cache_stays_dv_major(self):
        source = (ROOT / "python/sglang/srt/mem_cache/memory_pool.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("temporal_state_shape == (HV, V, K)", source)
        self.assertNotIn(
            "temporal_state = temporal_state.transpose(-1, -2)",
            source,
        )
        self.assertNotIn(
            "Shape: [num_layers, size + 1, speculative_num_draft_tokens, HV, K, V]",
            source,
        )

    def test_npu_chunk_boundary_transposes_to_kernel_layout_only(self):
        source = (
            ROOT / "python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py"
        ).read_text(encoding="utf-8")
        compact = " ".join(source.split())
        self.assertIn(
            "recurrent_state = ssm_states[cache_indices].transpose(-1, -2).contiguous()",
            compact,
        )
        self.assertIn(
            "last_recurrent_state.transpose(-1, -2).contiguous()",
            compact,
        )
        self.assertIn("h = h.transpose(-1, -2).contiguous()", compact)

    def test_ascend_verify_intermediate_state_uses_dv_major_view(self):
        source = (
            ROOT
            / "python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py"
        ).read_text(encoding="utf-8")
        compact = " ".join(source.split())
        self.assertIn(
            "-1, num_value_heads, head_v_dim, head_k_dim",
            compact,
        )
        self.assertIn(
            "-1, seq_len, num_value_heads, head_v_dim, head_k_dim",
            compact,
        )
        self.assertNotIn(
            "-1, num_value_heads, head_k_dim, head_v_dim",
            compact,
        )


if __name__ == "__main__":
    unittest.main()
