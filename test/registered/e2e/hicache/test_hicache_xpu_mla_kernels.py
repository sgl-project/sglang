"""MLA (DeepSeek-family) *_mla_* KV-transfer kernels on Intel XPU.

Kernel selection (see pool_host/mla.py load_to_device_per_layer /
backup_from_device_all_layer):

    io_backend  mem_layout    load kernel                      backup kernel
    ----------  ------------  -------------------------------  -------------------------------
    kernel      layer_first   transfer_kv_per_layer_mla        transfer_kv_all_layer_mla
    kernel      page_first    transfer_kv_per_layer_mla_pf_lf  transfer_kv_all_layer_mla_lf_pf

The table holds on XPU because MLATokenToKVPoolHost gates can_use_jit on
(_is_cuda or _is_hip) in mla.py, so the jit_transfer_hicache_*_mla branch
preceding each transfer_kv_* call never fires here.
"""

import os
import shutil
import tempfile
import unittest

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.hicache_xpu_common import (
    COMPARE_TOKENS,
    EVICT_DEVICE_POOL_TOKENS,
    EVICT_HICACHE_RATIO,
    XPU_AVAILABLE,
    complete,
    launch_server,
    load_back_tokens,
    prime_cache,
    resolve_base_url,
    shared_prefix_len,
    wait_load_back_tokens,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
    terminate_and_kill_process_tree,
)

register_xpu_ci(est_time=300, suite="nightly-xpu-1-gpu", nightly=True)


def _write_tiny_mla_model(model_dir):
    """Save a random 2-layer DeepseekV3 with the real MLA dims to model_dir.

    Built locally so the test needs no private or third-party HF repo;
    DEFAULT_MODEL_NAME_FOR_TEST_MLA is private. The tokenizer is borrowed from
    the Qwen model the other XPU tests already download. Accuracy is
    irrelevant: a restore reloads the same KV bytes it wrote.
    """
    import torch
    from transformers import AutoTokenizer, DeepseekV3Config, DeepseekV3ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN)
    tokenizer.save_pretrained(model_dir)
    config = DeepseekV3Config(
        vocab_size=-(-len(tokenizer) // 64) * 64,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        # Every layer dense, so no MoE weights are built. Past num_hidden_layers + 1
        # because sglang also probes layer_id + 1, which reads the moe_layer_freq
        # that transformers 5 no longer saves.
        first_k_dense_replace=3,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        moe_intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=2,
        # DeepSeek-V3's MLA dims: these set the KV-cache shape the kernels move.
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        max_position_embeddings=4096,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = DeepseekV3ForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(model_dir)


class _MlaKernelServer(CustomTestCase):
    """One MLA HiCache server with a host tier only, plus the scenario it runs."""

    model = None
    io_backend = None
    mem_layout = None
    process = None
    tmp_dir = None

    @classmethod
    def setUpClass(cls):
        if cls is _MlaKernelServer:
            raise unittest.SkipTest(
                "abstract base; concrete subclasses set io_backend and mem_layout"
            )
        cls.base_url = resolve_base_url()
        cls.tmp_dir = tempfile.mkdtemp(
            prefix=f"hc_xpu_mla_{cls.io_backend}_{cls.mem_layout}_"
        )
        cls.model = os.path.join(cls.tmp_dir, "model")
        _write_tiny_mla_model(cls.model)
        cls.process = launch_server(
            cls.model,
            cls.base_url,
            [
                "--tp",
                1,
                "--trust-remote-code",
                "--attention-backend",
                "triton",
                "--enable-hierarchical-cache",
                # The device KV pool is (mem-fraction x device mem), so 0.1
                # keeps the host pin under ~3 GB on a 22 GB Arc and leaves the
                # rest of the card for the fixture's weights.
                "--mem-fraction-static",
                0.1,
                "--max-total-tokens",
                EVICT_DEVICE_POOL_TOKENS,
                "--hicache-ratio",
                EVICT_HICACHE_RATIO,
                "--page-size",
                16,
                "--hicache-io-backend",
                cls.io_backend,
                "--hicache-mem-layout",
                cls.mem_layout,
                "--enable-cache-report",
                "--enable-metrics",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
        if cls.tmp_dir is not None:
            shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_offload_reload_round_trip(self):
        """Evict a long prefix to the host tier, reload it, and compare greedily.

        One method rather than several: the cached-token assertions are only
        meaningful after the eviction has run, and unittest orders methods
        alphabetically, not causally.
        """
        # 444 + 8 x 128 tokens overflow the 1024-token device pool, forcing a reload.
        prefix = "The history of computing is a long and fascinating story. " * 40
        prime_cache(
            self.base_url,
            self.model,
            prefix + " In summary,",
            max_tokens=COMPARE_TOKENS,
            timeout=120,
        )
        out1, out1_cached = complete(
            self.base_url,
            self.model,
            prefix + " In summary,",
            max_tokens=COMPARE_TOKENS,
            timeout=120,
            want_cached=True,
        )
        for i in range(8):
            complete(
                self.base_url,
                self.model,
                f"Unrelated filler request {i}: " + "lorem ipsum " * 60,
                max_tokens=16,
                timeout=120,
            )
        loaded_before = load_back_tokens(self.base_url)
        out2, cached = complete(
            self.base_url,
            self.model,
            prefix + " In summary,",
            max_tokens=COMPARE_TOKENS,
            timeout=120,
            want_cached=True,
        )

        tag = f"[{self.io_backend}/{self.mem_layout}]"
        loaded_after = wait_load_back_tokens(self.base_url, loaded_before)
        self.assertGreater(
            loaded_after,
            loaded_before,
            f"{tag} no tokens loaded back from the host tier -> the reload was "
            f"served without running the load kernel, so the assertions below "
            f"cannot distinguish it from a device-tier hit",
        )
        self.assertGreater(len(out1.strip()), 0, f"{tag} empty output")
        self.assertGreater(
            cached, 0, f"{tag} reload reported 0 cached tokens -> KV was recomputed"
        )
        self.assertEqual(
            cached,
            out1_cached,
            f"{tag} reload reused {cached} tokens vs the baseline's "
            f"{out1_cached} -> the tiers disagree on the prefix boundary",
        )
        spl = shared_prefix_len(out2, out1)
        self.assertEqual(
            out2,
            out1,
            f"{tag} reloaded continuation diverged after {spl} chars "
            f"(first={out1!r} reloaded={out2!r}) -> the host round trip "
            f"corrupted the KV",
        )


@unittest.skipUnless(XPU_AVAILABLE, "Intel XPU not available")
class TestMlaKernelLayerFirst(_MlaKernelServer):
    """kernel/layer_first: transfer_kv_per_layer_mla / transfer_kv_all_layer_mla."""

    io_backend = "kernel"
    mem_layout = "layer_first"


@unittest.skipUnless(XPU_AVAILABLE, "Intel XPU not available")
class TestMlaKernelPageFirst(_MlaKernelServer):
    """kernel/page_first: transfer_kv_per_layer_mla_pf_lf / *_all_layer_mla_lf_pf."""

    io_backend = "kernel"
    mem_layout = "page_first"


if __name__ == "__main__":
    unittest.main(verbosity=2)
