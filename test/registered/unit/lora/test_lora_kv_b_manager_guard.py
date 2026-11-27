"""LoRAManager refuses a kv_b_proj target the launch rules could not see.

Targets inferred from --lora-paths adapter configs surface only in
``init_lora_shapes``; by then the attention modules have snapshotted the
chunked-prefix flag, so where the LoRA backend cannot route cached-prefix rows
the manager fails with the flag spelled out instead of serving cached prefixes
through the MHA paths (see lora_hook).

    python -m pytest test/registered/unit/lora/test_lora_kv_b_manager_guard.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.lora import lora_manager
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _init_lora_shapes(
    targets,
    *,
    disable_chunked_prefix_cache,
    prefill_backend="trtllm_mla",
    routes_cached_rows=False,
):
    manager = SimpleNamespace(
        base_model=None,
        configs={},
        lora_refs={},
        lora_added_tokens_size=0,
        lora_backend=SimpleNamespace(name="triton"),
    )
    with (
        patch.object(
            lora_manager,
            "routes_kv_b_lora_cached_rows",
            return_value=routes_cached_rows,
        ),
        patch.object(
            lora_manager,
            "get_exec",
            return_value=SimpleNamespace(
                graph=SimpleNamespace(
                    cuda_graph_config=SimpleNamespace(
                        prefill=SimpleNamespace(backend="breakable")
                    )
                )
            ),
        ),
        patch.object(
            lora_manager,
            "attention_backends",
            return_value=(prefill_backend, prefill_backend),
        ),
        patch.object(
            lora_manager,
            "get_schedule",
            return_value=SimpleNamespace(
                disable_chunked_prefix_cache=disable_chunked_prefix_cache
            ),
        ),
        patch.object(
            lora_manager,
            "get_parallel",
            return_value=SimpleNamespace(enable_kv_cache_sharding=False),
        ),
        patch.object(
            lora_manager,
            "get_platform",
            return_value=SimpleNamespace(is_sm100_or_sm110=True),
        ),
    ):
        LoRAManager.init_lora_shapes(
            manager, max_lora_rank=16, target_modules=set(targets)
        )
    return manager


class TestKvBLoraManagerGuard(CustomTestCase):
    def test_kv_b_proj_with_chunked_prefix_cache_on_is_refused(self):
        with self.assertRaisesRegex(ValueError, "--disable-chunked-prefix-cache"):
            _init_lora_shapes(
                ["kv_b_proj", "o_proj"], disable_chunked_prefix_cache=False
            )

    def test_kv_b_proj_with_the_flag_loads(self):
        manager = _init_lora_shapes(
            ["kv_b_proj", "o_proj"], disable_chunked_prefix_cache=True
        )
        self.assertEqual(manager.target_modules, {"kv_b_proj", "o_proj"})

    def test_kv_b_proj_on_a_backend_that_ignores_the_flag_is_refused(self):
        with self.assertRaisesRegex(ValueError, "dsa"):
            _init_lora_shapes(
                ["kv_b_proj"], disable_chunked_prefix_cache=True, prefill_backend="dsa"
            )

    def test_routed_cached_rows_need_no_flag_on_any_backend(self):
        for prefill_backend in ("trtllm_mla", "dsa", "nsa"):
            with self.subTest(prefill_backend=prefill_backend):
                manager = _init_lora_shapes(
                    ["kv_b_proj", "o_proj"],
                    disable_chunked_prefix_cache=False,
                    prefill_backend=prefill_backend,
                    routes_cached_rows=True,
                )
                self.assertEqual(manager.target_modules, {"kv_b_proj", "o_proj"})

    def test_other_targets_do_not_need_the_flag(self):
        manager = _init_lora_shapes(
            ["q_b_proj", "o_proj"], disable_chunked_prefix_cache=False
        )
        self.assertEqual(manager.target_modules, {"q_b_proj", "o_proj"})


if __name__ == "__main__":
    unittest.main()
