import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache import kv_cache_configurator as kvc


class TestHybridNPUDSAPoolSelection(unittest.TestCase):
    def test_npu_hybrid_dsa_uses_npu_pool(self):
        class NPUPool:
            pass

        class GenericDSAPool:
            pass

        config = object.__new__(kvc.KVCacheConfigurator)
        config.use_mla_backend = True
        config.model_config = SimpleNamespace(hf_config=object())
        npu_module = SimpleNamespace(NPUMLATokenToKVPool=NPUPool)

        with (
            patch.object(kvc, "_is_npu", True),
            patch.object(kvc, "is_deepseek_dsa", return_value=True),
            patch.dict(
                sys.modules,
                {"sglang.srt.hardware_backend.npu.memory_pool_npu": npu_module},
            ),
        ):
            selected = config._hybrid_full_attention_pool_class(
                mha_pool_class=object,
                mla_pool_class=object,
                dsa_pool_class=GenericDSAPool,
            )

        self.assertIs(selected, NPUPool)


if __name__ == "__main__":
    unittest.main()
