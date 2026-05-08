# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.srt.managers.multimodal_processor import (
    PROCESSOR_MAPPING,
    import_processors,
)
from sglang.srt.models.sensenova_u1 import NEOChatModel
from sglang.srt.multimodal.processors.sensenova_u1 import (
    SenseNovaU1MultimodalProcessor,
)


class TestSenseNovaU1Processor(unittest.TestCase):
    def test_processor_is_registered_for_neo_chat_model(self):
        import_processors("sglang.srt.multimodal.processors")

        self.assertIs(PROCESSOR_MAPPING[NEOChatModel], SenseNovaU1MultimodalProcessor)


if __name__ == "__main__":
    unittest.main()
