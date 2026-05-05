import unittest


class TestBazelImportSmoke(unittest.TestCase):
    def test_import_sglang(self):
        __import__("sglang")

    def test_import_platform_plugin_constants(self):
        from sglang.srt.plugins import GENERAL_PLUGINS_GROUP, PLATFORM_PLUGINS_GROUP

        self.assertEqual(PLATFORM_PLUGINS_GROUP, "sglang.srt.platforms")
        self.assertEqual(GENERAL_PLUGINS_GROUP, "sglang.srt.plugins")


if __name__ == "__main__":
    unittest.main()
