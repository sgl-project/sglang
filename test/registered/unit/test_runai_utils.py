import unittest
from pathlib import Path

from sglang.srt.configs.load_config import LoadFormat
from sglang.srt.utils.runai_utils import ObjectStorageModel, is_runai_obj_uri
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestRunaiUtils(CustomTestCase):
    def test_is_runai_obj_uri_s3(self):
        self.assertTrue(is_runai_obj_uri("s3://bucket/model/"))
        self.assertTrue(is_runai_obj_uri("S3://Bucket/Model/"))

    def test_is_runai_obj_uri_gs(self):
        self.assertTrue(is_runai_obj_uri("gs://bucket/model/"))
        self.assertTrue(is_runai_obj_uri("GS://Bucket/Model/"))

    def test_is_runai_obj_uri_az(self):
        self.assertTrue(is_runai_obj_uri("az://container/model/"))
        self.assertTrue(is_runai_obj_uri("AZ://Container/Model/"))

    def test_is_runai_obj_uri_local_paths(self):
        self.assertFalse(is_runai_obj_uri("/path/to/model"))
        self.assertFalse(is_runai_obj_uri("./relative/path"))
        self.assertFalse(is_runai_obj_uri("meta-llama/Llama-3.2-1B"))

    def test_is_runai_obj_uri_other_schemes(self):
        self.assertFalse(is_runai_obj_uri("http://example.com/model"))
        self.assertFalse(is_runai_obj_uri("https://example.com/model"))
        self.assertFalse(is_runai_obj_uri("ftp://example.com/model"))

    def test_is_runai_obj_uri_pathlib(self):
        self.assertFalse(is_runai_obj_uri(Path("/local/model")))

    def test_get_path_deterministic(self):
        path1 = ObjectStorageModel.get_path("s3://bucket/model/")
        path2 = ObjectStorageModel.get_path("s3://bucket/model/")
        self.assertEqual(path1, path2)

    def test_get_path_different_uris(self):
        path1 = ObjectStorageModel.get_path("s3://bucket/model-a/")
        path2 = ObjectStorageModel.get_path("s3://bucket/model-b/")
        self.assertNotEqual(path1, path2)

    def test_get_path_contains_model_streamer(self):
        path = ObjectStorageModel.get_path("s3://bucket/model/")
        self.assertIn("model_streamer", path)

    def test_load_format_enum(self):
        self.assertEqual(LoadFormat.RUNAI_STREAMER.value, "runai_streamer")


if __name__ == "__main__":
    unittest.main()
