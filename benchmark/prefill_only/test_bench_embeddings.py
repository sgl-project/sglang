import unittest
from unittest.mock import patch


class TestEmbeddingResponseValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The benchmark builds its input text at import time.
        with patch("transformers.AutoTokenizer.from_pretrained") as load_tokenizer:
            load_tokenizer.return_value.encode.return_value = [1]
            import bench_embeddings

        cls.benchmark = bench_embeddings

    def test_empty_or_malformed_response(self):
        for response in (None, {}, {"data": []}, {"data": [{}]}, {"data": [None]}):
            with self.subTest(response=response):
                self.assertFalse(self.benchmark.validate_embeddings_response(response))

    def test_checks_every_embedding(self):
        for dimensions, second, expected in (
            (None, [], False),
            (None, None, False),
            (None, "ab", False),
            (None, [1.0], True),
            (2, [1.0], False),
            (2, [0.0, 0.0], True),
        ):
            with self.subTest(dimensions=dimensions, second=second):
                response = {"data": [{"embedding": [0.1, 0.2]}, {"embedding": second}]}
                with patch.object(self.benchmark, "MATRYOSHKA_DIMENSIONS", dimensions):
                    self.assertEqual(
                        self.benchmark.validate_embeddings_response(response), expected
                    )


if __name__ == "__main__":
    unittest.main()
else:
    pass
