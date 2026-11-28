"""Unit tests for srt/connector/s3.py object listing"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.connector.s3 import list_files
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_paginated_client(pages):
    """Build a mock S3 client whose paginator yields the given pages.

    The client only exposes ``get_paginator``: if the listing code ever
    falls back to a single ``list_objects_v2`` request, these tests fail
    with an AttributeError instead of silently passing.
    """
    client = MagicMock(spec=["get_paginator"])
    client.get_paginator.return_value.paginate.return_value = pages
    return client


class TestListFiles(CustomTestCase):
    def test_objects_beyond_the_first_page_are_listed(self):
        """list_objects_v2 returns at most 1000 keys per request; objects on
        later pages (e.g. large sharded models) must not be dropped."""
        pages = [
            {
                "Contents": [
                    {"Key": f"model/shard-{i:05d}.safetensors"} for i in range(1000)
                ]
            },
            {"Contents": [{"Key": "model/shard-01000.safetensors"}]},
        ]
        client = _make_paginated_client(pages)

        _, _, files = list_files(client, "s3://bucket/model")

        self.assertEqual(len(files), 1001)
        self.assertIn("model/shard-01000.safetensors", files)

    def test_listing_requests_every_page_from_the_bucket(self):
        """The paginator must be driven with the parsed bucket and prefix."""
        pages = [{"Contents": [{"Key": "model/config.json"}]}]
        client = _make_paginated_client(pages)

        bucket_name, prefix, _ = list_files(client, "s3://bucket/model")

        self.assertEqual(bucket_name, "bucket")
        self.assertEqual(prefix, "model")
        client.get_paginator.assert_called_once_with("list_objects_v2")
        kwargs = client.get_paginator.return_value.paginate.call_args.kwargs
        self.assertEqual(kwargs["Bucket"], "bucket")
        self.assertEqual(kwargs["Prefix"], "model")

    def test_empty_bucket_prefix_yields_no_files(self):
        client = _make_paginated_client([{}])

        _, _, files = list_files(client, "s3://bucket/model")

        self.assertEqual(files, [])

    def test_pattern_filtering_still_applies_to_the_full_listing(self):
        """allow/ignore patterns must filter the aggregated multi-page result."""
        pages = [
            {"Contents": [{"Key": "model/a.safetensors"}, {"Key": "model/b.bin"}]},
            {
                "Contents": [
                    {"Key": "model/c.safetensors"},
                    {"Key": "model/d.safetensors"},
                ]
            },
        ]
        client = _make_paginated_client(pages)

        _, _, files = list_files(
            client,
            "s3://bucket/model",
            allow_pattern=["*.safetensors"],
            ignore_pattern=["model/d*"],
        )

        self.assertEqual(files, ["model/a.safetensors", "model/c.safetensors"])

    def test_directory_placeholder_keys_are_dropped(self):
        """S3 conventionally stores zero-byte "dir/" placeholders; they must
        not be reported as downloadable files."""
        pages = [
            {
                "Contents": [
                    {"Key": "model/"},
                    {"Key": "model/sub/"},
                    {"Key": "model/config.json"},
                ]
            }
        ]
        client = _make_paginated_client(pages)

        _, _, files = list_files(client, "s3://bucket/model")

        self.assertEqual(files, ["model/config.json"])


if __name__ == "__main__":
    unittest.main()
