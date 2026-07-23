"""Unit tests for S3Connector endpoint_url / region_name / credential plumbing."""

import unittest
from unittest.mock import ANY, MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

# Import the connector stack ONCE here, at module load, OUTSIDE any
# ``patch.dict("sys.modules", ...)`` block. patch.dict snapshots sys.modules on
# enter and restores it on exit, which would evict modules imported inside the
# block (e.g. safetensors, a PyO3 extension that cannot be re-initialized),
# breaking subsequent tests. boto3 is imported lazily inside
# ``S3Connector.__init__``, so patching it around construction still works.
from sglang.srt.connector import create_remote_connector  # noqa: E402
from sglang.srt.connector.s3 import S3Connector  # noqa: E402

try:
    import botocore.config  # noqa: F401

    _HAS_BOTOCORE = True
except ImportError:
    _HAS_BOTOCORE = False


@unittest.skipUnless(_HAS_BOTOCORE, "botocore not installed")
class TestS3ConnectorEndpointUrl(unittest.TestCase):
    """Verify endpoint_url / region_name / creds flow through to boto3.client."""

    def _build(self, url, **kwargs):
        # boto3 is imported lazily in S3Connector.__init__, so patching it only
        # around construction is enough (and avoids importing anything heavy
        # inside the sys.modules snapshot).
        fake_boto3 = MagicMock()
        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            connector = S3Connector(url, **kwargs)
        return fake_boto3, connector

    def test_query_string_endpoint_url(self):
        url = (
            "s3://b/k?endpoint_url=https://s3.us-west-004.backblazeb2.com"
            "&region_name=us-west-004"
        )
        fake_boto3, conn = self._build(url)
        fake_boto3.client.assert_called_once_with(
            "s3",
            config=ANY,
            endpoint_url="https://s3.us-west-004.backblazeb2.com",
            region_name="us-west-004",
        )
        self.assertEqual(conn.url, "s3://b/k")

    def test_env_vars_are_not_injected_into_kwargs(self):
        # AWS_* env vars must be read by boto3's default chain, not re-injected
        # by ``_parse_s3_kwargs``. Re-injecting would override an explicit
        # ``botocore.config.Config(region_name=...)`` passed via ``config=``.
        with patch.dict(
            "os.environ",
            {
                "AWS_ENDPOINT_URL": "https://from-env",
                "AWS_DEFAULT_REGION": "us-east-1",
            },
        ):
            fake_boto3, _ = self._build("s3://bucket/path")
        fake_boto3.client.assert_called_once_with("s3", config=ANY)

    def test_explicit_kwargs_win(self):
        url = "s3://b/k?endpoint_url=https://from-url"
        fake_boto3, _ = self._build(url, endpoint_url="https://explicit")
        fake_boto3.client.assert_called_once_with(
            "s3", config=ANY, endpoint_url="https://explicit"
        )

    def test_no_overrides_passes_no_kwargs(self):
        # No query string, no explicit kwargs => only the sglang Config flows.
        fake_boto3, _ = self._build("s3://bucket/path")
        fake_boto3.client.assert_called_once_with("s3", config=ANY)

    def test_credential_query_params(self):
        url = (
            "s3://b/k?endpoint_url=https://s3.example.com"
            "&aws_access_key_id=AKIA_EXAMPLE"
            "&aws_secret_access_key=SECRET_EXAMPLE"
        )
        fake_boto3, conn = self._build(url)
        fake_boto3.client.assert_called_once_with(
            "s3",
            config=ANY,
            endpoint_url="https://s3.example.com",
            aws_access_key_id="AKIA_EXAMPLE",
            aws_secret_access_key="SECRET_EXAMPLE",
        )
        self.assertEqual(conn.url, "s3://b/k")

    def test_session_token_query_param(self):
        # ``aws_session_token`` (STS / temp creds) flows through from the URI.
        url = (
            "s3://b/k?endpoint_url=https://s3.example.com"
            "&aws_access_key_id=AKIA_X"
            "&aws_secret_access_key=SEC_X"
            "&aws_session_token=TOKEN_X"
        )
        fake_boto3, conn = self._build(url)
        fake_boto3.client.assert_called_once_with(
            "s3",
            config=ANY,
            endpoint_url="https://s3.example.com",
            aws_access_key_id="AKIA_X",
            aws_secret_access_key="SEC_X",
            aws_session_token="TOKEN_X",
        )
        self.assertEqual(conn.url, "s3://b/k")

    def test_region_query_alias_raises(self):
        # The deprecated ``?region=`` alias is no longer recognized. It must
        # surface as ``ValueError`` rather than being silently stripped.
        with self.assertRaises(ValueError) as ctx:
            self._build("s3://b/k?region=us-west-004")
        self.assertIn("region", str(ctx.exception))

    def test_typoed_query_param_raises(self):
        # Typos like ``?endpiont_url=...`` must fail loudly instead of being
        # silently dropped and producing a credentials/endpoint mismatch later.
        with self.assertRaises(ValueError) as ctx:
            self._build("s3://b/k?endpiont_url=https://x")
        self.assertIn("endpiont_url", str(ctx.exception))

    def test_blank_value_query_param_raises(self):
        # ``?endpoint_url=`` (no value) is misconfiguration, not a valid
        # opt-out -- raise rather than silently dropping the key.
        with self.assertRaises(ValueError) as ctx:
            self._build("s3://b/k?endpoint_url=")
        self.assertIn("endpoint_url", str(ctx.exception))

    def test_malformed_query_string_raises(self):
        # ``?endpoint_url`` (no ``=``) is malformed and must be rejected.
        with self.assertRaises(ValueError):
            self._build("s3://b/k?endpoint_url")

    def test_malformed_query_does_not_leak_raw_fragment(self):
        # ``parse_qs``'s native ValueError message embeds the offending
        # fragment, which can include credential bytes. Our re-raised error
        # must not echo it (and the cause chain is suppressed).
        secret = "VERY_SECRET_AKIA_DO_NOT_LEAK"
        with self.assertRaises(ValueError) as ctx:
            self._build(f"s3://b/k?aws_access_key_id{secret}")
        self.assertNotIn(secret, str(ctx.exception))
        # Cause chain is suppressed so the raw value won't show up in
        # tracebacks either.
        self.assertIsNone(ctx.exception.__cause__)

    def test_fragment_is_stripped_from_clean_url(self):
        # ``s3://b/k?...#frag`` would leave ``#frag`` in ``self.url`` and
        # break downstream ``path.split("/")`` listing/downloading.
        _, conn = self._build("s3://b/k?endpoint_url=https://x#frag")
        self.assertEqual(conn.url, "s3://b/k")

    def test_fragment_only_url_is_stripped(self):
        _, conn = self._build("s3://b/k#frag")
        self.assertEqual(conn.url, "s3://b/k")

    def test_explicit_region_name_wins(self):
        url = "s3://b/k?region_name=us-west-004"
        fake_boto3, _ = self._build(url, region_name="us-east-1")
        fake_boto3.client.assert_called_once_with(
            "s3", config=ANY, region_name="us-east-1"
        )

    def test_user_agent_extra_includes_sglang_version(self):
        from sglang.version import __version__

        fake_boto3, _ = self._build("s3://bucket/path")
        _, kwargs = fake_boto3.client.call_args
        self.assertIn(f"sglang/{__version__}", kwargs["config"].user_agent_extra)

    def test_user_config_is_merged_not_dropped(self):
        from botocore.config import Config

        from sglang.version import __version__

        user_cfg = Config(user_agent_extra="caller/1.2.3", region_name="us-east-2")
        fake_boto3, _ = self._build("s3://bucket/path", config=user_cfg)
        _, kwargs = fake_boto3.client.call_args
        merged = kwargs["config"]
        self.assertIn("caller/1.2.3", merged.user_agent_extra)
        self.assertIn(f"sglang/{__version__}", merged.user_agent_extra)
        self.assertEqual(merged.region_name, "us-east-2")

    def test_explicit_none_skips_kwarg(self):
        # ``endpoint_url=None`` opts back out of any query-derived value so
        # boto3's default chain (incl. ``AWS_ENDPOINT_URL`` env) takes over.
        fake_boto3, _ = self._build(
            "s3://b/k?endpoint_url=https://from-url", endpoint_url=None
        )
        _, kwargs = fake_boto3.client.call_args
        self.assertNotIn("endpoint_url", kwargs)

    def test_region_name_none_clears_query_region(self):
        # ``region_name=None`` clears the query-string-derived ``region_name``.
        fake_boto3, _ = self._build(
            "s3://b/k?region_name=us-west-004", region_name=None
        )
        _, kwargs = fake_boto3.client.call_args
        self.assertNotIn("region_name", kwargs)

    def test_config_must_be_botocore_config(self):
        # A non-``botocore.config.Config`` ``config=`` must raise ``TypeError``
        # rather than failing later on ``.merge()`` / ``.user_agent_extra``.
        with self.assertRaises(TypeError):
            self._build("s3://bucket/path", config={"user_agent_extra": "bad"})

    def test_duplicate_query_param_raises(self):
        # Two ``endpoint_url=...`` in the URI must surface the conflict
        # explicitly rather than silently picking one.
        with self.assertRaises(ValueError) as ctx:
            self._build("s3://b/k?endpoint_url=A&endpoint_url=B")
        self.assertIn("endpoint_url", str(ctx.exception))

    def test_query_region_name_wins_over_config_region(self):
        # Documented precedence: a query-string ``?region_name=`` is passed to
        # boto3 as a direct kwarg, which boto3 ranks above ``config.region_name``,
        # so the URI value wins when both are supplied. (An explicit
        # ``region_name=`` kwarg still beats the query -- see
        # ``test_explicit_region_name_wins``.)
        from botocore.config import Config

        user_cfg = Config(region_name="us-east-2")
        fake_boto3, _ = self._build("s3://b/k?region_name=us-west-004", config=user_cfg)
        _, kwargs = fake_boto3.client.call_args
        self.assertEqual(kwargs["region_name"], "us-west-004")
        # The caller Config is still merged (not dropped); boto3 resolves the
        # conflict in favor of the direct kwarg.
        self.assertEqual(kwargs["config"].region_name, "us-east-2")

    def test_unknown_param_name_is_truncated_in_error(self):
        # A hostile/garbage key must not bloat the (possibly logged) error;
        # it is truncated rather than echoed in full.
        long_key = "x" * 500
        with self.assertRaises(ValueError) as ctx:
            self._build(f"s3://b/k?{long_key}=1")
        msg = str(ctx.exception)
        self.assertNotIn(long_key, msg)
        self.assertIn("...", msg)

    def test_unknown_param_name_control_chars_sanitized(self):
        # Control characters in an unknown key are replaced so they can't
        # corrupt terminals/logs when the error surfaces. ``%09`` == TAB.
        with self.assertRaises(ValueError) as ctx:
            self._build("s3://b/k?a%09b=1")
        self.assertNotIn("\t", str(ctx.exception))

    def test_many_unknown_params_are_capped_in_error(self):
        # A URL stuffed with junk keys must not amplify into an unbounded
        # (possibly logged) error string: only the first N are listed.
        from sglang.srt.connector.s3 import _MAX_UNKNOWN_KEYS_SHOWN

        keys = "&".join(f"junk{i}=1" for i in range(_MAX_UNKNOWN_KEYS_SHOWN + 40))
        with self.assertRaises(ValueError) as ctx:
            self._build(f"s3://b/k?{keys}")
        msg = str(ctx.exception)
        self.assertIn("more)", msg)
        self.assertLessEqual(msg.count("junk"), _MAX_UNKNOWN_KEYS_SHOWN)

    def test_sanitize_query_key_respects_max_len(self):
        # The cap must bound the final length *including* the ``...`` marker,
        # not overshoot it by 3 chars.
        from sglang.srt.connector.s3 import _sanitize_query_key

        out = _sanitize_query_key("x" * 500, max_len=64)
        self.assertLessEqual(len(out), 64)
        self.assertTrue(out.endswith("..."))

    def test_sanitize_query_key_small_max_len_never_exceeds(self):
        # Even for caps smaller than the ``...`` marker, the result must never
        # exceed ``max_len`` (the marker itself gets truncated).
        from sglang.srt.connector.s3 import _sanitize_query_key

        for max_len in (0, 1, 2, 3, 4):
            out = _sanitize_query_key("x" * 50, max_len=max_len)
            self.assertLessEqual(len(out), max_len, f"max_len={max_len}")


@unittest.skipUnless(_HAS_BOTOCORE, "botocore not installed")
class TestCreateRemoteConnectorS3Forwarding(unittest.TestCase):
    """``create_remote_connector`` forwards only whitelisted s3 kwargs."""

    def _build(self, url, **kwargs):
        fake_boto3 = MagicMock()
        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            connector = create_remote_connector(url, **kwargs)
        return fake_boto3, connector

    def test_known_s3_kwargs_are_forwarded(self):
        fake_boto3, _ = self._build(
            "s3://bucket/path",
            endpoint_url="https://s3.example.com",
            region_name="eu-west-1",
        )
        fake_boto3.client.assert_called_once_with(
            "s3",
            config=ANY,
            endpoint_url="https://s3.example.com",
            region_name="eu-west-1",
        )

    def test_unknown_kwargs_are_silently_dropped(self):
        fake_boto3, _ = self._build(
            "s3://bucket/path",
            endpoint_url="https://s3.example.com",
            some_other_connector_option="ignored",
        )
        _, kwargs = fake_boto3.client.call_args
        self.assertEqual(kwargs.get("endpoint_url"), "https://s3.example.com")
        self.assertNotIn("some_other_connector_option", kwargs)

    def test_region_alias_is_not_forwarded(self):
        # ``region`` is intentionally absent from the whitelist; passing it
        # via ``create_remote_connector`` must not propagate to boto3.
        fake_boto3, _ = self._build("s3://bucket/path", region="us-west-1")
        _, kwargs = fake_boto3.client.call_args
        self.assertNotIn("region", kwargs)
        self.assertNotIn("region_name", kwargs)

    def test_session_token_is_forwarded(self):
        fake_boto3, _ = self._build(
            "s3://bucket/path",
            aws_access_key_id="AKIA_X",
            aws_secret_access_key="SEC_X",
            aws_session_token="TOKEN_X",
        )
        fake_boto3.client.assert_called_once_with(
            "s3",
            config=ANY,
            aws_access_key_id="AKIA_X",
            aws_secret_access_key="SEC_X",
            aws_session_token="TOKEN_X",
        )

    def test_dropped_kwargs_are_logged_at_debug(self):
        # ``region`` (typo for ``region_name``) is silently dropped, but the
        # drop must show up at DEBUG so it is debuggable.
        with self.assertLogs("sglang.srt.connector", level="DEBUG") as cm:
            self._build(
                "s3://bucket/path",
                region="us-west-1",
                endpoint_url="https://s3.example.com",
            )
        joined = "\n".join(cm.output)
        self.assertIn("region", joined)
        self.assertIn("ignoring", joined)


if __name__ == "__main__":
    unittest.main()
