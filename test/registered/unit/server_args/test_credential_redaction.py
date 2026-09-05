"""Credentials must not leave the record through the diagnostic projection.

`resolved_dict()` is what the quotable exits publish: the `server_args=` line
`Engine.__init__` logs, `/server_info` and its gRPC and in-process twins. Those
land in log aggregation and answer callers holding only `api_key`, so a
credential projected verbatim is a credential handed out -- and a published
`admin_api_key` is a privilege escalation, since `/server_info` is
`AuthLevel.NORMAL` while the endpoints that key unlocks are `ADMIN_FORCE`.

The pattern guard below is the half that survives us: it fails when a *new*
credential-shaped field arrives unclassified, so the next `--*-key` cannot
reach the projection without someone deciding what it is.
"""

import dataclasses
import json
import unittest

from sglang.srt.arg_groups.arg_utils import REDACTED, secret_fields
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.auth import AuthLevel, decide_request_auth
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

API_KEY = "sentinel-api-key-2b7f"
ADMIN_API_KEY = "sentinel-admin-api-key-9c31"
SSL_KEYFILE_PASSWORD = "sentinel-ssl-keyfile-password-4e08"

# A credential word spelled as a whole `_`-separated part of the field name.
# `token` is deliberately here and `tokens` is deliberately not: the plural is
# always the LLM sense (a count of them), the singular is ambiguous enough that
# a human should look.
_CREDENTIAL_WORDS = frozenset(
    {
        "key",
        "keys",
        "token",
        "secret",
        "secrets",
        "password",
        "passwd",
        "passphrase",
        "credential",
        "credentials",
        "cred",
        "creds",
        "bearer",
    }
)

# Credential-shaped names that are not credentials, each read and classified by
# hand. Every one is the LLM sense of "token"; adding a field here is the
# reviewed alternative to marking it `secret=True`.
_REVIEWED_PUBLIC = frozenset(
    {
        "bucket_inter_token_latency",
        "bucket_time_to_first_token",
        "kt_max_deferred_experts_per_token",
        "prefill_delayer_token_usage_low_watermark",
        "speculative_token_map",
        "tbo_token_distribution_threshold",
    }
)


def _credential_shaped(name: str) -> bool:
    return bool(_CREDENTIAL_WORDS & set(name.split("_")))


def _args_with_credentials() -> ServerArgs:
    return ServerArgs(
        model_path="dummy",
        api_key=API_KEY,
        admin_api_key=ADMIN_API_KEY,
        ssl_keyfile_password=SSL_KEYFILE_PASSWORD,
    )


class TestCredentialsAreRedactedFromTheProjection(CustomTestCase):
    """No configured credential reaches `resolved_dict()` in the clear."""

    def test_configured_credentials_project_as_the_marker(self):
        dump = _args_with_credentials().resolved_dict()

        for name in ("api_key", "admin_api_key", "ssl_keyfile_password"):
            with self.subTest(field=name):
                self.assertEqual(dump[name], REDACTED)

    def test_no_sentinel_survives_serialization(self):
        # The exits differ in how they render the dict -- the log line takes
        # `str`, `/server_info` and the gRPC twin take JSON -- so pin the
        # sentinel out of both renderings rather than out of one key.
        dump = _args_with_credentials().resolved_dict()
        renderings = (str(dump), json.dumps(dump, default=str))

        for rendering in renderings:
            for sentinel in (API_KEY, ADMIN_API_KEY, SSL_KEYFILE_PASSWORD):
                with self.subTest(sentinel=sentinel):
                    self.assertNotIn(sentinel, rendering)

    def test_an_unset_credential_projects_as_none(self):
        # "No key is configured" is operational information, and it is already
        # visible from outside: an unauthenticated request is answered rather
        # than refused. Publishing the marker for an unset field would instead
        # read as a key that exists.
        dump = ServerArgs(model_path="dummy").resolved_dict()

        for name in ("api_key", "admin_api_key", "ssl_keyfile_password"):
            with self.subTest(field=name):
                self.assertIsNone(dump[name])

    def test_the_record_keeps_the_real_values(self):
        # Redaction is a property of the projection alone; the auth middleware
        # reads the fields, and a redacted field would lock every caller out.
        server_args = _args_with_credentials()
        server_args.resolved_dict()

        self.assertEqual(server_args.api_key, API_KEY)
        self.assertEqual(server_args.admin_api_key, ADMIN_API_KEY)
        self.assertEqual(server_args.ssl_keyfile_password, SSL_KEYFILE_PASSWORD)

    def test_the_projected_admin_key_no_longer_unlocks_admin_endpoints(self):
        # The escalation this closes: `/server_info` is `AuthLevel.NORMAL`, so
        # a caller holding only `api_key` reads it, and used to find
        # `admin_api_key` in the body.
        server_args = _args_with_credentials()
        dump = server_args.resolved_dict()

        self.assertTrue(
            decide_request_auth(
                method="GET",
                path="/server_info",
                authorization_header=f"Bearer {API_KEY}",
                api_key=server_args.api_key,
                admin_api_key=server_args.admin_api_key,
                auth_level=AuthLevel.NORMAL,
            ).allowed,
            "the low-privilege key still reads /server_info; that is the "
            "premise of this test, not a regression",
        )
        self.assertFalse(
            decide_request_auth(
                method="POST",
                path="/clear_hicache_storage_backend",
                authorization_header=f"Bearer {dump['admin_api_key']}",
                api_key=server_args.api_key,
                admin_api_key=server_args.admin_api_key,
                auth_level=AuthLevel.ADMIN_FORCE,
            ).allowed
        )


class TestTheOverlayCannotUnredact(CustomTestCase):
    """`/server_info` also carries the overlay, which writes raw values."""

    def test_a_credential_overridden_at_runtime_stays_redacted(self):
        server_args = _args_with_credentials()
        publish(server_args, role="tokenizer")
        try:
            rotated = "sentinel-rotated-admin-key-7d52"
            get_context().override("test", admin_api_key=rotated)

            overlaid = get_context().resolved_server_args_dict()

            self.assertEqual(overlaid["admin_api_key"], REDACTED)
            self.assertNotIn(rotated, json.dumps(overlaid, default=str))
        finally:
            reset_context()


class TestRedactionIsNarrow(CustomTestCase):
    """Redaction takes the credentials and nothing else."""

    def test_only_the_marked_fields_are_redacted(self):
        marked = secret_fields(ServerArgs)
        self.assertEqual(marked, {"api_key", "admin_api_key", "ssl_keyfile_password"})

        dump = _args_with_credentials().resolved_dict()
        redacted = {name for name, value in dump.items() if value == REDACTED}

        self.assertEqual(redacted, marked)

    def test_ordinary_fields_still_carry_their_resolved_values(self):
        server_args = _args_with_credentials()

        dump = server_args.resolved_dict()

        self.assertEqual(dump["model_path"], "dummy")
        self.assertEqual(dump["port"], server_args.port)
        self.assertEqual(dump["tp_size"], server_args.tp_size)


class TestNewCredentialFieldsMustBeClassified(CustomTestCase):
    """The guard that outlives this fix.

    Only top-level `ServerArgs` fields are checked. A credential reached
    through a nested config dataclass would need its own marker; there is no
    such field today.
    """

    def test_no_credential_shaped_field_is_unclassified(self):
        marked = secret_fields(ServerArgs)
        unclassified = sorted(
            field.name
            for field in dataclasses.fields(ServerArgs)
            if _credential_shaped(field.name)
            and field.name not in marked
            and field.name not in _REVIEWED_PUBLIC
        )

        self.assertEqual(
            unclassified,
            [],
            "these ServerArgs fields are named like credentials and are "
            "published verbatim by /server_info and the startup log line. "
            "Mark each one Arg(secret=True) if it holds a credential, or add "
            "it to _REVIEWED_PUBLIC in this file if it does not.",
        )

    def test_the_reviewed_public_list_has_not_gone_stale(self):
        # A name that no longer exists, or no longer looks like a credential,
        # stops being an exemption and starts being cover for the next one.
        stale = sorted(
            name
            for name in _REVIEWED_PUBLIC
            if name not in {field.name for field in dataclasses.fields(ServerArgs)}
            or not _credential_shaped(name)
        )

        self.assertEqual(stale, [])

    def test_the_pattern_catches_the_fields_we_know_about(self):
        # Guards the guard: a pattern that matched nothing would pass the
        # unclassified check for free.
        for name in ("api_key", "admin_api_key", "ssl_keyfile_password"):
            with self.subTest(field=name):
                self.assertTrue(_credential_shaped(name))


if __name__ == "__main__":
    unittest.main()
