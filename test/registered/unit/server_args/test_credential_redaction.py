"""Credentials must not leave the record through the readbacks.

Two strings publish the whole record: `resolved_dict()` (the `server_args=`
line `Engine.__init__` and `_launch_subprocesses` log, `/server_info` and its
gRPC and in-process twins) and `launch_command` (the argv the launcher parsed,
or the `Engine(...)` call, carried by the same three readbacks). Both land in
log aggregation and answer callers holding only `api_key`, so a credential
published verbatim is a credential handed out, and a published
`admin_api_key` is a privilege escalation: `/server_info` is
`AuthLevel.NORMAL` while the endpoints that key unlocks are `ADMIN_FORCE`.

The pattern guard at the end is the half that outlives this fix: it fails when
a new credential-shaped field arrives unclassified, so the next `--*-key`
cannot reach a readback without someone deciding what it is.
"""

import glob
import json
import os
import pickle
import tempfile
import unittest
from collections import deque
from types import SimpleNamespace

import msgspec.structs

from sglang.srt.arg_groups.arg_utils import (
    REDACTED,
    cli_flags,
    redacted_argv,
    redacted_call,
    redacted_value,
    secret_fields,
)
from sglang.srt.environ import envs
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.srt.utils.auth import AuthLevel, decide_request_auth
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _sentinel(*parts: str) -> str:
    """A recognisable fake credential, assembled at runtime so a secret scanner
    reading this file does not see a literal credential assignment."""
    return "-".join(("sentinel", *parts))


API_KEY = _sentinel("api", "key", "2b7f")
ADMIN_API_KEY = _sentinel("admin", "api", "key", "9c31")
SSL_KEYFILE_PASSWORD = _sentinel("ssl", "keyfile", "pw", "4e08")
SENTINELS = (API_KEY, ADMIN_API_KEY, SSL_KEYFILE_PASSWORD)
CREDENTIAL_FIELDS = ("api_key", "admin_api_key", "ssl_keyfile_password")

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


def _field_names() -> set:
    return {field.name for field in msgspec.structs.fields(ServerArgs)}


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

        for name in CREDENTIAL_FIELDS:
            with self.subTest(field=name):
                self.assertEqual(dump[name], "<redacted:1 value>")

    def test_the_marker_keeps_the_count_of_values_and_nothing_else(self):
        # The one diagnostic a redacted field keeps (review on #18518): whether
        # auth is configured, and, for a field that carries a set of keys, how
        # many parsed. A single string is one value; a collection is its length.
        self.assertEqual(redacted_value(API_KEY), "<redacted:1 value>")
        self.assertEqual(
            redacted_value(("k1", "k2", "k3", "k4", "k5", "k6")), "<redacted:6 values>"
        )
        self.assertEqual(redacted_value([]), "<redacted:0 values>")

        # A record whose credential field holds several values (the shape a
        # comma-separated `--api-key` would parse into) publishes the count.
        many = ServerArgs(
            model_path="dummy", api_key=[_sentinel("k", str(i)) for i in range(3)]
        )
        dump = many.resolved_dict()
        self.assertEqual(dump["api_key"], "<redacted:3 values>")
        for i in range(3):
            self.assertNotIn(_sentinel("k", str(i)), json.dumps(dump, default=str))

    def test_no_sentinel_survives_serialization(self):
        # The exits differ in how they render the dict: the log line takes
        # `str`, `/server_info` and the gRPC twin take JSON. Pin the sentinel
        # out of both renderings rather than out of one key.
        dump = _args_with_credentials().resolved_dict()

        for rendering in (str(dump), json.dumps(dump, default=str)):
            for sentinel in SENTINELS:
                with self.subTest(sentinel=sentinel):
                    self.assertNotIn(sentinel, rendering)

    def test_an_unset_credential_projects_as_none(self):
        # "No key is configured" is operational information, and it is already
        # visible from outside: an unauthenticated request is answered rather
        # than refused. Publishing the marker for an unset field would instead
        # read as a key that exists.
        dump = ServerArgs(model_path="dummy").resolved_dict()

        for name in CREDENTIAL_FIELDS:
            with self.subTest(field=name):
                self.assertIsNone(dump[name])

    def test_the_record_keeps_the_real_values(self):
        # Redaction is a property of the readbacks alone; the auth middleware
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


class TestTheLaunchCommandIsRedacted(CustomTestCase):
    """`launch_command` is the other whole-record string the readbacks carry."""

    def test_the_launcher_redacts_both_argparse_spellings(self):
        server_args = prepare_server_args(
            [
                "--model-path",
                "/tmp/x",
                "--api-key",
                API_KEY,
                f"--admin-api-key={ADMIN_API_KEY}",
                "--ssl-keyfile-password",
                SSL_KEYFILE_PASSWORD,
                "--log-level",
                "warning",
            ]
        )

        self.assertEqual(
            server_args.launch_command,
            f"--model-path /tmp/x --api-key {REDACTED} "
            f"--admin-api-key={REDACTED} --ssl-keyfile-password {REDACTED} "
            "--log-level warning",
        )
        # The record still parsed the real values.
        self.assertEqual(server_args.api_key, API_KEY)
        self.assertEqual(server_args.admin_api_key, ADMIN_API_KEY)
        self.assertEqual(server_args.ssl_keyfile_password, SSL_KEYFILE_PASSWORD)

    def test_a_value_that_merely_looks_like_a_flag_is_kept(self):
        # Only the token after a credential flag is hidden; a later flag's
        # value that happens to contain "key" is configuration, not a secret.
        argv = ["--model-path", "/models/api-key-test", "--api-key", API_KEY]

        self.assertEqual(
            redacted_argv(ServerArgs, argv),
            ["--model-path", "/models/api-key-test", "--api-key", REDACTED],
        )

    def test_abbreviated_credential_flags_are_redacted(self):
        # argparse accepts any unique prefix of a flag (`allow_abbrev` is on),
        # so `--api-k` parses into `api_key` exactly like `--api-key`. The
        # launch command must hide the value under the spelling as typed.
        server_args = prepare_server_args(
            [
                "--model-path",
                "/tmp/x",
                "--api-k",
                API_KEY,
                f"--admin-api-k={ADMIN_API_KEY}",
                "--ssl-keyfile-p",
                SSL_KEYFILE_PASSWORD,
            ]
        )

        self.assertEqual(
            server_args.launch_command,
            f"--model-path /tmp/x --api-k {REDACTED} --admin-api-k={REDACTED} "
            f"--ssl-keyfile-p {REDACTED}",
        )
        self.assertEqual(server_args.api_key, API_KEY)
        self.assertEqual(server_args.admin_api_key, ADMIN_API_KEY)
        self.assertEqual(server_args.ssl_keyfile_password, SSL_KEYFILE_PASSWORD)

    def test_the_equals_spelling_of_every_credential_flag_is_redacted(self):
        argv = [
            f"--api-key={API_KEY}",
            f"--admin-api-key={ADMIN_API_KEY}",
            f"--ssl-keyfile-password={SSL_KEYFILE_PASSWORD}",
        ]

        self.assertEqual(
            redacted_argv(ServerArgs, argv),
            [
                f"--api-key={REDACTED}",
                f"--admin-api-key={REDACTED}",
                f"--ssl-keyfile-password={REDACTED}",
            ],
        )

    def test_a_flag_that_is_itself_a_prefix_of_a_credential_flag_is_kept(self):
        # `--ssl-keyfile` is a registered flag and a prefix of the credential
        # flag. argparse takes the exact match; so does the scrubber.
        self.assertIn("--ssl-keyfile", cli_flags(ServerArgs))
        argv = [
            "--ssl-keyfile",
            "/etc/sglang/key.pem",
            "--ssl-keyfile-password",
            SSL_KEYFILE_PASSWORD,
        ]

        self.assertEqual(
            redacted_argv(ServerArgs, argv),
            [
                "--ssl-keyfile",
                "/etc/sglang/key.pem",
                "--ssl-keyfile-password",
                REDACTED,
            ],
        )

    def test_an_ambiguous_abbreviation_is_not_guessed(self):
        # `--a` is a prefix of many flags. argparse rejects it, so there is no
        # parse to redact; a scrubber that guessed would hide configuration.
        self.assertGreater(
            sum(flag.startswith("--a") for flag in cli_flags(ServerArgs)), 1
        )

        self.assertEqual(redacted_argv(ServerArgs, ["--a", API_KEY]), ["--a", API_KEY])

    def test_a_parsed_credential_is_hidden_under_a_spelling_the_metadata_does_not_know(
        self,
    ):
        # The backstop: a flag registered outside the record with a credential
        # `dest` is invisible to the flag pass, but the value it parsed is on
        # the record, and the record is what the launcher hands over.
        record = SimpleNamespace(
            api_key=API_KEY, admin_api_key=ADMIN_API_KEY, ssl_keyfile_password=None
        )
        argv = ["--future-alias", API_KEY, f"--other-alias={ADMIN_API_KEY}"]

        self.assertEqual(
            redacted_argv(ServerArgs, argv, record),
            ["--future-alias", REDACTED, f"--other-alias={REDACTED}"],
        )
        # Without the record the flag pass alone runs, and it does not know
        # these spellings: the backstop is what closes them.
        self.assertEqual(redacted_argv(ServerArgs, argv), argv)

    def test_a_value_that_shares_a_prefix_with_a_credential_is_kept(self):
        # The backstop compares whole tokens. A name that merely starts with
        # the key is configuration and stays readable.
        lookalike = f"{API_KEY}-public"
        server_args = prepare_server_args(
            ["--model-path", "/tmp/x", "--served-model-name", lookalike]
            + ["--api-key", API_KEY]
        )

        self.assertEqual(
            server_args.launch_command,
            f"--model-path /tmp/x --served-model-name {lookalike} --api-key {REDACTED}",
        )

    def test_a_credential_reused_as_a_known_flags_value_stays_readable(self):
        # The backstop compares whole tokens, so a short key collides with
        # ordinary configuration: `--api-key 1 --tp 1` would hide the `--tp`
        # value. The value of a flag argparse bound to a public field of the
        # record (exact spelling, alias, or unique abbreviation, as the next
        # token or the `=` half) is left as typed; the metadata knows that
        # field is not a secret. No length threshold: that would under-hide
        # short keys in the positions where nothing says what they are.
        cases = [
            (
                ["--api-key", "1", "--tp", "1"],
                f"--api-key {REDACTED} --tp 1",
            ),
            (
                ["--model-path", "m", "--api-key", "m"],
                f"--model-path m --api-key {REDACTED}",
            ),
            (
                ["--api-key", "info", "--log-level", "info"],
                f"--api-key {REDACTED} --log-level info",
            ),
            (
                ["--served-model-name", API_KEY, "--api-key", API_KEY],
                f"--served-model-name {API_KEY} --api-key {REDACTED}",
            ),
            (
                [f"--served-model-name={API_KEY}", f"--api-key={API_KEY}"],
                f"--served-model-name={API_KEY} --api-key={REDACTED}",
            ),
        ]
        for argv, expected in cases:
            with self.subTest(argv=argv):
                if "--model-path" not in argv:
                    argv = ["--model-path", "/tmp/x"] + argv
                    expected = "--model-path /tmp/x " + expected
                server_args = prepare_server_args(argv)
                self.assertEqual(server_args.launch_command, expected)

    def test_a_credential_reused_anywhere_else_is_still_hidden(self):
        # Only a known public flag vouches for the token after it. After a flag
        # the metadata does not know, or standing alone, a matching token is
        # hidden: nothing says what argparse bound it to, and the other way
        # round publishes the key.
        record = SimpleNamespace(
            api_key=API_KEY, admin_api_key=ADMIN_API_KEY, ssl_keyfile_password=None
        )
        argv = [
            "--future-alias",
            API_KEY,
            f"--other-alias={ADMIN_API_KEY}",
            API_KEY,
            "--tp=1",
            ADMIN_API_KEY,
        ]

        self.assertEqual(
            redacted_argv(ServerArgs, argv, record),
            [
                "--future-alias",
                REDACTED,
                f"--other-alias={REDACTED}",
                REDACTED,
                "--tp=1",
                REDACTED,
            ],
        )

    def test_a_trailing_credential_flag_without_a_value_is_left_alone(self):
        # argparse rejects it; the scrubber must not invent a value.
        self.assertEqual(
            redacted_argv(ServerArgs, ["--model-path", "/tmp/x", "--api-key"]),
            ["--model-path", "/tmp/x", "--api-key"],
        )

    def test_the_engine_call_redacts_its_credential_kwargs(self):
        # `Engine.__init__` records the call that built the record through
        # `redacted_call`; an unset credential kwarg is spelled as typed.
        kwargs = {
            "model_path": "dummy",
            "api_key": API_KEY,
            "admin_api_key": None,
            "log_level": "error",
        }

        self.assertEqual(
            redacted_call(ServerArgs, "Engine", kwargs),
            "Engine(model_path='dummy', api_key='<redacted:1 value>', "
            "admin_api_key=None, log_level='error')",
        )
        # a kwarg that carries several credentials publishes their count
        self.assertEqual(
            redacted_call(ServerArgs, "Engine", {"api_key": [API_KEY, ADMIN_API_KEY]}),
            "Engine(api_key='<redacted:2 values>')",
        )


class TestTheCrashDumpCarriesTheRedactedLaunchCommand(CustomTestCase):
    """The tokenizer crash dump wrote `" ".join(sys.argv)` next to a record
    whose `launch_command` was already redacted: one more file with the key in
    plain text. The pickled record itself keeps the real values, so the dump
    can be replayed."""

    def test_the_dump_writes_the_records_launch_command(self):
        # `dummy` is the model path that ends resolution before the device
        # probe, so the record can be published on a machine without one.
        server_args = prepare_server_args(
            ["--model-path", "dummy", "--api-key", API_KEY]
        )
        # `__new__` skips `__init__`, which would open the ZMQ sockets; the
        # dump reads only what is set here.
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.server_args = server_args
        manager.model_path = server_args.model_path
        manager.served_model_name = server_args.served_model_name
        manager.crash_dump_performed = False
        manager.crash_dump_request_list = deque([("req", {"text": "out"}, 0.0, 1.0)])
        manager.rid_to_state = {}
        publish(server_args, role="tokenizer")
        try:
            with tempfile.TemporaryDirectory() as folder:
                manager.crash_dump_folder = folder
                # Off, or the dump sleeps and runs py-spy on this process.
                with (
                    envs.SGLANG_PYSPY_DUMP_BEFORE_CRASH.override(False),
                    envs.SGLANG_CUDA_COREDUMP_BEFORE_CRASH.override(False),
                ):
                    manager.dump_requests_before_crash(hostname="host")
                files = glob.glob(os.path.join(folder, "host", "crash_dump_*.pkl"))
                self.assertEqual(len(files), 1)
                with open(files[0], "rb") as f:
                    dump = pickle.load(f)
        finally:
            reset_context()

        self.assertEqual(
            dump["launch_command"], f"--model-path dummy --api-key {REDACTED}"
        )
        self.assertNotIn(API_KEY, dump["launch_command"])
        self.assertEqual(dump["resolved_config"]["api_key"], REDACTED)
        # The record is the replayable half and keeps the value, by design.
        self.assertEqual(dump["server_args"].api_key, API_KEY)


class TestTheOverlayCannotUnredact(CustomTestCase):
    """`/server_info` also carries the overlay, which writes raw values."""

    def test_a_credential_overridden_at_runtime_stays_redacted(self):
        server_args = _args_with_credentials()
        publish(server_args, role="tokenizer")
        try:
            rotated = _sentinel("rotated", "admin", "key", "7d52")
            get_context().override("test", admin_api_key=rotated)

            overlaid = get_context().resolved_server_args_dict()

            self.assertEqual(overlaid["admin_api_key"], "<redacted:1 value>")
            self.assertNotIn(rotated, json.dumps(overlaid, default=str))
        finally:
            reset_context()


class TestRedactionIsNarrow(CustomTestCase):
    """Redaction takes the credentials and nothing else."""

    def test_only_the_marked_fields_are_redacted(self):
        marked = secret_fields(ServerArgs)
        self.assertEqual(marked, set(CREDENTIAL_FIELDS))

        dump = _args_with_credentials().resolved_dict()
        redacted = {
            name for name, value in dump.items() if value == "<redacted:1 value>"
        }

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
    through a nested config record would need its own marker; there is no
    such field today.
    """

    def test_no_credential_shaped_field_is_unclassified(self):
        marked = secret_fields(ServerArgs)
        unclassified = sorted(
            name
            for name in _field_names()
            if _credential_shaped(name)
            and name not in marked
            and name not in _REVIEWED_PUBLIC
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
        names = _field_names()
        stale = sorted(
            name
            for name in _REVIEWED_PUBLIC
            if name not in names or not _credential_shaped(name)
        )

        self.assertEqual(stale, [])

    def test_the_pattern_catches_the_fields_we_know_about(self):
        # Guards the guard: a pattern that matched nothing would pass the
        # unclassified check for free.
        for name in CREDENTIAL_FIELDS:
            with self.subTest(field=name):
                self.assertTrue(_credential_shaped(name))


if __name__ == "__main__":
    unittest.main()
