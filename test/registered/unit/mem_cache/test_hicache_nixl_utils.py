"""Unit tests for the nixl_utils config parsing and O_DIRECT probe helpers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import ctypes
import errno
import fcntl
import mmap
import os
import shutil
import tempfile
import unittest
from unittest import mock

from sglang.srt.mem_cache.storage.nixl import nixl_utils
from sglang.srt.mem_cache.storage.nixl.nixl_utils import (
    NixlBackendConfig,
    NixlFileManager,
)
from sglang.test.test_utils import CustomTestCase

_UTILS_LOGGER = "sglang.srt.mem_cache.storage.nixl.nixl_utils"
_PAGE_BYTES = 4096

# SGLang-only keys and a value that differs from the built-in default, so a
# dropped promotion is visible in the parsed config.
_SGLANG_KEYS = {
    "use_direct_io": False,
    "l3_cleaner_enabled": False,
    "l3_cleaner_high_watermark": 85,
    "l3_cleaner_low_watermark": 75,
}


class TestNixlBackendInitParams(CustomTestCase):
    def test_initparams_keep_lowercase_booleans(self):
        """JSON booleans must reach NIXL lowercase; its plugins match case-sensitively
        and silently ignore ``"True"``."""
        cfg = NixlBackendConfig(
            {
                "plugin": {
                    "posix": {
                        "active": True,
                        "use_uring": True,
                        "use_aio": False,
                        "num_threads": 4,
                        "bucket": "Cache",
                    }
                }
            }
        )

        self.assertEqual(
            cfg.get_backend_initparams("POSIX"),
            {
                "use_uring": "true",
                "use_aio": "false",
                "num_threads": "4",
                "bucket": "Cache",
            },
        )

    def test_only_the_bool_repr_of_a_string_value_is_folded(self):
        """An option value is an opaque case-sensitive string to NIXL, so folding
        every casing would corrupt a value that really is ``"TRUE"``.
        """
        cfg = NixlBackendConfig({"plugin": {"posix": {"secret_key": "TRUE"}}})

        self.assertEqual(cfg.get_backend_initparams("POSIX"), {"secret_key": "TRUE"})

    def test_folding_a_stringified_bool_is_reported(self):
        """Folding is a guess about a case-sensitive value, so it cannot be silent."""
        cfg = NixlBackendConfig({"plugin": {"posix": {"use_gds": "True"}}})

        with self.assertLogs(_UTILS_LOGGER, level="WARNING") as logs:
            initparams = cfg.get_backend_initparams("POSIX")

        self.assertEqual(initparams, {"use_gds": "true"})
        self.assertTrue(any("use_gds" in line for line in logs.output))

    def test_plugin_selector_is_not_forwarded_to_the_plugin(self):
        """``active`` picks the plugin for SGLang; no NIXL plugin declares it, so
        forwarding it passes create_backend an unknown option.
        """
        cfg = NixlBackendConfig({"plugin": {"posix": {"active": True}}})

        self.assertEqual(cfg.get_specified_plugin(), "POSIX")
        self.assertEqual(cfg.get_backend_initparams("POSIX"), {})

    def test_flat_form_initparams_are_normalized_too(self):
        """The flat form is the spelling for a single selected plugin and reaches
        the same NIXL create_backend call, so it needs the same normalization.
        """
        cfg = NixlBackendConfig({"use_uring": True, "num_threads": 4})

        self.assertEqual(
            cfg.get_backend_initparams("POSIX"),
            {"use_uring": "true", "num_threads": "4"},
        )

    def test_sglang_keys_nested_under_plugin_are_promoted(self):
        """SGLang-only keys are read from the top level, so nesting them the way
        the fully-qualified form nests plugin options left them at their defaults.
        """
        with self.assertLogs(_UTILS_LOGGER, level="WARNING") as logs:
            cfg = NixlBackendConfig(
                {"plugin": {"posix": {"active": True, **_SGLANG_KEYS}}}
            )

        cleaner = cfg.get_l3_cleaner_config()
        self.assertFalse(cfg.get_use_direct_io())
        self.assertFalse(cleaner["enabled"])
        self.assertEqual(cleaner["high_watermark"], 85.0)
        self.assertEqual(cleaner["low_watermark"], 75.0)
        self.assertEqual(cfg.get_backend_initparams("POSIX"), {})
        for key in _SGLANG_KEYS:
            self.assertTrue(
                any(key in line for line in logs.output),
                f"no warning names the misplaced key {key}",
            )

    def test_top_level_sglang_key_wins_over_nested(self):
        """Promotion must not override the supported top-level spelling."""
        with self.assertLogs(_UTILS_LOGGER, level="WARNING"):
            cfg = NixlBackendConfig(
                {
                    "plugin": {"posix": {"active": True, "use_direct_io": True}},
                    "use_direct_io": False,
                }
            )

        self.assertFalse(cfg.get_use_direct_io())

    def test_conflicting_nested_sglang_keys_are_not_promoted(self):
        """A key nested under two plugins with different values has no single
        meaning at the top level, so neither value may be picked.
        """
        with self.assertLogs(_UTILS_LOGGER, level="WARNING"):
            cfg = NixlBackendConfig(
                {
                    "plugin": {
                        "posix": {"active": True, "use_direct_io": True},
                        "gds": {"use_direct_io": False},
                    }
                }
            )

        self.assertNotIn("use_direct_io", cfg.config)

    def test_wrongly_typed_nested_sglang_keys_are_not_promoted(self):
        """A mistyped nested SGLang key is ignored with a warning: it neither takes
        effect nor fails startup."""
        with mock.patch.dict(os.environ, {"SGLANG_HICACHE_NIXL_USE_DIRECT_IO": "1"}):
            with self.assertLogs(_UTILS_LOGGER, level="WARNING") as logs:
                cfg = NixlBackendConfig(
                    {
                        "plugin": {
                            "posix": {
                                "active": True,
                                "use_direct_io": "false",
                                "l3_cleaner_enabled": "true",
                                "l3_cleaner_high_watermark": "high",
                            }
                        }
                    }
                )

            self.assertTrue(cfg.get_use_direct_io())
        self.assertEqual(
            cfg.get_l3_cleaner_config(),
            {"enabled": True, "high_watermark": 80.0, "low_watermark": 70.0},
        )
        self.assertTrue(
            all("wrong type and is ignored" in line for line in logs.output)
        )

    def test_wrongly_typed_top_level_sglang_keys_are_rejected(self):
        """A string ``"false"`` for use_direct_io or a boolean watermark must raise,
        not be coerced into the opposite or a wrong value."""
        cfg = NixlBackendConfig({"use_direct_io": "false"})
        with self.assertRaisesRegex(ValueError, "use_direct_io must be a boolean"):
            cfg.get_use_direct_io()

        cfg = NixlBackendConfig({"l3_cleaner_high_watermark": True})
        with self.assertRaisesRegex(ValueError, "l3_cleaner_high_watermark"):
            cfg.get_l3_cleaner_config()

    def test_promotion_leaves_the_caller_extra_config_untouched(self):
        """extra_config is shared server-wide; promoting into it would make the
        parse order of two backends observable.
        """
        extra_config = {"plugin": {"posix": {"active": True, "use_direct_io": False}}}

        with self.assertLogs(_UTILS_LOGGER, level="WARNING"):
            NixlBackendConfig(extra_config)

        self.assertEqual(
            extra_config,
            {"plugin": {"posix": {"active": True, "use_direct_io": False}}},
        )

    def test_plugin_section_without_an_active_plugin_uses_the_env_selector(self):
        """A plugin section with no active plugin, or that is not a mapping, must fall
        back to the env selector instead of raising at startup."""
        for plugin in (
            {},
            {"posix": {}},
            "posix",
            {"posix": "active"},
            [{"posix": {}}],
        ):
            with self.subTest(plugin=plugin):
                cfg = NixlBackendConfig({"plugin": plugin})
                # The scan for misplaced keys must pass a malformed section through.
                self.assertEqual(cfg.config["plugin"], plugin)

                with mock.patch.dict(
                    os.environ, {"SGLANG_HICACHE_NIXL_BACKEND_PLUGIN": "POSIX"}
                ):
                    with self.assertLogs(_UTILS_LOGGER, level="WARNING"):
                        self.assertEqual(cfg.get_specified_plugin(), "POSIX")
                # Top-level keys are not init params for a fully-qualified config, so
                # the malformed section leaves the plugin with its own defaults.
                self.assertEqual(cfg.get_backend_initparams("POSIX"), {})


@unittest.skipUnless(hasattr(os, "O_DIRECT"), "O_DIRECT not available on this platform")
class TestNixlDirectIOProbe(CustomTestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_nixl_direct_io_probe_")
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)
        self._require_direct_io_filesystem()
        self.file_manager = NixlFileManager(self.test_dir, use_direct_io=True)

    def _aligned_buffer(self, size: int = _PAGE_BYTES) -> int:
        """Return the address of an anonymous mapping, which is page-aligned."""
        buf = mmap.mmap(-1, size)
        self.addCleanup(buf.close)
        return ctypes.addressof(ctypes.c_char.from_buffer(buf))

    def _require_direct_io_filesystem(self) -> None:
        """Skip when the temp filesystem rejects O_DIRECT; checked with raw syscalls,
        not the code under test."""
        addr = self._aligned_buffer()
        path = os.path.join(self.test_dir, "fs_support_probe")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o644)
        except OSError as e:
            self.skipTest(f"{self.test_dir} rejects O_DIRECT opens: {e}")
        try:
            written = os.pwrite(fd, (ctypes.c_char * _PAGE_BYTES).from_address(addr), 0)
        except OSError as e:
            self.skipTest(f"{self.test_dir} rejects O_DIRECT writes: {e}")
        finally:
            os.close(fd)
            os.unlink(path)
        if written != _PAGE_BYTES:
            self.skipTest(f"{self.test_dir} short-wrote an O_DIRECT page")

    def _leftover_files(self, directory: str) -> list:
        return [
            name
            for name in os.listdir(directory)
            if not os.path.isdir(os.path.join(directory, name))
        ]

    def test_probe_reports_source_buffer_o_direct_rejects(self):
        """O_DIRECT pins its source buffer with get_user_pages(), which returns
        EFAULT for some page-aligned allocations; only a write attempt detects it.
        """
        # 0x1000 is page-aligned and always below vm.mmap_min_addr (0x10000 by
        # default), so it is unmapped and reproduces the EFAULT an XPU pinned host
        # allocation returns.
        error = self.file_manager.direct_io_error(0x1000, _PAGE_BYTES)

        self.assertIsNotNone(error)
        self.assertIn(f"errno {errno.EFAULT}", error)
        self.assertEqual(self._leftover_files(self.test_dir), [])

    def test_probe_keeps_o_direct_on_a_disk_error(self):
        """A full or flaky disk at startup says nothing about the buffer, so it must
        not switch the whole process to buffered I/O for its lifetime.
        """
        for code in (errno.ENOSPC, errno.EIO):
            with self.subTest(errno=errno.errorcode[code]):
                with (
                    mock.patch.object(
                        nixl_utils.os, "pwrite", side_effect=OSError(code, "disk")
                    ),
                    self.assertLogs(_UTILS_LOGGER, level="WARNING"),
                ):
                    error = self.file_manager.direct_io_error(
                        self._aligned_buffer(), _PAGE_BYTES
                    )

                self.assertIsNone(error)
                self.assertEqual(self._leftover_files(self.test_dir), [])

    def test_probe_rejects_on_einval(self):
        """EINVAL is how O_DIRECT rejects a buffer it cannot use, so it must still
        disable O_DIRECT; a filter that only kept EFAULT would miss it.
        """
        with mock.patch.object(
            nixl_utils.os, "pwrite", side_effect=OSError(errno.EINVAL, "invalid")
        ):
            error = self.file_manager.direct_io_error(
                self._aligned_buffer(), _PAGE_BYTES
            )

        self.assertIsNotNone(error)
        self.assertIn(f"errno {errno.EINVAL}", error)

    def test_probe_accepts_page_aligned_host_buffer(self):
        """A probe that always failed would silently disable O_DIRECT everywhere."""
        self.assertIsNone(
            self.file_manager.direct_io_error(self._aligned_buffer(), _PAGE_BYTES)
        )
        # The L3 cleaner only walks bucket directories, so a scratch file left in
        # a base directory is never reclaimed.
        self.assertEqual(self._leftover_files(self.test_dir), [])

    def test_probe_checks_every_base_directory(self):
        """Base directories are separate mounts, so a later one that rejects O_DIRECT
        must be caught even when the first is usable."""
        unsupported = os.path.join(self.test_dir, "unsupported")
        os.mkdir(unsupported)
        self.file_manager.base_dirs.append(unsupported)
        open_probe_file = self.file_manager._open_probe_file

        def reject_unsupported(base):
            if base == unsupported:
                raise OSError(errno.EINVAL, "Invalid argument")
            return open_probe_file(base)

        with mock.patch.object(
            self.file_manager, "_open_probe_file", side_effect=reject_unsupported
        ):
            error = self.file_manager.direct_io_error(
                self._aligned_buffer(), _PAGE_BYTES
            )

        self.assertIsNotNone(error)
        self.assertIn("unsupported", error)
        self.assertIn(f"errno {errno.EINVAL}", error)

    def test_probe_keeps_o_direct_when_the_scratch_open_fails_transiently(self):
        """A full disk or an fd-tight startup says nothing about O_DIRECT, so a
        failed probe open must not switch the process to buffered I/O for good.
        """
        for code in (errno.ENOSPC, errno.EMFILE):
            with self.subTest(errno=errno.errorcode[code]):
                with (
                    mock.patch.object(
                        self.file_manager,
                        "_open_probe_file",
                        side_effect=OSError(code, "transient"),
                    ),
                    self.assertLogs(_UTILS_LOGGER, level="WARNING"),
                ):
                    error = self.file_manager.direct_io_error(
                        self._aligned_buffer(), _PAGE_BYTES
                    )

                self.assertIsNone(error)

    def test_probe_file_has_no_directory_entry_while_open(self):
        """A SIGKILL between open and unlink must not leave a file the L3 cleaner
        never reclaims, so the probe file is unlinked by construction.
        """
        self._require_o_tmpfile_filesystem()

        fd, path = self.file_manager._open_probe_file(self.test_dir)
        try:
            self.assertIsNone(path)
            self.assertEqual(self._leftover_files(self.test_dir), [])
        finally:
            os.close(fd)

    def _require_o_tmpfile_filesystem(self) -> None:
        """Skip only where the filesystem has no O_TMPFILE; checked with a raw syscall,
        not the code under test."""
        if not hasattr(os, "O_TMPFILE"):
            self.skipTest("O_TMPFILE not available on this platform")
        try:
            fd = os.open(self.test_dir, os.O_WRONLY | os.O_TMPFILE, 0o644)
        except OSError as e:
            self.skipTest(f"{self.test_dir} does not support O_TMPFILE: {e}")
        os.close(fd)

    def test_probe_falls_back_to_a_named_file_without_o_tmpfile(self):
        """O_TMPFILE is Linux 3.11+ and not on every filesystem; the probe still
        has to answer, and still has to clean up after itself.
        """
        with mock.patch.object(nixl_utils, "_O_TMPFILE", 0):
            fd, path = self.file_manager._open_probe_file(self.test_dir)
            os.close(fd)
            self.assertIsNotNone(path)
            os.unlink(path)

            self.assertIsNone(
                self.file_manager.direct_io_error(self._aligned_buffer(), _PAGE_BYTES)
            )
        self.assertEqual(self._leftover_files(self.test_dir), [])

    def test_probe_reports_platforms_without_o_direct(self):
        """On a platform where the flag does not exist the probe cannot open
        anything, so it must report that instead of claiming O_DIRECT works.
        """
        with mock.patch.object(nixl_utils, "_O_DIRECT", 0):
            error = self.file_manager.direct_io_error(
                self._aligned_buffer(), _PAGE_BYTES
            )

        self.assertIsNotNone(error)
        self.assertIn("not available", error)

    def test_disable_direct_io_applies_to_later_opens(self):
        """The fallback has to reach files opened after the probe, not just the flag."""
        path = self.file_manager.get_file_path("page-123")
        fd = self.file_manager.open_file(path, create=True)
        try:
            # Without this the assertion below also passes on a manager that never
            # sets O_DIRECT in the first place.
            self.assertTrue(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_DIRECT)
        finally:
            os.close(fd)

        self.file_manager.disable_direct_io("probe failed")

        fd = self.file_manager.open_file(path, create=True)
        try:
            self.assertFalse(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_DIRECT)
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
