"""Unit tests for NIXL FILE tenant directory helpers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import shutil
import tempfile
import unittest

from sglang.srt.mem_cache.storage.nixl.nixl_tenant import (
    STORAGE_LAYOUT_VERSION,
    prepare_tenant_storage_dirs,
    resolve_tenant_key,
    start_tenant_cleanup,
)
from sglang.test.test_utils import CustomTestCase


class TestNixlTenantStorage(CustomTestCase):
    """Tests for tenant-scoped NIXL FILE storage directories."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_nixl_tenant_")
        self.base_dirs = [os.path.join(self.test_dir, f"disk{i}") for i in range(2)]
        for base_dir in self.base_dirs:
            os.makedirs(base_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _write_file(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x")

    def test_resolve_tenant_key_is_stable_and_safe(self):
        """Tenant keys include model, TP, disk count, and layout version."""
        key = resolve_tenant_key("/models/foo/bar", tp_size=8, num_disks=2)
        self.assertEqual(key, f"models-foo-bar-tp8-n2-{STORAGE_LAYOUT_VERSION}")

        explicit = resolve_tenant_key(
            "ignored", tp_size=4, num_disks=1, explicit_tenant="prod/app@1"
        )
        self.assertEqual(explicit, f"prod-app-1-tp4-n1-{STORAGE_LAYOUT_VERSION}")

    def test_prepare_cleans_only_foreign_tenant_dirs(self):
        """Startup cleanup deletes tenant siblings, not arbitrary base-dir content."""
        current = resolve_tenant_key("model-a", tp_size=8, num_disks=2)
        foreign = resolve_tenant_key("model-b", tp_size=8, num_disks=2)
        unmarked = resolve_tenant_key("manual-data", tp_size=8, num_disks=2)

        for base_dir in self.base_dirs:
            self._write_file(os.path.join(base_dir, foreign, "00", "old"))
            self._write_file(os.path.join(base_dir, foreign, ".sglang-nixl-tenant"))
            self._write_file(os.path.join(base_dir, unmarked, "00", "keep"))
            self._write_file(os.path.join(base_dir, "not-a-tenant", "keep"))

        tenant_dirs, thread = prepare_tenant_storage_dirs(
            self.base_dirs,
            current,
            tp_rank=0,
            force_clean_all=False,
            run_id="run-1",
        )
        if thread is not None:
            thread.join()

        self.assertTrue(all(os.path.isdir(path) for path in tenant_dirs))
        for base_dir, tenant_dir in zip(self.base_dirs, tenant_dirs):
            self.assertFalse(os.path.exists(os.path.join(base_dir, foreign)))
            self.assertTrue(os.path.exists(os.path.join(base_dir, unmarked)))
            self.assertTrue(os.path.exists(os.path.join(base_dir, "not-a-tenant")))
            marker = os.path.join(tenant_dir, ".sglang-nixl-tenant")
            with open(marker) as f:
                self.assertIn("run_id=run-1", f.read())

    def test_force_clean_removes_current_tenant_before_recreate(self):
        """Force cleanup waits for old current-tenant contents to be removed."""
        tenant = resolve_tenant_key("model-a", tp_size=8, num_disks=2)
        for base_dir in self.base_dirs:
            self._write_file(os.path.join(base_dir, tenant, "00", "old"))
            self._write_file(os.path.join(base_dir, tenant, ".sglang-nixl-tenant"))

        tenant_dirs, thread = prepare_tenant_storage_dirs(
            self.base_dirs,
            tenant,
            tp_rank=0,
            force_clean_all=True,
            run_id="run-2",
        )

        self.assertTrue(thread is None or not thread.is_alive())
        for tenant_dir in tenant_dirs:
            self.assertTrue(os.path.isdir(tenant_dir))
            self.assertFalse(os.path.exists(os.path.join(tenant_dir, "00", "old")))
            self.assertTrue(
                os.path.exists(os.path.join(tenant_dir, ".sglang-nixl-tenant"))
            )

    def test_missing_base_dir_is_tolerated(self):
        """Missing storage bases should not make startup cleanup fail."""
        missing = os.path.join(self.test_dir, "missing")
        tenant = resolve_tenant_key("model-a", tp_size=1)
        thread = start_tenant_cleanup([missing], tenant, force_clean_all=False)
        self.assertIsNone(thread)


if __name__ == "__main__":
    unittest.main()
