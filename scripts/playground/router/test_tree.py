import random
import string
import time
import unittest
from typing import Dict, List, Tuple

from tree import MultiTenantRadixTree


class TestMultiTenantRadixTree(unittest.TestCase):
    def setUp(self):
        self.tree = MultiTenantRadixTree()

    def test_insert_exact_match(self):
        """Test 1: Basic insert and exact match operations"""
        # Insert a single string for one tenant
        self.tree.insert("hello", "tenant1")
        matched, tenant = self.tree.prefix_match("hello")
        self.assertEqual(matched, "hello")
        self.assertEqual(tenant, "tenant1")

        # Insert same string for different tenant
        self.tree.insert("hello", "tenant2")
        matched, tenant = self.tree.prefix_match("hello")
        self.assertIn(tenant, ["tenant1", "tenant2"])

        # Insert different string for same tenant
        self.tree.insert("world", "tenant1")
        matched, tenant = self.tree.prefix_match("world")
        self.assertEqual(matched, "world")
        self.assertEqual(tenant, "tenant1")

        print(self.tree.pretty_print())

    def test_insert_partial_match(self):
        """Test 2: Insert with partial matching scenarios"""
        # Test partial matches with common prefixes
        self.tree.insert("hello", "tenant1")
        print(self.tree.pretty_print())
        self.tree.insert("help", "tenant2")
        print(self.tree.pretty_print())

        # Match exact strings
        matched, tenant = self.tree.prefix_match("hello")
        self.assertEqual(matched, "hello")
        self.assertEqual(tenant, "tenant1")

        matched, tenant = self.tree.prefix_match("help")
        self.assertEqual(matched, "help")
        self.assertEqual(tenant, "tenant2")

        # Match partial string
        matched, tenant = self.tree.prefix_match("hel")
        self.assertEqual(matched, "hel")
        self.assertIn(tenant, ["tenant1", "tenant2"])

        # Match longer string
        matched, tenant = self.tree.prefix_match("hello_world")
        self.assertEqual(matched, "hello")
        self.assertEqual(tenant, "tenant1")

    def test_insert_edge_cases(self):
        """Test 3: Edge cases for insert and match operations"""
        # Empty string
        self.tree.insert("", "tenant1")
        matched, tenant = self.tree.prefix_match("")
        self.assertEqual(matched, "")
        self.assertEqual(tenant, "tenant1")

        # Single character
        self.tree.insert("a", "tenant1")
        matched, tenant = self.tree.prefix_match("a")
        self.assertEqual(matched, "a")
        self.assertEqual(tenant, "tenant1")

        # Very long string
        long_str = "a" * 1000
        self.tree.insert(long_str, "tenant1")
        matched, tenant = self.tree.prefix_match(long_str)
        self.assertEqual(matched, long_str)
        self.assertEqual(tenant, "tenant1")

        # Unicode characters
        self.tree.insert("你好", "tenant1")
        matched, tenant = self.tree.prefix_match("你好")
        self.assertEqual(matched, "你好")
        self.assertEqual(tenant, "tenant1")

    def test_simple_eviction(self):
        """Test 4: Simple eviction scenarios
        Tenant1: limit 10 chars
        Tenant2: limit 5 chars

        Should demonstrate:
        1. Basic eviction when size limit exceeded
        2. Proper eviction based on last access time
        3. Verification that shared nodes remain intact for other tenants
        """
        # Set up size limits
        max_size = {"tenant1": 10, "tenant2": 5}

        # Insert strings for both tenants
        self.tree.insert("hello", "tenant1")  # size 5
        self.tree.insert("hello", "tenant2")  # size 5
        self.tree.insert("world", "tenant2")  # size 5, total for tenant2 = 10

        # Verify initial sizes
        sizes_before = self.tree.get_used_size_per_tenant()
        self.assertEqual(sizes_before["tenant1"], 5)  # "hello" = 5
        self.assertEqual(sizes_before["tenant2"], 10)  # "hello" + "world" = 10

        # Evict - should remove "hello" from tenant2 as it's the oldest
        self.tree.evict_tenant_data(max_size)

        # Verify sizes after eviction
        sizes_after = self.tree.get_used_size_per_tenant()
        self.assertEqual(sizes_after["tenant1"], 5)  # Should be unchanged
        self.assertEqual(sizes_after["tenant2"], 5)  # Only "world" remains

        # Verify "world" remains for tenant2 (was accessed more recently)
        matched, tenant = self.tree.prefix_match("world")
        self.assertEqual(matched, "world")
        self.assertEqual(tenant, "tenant2")

    def test_medium_eviction(self):
        """Test 5: Medium complexity eviction scenarios with shared prefixes
        Tenant1: limit 10 chars
        Tenant2: limit 7 chars (forces one string to be evicted)

        Tree structure after inserts:
        └── 'h' [t1, t2]
            ├── 'i' [t1, t2]      # Oldest for t2
            └── 'e' [t1, t2]
                ├── 'llo' [t1, t2]
                └── 'y' [t2]      # Newest for t2

        Size calculations:
        tenant1: "h"(1) + "i"(1) + "e"(1) + "llo"(3) = 6 chars
        tenant2: "h"(1) + "i"(1) + "e"(1) + "llo"(3) + "y"(1) = 7 chars

        After eviction (tenant2 exceeds limit by 1 char):
        "hi" should be removed from tenant2 as it's the oldest access
        """
        max_size = {
            "tenant1": 10,
            "tenant2": 6,
        }  # tenant2 will need to evict one string

        # Create a tree with overlapping prefixes
        self.tree.insert("hi", "tenant1")
        self.tree.insert("hi", "tenant2")  # OLDEST for t2

        self.tree.insert("hello", "tenant1")
        self.tree.insert("hello", "tenant2")

        self.tree.insert("hey", "tenant2")  # NEWEST for t2

        # Verify initial sizes
        sizes_before = self.tree.get_used_size_per_tenant()
        self.assertEqual(sizes_before["tenant1"], 6)  # h(1) + i(1) + e(1) + llo(3) = 6
        self.assertEqual(
            sizes_before["tenant2"], 7
        )  # h(1) + i(1) + e(1) + llo(3) + y(1) = 7

        print("\nTree before eviction:")
        print(self.tree.pretty_print())

        # Evict - should remove "hi" from tenant2 as it's the oldest
        self.tree.evict_tenant_data(max_size)

        print("\nTree after eviction:")
        print(self.tree.pretty_print())

        # Verify sizes after eviction
        sizes_after = self.tree.get_used_size_per_tenant()
        self.assertEqual(sizes_after["tenant1"], 6)  # Should be unchanged
        self.assertEqual(sizes_after["tenant2"], 6)  # h(1) + e(1) + llo(3) + y(1) = 6

    def test_advanced_eviction(self):
        ...
        # Create 4 tenants
        # Each tenants keeps adding strings with shared prefixes to thousands usage
        # Set a strict limit for each tenant to only 100
        # At the end, check whether all of the tenant is under 100 after eviction

        max_size = {"tenant1": 100, "tenant2": 100, "tenant3": 100, "tenant4": 100}

        prefixes = ["aqwefcisdf", "iajsdfkmade", "kjnzxcvewqe", "iejksduqasd"]
        for i in range(100):
            for j, prefix in enumerate(prefixes):
                random_suffix = "".join(random.choices(string.ascii_letters, k=10))
                self.tree.insert(prefix + random_suffix, f"tenant{j+1}")

        sizes_before = self.tree.get_used_size_per_tenant()
        print(sizes_before)

        self.tree.evict_tenant_data(max_size)

        sizes_after = self.tree.get_used_size_per_tenant()
        print(sizes_after)
        # ensure size_after is below max_size
        for tenant, size in sizes_after.items():
            self.assertLessEqual(size, max_size[tenant])


if __name__ == "__main__":
    unittest.main()
