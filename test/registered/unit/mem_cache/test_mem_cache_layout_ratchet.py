"""Ratchet guard: the ``mem_cache`` allocator / pool / pool_host layout may only
improve.

Four pins, each shrink-only, each covering a way the layout regressed after the
restructure was agreed:

- ``_LEGACY_HOMES`` -- a class filed under the wrong layer directory.
- ``_TOP_LEVEL_MODULES`` -- a new catch-all module at the ``mem_cache`` root.
- ``_SHRINKING_MODULES`` -- a new class appended to a module already slated for
  deletion, which is how ``memory_pool.py`` grew from 2257 to 5042 lines while
  the split was in progress.
- ``_FORBIDDEN_DEPS`` -- an import that reverses a layer boundary.

Target tree and roadmap: https://github.com/sgl-project/sglang/issues/25371
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import unittest
from pathlib import Path

import sglang.srt
from sglang.test.test_utils import CustomTestCase

_MEM_CACHE = Path(next(iter(sglang.srt.__path__))) / "mem_cache"
_ROADMAP = "https://github.com/sgl-project/sglang/issues/25371"

# A class belongs to a layer by what it inherits, never by what it is named:
# HostTensorAllocator is named Allocator but hands out pinned host memory, not
# KV slots, and stays in pool_host/ and storage/.
_ROLE_ROOTS = {
    "allocator": frozenset({"BaseTokenToKVPoolAllocator"}),
    "pool_host": frozenset({"HostKVCache"}),
    "pool": frozenset({"KVCache", "BaseSWAKVPool", "ReqToTokenPool", "MambaPool"}),
}

# "pool" is deliberately absent: mem_cache/pool/ does not exist yet, and an error
# telling someone to file a class into a package nobody has created is a trap,
# not a rule. _SHRINKING_MODULES holds the device pools until the first pool/ PR
# lands, at which point "pool" joins this tuple and its classes move from
# _SHRINKING_MODULES to _LEGACY_HOMES.
_ENFORCED_ROLES = ("allocator", "pool_host")

_LEGACY_HOMES = {
    "MultiEndedAllocator": "multi_ended_allocator.py",
    "UnifiedMambaTokenToKVPoolAllocator": "multi_ended_allocator.py",
    "UnifiedSWATokenToKVPoolAllocator": "multi_ended_allocator.py",
    # Landed after this pin was first written; recorded as debt, not endorsed.
    "FloatMultiEndedAllocator": "multi_ended_allocator.py",
    "UnifiedMambaSWATokenToKVPoolAllocator": "multi_ended_allocator.py",
    "DeepSeekV4PagedHostPool": "memory_pool_host.py",
    "DeepSeekV4StateHostPool": "memory_pool_host.py",
}

_TOP_LEVEL_MODULES = frozenset(
    {
        "allocation.py",
        "allocation_sizing.py",
        "base_prefix_cache.py",
        "base_swa_memory_pool.py",
        "cache_init_params.py",
        "chunk_cache.py",
        "common.py",
        "deepseek_v4_compress_state.py",
        "deepseek_v4_memory_pool.py",
        "dsa_cache_layer_split.py",
        "embedding_cache_controller.py",
        "embedding_store.py",
        "events.py",
        "evict_policy.py",
        "flush_cache.py",
        "hicache_storage.py",
        "hiradix_cache.py",
        "hisparse_memory_pool.py",
        "index_key_cache.py",
        # Landed after this pin was first written; recorded as debt, not endorsed.
        "kv_index_translator.py",
        "kv_cache_builder.py",
        "kv_cache_configurator.py",
        "kv_cache_dtype.py",
        "kv_vmm_backing.py",
        "l2_transfer.py",
        "mamba_checkpoint_pool.py",
        "mamba_radix_cache.py",
        "mamba_slot_fused.py",
        "memory_pool.py",
        "memory_pool_host.py",
        "multi_ended_allocator.py",
        "multimodal_cache.py",
        "pure_swa_radix_cache.py",
        "radix_cache.py",
        "radix_cache_cpp.py",
        "registry.py",
        "swa_memory_pool.py",
        "swa_radix_cache.py",
        "unified_memory_pool.py",
        "unified_radix_cache.py",
        "utils.py",
    }
)

# Modules the roadmap deletes outright. Their class lists are frozen so the
# not-yet-guarded pool layer cannot keep growing inside them.
_SHRINKING_MODULES = {
    "memory_pool.py": frozenset(
        {
            "DSATokenToKVPool",
            "HybridLinearKVPool",
            "HybridReqToTokenPool",
            "KVCache",
            "KVWriteLoc",
            "KvBufferDesc",
            "MHATokenToKOnlyPool",
            "MHATokenToKVPool",
            "MHATokenToKVPoolFP4",
            "MHATokenToKVPoolMXFP8",
            "MLATokenToKVPool",
            "MLATokenToKVPoolFP4",
            "MambaPool",
            "MiniMaxSparseKVPool",
            "NoOpMHATokenToKVPool",
            "PageMajorMHATokenToKVPool",
            "ReqToTokenPool",
        }
    ),
    "memory_pool_host.py": frozenset(
        {
            "DeepSeekV4PagedHostPool",
            "DeepSeekV4StateHostPool",
            "LogicalHostPool",
        }
    ),
    "deepseek_v4_memory_pool.py": frozenset(
        {
            "DeepSeekV4IndexerPool",
            "DeepSeekV4LayerItem",
            "DeepSeekV4SingleKVPool",
            "DeepSeekV4TokenToKVPool",
            "DeepSeekV4UnifiedKVPool",
            "HiSparseC4DevicePool",
        }
    ),
    "deepseek_v4_compress_state.py": frozenset({"CompressStatePool", "KVAndScore"}),
    "swa_memory_pool.py": frozenset({"SWAKVPool"}),
    "base_swa_memory_pool.py": frozenset({"BaseSWAKVPool"}),
    "hisparse_memory_pool.py": frozenset({"HiSparseDSATokenToKVPool"}),
    "unified_memory_pool.py": frozenset(
        {
            "MHASubPoolSpec",
            "MLASubPoolSpec",
            "MambaSubPoolSpec",
            "SubPoolSpec",
            "UnifiedHybridLinearKVPool",
            "UnifiedHybridReqToTokenPool",
            "UnifiedKVPool",
            "UnifiedMHATokenToKVPool",
            "UnifiedMLATokenToKVPool",
            "UnifiedMambaPool",
            "UnifiedMambaSlotAllocator",
            "UnifiedPoolBundle",
            "UnifiedSWAKVPool",
            "UnifiedSWAPoolBundle",
        }
    ),
    "multi_ended_allocator.py": frozenset(
        {
            "MultiEndedAllocator",
            "UnifiedMambaTokenToKVPoolAllocator",
            "UnifiedSWATokenToKVPoolAllocator",
            # The three below landed after this pin was first written. Recorded
            # as debt, not endorsed -- this module gained three classes in the
            # two weeks the pin sat unmerged, which is the drift it exists for.
            "FloatMultiEndedAllocator",
            "UnifiedMambaSWATokenToKVPoolAllocator",
            "_CapacityField",
        }
    ),
    "mamba_checkpoint_pool.py": frozenset(
        {"Int8CheckpointStore", "MambaCheckpointPool"}
    ),
    "dsa_cache_layer_split.py": frozenset(
        {"LayerSplitDSATokenToKVPool", "LayerSplitIndexKeyCache"}
    ),
    "index_key_cache.py": frozenset({"IndexKeyCache"}),
}

# Storage layers never reach back up into indexing or orchestration, and none of
# the three reach into the construction layer that builds them.
_CONSTRUCTION = (
    "allocation_sizing",
    "cache_init_params",
    "kv_cache_builder",
    "kv_cache_configurator",
    "kv_cache_dtype",
    "kv_vmm_backing",
)
_FORBIDDEN_DEPS = {
    "allocator": ("allocation", "hybrid_cache") + _CONSTRUCTION,
    "pool_host": ("allocation", "allocator", "hybrid_cache") + _CONSTRUCTION,
    "pool": ("allocation", "allocator", "hybrid_cache") + _CONSTRUCTION,
}


def _modules():
    """Relative posix path -> parsed module, for every module under mem_cache/."""
    return {
        path.relative_to(_MEM_CACHE).as_posix(): ast.parse(path.read_text())
        for path in sorted(_MEM_CACHE.rglob("*.py"))
    }


def _base_names(node: ast.ClassDef) -> list[str]:
    names = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _class_defs(modules) -> list[tuple[str, str]]:
    """(defining module, class name) for every module-level class.

    Nested classes are skipped: MambaPool.State is part of MambaPool, not a peer
    that could be filed anywhere else.
    """
    return [
        (rel, node.name)
        for rel, tree in sorted(modules.items())
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]


def _bases_by_name(modules) -> dict[str, set[str]]:
    """Class name -> union of the direct bases of every class with that name.

    Bases are matched by name because resolving them properly would mean
    resolving imports. Names do collide (three modules define TreeNode), and
    unioning over-approximates ancestry rather than dropping all but one
    definition -- for a guard, a false positive is a review conversation and a
    false negative is a silent miss.
    """
    bases = {}
    for rel, tree in modules.items():
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases.setdefault(node.name, set()).update(_base_names(node))
    return bases


def _ancestors(name: str, bases, seen=None) -> set[str]:
    """Transitive base names. Inheritance crosses modules, so a direct base name
    is not enough to tell a KVCache subclass from an unrelated class."""
    seen = seen or set()
    found = {name}
    for base in bases.get(name, ()):
        if base not in seen:
            found |= _ancestors(base, bases, seen | {base})
    return found


def _role(name: str, bases) -> str | None:
    ancestry = _ancestors(name, bases)
    for role, roots in _ROLE_ROOTS.items():
        if ancestry & roots:
            return role
    return None


def _imported_submodules(tree: ast.Module, rel: str) -> set[str]:
    """mem_cache-relative dotted module paths this module imports."""
    prefix = "sglang.srt.mem_cache."
    package = rel.rsplit("/", 1)[0] if "/" in rel else ""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split("/") if package else []
                parts = parts[: len(parts) - node.level + 1]
                target = ".".join(filter(None, parts + [node.module or ""]))
                if target:
                    found.add(target)
            elif node.module and node.module.startswith(prefix):
                found.add(node.module[len(prefix) :])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(prefix):
                    found.add(alias.name[len(prefix) :])
    return found


class TestMemCacheLayoutRatchet(CustomTestCase):
    """See module docstring; every failure here is fixed by moving code, not by
    growing a pin -- unless review agrees the pin should grow."""

    def setUp(self):
        self.modules = _modules()
        self.bases = _bases_by_name(self.modules)

    def test_layer_classes_live_in_their_layer_package(self):
        for rel, name in _class_defs(self.modules):
            role = _role(name, self.bases)
            if role not in _ENFORCED_ROLES:
                continue
            if rel.startswith(f"{role}/"):
                self.assertNotIn(
                    name,
                    _LEGACY_HOMES,
                    f"{name} now lives in {rel}; drop it from _LEGACY_HOMES to "
                    "lock in the progress.",
                )
                continue
            self.assertEqual(
                _LEGACY_HOMES.get(name),
                rel,
                f"{name} inherits from {sorted(_ROLE_ROOTS[role])}, so it belongs "
                f"in mem_cache/{role}/<family>.py, not {rel}. See {_ROADMAP}.",
            )

    def test_no_new_top_level_modules(self):
        found = {path.name for path in _MEM_CACHE.glob("*.py")}
        added = found - _TOP_LEVEL_MODULES
        self.assertFalse(
            added,
            f"new top-level mem_cache modules {sorted(added)}; file them under "
            f"allocator/, pool/, pool_host/, or storage/ instead. Growing this "
            f"pin needs an explicit reason in review. See {_ROADMAP}.",
        )
        removed = _TOP_LEVEL_MODULES - found
        self.assertFalse(
            removed,
            f"{sorted(removed)} no longer exist; shrink _TOP_LEVEL_MODULES to "
            "lock in the progress.",
        )

    def test_modules_slated_for_deletion_only_shrink(self):
        for rel, pinned in _SHRINKING_MODULES.items():
            tree = self.modules.get(rel)
            if tree is None:
                self.fail(
                    f"{rel} is gone; drop its _SHRINKING_MODULES entry to lock in "
                    "the progress."
                )
            declared = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
            added = declared - pinned
            self.assertFalse(
                added,
                f"{rel} is slated for deletion but gained {sorted(added)}; put new "
                f"classes in their layer package. See {_ROADMAP}.",
            )
            removed = pinned - declared
            self.assertFalse(
                removed,
                f"{rel} no longer defines {sorted(removed)}; shrink its "
                "_SHRINKING_MODULES entry to lock in the progress.",
            )

    def test_layer_packages_do_not_import_upwards(self):
        for rel, tree in self.modules.items():
            layer = rel.split("/", 1)[0]
            forbidden = _FORBIDDEN_DEPS.get(layer)
            if forbidden is None or "/" not in rel:
                continue
            for imported in _imported_submodules(tree, rel):
                head = imported.split(".", 1)[0]
                self.assertNotIn(
                    head,
                    forbidden,
                    f"{rel} imports {imported}: {layer}/ is below it and must not "
                    f"depend on it. See {_ROADMAP}.",
                )


if __name__ == "__main__":
    unittest.main()
