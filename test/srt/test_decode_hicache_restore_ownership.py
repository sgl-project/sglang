import ast
import pathlib
import types
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "python/sglang/srt/disaggregation/decode_hicache_mixin.py"


def _load_method(name, namespace):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DecodeHiCacheTransferMixin"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


class DecodeHiCacheRestoreOwnershipTest(unittest.TestCase):
    def test_coverage_failure_restores_prealloc_ownership_metadata(self):
        failed = object()
        original_prefix = [101, 102, 103, 104, 105]
        original_last_node = object()
        req = types.SimpleNamespace(
            rid="restore-failure",
            req_pool_idx=0,
            origin_input_ids=list(range(10)),
            prefix_indices=original_prefix,
            last_node=original_last_node,
            last_host_node=object(),
            best_match_node=object(),
            host_hit_length=1,
            swa_host_hit_length=2,
            mamba_host_hit_length=3,
            cache_protected_len=6,
        )

        rematched_node = object()

        def rematch(_tree, target_req, _tokens, **_kwargs):
            target_req.prefix_indices = [201, 202, 203, 204, 205]
            target_req.last_node = rematched_node
            target_req.last_host_node = rematched_node
            target_req.best_match_node = rematched_node
            target_req.host_hit_length = 0
            target_req.swa_host_hit_length = 0
            target_req.mamba_host_hit_length = 0
            target_req.cache_protected_len = 5
            return types.SimpleNamespace(
                device_indices=target_req.prefix_indices,
                best_match_node=rematched_node,
                host_hit_length=0,
            )

        namespace = {
            "DecodeRequest": object,
            "match_prefix_for_req": rematch,
            "InitLoadBackParams": lambda **kwargs: kwargs,
            "HiCacheRestoreResult": types.SimpleNamespace(FAILED=failed),
            "logger": types.SimpleNamespace(warning=lambda *_args, **_kwargs: None),
            "torch": types.SimpleNamespace(cat=lambda values: sum(values, [])),
        }
        method = _load_method("_try_hicache_queue_load_back", namespace)
        prefix_match = types.SimpleNamespace(
            l1_prefix_len=5,
            l2_host_hit_length=0,
            l3_storage_hit_length=1,
            decode_prefix_len=6,
        )
        decode_req = types.SimpleNamespace(
            req=req,
            prefix_match=prefix_match,
            hicache_restore_status=None,
        )
        tree = types.SimpleNamespace(
            req_to_token_pool=types.SimpleNamespace(
                req_to_token=[[101, 102, 103, 104, 105, 999]]
            ),
            check_prefetch_progress=lambda _rid: True,
            pop_prefetch_loaded_tokens=lambda _rid: None,
            init_load_back=lambda _params: ([], object()),
        )
        owner = types.SimpleNamespace(tree_cache=tree)

        self.assertFalse(method(owner, decode_req))
        self.assertIs(decode_req.hicache_restore_status, failed)
        self.assertEqual(req.prefix_indices, original_prefix)
        self.assertIs(req.last_node, original_last_node)
        self.assertEqual(req.cache_protected_len, 6)
        self.assertEqual(req.host_hit_length, 1)
        self.assertEqual(req.swa_host_hit_length, 2)
        self.assertEqual(req.mamba_host_hit_length, 3)


if __name__ == "__main__":
    unittest.main()
