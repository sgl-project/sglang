"""CPU-only mocked policy tests; no CUDA kernels or end-to-end model coverage.

Execute repository function ASTs with explicit dependency doubles, avoiding the
heavy runtime/package initializers and making native FlashMLA imports fail.
"""

from __future__ import annotations

import ast
import os
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=2, suite="base-a-test-cpu")
PYTHON = ROOT / "python/sglang"
# Load the real stdlib-only EnvField API without package/runtime initialization.
ENVS = runpy.run_path(str(PYTHON / "srt/environ.py"))["envs"]
BACKEND = PYTHON / "srt/layers/attention/deepseek_v4_backend.py"
POOL = PYTHON / "srt/mem_cache/deepseek_v4_memory_pool.py"
MODEL = PYTHON / "srt/models/deepseek_v4.py"
HELPER = PYTHON / "kernels/ops/attention/dsv4/sm80_utils.py"
FLAG = "SGLANG_DSV41_SM80_EXPERIMENTAL"


def load(path, name, namespace, cls=None, remove_imports=False):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if cls:
        nodes = next(
            n.body for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls
        )
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    if remove_imports:

        class RemoveImports(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return ast.copy_location(ast.Pass(), node)

        node = RemoveImports().visit(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    # Execute only the selected repository-owned function, never external code.
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102
    return namespace[name]


def fake_torch(cap=(8, 0), hip=None, available=True):
    return NS(
        device=lambda value: NS(type=str(value).split(":")[0]),
        version=NS(hip=hip, cuda="12.8"),
        cuda=NS(
            is_available=Mock(return_value=available),
            get_device_capability=Mock(return_value=cap),
        ),
    )


class Field:
    def __init__(self, name, default):
        self.name, self.default = name, default

    def get(self):
        return os.environ.get(self.name, self.default)

    def is_set(self):
        return self.name in os.environ


class Tensor:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)

    def __getitem__(self, key):
        return self

    def view(self, *shape):
        return Tensor(shape)

    def unsqueeze(self, dim):
        return Tensor(self.shape[:dim] + (1,) + self.shape[dim:])

    def squeeze(self, dim):
        return Tensor(self.shape[:dim] + self.shape[dim + 1 :])


class RuntimePolicyTests(unittest.TestCase):
    def test_helper_matrix_exact_opt_in(self):
        for flag in (None, "0", "true", "True", "yes", "01", " 1", "1"):
            for model in (None, "deepseek_v4", "deepseek_v41", "other"):
                for cap in ((8, 0), (8, 6), (9, 0), (10, 0), (12, 0)):
                    with (
                        self.subTest(flag=flag, model=model, cap=cap),
                        patch.dict(os.environ, {}, clear=True),
                    ):
                        if flag is not None:
                            os.environ[FLAG] = flag
                        torch = fake_torch(cap)
                        helper = load(
                            HELPER,
                            "is_dsv41_sm80_experimental",
                            {"envs": ENVS, "torch": torch},
                        )
                        self.assertEqual(
                            helper(model, "cuda:3"),
                            flag == "1" and model == "deepseek_v41" and cap == (8, 0),
                        )
                        if torch.cuda.get_device_capability.called:
                            self.assertEqual(
                                torch.cuda.get_device_capability.call_args.args[0].type,
                                "cuda",
                            )
        with patch.dict(os.environ, {FLAG: "1"}):
            for device, torch in (
                ("cpu", fake_torch()),
                ("cuda:0", fake_torch(hip="6.0")),
                ("cuda:0", fake_torch(available=False)),
            ):
                helper = load(
                    HELPER, "is_dsv41_sm80_experimental", {"envs": ENVS, "torch": torch}
                )
                self.assertFalse(helper("deepseek_v41", device))
                torch.cuda.get_device_capability.assert_not_called()

    def selector(self, cap=(8, 0), backend="flashmla", hip=False):
        torch = fake_torch(cap, hip="6" if hip else None)
        helper = load(HELPER, "is_dsv41_sm80_experimental", {"envs": ENVS, "torch": torch})
        envs = NS(
            SGLANG_DSV4_KV_LAYOUT=Field("SGLANG_DSV4_KV_LAYOUT", "v4"),
            SGLANG_DSV4_COMPRESSED_KV_LAYOUT=Field(
                "SGLANG_DSV4_COMPRESSED_KV_LAYOUT", "auto"
            ),
        )
        ns = {
            "torch": torch,
            "envs": envs,
            "is_dsv41_sm80_experimental": helper,
            "get_exec": lambda: NS(kernel=NS(dsv4_attn_backend=backend)),
            "KVLayout": NS(V4="v4", V41="v41"),
            "_is_hip": hip,
            "is_gfx95_supported": lambda: False,
            "resolve_hip_flashmla_backend": lambda: "other",
        }
        return load(POOL, "select_dsv4_kv_layout", ns)

    def test_pool_modes_and_other_models_unchanged(self):
        with patch.dict(os.environ, {FLAG: "1"}, clear=True):
            select = self.selector()
            self.assertEqual(select(is_dsv41=True, device="cuda:3"), ("v41", "auto"))
            self.assertEqual(select(is_dsv41=False, device="cuda:3"), ("v4", None))
            for mode in ("auto", "v41"):
                for option in ("auto", "fp8", "fp4"):
                    os.environ.update(
                        SGLANG_DSV4_KV_LAYOUT=mode,
                        SGLANG_DSV4_COMPRESSED_KV_LAYOUT=option,
                    )
                    self.assertEqual(
                        select(is_dsv41=True, device="cuda:3"), ("v41", option)
                    )
            os.environ["SGLANG_DSV4_KV_LAYOUT"] = "v4"
            with self.assertRaisesRegex(ValueError, "explicit V4"):
                select(is_dsv41=True, device="cuda:3")
            with self.assertRaisesRegex(ValueError, "trtllm"):
                self.selector(backend="trtllm")(is_dsv41=True, device="cuda:3")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                self.selector()(is_dsv41=True, device="cuda:0"), ("v4", None)
            )
            os.environ.update({FLAG: "1", "SGLANG_DSV4_KV_LAYOUT": "auto"})
            self.assertEqual(
                self.selector(cap=(8, 6))(is_dsv41=True, device="cuda:0"), ("v4", None)
            )
            self.assertEqual(
                self.selector(cap=(10, 0))(is_dsv41=False, device="cuda:0"),
                ("v41", "auto"),
            )
            os.environ["SGLANG_DSV4_KV_LAYOUT"] = "v41"
            with self.assertRaisesRegex(AssertionError, "SM100"):
                self.selector()(is_dsv41=False, device="cuda:0")

    def test_skip_metadata_before_platform_or_native_import(self):
        create = load(BACKEND, "_create_flashmla_metadata", {})
        with patch("builtins.__import__", side_effect=AssertionError("native import")):
            self.assertIsNone(create(skip=True))

    def test_all_metadata_initialization_and_warmup_use_stored_flag(self):
        create = Mock(return_value=None)
        ns = {"_create_flashmla_metadata": create}
        init = load(BACKEND, "init_flashmla_related", ns, "DSV4AttnMetadata")
        core = NS(
            index_topk=512,
            has_c4=False,
            has_c128=True,
            low_ratios=(1, 2),
            skip_flashmla_metadata=True,
        )
        with patch.dict(os.environ, {FLAG: "0"}):
            init(core, low_ratio_buffers=tuple(range(8)))
        self.assertEqual(create.call_count, 4)
        self.assertTrue(all(c.kwargs == {"skip": True} for c in create.call_args_list))
        create.reset_mock()
        ns.update(DSV4Metadata=NS, DSV4AttnMetadata=NS)
        warmup = load(
            BACKEND, "on_after_cuda_graph_warmup", ns, "DeepseekV4AttnBackend"
        )
        core.has_c4 = True
        backend = NS(
            forward_metadata=NS(core_attn_metadata=core),
            use_dsv41_sm80_experimental=True,
        )
        with patch.dict(os.environ, {FLAG: "0"}):
            warmup(backend)
        self.assertEqual(create.call_count, 5)
        self.assertTrue(all(c.kwargs == {"skip": True} for c in create.call_args_list))

    def test_make_core_without_compression_preserves_skip_flag(self):
        class Core:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        prep = NS(
            seq_lens_casual=Tensor((2,)),
            positions_casual=Tensor((2,)),
            page_table=Tensor((2, 1)),
            swa_topk_lengths=Tensor((2,)),
        )
        create = Mock(return_value=None)
        ns = {
            "BuildPageTablePositions": NS(execute=Mock(return_value=prep)),
            "BuildCausalSwaPageIndices": NS(execute=Mock(return_value=Tensor((2, 64)))),
            "DSV4AttnMetadata": Core,
            "_create_flashmla_metadata": create,
            "SWA_WINDOW": 128,
            "PAGE_INDEX_ALIGNED_SIZE": 64,
        }
        make_core = load(
            BACKEND, "make_core_attn_metadata", ns, "DeepseekV4AttnBackend"
        )
        for active in (True, False):
            backend = NS(
                page_size=256,
                token_to_kv_pool=NS(
                    request_window=None, full_to_swa_index_mapping=None
                ),
                req_to_token=None,
                index_topk=512,
                present_ratios=(1, 2),
                low_ratios=(1, 2),
                cuda_int32_kwargs={},
                use_dsv41_sm80_experimental=active,
                trtllm_attn=False,
            )
            with patch.dict(os.environ, {FLAG: "0" if active else "1"}):
                core = make_core(
                    backend,
                    None,
                    None,
                    Tensor((2,)),
                    2,
                    Tensor((2,)),
                    need_compress=False,
                    is_prefill=True,
                )
            self.assertEqual(core.skip_flashmla_metadata, active)
            create.assert_called_with(skip=active)
            self.assertIsNone(core.c0_flashmla_metadata)

    def test_native_head_width_and_sparse_prefill_bypass(self):
        helper = Mock(return_value=True)
        heads = load(
            MODEL,
            "_kernel_num_heads",
            {"is_dsv41_sm80_experimental": helper},
            "MqaAttentionBase",
        )
        model = NS(
            is_dsv41=True,
            attn_sink=NS(device="cuda:3"),
            n_local_heads=16,
            attn_tp_size=8,
            n_heads=128,
        )
        self.assertEqual(heads(model, 1), 16)
        helper.assert_called_once_with("deepseek_v41", "cuda:3")
        use_prefill = load(BACKEND, "_use_sparse_prefill", {}, "DeepseekV4AttnBackend")
        self.assertFalse(
            use_prefill(NS(use_dsv41_sm80_experimental=True), None, num_qo_tokens=9999)
        )
        helper.return_value = False
        heads.__globals__["get_platform"] = lambda: NS(is_sm120=False)
        self.assertEqual(heads(model, 1), 64)

    def test_forward_calls_adapter_after_store_and_view_never_native_metadata(self):
        class Pool:
            get_swa_key_buffer_radix = Mock(return_value=Tensor((2, 256 * 528)))
            get_extra_key_buffer = Mock(return_value=Tensor((3, 128 * 288)))
            get_swa_key_page_size = Mock(return_value=256)
            get_swa_key_bytes_per_token = Mock(return_value=528)
            get_extra_key_page_size = Mock(return_value=128)
            get_extra_key_bytes_per_token = Mock(return_value=288)

        class Core:
            swa_page_indices = Tensor((2, 64))
            swa_topk_lengths = Tensor((2,))
            sparse_page_indices = Mock(return_value=Tensor((2, 512)))
            sparse_topk_lengths = Mock(return_value=Tensor((2,)))
            get_flashmla_metadata = Mock(side_effect=AssertionError("native metadata"))

        store = Mock()

        def adapter(q, cache, head_dim_v, scale, **kwargs):
            store.assert_called_once()
            self.assertEqual(q.shape, (2, 1, 16, 512))
            self.assertEqual(cache.shape, (2, 256, 1, 528))
            self.assertEqual(kwargs["extra_k_cache"].shape, (3, 128, 1, 288))
            self.assertEqual(kwargs["indices"].shape, (2, 1, 64))
            self.assertEqual(kwargs["extra_indices_in_kvcache"].shape, (2, 1, 512))
            self.assertEqual(
                set(kwargs),
                {
                    "indices",
                    "topk_length",
                    "attn_sink",
                    "extra_k_cache",
                    "extra_indices_in_kvcache",
                    "extra_topk_length",
                },
            )
            return Tensor((2, 1, 16, 512)), None

        ns = {
            "DeepSeekV4TokenToKVPool": Pool,
            "DSV4AttnMetadata": Core,
            "flash_mla_with_kvcache_sm80": adapter,
        }
        forward = load(
            BACKEND,
            "_forward_attention",
            ns,
            "DeepseekV4AttnBackend",
            remove_imports=True,
        )
        backend = NS(
            mtp_enabled=False,
            trtllm_attn=False,
            forward_metadata=NS(core_attn_metadata=Core()),
            token_to_kv_pool=Pool(),
            store_cache=store,
            use_dsv41_sm80_experimental=True,
            head_dim_v=512,
            softmax_scale=0.1,
        )
        kv = object()
        out = forward(
            backend,
            Tensor((2, 16, 512)),
            kv,
            kv,
            NS(layer_id=7),
            object(),
            2,
            attn_sink=object(),
        )
        self.assertEqual(out.shape, (2, 16, 512))
        Core.get_flashmla_metadata.assert_not_called()
        ns["flash_mla_with_kvcache_sm80"] = Mock(
            side_effect=ValueError("adapter rejects")
        )
        with self.assertRaisesRegex(ValueError, "adapter rejects"):
            forward(
                backend,
                Tensor((2, 16, 512)),
                kv,
                kv,
                NS(layer_id=7),
                object(),
                2,
                attn_sink=object(),
            )
        Core.get_flashmla_metadata.assert_not_called()

    def test_all_creation_sites_explicitly_guarded_and_callers_pass_identity(self):
        tree = ast.parse(BACKEND.read_text())
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_create_flashmla_metadata"
        ]
        self.assertEqual(len(calls), 13)
        for call in calls:
            self.assertEqual([k.arg for k in call.keywords], ["skip"])
        for path in (
            PYTHON / "srt/model_executor/pool_configurator.py",
            PYTHON / "srt/mem_cache/kv_cache_configurator.py",
        ):
            tree = ast.parse(path.read_text())
            calls = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "select_dsv4_kv_layout"
            ]
            self.assertEqual(len(calls), 1)
            self.assertEqual({k.arg for k in calls[0].keywords}, {"is_dsv41", "device"})


if __name__ == "__main__":
    unittest.main()
