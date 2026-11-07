"""Compiled rerank templates must be reused without retaining request context."""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # Must precede imports that transitively load sgl_kernel.

import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from jinja2 import TemplateSyntaxError, UndefinedError
from jinja2.sandbox import SecurityError

from sglang.srt.entrypoints.openai.serving_rerank import (
    _compile_rerank_template,
    _get_jinja_env,
    _render_jinja_chat_template,
    _render_vl_jinja_template,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TEMPLATE_DIR = Path(__file__).resolve().parents[5] / "examples/chat_template"
_TEXT_TEMPLATE = (
    "{{ instruct | default('DEF') }}|"
    "{{ messages[0]['content'] }}|{{ messages[1]['content'] }}"
)
_VL_TEMPLATE = (
    "{{ instruct | default('DEF') }}|{{ query[0].text }}|{{ document[0].text }}"
)


class TestRerankTemplateCache(CustomTestCase):
    def setUp(self):
        _compile_rerank_template.cache_clear()
        self.addCleanup(_compile_rerank_template.cache_clear)

    @staticmethod
    def render(vl, template, query="Q", document="D", instruct=None):
        if vl:
            return _render_vl_jinja_template(
                template,
                query=[{"type": "text", "text": query}],
                document=[{"type": "text", "text": document}],
                instruct=instruct,
            )
        return _render_jinja_chat_template(
            template, query=query, document=document, instruct=instruct
        )

    def test_helpers_reuse_templates_with_fresh_request_data(self):
        """Changing a document or instruction must hit the same compiled template."""
        for vl, template in ((False, _TEXT_TEMPLATE), (True, _VL_TEMPLATE)):
            for i, instruct in enumerate(("custom", None, "", "next")):
                with self.subTest(vl=vl, instruct=instruct):
                    self.assertEqual(
                        self.render(vl, template, f"Q{i}", f"D{i}", instruct),
                        f"{instruct or 'DEF'}|Q{i}|D{i}",
                    )
        info = _compile_rerank_template.cache_info()
        self.assertEqual((info.misses, info.hits, info.currsize), (2, 6, 2))

    def test_repository_templates_match_uncached_rendering(self):
        """Caching must preserve whitespace, defaults, and multimodal placeholders."""
        for vl, filename in (
            (False, "qwen3_reranker.jinja"),
            (True, "qwen3_vl_reranker.jinja"),
        ):
            template = (_TEMPLATE_DIR / filename).read_text()
            for instruct in (None, "", "Find <relevant> passages & images."):
                query = [{"type": "text", "text": "Query <&> {{ literal }}"}]
                document = [
                    {"type": "text", "text": "First\npassage"},
                    {"type": "image"},
                    {"type": "video"},
                    {"type": "text", "text": "Second passage"},
                ]
                if vl:
                    render = _render_vl_jinja_template
                    kwargs = {"query": query, "document": document}
                else:
                    render = _render_jinja_chat_template
                    kwargs = {
                        "messages": [
                            {"role": "user", "content": "Query <&> {{ literal }}"},
                            {
                                "role": "user",
                                "content": "First\npassage Second passage",
                            },
                        ]
                    }
                if instruct:
                    kwargs["instruct"] = instruct
                expected = _get_jinja_env().from_string(template).render(**kwargs)
                with self.subTest(vl=vl, instruct=instruct):
                    self.assertEqual(
                        render(
                            template, query=query, document=document, instruct=instruct
                        ),
                        expected,
                    )

    def test_template_changes_and_eviction(self):
        """Replacing template source must take effect, including after eviction."""
        limit = _compile_rerank_template.cache_info().maxsize
        self.assertIsNotNone(limit)
        templates = [f"{i}:" + _TEXT_TEMPLATE for i in range(limit + 1)]
        first = _compile_rerank_template(templates[0])
        for i, template in enumerate(templates):
            self.assertEqual(self.render(False, template), f"{i}:DEF|Q|D")
        self.assertEqual(_compile_rerank_template.cache_info().currsize, limit)
        self.assertIsNot(_compile_rerank_template(templates[0]), first)
        self.assertEqual(
            self.render(False, templates[0], "new", "doc"), "0:DEF|new|doc"
        )

    def test_overlapping_renders_keep_separate_contexts(self):
        """Hold several renders inside the same template with distinct request data."""
        workers = 4
        barrier = Barrier(workers, timeout=10)

        class BlockingText(str):
            def __str__(self):
                barrier.wait()
                return super().__str__()

        for template in (_TEXT_TEMPLATE, _VL_TEMPLATE):
            _compile_rerank_template(template)

        def render_request(i):
            instruct = f"I{i}" if i % 2 else None
            return [
                self.render(vl, template, BlockingText(f"Q{i}"), f"D{i}", instruct)
                for vl, template in ((False, _TEXT_TEMPLATE), (True, _VL_TEMPLATE))
            ]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(render_request, range(workers)))
        for i, result in enumerate(results):
            expected = f"{'I' + str(i) if i % 2 else 'DEF'}|Q{i}|D{i}"
            self.assertEqual(result, [expected, expected])

    def test_sandbox_and_errors_on_cold_and_warm_cache(self):
        for vl in (False, True):
            for template, error in (
                ("{% if %}", TemplateSyntaxError),
                ("{{ missing.foo }}", UndefinedError),
                ("{{ ''.__class__.__mro__ }}", SecurityError),
                ("{{ [].append(1) }}", SecurityError),
                ("{{ 1 / 0 }}", ZeroDivisionError),
            ):
                _compile_rerank_template.cache_clear()
                for attempt in range(2):
                    with self.subTest(vl=vl, template=template, attempt=attempt):
                        with self.assertRaises(error):
                            self.render(vl, template)
                self.assertEqual(
                    _compile_rerank_template.cache_info().currsize,
                    0 if error is TemplateSyntaxError else 1,
                )

    def test_missing_jinja_keeps_installation_error(self):
        with patch.dict("sys.modules", {"jinja2": None}):
            for vl, template in ((False, _TEXT_TEMPLATE), (True, _VL_TEMPLATE)):
                with self.subTest(vl=vl):
                    with self.assertRaisesRegex(ValueError, "requires `jinja2`"):
                        self.render(vl, template)
        self.assertEqual(_compile_rerank_template.cache_info().currsize, 0)
        self.assertEqual(self.render(False, _TEXT_TEMPLATE), "DEF|Q|D")


if __name__ == "__main__":
    unittest.main()
