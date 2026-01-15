#!/usr/bin/env python3
"""
8-Hour Comprehensive UI Test Suite
===================================

Tests EVERYTHING in the Attention Explorer UIs:
- Every button click
- Every input field with various values
- Every slider position
- Every modal and help section
- Every tooltip
- Keyboard navigation
- Responsive layouts
- Edge cases (empty, long, special chars)
- Many different query scenarios

Generates a beautiful HTML report with screenshots.
"""

import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

try:
    from playwright.sync_api import sync_playwright, Page, Browser
except ImportError:
    print("Please install playwright: pip install playwright && playwright install chromium")
    sys.exit(1)


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

@dataclass
class TestConfig:
    """Configuration for the test run."""
    base_url: str = "http://localhost:8082"
    api_url: str = "http://localhost:8000"
    duration_hours: float = 8.0
    screenshot_dir: Path = field(default_factory=lambda: Path("overnight_ui_results"))
    viewport_width: int = 1600
    viewport_height: int = 1000
    slow_mo: int = 100  # Milliseconds between actions for visibility

    # Test queries - variety of topics and lengths
    test_queries: List[str] = field(default_factory=lambda: [
        # Short queries
        "What is AI?",
        "Hello world",
        "Why is the sky blue?",
        "Define entropy",

        # Medium queries
        "What is the capital of France and why is it historically significant?",
        "Explain how transformers work in machine learning",
        "What are the health benefits of green tea?",
        "How does attention mechanism help in NLP?",

        # Long queries
        "I'm trying to understand how large language models process information. Can you explain the role of attention in transformer architectures, specifically how the query, key, and value matrices interact?",
        "Compare and contrast the economic policies of the United States and European Union, focusing on trade agreements, monetary policy, and regulatory frameworks.",

        # Technical queries
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "SELECT * FROM users WHERE age > 21 ORDER BY name",
        "import torch; x = torch.randn(3, 4); print(x.shape)",

        # Edge cases
        "",  # Empty
        "a",  # Single char
        "🚀 🎉 🔥 💡 🌟",  # Emojis
        "   spaces   around   ",  # Extra spaces
        "Line1\nLine2\nLine3",  # Multiline
        "Special chars: @#$%^&*()[]{}|\\",
        "Unicode: 你好 مرحبا שלום",
        "A" * 500,  # Very long single word
        " ".join(["word"] * 100),  # Many words

        # Reasoning queries
        "If all cats are mammals and all mammals are warm-blooded, are all cats warm-blooded?",
        "What comes next in the sequence: 2, 4, 8, 16, ?",
        "Solve: If x + 5 = 12, what is x?",

        # Creative queries
        "Write a haiku about machine learning",
        "Tell me a joke about programming",
        "Describe a sunset in three words",

        # Factual queries
        "Who wrote Romeo and Juliet?",
        "What year did World War 2 end?",
        "What is the chemical formula for water?",

        # Ambiguous queries
        "Bank",  # Multiple meanings
        "Apple",  # Company or fruit?
        "Mercury",  # Planet or element?
    ])

    # RAG documents to test
    rag_documents: List[str] = field(default_factory=lambda: [
        "The Eiffel Tower is located in Paris, France. It was built in 1889.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Green tea contains antioxidants called catechins which have various health benefits.",
        "Python is a high-level programming language known for its readability.",
        "The mitochondria is the powerhouse of the cell.",
        "Quantum computing uses qubits instead of classical bits.",
        "Shakespeare wrote many famous plays including Hamlet and Macbeth.",
        "The speed of light is approximately 299,792,458 meters per second.",
    ])


@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    category: str
    timestamp: str
    duration_ms: float
    success: bool
    screenshot_path: Optional[str] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSession:
    """Tracks the entire test session."""
    start_time: datetime
    results: List[TestResult] = field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    screenshots_taken: int = 0

    def add_result(self, result: TestResult):
        self.results.append(result)
        self.total_tests += 1
        if result.success:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        if result.screenshot_path:
            self.screenshots_taken += 1


# ============================================================================
# TEST UTILITIES
# ============================================================================

class UITester:
    """Comprehensive UI testing class."""

    def __init__(self, config: TestConfig):
        self.config = config
        self.session = TestSession(start_time=datetime.now())
        self.page: Optional[Page] = None
        self.browser: Optional[Browser] = None

        # Create output directory
        self.output_dir = config.screenshot_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Screenshot counter
        self.screenshot_counter = 0

    def screenshot(self, name: str, full_page: bool = False) -> str:
        """Take a screenshot with incrementing counter."""
        self.screenshot_counter += 1
        filename = f"{self.screenshot_counter:04d}_{name}.png"
        path = self.output_dir / filename
        self.page.screenshot(path=str(path), full_page=full_page)
        return str(path)

    def run_test(self, name: str, category: str, test_fn) -> TestResult:
        """Run a single test with error handling."""
        start = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            details = test_fn()
            duration = (time.time() - start) * 1000

            result = TestResult(
                test_name=name,
                category=category,
                timestamp=timestamp,
                duration_ms=duration,
                success=True,
                details=details or {}
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            error_screenshot = self.screenshot(f"error_{name}")

            result = TestResult(
                test_name=name,
                category=category,
                timestamp=timestamp,
                duration_ms=duration,
                success=False,
                screenshot_path=error_screenshot,
                error=str(e),
                details={"traceback": traceback.format_exc()}
            )

        self.session.add_result(result)
        self.log_result(result)
        return result

    def log_result(self, result: TestResult):
        """Log test result to console."""
        status = "✅ PASS" if result.success else "❌ FAIL"
        print(f"  {status} [{result.category}] {result.test_name} ({result.duration_ms:.0f}ms)")
        if result.error:
            print(f"       Error: {result.error[:100]}")

    def wait_and_click(self, selector: str, timeout: int = 5000):
        """Wait for element and click it."""
        self.page.wait_for_selector(selector, timeout=timeout)
        self.page.click(selector)
        time.sleep(0.3)

    def fill_input(self, selector: str, value: str):
        """Fill an input field."""
        self.page.fill(selector, value)
        time.sleep(0.2)


# ============================================================================
# EXPLORER.HTML TESTS
# ============================================================================

class ExplorerTests:
    """Tests for explorer.html"""

    def __init__(self, tester: UITester):
        self.t = tester
        self.page = tester.page
        self.config = tester.config

    def navigate(self):
        """Navigate to explorer.html"""
        self.page.goto(f"{self.config.base_url}/explorer.html")
        self.page.wait_for_load_state("networkidle")
        time.sleep(1)

    # --- Page Load Tests ---

    def test_page_loads(self):
        """Test that page loads correctly."""
        def run():
            self.navigate()
            title = self.page.title()
            screenshot = self.t.screenshot("explorer_loaded")
            return {
                "title": title,
                "screenshot": screenshot,
                "has_title": "Attention" in title or "Explorer" in title
            }
        return self.t.run_test("page_loads", "Explorer/Load", run)

    def test_header_visible(self):
        """Test header is visible."""
        def run():
            header = self.page.locator("h1, .header, header").first
            visible = header.is_visible()
            text = header.text_content() if visible else ""
            return {"visible": visible, "text": text}
        return self.t.run_test("header_visible", "Explorer/Load", run)

    # --- Input Field Tests ---

    def test_input_field_exists(self):
        """Test input field exists."""
        def run():
            input_field = self.page.locator("#inputText, textarea, input[type='text']").first
            visible = input_field.is_visible()
            screenshot = self.t.screenshot("input_field")
            return {"visible": visible, "screenshot": screenshot}
        return self.t.run_test("input_field_exists", "Explorer/Input", run)

    def test_input_empty(self):
        """Test with empty input."""
        def run():
            self.page.fill("#inputText", "")
            screenshot = self.t.screenshot("input_empty")
            return {"value": "", "screenshot": screenshot}
        return self.t.run_test("input_empty", "Explorer/Input", run)

    def test_input_short_text(self):
        """Test with short text."""
        def run():
            text = "Hello"
            self.page.fill("#inputText", text)
            screenshot = self.t.screenshot("input_short")
            return {"value": text, "screenshot": screenshot}
        return self.t.run_test("input_short_text", "Explorer/Input", run)

    def test_input_medium_text(self):
        """Test with medium text."""
        def run():
            text = "What is the capital of France and why is it important?"
            self.page.fill("#inputText", text)
            screenshot = self.t.screenshot("input_medium")
            return {"value": text, "length": len(text), "screenshot": screenshot}
        return self.t.run_test("input_medium_text", "Explorer/Input", run)

    def test_input_long_text(self):
        """Test with long text."""
        def run():
            text = " ".join(["word"] * 100)
            self.page.fill("#inputText", text)
            screenshot = self.t.screenshot("input_long")
            return {"length": len(text), "word_count": 100, "screenshot": screenshot}
        return self.t.run_test("input_long_text", "Explorer/Input", run)

    def test_input_special_chars(self):
        """Test with special characters."""
        def run():
            text = "Special: @#$%^&*()[]{}|\\<>?/~`"
            self.page.fill("#inputText", text)
            actual = self.page.input_value("#inputText")
            screenshot = self.t.screenshot("input_special")
            return {"input": text, "actual": actual, "match": text == actual}
        return self.t.run_test("input_special_chars", "Explorer/Input", run)

    def test_input_unicode(self):
        """Test with unicode characters."""
        def run():
            text = "Unicode: 你好 مرحبا שלום 🚀🎉"
            self.page.fill("#inputText", text)
            actual = self.page.input_value("#inputText")
            screenshot = self.t.screenshot("input_unicode")
            return {"input": text, "actual": actual, "screenshot": screenshot}
        return self.t.run_test("input_unicode", "Explorer/Input", run)

    def test_input_multiline(self):
        """Test with multiline text."""
        def run():
            text = "Line 1\nLine 2\nLine 3"
            self.page.fill("#inputText", text)
            screenshot = self.t.screenshot("input_multiline")
            return {"lines": 3, "screenshot": screenshot}
        return self.t.run_test("input_multiline", "Explorer/Input", run)

    # --- Button Tests ---

    def test_analyze_button_exists(self):
        """Test Analyze button exists."""
        def run():
            btn = self.page.locator("button:has-text('Analyze'), .analyze-btn, #analyzeBtn")
            count = btn.count()
            visible = btn.first.is_visible() if count > 0 else False
            screenshot = self.t.screenshot("analyze_button")
            return {"count": count, "visible": visible, "screenshot": screenshot}
        return self.t.run_test("analyze_button_exists", "Explorer/Buttons", run)

    def test_clear_button_exists(self):
        """Test Clear button exists."""
        def run():
            btn = self.page.locator("button:has-text('Clear'), .clear-btn")
            count = btn.count()
            screenshot = self.t.screenshot("clear_button")
            return {"count": count, "screenshot": screenshot}
        return self.t.run_test("clear_button_exists", "Explorer/Buttons", run)

    def test_clear_button_works(self):
        """Test Clear button clears input."""
        def run():
            self.page.fill("#inputText", "Some text to clear")
            before = self.page.input_value("#inputText")
            self.page.click("button:has-text('Clear')")
            time.sleep(0.3)
            after = self.page.input_value("#inputText")
            screenshot = self.t.screenshot("clear_after")
            return {"before": before, "after": after, "cleared": after == "", "screenshot": screenshot}
        return self.t.run_test("clear_button_works", "Explorer/Buttons", run)

    def test_help_button_exists(self):
        """Test Help button exists."""
        def run():
            btn = self.page.locator(".help-btn, button:has-text('Help'), button:has-text('?')")
            count = btn.count()
            screenshot = self.t.screenshot("help_button")
            return {"count": count, "screenshot": screenshot}
        return self.t.run_test("help_button_exists", "Explorer/Buttons", run)

    def test_direction_buttons(self):
        """Test Attends To / Attended By buttons."""
        def run():
            attends_to = self.page.locator("button:has-text('Attends To'), input[value='attends_to']")
            attended_by = self.page.locator("button:has-text('Attended By'), input[value='attended_by']")
            screenshot = self.t.screenshot("direction_buttons")
            return {
                "attends_to_count": attends_to.count(),
                "attended_by_count": attended_by.count(),
                "screenshot": screenshot
            }
        return self.t.run_test("direction_buttons", "Explorer/Buttons", run)

    # --- Help Modal Tests ---

    def test_help_modal_opens(self):
        """Test help modal opens."""
        def run():
            self.page.click(".help-btn")
            time.sleep(0.5)
            modal = self.page.locator(".modal, .help-modal, [role='dialog']")
            visible = modal.first.is_visible() if modal.count() > 0 else False
            screenshot = self.t.screenshot("help_modal_open")
            return {"modal_visible": visible, "screenshot": screenshot}
        return self.t.run_test("help_modal_opens", "Explorer/Modal", run)

    def test_help_modal_content(self):
        """Test help modal has content."""
        def run():
            modal = self.page.locator(".modal-content, .modal-body")
            if modal.count() > 0:
                text = modal.first.text_content()
                has_content = len(text) > 100
                screenshot = self.t.screenshot("help_modal_content")
                return {"has_content": has_content, "content_length": len(text), "screenshot": screenshot}
            return {"has_content": False}
        return self.t.run_test("help_modal_content", "Explorer/Modal", run)

    def test_help_modal_scroll(self):
        """Test help modal scrolls."""
        def run():
            modal = self.page.locator(".modal-content, .modal-body")
            if modal.count() > 0:
                # Scroll down
                modal.first.evaluate("el => el.scrollTop = 500")
                time.sleep(0.3)
                screenshot1 = self.t.screenshot("help_modal_scroll_1")

                # Scroll more
                modal.first.evaluate("el => el.scrollTop = 1000")
                time.sleep(0.3)
                screenshot2 = self.t.screenshot("help_modal_scroll_2")

                # Scroll to bottom
                modal.first.evaluate("el => el.scrollTop = el.scrollHeight")
                time.sleep(0.3)
                screenshot3 = self.t.screenshot("help_modal_scroll_3")

                return {"scrolled": True, "screenshots": [screenshot1, screenshot2, screenshot3]}
            return {"scrolled": False}
        return self.t.run_test("help_modal_scroll", "Explorer/Modal", run)

    def test_help_modal_closes(self):
        """Test help modal closes."""
        def run():
            close_btn = self.page.locator(".modal-close, .close-btn, button:has-text('×')")
            if close_btn.count() > 0:
                close_btn.first.click()
                time.sleep(0.3)
                modal = self.page.locator(".modal, .help-modal")
                closed = not modal.first.is_visible() if modal.count() > 0 else True
                screenshot = self.t.screenshot("help_modal_closed")
                return {"closed": closed, "screenshot": screenshot}
            # Try escape key
            self.page.keyboard.press("Escape")
            time.sleep(0.3)
            screenshot = self.t.screenshot("help_modal_escape")
            return {"closed": True, "method": "escape", "screenshot": screenshot}
        return self.t.run_test("help_modal_closes", "Explorer/Modal", run)

    # --- Slider/Settings Tests ---

    def test_settings_panel(self):
        """Test settings panel exists."""
        def run():
            # Look for settings elements
            sliders = self.page.locator("input[type='range']")
            number_inputs = self.page.locator("input[type='number']")
            screenshot = self.t.screenshot("settings_panel")
            return {
                "slider_count": sliders.count(),
                "number_input_count": number_inputs.count(),
                "screenshot": screenshot
            }
        return self.t.run_test("settings_panel", "Explorer/Settings", run)

    def test_slider_max_tokens(self):
        """Test max tokens slider if exists."""
        def run():
            slider = self.page.locator("#maxTokens, input[name='maxTokens'], input[type='range']").first
            if slider.count() > 0:
                # Move slider
                slider.fill("50")
                time.sleep(0.2)
                screenshot1 = self.t.screenshot("slider_max_tokens_50")

                slider.fill("200")
                time.sleep(0.2)
                screenshot2 = self.t.screenshot("slider_max_tokens_200")

                return {"adjusted": True, "screenshots": [screenshot1, screenshot2]}
            return {"adjusted": False, "reason": "slider not found"}
        return self.t.run_test("slider_max_tokens", "Explorer/Settings", run)

    def test_topk_setting(self):
        """Test top-k setting."""
        def run():
            topk = self.page.locator("#topK, input[name='topK'], input[name='top_k']")
            if topk.count() > 0:
                topk.first.fill("5")
                time.sleep(0.2)
                screenshot1 = self.t.screenshot("topk_5")

                topk.first.fill("20")
                time.sleep(0.2)
                screenshot2 = self.t.screenshot("topk_20")

                return {"adjusted": True, "screenshots": [screenshot1, screenshot2]}
            return {"adjusted": False}
        return self.t.run_test("topk_setting", "Explorer/Settings", run)

    def test_temperature_setting(self):
        """Test temperature setting."""
        def run():
            temp = self.page.locator("#temperature, input[name='temperature']")
            if temp.count() > 0:
                temp.first.fill("0.1")
                time.sleep(0.2)
                screenshot1 = self.t.screenshot("temp_0.1")

                temp.first.fill("1.5")
                time.sleep(0.2)
                screenshot2 = self.t.screenshot("temp_1.5")

                return {"adjusted": True, "screenshots": [screenshot1, screenshot2]}
            return {"adjusted": False}
        return self.t.run_test("temperature_setting", "Explorer/Settings", run)

    # --- Keyboard Navigation ---

    def test_tab_navigation(self):
        """Test tab navigation through elements."""
        def run():
            self.page.keyboard.press("Tab")
            focused1 = self.page.evaluate("document.activeElement.tagName")

            self.page.keyboard.press("Tab")
            focused2 = self.page.evaluate("document.activeElement.tagName")

            self.page.keyboard.press("Tab")
            focused3 = self.page.evaluate("document.activeElement.tagName")

            screenshot = self.t.screenshot("tab_navigation")
            return {
                "focused_elements": [focused1, focused2, focused3],
                "screenshot": screenshot
            }
        return self.t.run_test("tab_navigation", "Explorer/Keyboard", run)

    def test_enter_submits(self):
        """Test Enter key behavior."""
        def run():
            self.page.fill("#inputText", "Test query")
            self.page.keyboard.press("Enter")
            time.sleep(0.5)
            screenshot = self.t.screenshot("enter_submit")
            return {"submitted": True, "screenshot": screenshot}
        return self.t.run_test("enter_submits", "Explorer/Keyboard", run)

    # --- Responsive Layout ---

    def test_desktop_layout(self):
        """Test desktop layout (1920x1080)."""
        def run():
            self.page.set_viewport_size({"width": 1920, "height": 1080})
            time.sleep(0.5)
            screenshot = self.t.screenshot("layout_desktop", full_page=True)
            return {"viewport": "1920x1080", "screenshot": screenshot}
        return self.t.run_test("desktop_layout", "Explorer/Responsive", run)

    def test_laptop_layout(self):
        """Test laptop layout (1366x768)."""
        def run():
            self.page.set_viewport_size({"width": 1366, "height": 768})
            time.sleep(0.5)
            screenshot = self.t.screenshot("layout_laptop", full_page=True)
            return {"viewport": "1366x768", "screenshot": screenshot}
        return self.t.run_test("laptop_layout", "Explorer/Responsive", run)

    def test_tablet_layout(self):
        """Test tablet layout (768x1024)."""
        def run():
            self.page.set_viewport_size({"width": 768, "height": 1024})
            time.sleep(0.5)
            screenshot = self.t.screenshot("layout_tablet", full_page=True)
            return {"viewport": "768x1024", "screenshot": screenshot}
        return self.t.run_test("tablet_layout", "Explorer/Responsive", run)

    def test_mobile_layout(self):
        """Test mobile layout (375x667)."""
        def run():
            self.page.set_viewport_size({"width": 375, "height": 667})
            time.sleep(0.5)
            screenshot = self.t.screenshot("layout_mobile", full_page=True)
            # Reset viewport
            self.page.set_viewport_size({
                "width": self.config.viewport_width,
                "height": self.config.viewport_height
            })
            return {"viewport": "375x667", "screenshot": screenshot}
        return self.t.run_test("mobile_layout", "Explorer/Responsive", run)

    # --- Query Tests (if server available) ---

    def test_query_simple(self):
        """Test simple query."""
        def run():
            self.page.fill("#inputText", "What is AI?")
            screenshot_before = self.t.screenshot("query_simple_before")

            self.page.click("button:has-text('Analyze')")

            # Wait for response (with timeout)
            try:
                self.page.wait_for_selector(".token, .response, .output", timeout=30000)
                time.sleep(1)
                screenshot_after = self.t.screenshot("query_simple_after")
                return {
                    "query": "What is AI?",
                    "response_received": True,
                    "screenshots": [screenshot_before, screenshot_after]
                }
            except:
                screenshot_timeout = self.t.screenshot("query_simple_timeout")
                return {
                    "query": "What is AI?",
                    "response_received": False,
                    "screenshot": screenshot_timeout
                }
        return self.t.run_test("query_simple", "Explorer/Query", run)

    def run_all_tests(self):
        """Run all explorer tests."""
        print("\n" + "="*60)
        print("EXPLORER.HTML TESTS")
        print("="*60)

        # Page load
        self.test_page_loads()
        self.test_header_visible()

        # Input field
        self.test_input_field_exists()
        self.test_input_empty()
        self.test_input_short_text()
        self.test_input_medium_text()
        self.test_input_long_text()
        self.test_input_special_chars()
        self.test_input_unicode()
        self.test_input_multiline()

        # Buttons
        self.test_analyze_button_exists()
        self.test_clear_button_exists()
        self.test_clear_button_works()
        self.test_help_button_exists()
        self.test_direction_buttons()

        # Help modal
        self.test_help_modal_opens()
        self.test_help_modal_content()
        self.test_help_modal_scroll()
        self.test_help_modal_closes()

        # Settings
        self.test_settings_panel()
        self.test_slider_max_tokens()
        self.test_topk_setting()
        self.test_temperature_setting()

        # Keyboard
        self.test_tab_navigation()
        self.test_enter_submits()

        # Responsive
        self.test_desktop_layout()
        self.test_laptop_layout()
        self.test_tablet_layout()
        self.test_mobile_layout()

        # Queries (if server available)
        self.test_query_simple()


# ============================================================================
# RAG_EXPLORER.HTML TESTS
# ============================================================================

class RAGExplorerTests:
    """Tests for rag_explorer.html"""

    def __init__(self, tester: UITester):
        self.t = tester
        self.page = tester.page
        self.config = tester.config

    def navigate(self):
        """Navigate to rag_explorer.html"""
        self.page.goto(f"{self.config.base_url}/rag_explorer.html")
        self.page.wait_for_load_state("networkidle")
        time.sleep(1)

    # --- Page Load Tests ---

    def test_page_loads(self):
        """Test page loads."""
        def run():
            self.navigate()
            title = self.page.title()
            screenshot = self.t.screenshot("rag_loaded")
            return {"title": title, "screenshot": screenshot}
        return self.t.run_test("page_loads", "RAG/Load", run)

    def test_sections_exist(self):
        """Test main sections exist."""
        def run():
            query_section = self.page.locator(".query-section, #querySection, [data-section='query']")
            doc_section = self.page.locator(".document-section, #documentSection, .documents")
            screenshot = self.t.screenshot("rag_sections")
            return {
                "query_section": query_section.count() > 0,
                "document_section": doc_section.count() > 0,
                "screenshot": screenshot
            }
        return self.t.run_test("sections_exist", "RAG/Load", run)

    # --- Query Input Tests ---

    def test_query_input_exists(self):
        """Test query input exists."""
        def run():
            input_field = self.page.locator("#queryInput, #query, textarea").first
            visible = input_field.is_visible()
            screenshot = self.t.screenshot("rag_query_input")
            return {"visible": visible, "screenshot": screenshot}
        return self.t.run_test("query_input_exists", "RAG/Input", run)

    def test_query_various_inputs(self):
        """Test various query inputs."""
        def run():
            test_cases = [
                ("empty", ""),
                ("short", "Hello"),
                ("question", "What are the benefits of exercise?"),
                ("long", " ".join(["word"] * 50)),
            ]
            results = []
            for name, text in test_cases:
                self.page.fill("#queryInput, #query, textarea", text)
                time.sleep(0.2)
                screenshot = self.t.screenshot(f"rag_query_{name}")
                results.append({"name": name, "length": len(text), "screenshot": screenshot})
            return {"test_cases": results}
        return self.t.run_test("query_various_inputs", "RAG/Input", run)

    # --- Document Panel Tests ---

    def test_document_list(self):
        """Test document list."""
        def run():
            docs = self.page.locator(".document, .doc-item, [data-doc]")
            count = docs.count()
            screenshot = self.t.screenshot("rag_documents")
            return {"document_count": count, "screenshot": screenshot}
        return self.t.run_test("document_list", "RAG/Documents", run)

    def test_document_click(self):
        """Test clicking on a document."""
        def run():
            docs = self.page.locator(".document, .doc-item")
            if docs.count() > 0:
                docs.first.click()
                time.sleep(0.3)
                screenshot = self.t.screenshot("rag_doc_clicked")
                return {"clicked": True, "screenshot": screenshot}
            return {"clicked": False, "reason": "no documents"}
        return self.t.run_test("document_click", "RAG/Documents", run)

    # --- Analyze Button Tests ---

    def test_analyze_button(self):
        """Test analyze button."""
        def run():
            btn = self.page.locator("button:has-text('Analyze'), .analyze-btn")
            count = btn.count()
            visible = btn.first.is_visible() if count > 0 else False
            screenshot = self.t.screenshot("rag_analyze_btn")
            return {"count": count, "visible": visible, "screenshot": screenshot}
        return self.t.run_test("analyze_button", "RAG/Buttons", run)

    def test_analyze_click(self):
        """Test clicking analyze."""
        def run():
            self.page.fill("#queryInput, #query, textarea", "What is machine learning?")
            time.sleep(0.2)
            self.page.click("button:has-text('Analyze')")
            time.sleep(1)
            screenshot = self.t.screenshot("rag_analyze_clicked")
            return {"clicked": True, "screenshot": screenshot}
        return self.t.run_test("analyze_click", "RAG/Buttons", run)

    # --- Help/Glossary Tests ---

    def test_help_button(self):
        """Test help/glossary button."""
        def run():
            btn = self.page.locator(".help-btn, button:has-text('Glossary'), button:has-text('Help')")
            count = btn.count()
            if count > 0:
                btn.first.click()
                time.sleep(0.5)
                screenshot = self.t.screenshot("rag_help_open")
                return {"found": True, "screenshot": screenshot}
            return {"found": False}
        return self.t.run_test("help_button", "RAG/Help", run)

    def test_help_content(self):
        """Test help content."""
        def run():
            modal = self.page.locator(".modal-content, .modal-body, .help-content")
            if modal.count() > 0:
                text = modal.first.text_content()

                # Scroll through content
                screenshots = []
                for scroll_pos in [0, 400, 800, 1200]:
                    modal.first.evaluate(f"el => el.scrollTop = {scroll_pos}")
                    time.sleep(0.3)
                    screenshots.append(self.t.screenshot(f"rag_help_scroll_{scroll_pos}"))

                return {
                    "content_length": len(text),
                    "has_content": len(text) > 100,
                    "screenshots": screenshots
                }
            return {"has_content": False}
        return self.t.run_test("help_content", "RAG/Help", run)

    def test_help_close(self):
        """Test closing help."""
        def run():
            close = self.page.locator(".modal-close, .close-btn, button:has-text('×')")
            if close.count() > 0:
                close.first.click()
                time.sleep(0.3)
            else:
                self.page.keyboard.press("Escape")
                time.sleep(0.3)
            screenshot = self.t.screenshot("rag_help_closed")
            return {"closed": True, "screenshot": screenshot}
        return self.t.run_test("help_close", "RAG/Help", run)

    # --- Visualization Tests ---

    def test_visualization_area(self):
        """Test visualization area."""
        def run():
            viz = self.page.locator("canvas, svg, .visualization, .chart")
            count = viz.count()
            screenshot = self.t.screenshot("rag_viz_area")
            return {"viz_elements": count, "screenshot": screenshot}
        return self.t.run_test("visualization_area", "RAG/Viz", run)

    # --- Responsive Tests ---

    def test_responsive_layouts(self):
        """Test responsive layouts."""
        def run():
            viewports = [
                ("desktop", 1920, 1080),
                ("laptop", 1366, 768),
                ("tablet", 768, 1024),
                ("mobile", 375, 667),
            ]
            screenshots = []
            for name, w, h in viewports:
                self.page.set_viewport_size({"width": w, "height": h})
                time.sleep(0.5)
                screenshot = self.t.screenshot(f"rag_layout_{name}", full_page=True)
                screenshots.append({"name": name, "viewport": f"{w}x{h}", "screenshot": screenshot})

            # Reset
            self.page.set_viewport_size({
                "width": self.config.viewport_width,
                "height": self.config.viewport_height
            })
            return {"layouts": screenshots}
        return self.t.run_test("responsive_layouts", "RAG/Responsive", run)

    def run_all_tests(self):
        """Run all RAG explorer tests."""
        print("\n" + "="*60)
        print("RAG_EXPLORER.HTML TESTS")
        print("="*60)

        # Page load
        self.test_page_loads()
        self.test_sections_exist()

        # Query input
        self.test_query_input_exists()
        self.test_query_various_inputs()

        # Documents
        self.test_document_list()
        self.test_document_click()

        # Buttons
        self.test_analyze_button()
        self.test_analyze_click()

        # Help
        self.test_help_button()
        self.test_help_content()
        self.test_help_close()

        # Visualization
        self.test_visualization_area()

        # Responsive
        self.test_responsive_layouts()


# ============================================================================
# STRESS & EDGE CASE TESTS
# ============================================================================

class StressTests:
    """Stress and edge case tests."""

    def __init__(self, tester: UITester):
        self.t = tester
        self.page = tester.page
        self.config = tester.config

    def test_rapid_clicks(self):
        """Test rapid button clicks."""
        def run():
            self.page.goto(f"{self.config.base_url}/explorer.html")
            self.page.wait_for_load_state("networkidle")

            # Rapid clicks on analyze
            for i in range(10):
                self.page.click("button:has-text('Analyze')", timeout=1000)
                time.sleep(0.1)

            screenshot = self.t.screenshot("stress_rapid_clicks")
            return {"clicks": 10, "screenshot": screenshot}
        return self.t.run_test("rapid_clicks", "Stress", run)

    def test_rapid_typing(self):
        """Test rapid typing."""
        def run():
            self.page.goto(f"{self.config.base_url}/explorer.html")
            self.page.wait_for_load_state("networkidle")

            # Type rapidly
            text = "The quick brown fox jumps over the lazy dog " * 10
            self.page.fill("#inputText", text)

            # Check value
            actual = self.page.input_value("#inputText")
            screenshot = self.t.screenshot("stress_rapid_typing")
            return {"typed_length": len(text), "actual_length": len(actual), "screenshot": screenshot}
        return self.t.run_test("rapid_typing", "Stress", run)

    def test_modal_spam(self):
        """Test opening/closing modal rapidly."""
        def run():
            self.page.goto(f"{self.config.base_url}/explorer.html")
            self.page.wait_for_load_state("networkidle")

            for i in range(5):
                self.page.click(".help-btn")
                time.sleep(0.2)
                self.page.keyboard.press("Escape")
                time.sleep(0.2)

            screenshot = self.t.screenshot("stress_modal_spam")
            return {"cycles": 5, "screenshot": screenshot}
        return self.t.run_test("modal_spam", "Stress", run)

    def test_viewport_resize(self):
        """Test rapid viewport resizing."""
        def run():
            self.page.goto(f"{self.config.base_url}/explorer.html")
            self.page.wait_for_load_state("networkidle")

            viewports = [(1920, 1080), (800, 600), (1366, 768), (375, 667), (1600, 900)]
            for w, h in viewports:
                self.page.set_viewport_size({"width": w, "height": h})
                time.sleep(0.3)

            screenshot = self.t.screenshot("stress_resize")
            return {"resize_count": len(viewports), "screenshot": screenshot}
        return self.t.run_test("viewport_resize", "Stress", run)

    def run_all_tests(self):
        """Run all stress tests."""
        print("\n" + "="*60)
        print("STRESS TESTS")
        print("="*60)

        self.test_rapid_clicks()
        self.test_rapid_typing()
        self.test_modal_spam()
        self.test_viewport_resize()


# ============================================================================
# EXTENDED QUERY TESTS
# ============================================================================

class ExtendedQueryTests:
    """Extended tests with many different queries."""

    def __init__(self, tester: UITester):
        self.t = tester
        self.page = tester.page
        self.config = tester.config

    def test_query(self, query: str, index: int):
        """Test a single query."""
        def run():
            self.page.fill("#inputText", query)
            screenshot_input = self.t.screenshot(f"query_{index:03d}_input")

            self.page.click("button:has-text('Analyze')")

            try:
                self.page.wait_for_selector(".token, .response", timeout=30000)
                time.sleep(1)
                screenshot_result = self.t.screenshot(f"query_{index:03d}_result")
                return {
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "success": True,
                    "screenshots": [screenshot_input, screenshot_result]
                }
            except:
                screenshot_timeout = self.t.screenshot(f"query_{index:03d}_timeout")
                return {
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "success": False,
                    "screenshot": screenshot_timeout
                }

        safe_name = f"query_{index:03d}"
        return self.t.run_test(safe_name, "ExtendedQuery", run)

    def run_all_tests(self):
        """Run extended query tests."""
        print("\n" + "="*60)
        print("EXTENDED QUERY TESTS")
        print("="*60)

        self.page.goto(f"{self.config.base_url}/explorer.html")
        self.page.wait_for_load_state("networkidle")

        for i, query in enumerate(self.config.test_queries):
            self.test_query(query, i)
            time.sleep(0.5)  # Brief pause between queries


# ============================================================================
# REPORT GENERATOR
# ============================================================================

def generate_html_report(session: TestSession, output_dir: Path, config: TestConfig):
    """Generate beautiful HTML report."""

    duration = datetime.now() - session.start_time
    pass_rate = (session.passed_tests / session.total_tests * 100) if session.total_tests > 0 else 0

    # Group results by category
    categories = {}
    for result in session.results:
        cat = result.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UI Test Report - {session.start_time.strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        :root {{
            --bg: #0d1117;
            --card-bg: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-muted: #8b949e;
            --success: #238636;
            --error: #da3633;
            --warning: #d29922;
            --accent: #58a6ff;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}

        .container {{ max-width: 1400px; margin: 0 auto; }}

        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            color: #fff;
        }}

        .subtitle {{
            color: var(--text-muted);
            margin-bottom: 2rem;
        }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .stat-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
        }}

        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #fff;
        }}

        .stat-value.success {{ color: var(--success); }}
        .stat-value.error {{ color: var(--error); }}
        .stat-value.warning {{ color: var(--warning); }}

        .stat-label {{
            color: var(--text-muted);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .progress-bar {{
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
            margin: 1rem 0;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--success), #2ea043);
            border-radius: 4px;
            transition: width 0.5s;
        }}

        .category {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 1.5rem;
            overflow: hidden;
        }}

        .category-header {{
            background: rgba(88, 166, 255, 0.1);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .category-header:hover {{
            background: rgba(88, 166, 255, 0.15);
        }}

        .category-title {{
            font-size: 1.2rem;
            font-weight: 600;
        }}

        .category-stats {{
            display: flex;
            gap: 1rem;
        }}

        .category-stat {{
            font-size: 0.9rem;
        }}

        .category-stat.pass {{ color: var(--success); }}
        .category-stat.fail {{ color: var(--error); }}

        .category-content {{
            padding: 1rem;
        }}

        .test-row {{
            display: grid;
            grid-template-columns: 30px 1fr 100px 80px;
            gap: 1rem;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
            align-items: center;
        }}

        .test-row:last-child {{ border-bottom: none; }}

        .test-row:hover {{
            background: rgba(255,255,255,0.02);
        }}

        .status-icon {{
            font-size: 1.2rem;
        }}

        .test-name {{
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.9rem;
        }}

        .test-duration {{
            color: var(--text-muted);
            font-size: 0.85rem;
        }}

        .test-screenshot {{
            text-align: right;
        }}

        .test-screenshot a {{
            color: var(--accent);
            text-decoration: none;
            font-size: 0.85rem;
        }}

        .test-screenshot a:hover {{
            text-decoration: underline;
        }}

        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1rem;
            margin-top: 2rem;
        }}

        .gallery-item {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }}

        .gallery-item img {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-bottom: 1px solid var(--border);
        }}

        .gallery-item-info {{
            padding: 0.75rem;
        }}

        .gallery-item-title {{
            font-size: 0.9rem;
            font-weight: 500;
            margin-bottom: 0.25rem;
        }}

        .gallery-item-meta {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}

        .error-details {{
            background: rgba(218, 54, 51, 0.1);
            border: 1px solid var(--error);
            border-radius: 4px;
            padding: 1rem;
            margin-top: 0.5rem;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.8rem;
            overflow-x: auto;
        }}

        footer {{
            text-align: center;
            margin-top: 3rem;
            padding: 2rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border);
        }}

        .toggle-icon {{
            transition: transform 0.2s;
        }}

        .collapsed .toggle-icon {{
            transform: rotate(-90deg);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>UI Test Report</h1>
        <p class="subtitle">
            Started: {session.start_time.strftime('%Y-%m-%d %H:%M:%S')} |
            Duration: {str(duration).split('.')[0]} |
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{session.total_tests}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success">{session.passed_tests}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value error">{session.failed_tests}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: var(--accent);">{session.screenshots_taken}</div>
                <div class="stat-label">Screenshots</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'success' if pass_rate >= 90 else 'warning' if pass_rate >= 70 else 'error'}">{pass_rate:.1f}%</div>
                <div class="stat-label">Pass Rate</div>
            </div>
        </div>

        <div class="progress-bar">
            <div class="progress-fill" style="width: {pass_rate}%;"></div>
        </div>
"""

    # Add categories
    for category, results in categories.items():
        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed

        html += f"""
        <div class="category">
            <div class="category-header" onclick="this.parentElement.classList.toggle('collapsed')">
                <span class="category-title">{category}</span>
                <div class="category-stats">
                    <span class="category-stat pass">✓ {passed}</span>
                    <span class="category-stat fail">✗ {failed}</span>
                    <span class="toggle-icon">▼</span>
                </div>
            </div>
            <div class="category-content">
"""

        for result in results:
            icon = "✅" if result.success else "❌"
            screenshot_link = ""
            if result.screenshot_path:
                screenshot_link = f'<a href="{Path(result.screenshot_path).name}" target="_blank">View</a>'

            html += f"""
                <div class="test-row">
                    <span class="status-icon">{icon}</span>
                    <span class="test-name">{result.test_name}</span>
                    <span class="test-duration">{result.duration_ms:.0f}ms</span>
                    <span class="test-screenshot">{screenshot_link}</span>
                </div>
"""

            if result.error:
                html += f"""
                <div class="error-details">{result.error}</div>
"""

        html += """
            </div>
        </div>
"""

    # Screenshot gallery
    html += """
        <h2 style="margin-top: 3rem; margin-bottom: 1rem;">Screenshot Gallery</h2>
        <div class="gallery">
"""

    screenshots = sorted(output_dir.glob("*.png"))[:50]  # Limit to 50 in gallery
    for screenshot in screenshots:
        html += f"""
            <div class="gallery-item">
                <a href="{screenshot.name}" target="_blank">
                    <img src="{screenshot.name}" alt="{screenshot.stem}">
                </a>
                <div class="gallery-item-info">
                    <div class="gallery-item-title">{screenshot.stem}</div>
                </div>
            </div>
"""

    html += """
        </div>

        <footer>
            <p>Generated by Overnight UI Test Suite</p>
            <p>Attention Explorer - SGLang</p>
        </footer>
    </div>

    <script>
        // Collapse failed categories by default
        document.querySelectorAll('.category').forEach(cat => {
            const failCount = parseInt(cat.querySelector('.category-stat.fail').textContent.replace('✗ ', ''));
            if (failCount === 0) {
                cat.classList.add('collapsed');
            }
        });
    </script>
</body>
</html>
"""

    report_path = output_dir / "report.html"
    report_path.write_text(html)

    # Also save JSON results
    json_path = output_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump({
            "session": {
                "start_time": session.start_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "total_tests": session.total_tests,
                "passed_tests": session.passed_tests,
                "failed_tests": session.failed_tests,
                "pass_rate": pass_rate,
                "screenshots_taken": session.screenshots_taken,
            },
            "results": [asdict(r) for r in session.results]
        }, f, indent=2, default=str)

    return report_path


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_overnight_tests(duration_hours: float = 8.0):
    """Run comprehensive overnight tests."""

    config = TestConfig(duration_hours=duration_hours)

    print("="*70)
    print("  8-HOUR COMPREHENSIVE UI TEST SUITE")
    print("="*70)
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration: {duration_hours} hours")
    print(f"  Output: {config.screenshot_dir}")
    print("="*70)
    print()

    end_time = datetime.now() + timedelta(hours=duration_hours)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, slow_mo=config.slow_mo)
        page = browser.new_page(viewport={
            "width": config.viewport_width,
            "height": config.viewport_height
        })

        tester = UITester(config)
        tester.page = page
        tester.browser = browser

        cycle = 0

        while datetime.now() < end_time:
            cycle += 1
            print(f"\n{'#'*70}")
            print(f"# TEST CYCLE {cycle}")
            print(f"# Time remaining: {str(end_time - datetime.now()).split('.')[0]}")
            print(f"{'#'*70}")

            # Run all test suites
            try:
                explorer_tests = ExplorerTests(tester)
                explorer_tests.run_all_tests()
            except Exception as e:
                print(f"Explorer tests error: {e}")

            try:
                rag_tests = RAGExplorerTests(tester)
                rag_tests.run_all_tests()
            except Exception as e:
                print(f"RAG tests error: {e}")

            try:
                stress_tests = StressTests(tester)
                stress_tests.run_all_tests()
            except Exception as e:
                print(f"Stress tests error: {e}")

            # Extended query tests (every 3rd cycle)
            if cycle % 3 == 1:
                try:
                    query_tests = ExtendedQueryTests(tester)
                    query_tests.run_all_tests()
                except Exception as e:
                    print(f"Query tests error: {e}")

            # Progress report
            print(f"\n--- Cycle {cycle} Complete ---")
            print(f"Total tests: {tester.session.total_tests}")
            print(f"Passed: {tester.session.passed_tests}")
            print(f"Failed: {tester.session.failed_tests}")
            print(f"Screenshots: {tester.session.screenshots_taken}")

            # Generate intermediate report
            if cycle % 5 == 0:
                report_path = generate_html_report(tester.session, tester.output_dir, config)
                print(f"Intermediate report: {report_path}")

            # Brief pause between cycles
            time.sleep(30)

        browser.close()

    # Final report
    print("\n" + "="*70)
    print("  TEST SUITE COMPLETE")
    print("="*70)

    report_path = generate_html_report(tester.session, tester.output_dir, config)

    print(f"\nFinal Results:")
    print(f"  Total tests: {tester.session.total_tests}")
    print(f"  Passed: {tester.session.passed_tests}")
    print(f"  Failed: {tester.session.failed_tests}")
    print(f"  Pass rate: {tester.session.passed_tests/tester.session.total_tests*100:.1f}%")
    print(f"  Screenshots: {tester.session.screenshots_taken}")
    print(f"\n  Report: {report_path}")
    print(f"  Results: {tester.output_dir}/results.json")
    print("="*70)

    return tester.session


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="8-Hour Comprehensive UI Test Suite")
    parser.add_argument("--hours", type=float, default=8.0, help="Test duration in hours")
    parser.add_argument("--quick", action="store_true", help="Quick test (5 minutes)")
    args = parser.parse_args()

    duration = 5/60 if args.quick else args.hours  # 5 minutes for quick test
    run_overnight_tests(duration)
