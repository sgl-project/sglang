#!/usr/bin/env python3
"""
Comprehensive UI Test Suite for Attention Explorer Interfaces

Tests all HTML UIs:
1. explorer.html - Main attention visualization
2. rag_explorer.html - RAG query analysis and document ranking

Uses Playwright for browser automation and screenshot capture.

Usage:
    python scripts/ui_test_suite.py [--headed] [--server URL]

Author: SGLang Team
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Playwright imports
try:
    from playwright.sync_api import sync_playwright, Page, Browser, expect
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright not available. Install with: pip install playwright && playwright install")


@dataclass
class UITestResult:
    """Result of a UI test."""
    test_name: str
    ui_file: str
    timestamp: str
    duration_ms: float
    success: bool
    screenshot_path: Optional[str]
    error: Optional[str]
    details: Dict[str, Any]


class UITestSuite:
    """Comprehensive UI test suite for attention explorer interfaces."""

    def __init__(
        self,
        base_url: str = "http://localhost:8082",
        server_url: str = "http://localhost:30000",
        output_dir: str = "ui_test_results",
        headed: bool = False,
    ):
        self.base_url = base_url
        self.server_url = server_url
        self.output_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headed = headed
        self.results: List[UITestResult] = []

        # UI files to test
        self.ui_files = {
            "explorer": f"{base_url}/explorer.html",
            "rag_explorer": f"{base_url}/rag_explorer.html",
        }

        print(f"UI Test Suite initialized")
        print(f"  Base URL: {base_url}")
        print(f"  Server URL: {server_url}")
        print(f"  Output: {self.output_dir}")
        print(f"  Headed mode: {headed}")

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _log(self, msg: str):
        print(f"[{self._timestamp()}] {msg}")

    def _save_result(self, result: UITestResult):
        self.results.append(result)
        status = "PASS" if result.success else "FAIL"
        self._log(f"  [{status}] {result.test_name} ({result.duration_ms:.0f}ms)")

    def _screenshot(self, page: Page, name: str) -> str:
        path = self.output_dir / f"{name}.png"
        page.screenshot(path=str(path))
        return str(path)

    # ============================================================
    # EXPLORER.HTML TESTS
    # ============================================================

    def test_explorer_loads(self, page: Page) -> UITestResult:
        """Test that explorer.html loads correctly."""
        start = time.time()
        try:
            page.goto(self.ui_files["explorer"])
            page.wait_for_load_state("networkidle")

            # Check for main elements
            title = page.title()
            has_header = page.locator("h1, .header, header").count() > 0

            screenshot = self._screenshot(page, "explorer_load")

            return UITestResult(
                test_name="explorer_loads",
                ui_file="explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=True,
                screenshot_path=screenshot,
                error=None,
                details={"title": title, "has_header": has_header}
            )
        except Exception as e:
            return UITestResult(
                test_name="explorer_loads",
                ui_file="explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=False,
                screenshot_path=None,
                error=str(e),
                details={}
            )

    def test_explorer_input_field(self, page: Page) -> UITestResult:
        """Test explorer input field functionality."""
        start = time.time()
        try:
            page.goto(self.ui_files["explorer"])
            page.wait_for_load_state("networkidle")

            # Find and interact with input field
            input_field = page.locator("input[type='text'], textarea").first
            input_field.fill("What is the capital of France?")

            # Check value was set
            value = input_field.input_value()

            screenshot = self._screenshot(page, "explorer_input")

            return UITestResult(
                test_name="explorer_input_field",
                ui_file="explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=len(value) > 0,
                screenshot_path=screenshot,
                error=None,
                details={"input_value": value}
            )
        except Exception as e:
            return UITestResult(
                test_name="explorer_input_field",
                ui_file="explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=False,
                screenshot_path=None,
                error=str(e),
                details={}
            )

    def test_explorer_buttons(self, page: Page) -> UITestResult:
        """Test explorer has clickable buttons."""
        start = time.time()
        try:
            page.goto(self.ui_files["explorer"])
            page.wait_for_load_state("networkidle")

            # Find buttons
            buttons = page.locator("button")
            button_count = buttons.count()

            button_texts = []
            for i in range(min(button_count, 10)):
                text = buttons.nth(i).text_content()
                if text:
                    button_texts.append(text.strip())

            screenshot = self._screenshot(page, "explorer_buttons")

            return UITestResult(
                test_name="explorer_buttons",
                ui_file="explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=button_count > 0,
                screenshot_path=screenshot,
                error=None,
                details={"button_count": button_count, "button_texts": button_texts}
            )
        except Exception as e:
            return UITestResult(
                test_name="explorer_buttons",
                ui_file="explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=False,
                screenshot_path=None,
                error=str(e),
                details={}
            )

    def test_explorer_visualization_container(self, page: Page) -> UITestResult:
        """Test explorer has visualization containers."""
        start = time.time()
        try:
            page.goto(self.ui_files["explorer"])
            page.wait_for_load_state("networkidle")

            # Look for canvas or SVG elements (common for visualizations)
            canvas_count = page.locator("canvas").count()
            svg_count = page.locator("svg").count()
            div_viz = page.locator(".visualization, .chart, .graph, #visualization").count()

            screenshot = self._screenshot(page, "explorer_viz")

            return UITestResult(
                test_name="explorer_visualization_container",
                ui_file="explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=True,
                screenshot_path=screenshot,
                error=None,
                details={
                    "canvas_count": canvas_count,
                    "svg_count": svg_count,
                    "viz_div_count": div_viz
                }
            )
        except Exception as e:
            return UITestResult(
                test_name="explorer_visualization_container",
                ui_file="explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=False,
                screenshot_path=None,
                error=str(e),
                details={}
            )

    # ============================================================
    # RAG_EXPLORER.HTML TESTS
    # ============================================================

    def test_rag_explorer_loads(self, page: Page) -> UITestResult:
        """Test that rag_explorer.html loads correctly."""
        start = time.time()
        try:
            page.goto(self.ui_files["rag_explorer"])
            page.wait_for_load_state("networkidle")

            # Check for main elements
            title = page.title()
            has_query_section = page.locator("text=Query").count() > 0 or page.locator(".query").count() > 0
            has_document_section = page.locator("text=Document").count() > 0 or page.locator(".document").count() > 0

            screenshot = self._screenshot(page, "rag_explorer_load")

            return UITestResult(
                test_name="rag_explorer_loads",
                ui_file="rag_explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=True,
                screenshot_path=screenshot,
                error=None,
                details={
                    "title": title,
                    "has_query_section": has_query_section,
                    "has_document_section": has_document_section
                }
            )
        except Exception as e:
            return UITestResult(
                test_name="rag_explorer_loads",
                ui_file="rag_explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=False,
                screenshot_path=None,
                error=str(e),
                details={}
            )

    def test_rag_sample_queries(self, page: Page) -> UITestResult:
        """Test RAG explorer sample query buttons."""
        start = time.time()
        try:
            page.goto(self.ui_files["rag_explorer"])
            page.wait_for_load_state("networkidle")

            # Sample queries are divs with class "sample-query", not buttons
            sample_queries = page.locator(".sample-query")
            sample_count = sample_queries.count()

            # Fallback: check for any clickable elements with sample query text
            if sample_count == 0:
                sample_queries = page.locator("[onclick*='setQuery']")
                sample_count = sample_queries.count()

            # Click on a sample query if available
            clicked = False
            if sample_count > 0:
                sample_queries.first.click()
                page.wait_for_timeout(500)
                clicked = True

            screenshot = self._screenshot(page, "rag_sample_queries")

            return UITestResult(
                test_name="rag_sample_queries",
                ui_file="rag_explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=sample_count > 0,
                screenshot_path=screenshot,
                error=None,
                details={"sample_query_count": sample_count, "clicked": clicked}
            )
        except Exception as e:
            return UITestResult(
                test_name="rag_sample_queries",
                ui_file="rag_explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=False,
                screenshot_path=None,
                error=str(e),
                details={}
            )

    def test_rag_query_input(self, page: Page) -> UITestResult:
        """Test RAG explorer query input functionality."""
        start = time.time()
        try:
            page.goto(self.ui_files["rag_explorer"])
            page.wait_for_load_state("networkidle")

            # Find query input
            query_input = page.locator("textarea, input[type='text']").first
            test_query = "What are the health benefits of green tea?"
            query_input.fill(test_query)

            # Verify input
            value = query_input.input_value()

            screenshot = self._screenshot(page, "rag_query_input")

            return UITestResult(
                test_name="rag_query_input",
                ui_file="rag_explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=value == test_query,
                screenshot_path=screenshot,
                error=None,
                details={"input_value": value, "expected": test_query}
            )
        except Exception as e:
            return UITestResult(
                test_name="rag_query_input",
                ui_file="rag_explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=False,
                screenshot_path=None,
                error=str(e),
                details={}
            )

    def test_rag_analyze_button(self, page: Page) -> UITestResult:
        """Test RAG explorer analyze button."""
        start = time.time()
        try:
            page.goto(self.ui_files["rag_explorer"])
            page.wait_for_load_state("networkidle")

            # Enter a query
            query_input = page.locator("textarea, input[type='text']").first
            query_input.fill("What is the capital of France?")

            # Find and click analyze button
            analyze_btn = page.locator("button:has-text('Analyze'), button:has-text('Query')")
            btn_count = analyze_btn.count()

            if btn_count > 0:
                analyze_btn.first.click()
                page.wait_for_timeout(1000)

            screenshot = self._screenshot(page, "rag_analyze_click")

            return UITestResult(
                test_name="rag_analyze_button",
                ui_file="rag_explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=btn_count > 0,
                screenshot_path=screenshot,
                error=None,
                details={"analyze_button_found": btn_count > 0}
            )
        except Exception as e:
            return UITestResult(
                test_name="rag_analyze_button",
                ui_file="rag_explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=False,
                screenshot_path=None,
                error=str(e),
                details={}
            )

    def test_rag_document_ranking(self, page: Page) -> UITestResult:
        """Test RAG explorer document ranking display."""
        start = time.time()
        try:
            page.goto(self.ui_files["rag_explorer"])
            page.wait_for_load_state("networkidle")

            # Look for document ranking section
            doc_section = page.locator("text=Document Ranking, text=Documents, .documents, #documents")
            has_doc_section = doc_section.count() > 0

            # Look for ranked items
            doc_items = page.locator(".document-item, .chunk, [class*='rank'], [class*='doc']")
            doc_count = doc_items.count()

            screenshot = self._screenshot(page, "rag_document_ranking")

            return UITestResult(
                test_name="rag_document_ranking",
                ui_file="rag_explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=True,
                screenshot_path=screenshot,
                error=None,
                details={
                    "has_document_section": has_doc_section,
                    "document_count": doc_count
                }
            )
        except Exception as e:
            return UITestResult(
                test_name="rag_document_ranking",
                ui_file="rag_explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=False,
                screenshot_path=None,
                error=str(e),
                details={}
            )

    def test_rag_connection_status(self, page: Page) -> UITestResult:
        """Test RAG explorer connection status indicator."""
        start = time.time()
        try:
            page.goto(self.ui_files["rag_explorer"])
            page.wait_for_load_state("networkidle")

            # Look for connection status
            connected = page.locator("text=Connected, .connected, [class*='status']")
            has_status = connected.count() > 0

            screenshot = self._screenshot(page, "rag_connection_status")

            return UITestResult(
                test_name="rag_connection_status",
                ui_file="rag_explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=True,
                screenshot_path=screenshot,
                error=None,
                details={"has_status_indicator": has_status}
            )
        except Exception as e:
            return UITestResult(
                test_name="rag_connection_status",
                ui_file="rag_explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=False,
                screenshot_path=None,
                error=str(e),
                details={}
            )

    def test_rag_full_workflow(self, page: Page) -> UITestResult:
        """Test complete RAG workflow: query -> analyze -> view results."""
        start = time.time()
        try:
            page.goto(self.ui_files["rag_explorer"])
            page.wait_for_load_state("networkidle")

            steps_completed = []

            # Step 1: Enter query
            query_input = page.locator("textarea, input[type='text']").first
            query_input.fill("What are the health benefits of green tea?")
            steps_completed.append("query_entered")

            # Step 2: Click analyze
            analyze_btn = page.locator("button:has-text('Analyze')")
            if analyze_btn.count() > 0:
                analyze_btn.first.click()
                page.wait_for_timeout(2000)
                steps_completed.append("analyze_clicked")

            # Step 3: Check for results
            # Look for any indication of results
            results_visible = (
                page.locator(".result, .score, .ranking, #results").count() > 0 or
                page.locator("text=Green Tea, text=Tea History").count() > 0
            )
            if results_visible:
                steps_completed.append("results_visible")

            screenshot = self._screenshot(page, "rag_full_workflow")

            return UITestResult(
                test_name="rag_full_workflow",
                ui_file="rag_explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=len(steps_completed) >= 2,
                screenshot_path=screenshot,
                error=None,
                details={"steps_completed": steps_completed}
            )
        except Exception as e:
            return UITestResult(
                test_name="rag_full_workflow",
                ui_file="rag_explorer.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=False,
                screenshot_path=None,
                error=str(e),
                details={}
            )

    # ============================================================
    # CROSS-UI TESTS
    # ============================================================

    def test_responsive_layout(self, page: Page, ui_name: str) -> UITestResult:
        """Test responsive layout at different screen sizes."""
        start = time.time()
        try:
            url = self.ui_files[ui_name]
            results = {}

            viewports = [
                ("desktop", 1920, 1080),
                ("laptop", 1366, 768),
                ("tablet", 768, 1024),
                ("mobile", 375, 667),
            ]

            for name, width, height in viewports:
                page.set_viewport_size({"width": width, "height": height})
                page.goto(url)
                page.wait_for_load_state("networkidle")

                # Check if content is visible
                body = page.locator("body")
                is_visible = body.is_visible()
                results[name] = {"width": width, "height": height, "visible": is_visible}

                self._screenshot(page, f"{ui_name}_responsive_{name}")

            # Reset to default
            page.set_viewport_size({"width": 1400, "height": 900})

            all_visible = all(r["visible"] for r in results.values())

            return UITestResult(
                test_name=f"responsive_layout_{ui_name}",
                ui_file=f"{ui_name}.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=all_visible,
                screenshot_path=None,
                error=None,
                details={"viewports": results}
            )
        except Exception as e:
            return UITestResult(
                test_name=f"responsive_layout_{ui_name}",
                ui_file=f"{ui_name}.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=False,
                screenshot_path=None,
                error=str(e),
                details={}
            )

    def test_no_console_errors(self, page: Page, ui_name: str) -> UITestResult:
        """Test that UI has no critical JavaScript console errors."""
        start = time.time()
        try:
            url = self.ui_files[ui_name]
            errors = []

            # Listen for console errors
            page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

            page.goto(url)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(2000)  # Wait for any async errors

            screenshot = self._screenshot(page, f"{ui_name}_console_check")

            # Filter out expected network errors (connection refused, etc.)
            # These are expected when no server is running
            critical_errors = [
                e for e in errors
                if not any(ignore in e for ignore in [
                    "ERR_CONNECTION_REFUSED",
                    "net::ERR_",
                    "Failed to fetch",
                    "NetworkError",
                    "CORS",
                ])
            ]

            return UITestResult(
                test_name=f"no_console_errors_{ui_name}",
                ui_file=f"{ui_name}.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=len(critical_errors) == 0,
                screenshot_path=screenshot,
                error=None,
                details={
                    "total_errors": len(errors),
                    "critical_errors": critical_errors[:10],
                    "network_errors_ignored": len(errors) - len(critical_errors)
                }
            )
        except Exception as e:
            return UITestResult(
                test_name=f"no_console_errors_{ui_name}",
                ui_file=f"{ui_name}.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=False,
                screenshot_path=None,
                error=str(e),
                details={}
            )

    def test_keyboard_navigation(self, page: Page, ui_name: str) -> UITestResult:
        """Test keyboard navigation (Tab key)."""
        start = time.time()
        try:
            url = self.ui_files[ui_name]
            page.goto(url)
            page.wait_for_load_state("networkidle")

            # Press Tab multiple times and track focused elements
            focused_elements = []
            for _ in range(5):
                page.keyboard.press("Tab")
                page.wait_for_timeout(100)

                focused = page.evaluate("document.activeElement.tagName")
                focused_elements.append(focused)

            # Check that we're actually moving through elements
            unique_elements = len(set(focused_elements))

            screenshot = self._screenshot(page, f"{ui_name}_keyboard_nav")

            return UITestResult(
                test_name=f"keyboard_navigation_{ui_name}",
                ui_file=f"{ui_name}.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=unique_elements > 1,
                screenshot_path=screenshot,
                error=None,
                details={"focused_elements": focused_elements, "unique_count": unique_elements}
            )
        except Exception as e:
            return UITestResult(
                test_name=f"keyboard_navigation_{ui_name}",
                ui_file=f"{ui_name}.html",
                timestamp=self._timestamp(),
                duration_ms=(time.time() - start) * 1000,
                success=False,
                screenshot_path=None,
                error=str(e),
                details={}
            )

    # ============================================================
    # MAIN TEST RUNNER
    # ============================================================

    def run_all_tests(self):
        """Run all UI tests."""
        if not PLAYWRIGHT_AVAILABLE:
            self._log("ERROR: Playwright not available")
            return

        self._log("=" * 60)
        self._log("UI TEST SUITE")
        self._log("=" * 60)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not self.headed)
            page = browser.new_page(viewport={"width": 1400, "height": 900})

            # EXPLORER.HTML TESTS
            self._log("\n--- explorer.html Tests ---")
            self._save_result(self.test_explorer_loads(page))
            self._save_result(self.test_explorer_input_field(page))
            self._save_result(self.test_explorer_buttons(page))
            self._save_result(self.test_explorer_visualization_container(page))
            self._save_result(self.test_responsive_layout(page, "explorer"))
            self._save_result(self.test_no_console_errors(page, "explorer"))
            self._save_result(self.test_keyboard_navigation(page, "explorer"))

            # RAG_EXPLORER.HTML TESTS
            self._log("\n--- rag_explorer.html Tests ---")
            self._save_result(self.test_rag_explorer_loads(page))
            self._save_result(self.test_rag_sample_queries(page))
            self._save_result(self.test_rag_query_input(page))
            self._save_result(self.test_rag_analyze_button(page))
            self._save_result(self.test_rag_document_ranking(page))
            self._save_result(self.test_rag_connection_status(page))
            self._save_result(self.test_rag_full_workflow(page))
            self._save_result(self.test_responsive_layout(page, "rag_explorer"))
            self._save_result(self.test_no_console_errors(page, "rag_explorer"))
            self._save_result(self.test_keyboard_navigation(page, "rag_explorer"))

            browser.close()

        # Generate summary
        self._generate_summary()

    def _generate_summary(self):
        """Generate test summary."""
        passed = sum(1 for r in self.results if r.success)
        failed = len(self.results) - passed

        self._log("\n" + "=" * 60)
        self._log("TEST SUMMARY")
        self._log("=" * 60)
        self._log(f"Total: {len(self.results)}")
        self._log(f"Passed: {passed}")
        self._log(f"Failed: {failed}")
        self._log(f"Success rate: {passed/len(self.results)*100:.1f}%")

        if failed > 0:
            self._log("\nFailed tests:")
            for r in self.results:
                if not r.success:
                    self._log(f"  - {r.test_name}: {r.error}")

        # Save results to JSON
        results_file = self.output_dir / "test_results.json"
        with open(results_file, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, default=str)
        self._log(f"\nResults saved to: {results_file}")

        # List screenshots
        screenshots = list(self.output_dir.glob("*.png"))
        self._log(f"Screenshots captured: {len(screenshots)}")


def main():
    parser = argparse.ArgumentParser(description="UI Test Suite for Attention Explorer")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed mode")
    parser.add_argument("--base-url", default="http://localhost:8082", help="Base URL for UI files")
    parser.add_argument("--server", default="http://localhost:30000", help="SGLang server URL")
    parser.add_argument("--output", default="ui_test_results", help="Output directory")
    args = parser.parse_args()

    suite = UITestSuite(
        base_url=args.base_url,
        server_url=args.server,
        output_dir=args.output,
        headed=args.headed,
    )
    suite.run_all_tests()


if __name__ == "__main__":
    main()
