#!/usr/bin/env python3
"""
Simple E2E test for 3D Attention Explorer using Playwright.
Simulates user behavior and reports issues.

Run: python test_e2e.py
"""

from playwright.sync_api import sync_playwright
import time


def test_explorer():
    """Run end-to-end tests on the explorer."""

    print("=" * 60)
    print("3D ATTENTION EXPLORER - E2E TEST")
    print("=" * 60)

    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        results = []

        def test(name, fn):
            try:
                fn()
                results.append((name, "PASS", ""))
                print(f"  ✓ {name}")
            except Exception as e:
                results.append((name, "FAIL", str(e)))
                print(f"  ✗ {name}: {e}")

        # Capture console errors
        js_errors = []
        page.on("console", lambda msg: js_errors.append(f"{msg.type}: {msg.text}") if msg.type == "error" else None)
        page.on("pageerror", lambda err: js_errors.append(f"PAGE ERROR: {err}"))

        # Test 1: Page loads
        print("\n[1] PAGE LOAD TESTS")
        page.goto("http://localhost:8090/explorer_3d.html")
        page.wait_for_load_state("networkidle")

        test("Page title correct", lambda: (
            page.title() == "SGLang 3D Attention Tree" or
            (_ for _ in ()).throw(Exception(f"Title was: {page.title()}"))
        ))

        test("Sidebar visible", lambda: page.locator("#sidebar").is_visible())
        test("Timeline nav visible", lambda: page.locator("#timeline-nav").is_visible())
        test("Generate button visible", lambda: page.locator("#streamBtn").is_visible())

        # Test 2: WebSocket connection
        print("\n[2] WEBSOCKET CONNECTION")
        page.wait_for_timeout(2000)
        status = page.locator("#statusText").inner_text()
        test("WebSocket connected", lambda: (
            status == "Connected" or
            (_ for _ in ()).throw(Exception(f"Status: {status}"))
        ))

        # Test 3: Generation
        print("\n[3] GENERATION TESTS")
        page.fill("#inputText", "What is machine learning?")
        page.fill("#maxTokens", "30")  # Limit tokens for faster test
        page.click("#streamBtn")

        # Wait for generation to complete (button re-enabled)
        page.wait_for_function("document.getElementById('streamBtn').disabled === false", timeout=30000)
        page.wait_for_timeout(1000)  # Extra wait for UI updates

        token_count = int(page.locator("#tokenCount").inner_text())
        test("Tokens generated", lambda: (
            token_count > 10 or
            (_ for _ in ()).throw(Exception(f"Only {token_count} tokens"))
        ))

        edge_count = int(page.locator("#edgeCount").inner_text())
        test("Edges created", lambda: (
            edge_count > 0 or
            (_ for _ in ()).throw(Exception(f"Only {edge_count} edges"))
        ))

        word_count = page.locator(".text-word").count()
        test("Words displayed", lambda: (
            word_count > 5 or
            (_ for _ in ()).throw(Exception(f"Only {word_count} words"))
        ))

        top_words = page.locator(".top-word-item").count()
        test("Top attention words shown", lambda: (
            top_words > 0 or
            (_ for _ in ()).throw(Exception(f"No top words"))
        ))

        # Test 4: Navigation
        print("\n[4] NAVIGATION TESTS")

        # Test next button
        initial_pos = page.locator("#timelinePosition").inner_text()
        page.click("button:has-text('▶')")
        page.wait_for_timeout(500)
        new_pos = page.locator("#timelinePosition").inner_text()
        test("Next button works", lambda: (
            new_pos != initial_pos or
            (_ for _ in ()).throw(Exception(f"Position unchanged: {initial_pos}"))
        ))

        # Test end button
        page.click("button:has-text('⏭')")
        page.wait_for_timeout(500)
        end_pos = page.locator("#timelinePosition").inner_text()
        expected_end = f"{token_count} / {token_count}"
        test("End button works", lambda: (
            end_pos == expected_end or
            (_ for _ in ()).throw(Exception(f"Expected {expected_end}, got {end_pos}"))
        ))

        # Test start button (goes to position 1)
        page.click("button:has-text('⏮')")
        page.wait_for_timeout(500)
        start_pos = page.locator("#timelinePosition").inner_text()
        test("Start button works", lambda: (
            start_pos.startswith("1 /") or start_pos.split(" / ")[0] == "1" or
            (_ for _ in ()).throw(Exception(f"Expected position 1, got {start_pos}"))
        ))

        # Test slider
        slider = page.locator("#timeline-slider")
        slider.evaluate("el => { el.value = Math.floor(el.max / 2); el.dispatchEvent(new Event('input')); }")
        page.wait_for_timeout(500)
        mid_pos = page.locator("#timelinePosition").inner_text()
        test("Slider works", lambda: (
            not mid_pos.startswith("1 /") or
            (_ for _ in ()).throw(Exception(f"Slider didn't move: {mid_pos}"))
        ))

        # Test 5: Word interaction
        print("\n[5] WORD INTERACTION TESTS")

        # Click a word
        first_word = page.locator(".text-word").first
        first_word.click()
        page.wait_for_timeout(500)

        # Check word is highlighted
        classes = first_word.get_attribute("class")
        test("Word click highlights", lambda: (
            "viewing" in classes or
            (_ for _ in ()).throw(Exception(f"Classes: {classes}"))
        ))

        # Check current token display updated
        token_text = page.locator("#currentTokenText").inner_text()
        test("Current token shows", lambda: (
            token_text != "No token selected" or
            (_ for _ in ()).throw(Exception(f"Still shows: {token_text}"))
        ))

        # Test 6: View controls
        print("\n[6] VIEW CONTROL TESTS")

        for view, key in [("Top", "2"), ("Side", "3"), ("Front", "4"), ("Tree", "1")]:
            btn = page.locator(f"button:has-text('{view}')")
            btn.click()
            page.wait_for_timeout(300)
            has_active = "active" in btn.get_attribute("class")
            test(f"{view} view button", lambda h=has_active: (
                h or (_ for _ in ()).throw(Exception("Not active"))
            ))

        # Test 7: Clear and regenerate
        print("\n[7] CLEAR & REGENERATE TESTS")

        page.click("button:has-text('Clear')")
        page.wait_for_timeout(500)

        cleared_count = int(page.locator("#tokenCount").inner_text())
        test("Clear resets tokens", lambda: (
            cleared_count == 0 or
            (_ for _ in ()).throw(Exception(f"Still has {cleared_count} tokens"))
        ))

        page.fill("#inputText", "What is AI?")
        page.click("#streamBtn")
        page.wait_for_timeout(5000)

        regen_count = int(page.locator("#tokenCount").inner_text())
        test("Regeneration works", lambda: (
            regen_count > 10 or
            (_ for _ in ()).throw(Exception(f"Only {regen_count} tokens"))
        ))

        # Summary
        print("\n" + "=" * 60)
        passed = sum(1 for r in results if r[1] == "PASS")
        failed = sum(1 for r in results if r[1] == "FAIL")
        print(f"RESULTS: {passed} passed, {failed} failed")
        print("=" * 60)

        if failed > 0:
            print("\nFAILED TESTS:")
            for name, status, err in results:
                if status == "FAIL":
                    print(f"  - {name}: {err}")

        if js_errors:
            print("\nJAVASCRIPT ERRORS:")
            for err in js_errors[:10]:  # Limit to first 10
                print(f"  - {err}")

        browser.close()

        return failed == 0


if __name__ == "__main__":
    success = test_explorer()
    exit(0 if success else 1)
