"""
End-to-End Playwright Tests for 3D Attention Explorer

Run with: pytest tests/test_explorer_e2e.py -v --headed
"""

import pytest
from playwright.sync_api import Page, expect
import time


BASE_URL = "http://localhost:8090/explorer_3d.html"


@pytest.fixture(scope="module")
def browser_context(playwright):
    """Create a browser context for testing."""
    browser = playwright.chromium.launch(headless=False, slow_mo=100)
    context = browser.new_context(viewport={"width": 1400, "height": 900})
    yield context
    context.close()
    browser.close()


@pytest.fixture
def page(browser_context):
    """Create a new page for each test."""
    page = browser_context.new_page()
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    yield page
    page.close()


class TestPageLoad:
    """Test initial page load and elements."""

    def test_page_loads(self, page: Page):
        """Test that page loads successfully."""
        expect(page).to_have_title("SGLang 3D Attention Tree")

    def test_sidebar_visible(self, page: Page):
        """Test sidebar sections are visible."""
        expect(page.locator("#sidebar")).to_be_visible()
        expect(page.locator("text=Statistics")).to_be_visible()
        expect(page.locator("text=Top Attention Words")).to_be_visible()
        expect(page.locator("text=Generated Text")).to_be_visible()

    def test_controls_visible(self, page: Page):
        """Test control buttons are visible."""
        expect(page.locator("text=Tree")).to_be_visible()
        expect(page.locator("text=Top")).to_be_visible()
        expect(page.locator("text=Side")).to_be_visible()

    def test_timeline_nav_visible(self, page: Page):
        """Test timeline navigation is visible."""
        expect(page.locator("#timeline-nav")).to_be_visible()
        expect(page.locator("#timeline-slider")).to_be_visible()

    def test_websocket_connects(self, page: Page):
        """Test WebSocket connection status."""
        # Wait for connection
        page.wait_for_timeout(2000)
        status = page.locator("#statusText")
        expect(status).to_have_text("Connected")


class TestGeneration:
    """Test text generation flow."""

    def test_generate_button_works(self, page: Page):
        """Test clicking Generate starts generation."""
        # Wait for connection
        page.wait_for_selector("#statusDot.connected", timeout=5000)

        # Enter prompt
        page.fill("#inputText", "What is AI?")

        # Click generate
        page.click("#streamBtn")

        # Wait for tokens to appear
        page.wait_for_timeout(3000)

        # Check token count updated
        token_count = page.locator("#tokenCount").inner_text()
        assert int(token_count) > 0, "Should have generated tokens"

    def test_text_display_shows_words(self, page: Page):
        """Test text display shows proper words."""
        page.wait_for_selector("#statusDot.connected", timeout=5000)
        page.fill("#inputText", "Explain neural networks.")
        page.click("#streamBtn")

        # Wait for generation
        page.wait_for_timeout(5000)

        # Check text display has content
        text_display = page.locator("#text-display")
        expect(text_display).not_to_be_empty()

        # Check for word elements
        words = page.locator(".text-word")
        assert words.count() > 0, "Should have word elements"

    def test_top_attention_words_populated(self, page: Page):
        """Test top attention words section is populated."""
        page.wait_for_selector("#statusDot.connected", timeout=5000)
        page.fill("#inputText", "What is machine learning?")
        page.click("#streamBtn")

        # Wait for generation
        page.wait_for_timeout(5000)

        # Check top attention list
        top_items = page.locator(".top-word-item")
        assert top_items.count() > 0, "Should have top attention words"


class TestNavigation:
    """Test navigation controls."""

    def _generate_first(self, page: Page):
        """Helper to generate content first."""
        page.wait_for_selector("#statusDot.connected", timeout=5000)
        page.fill("#inputText", "What is deep learning?")
        page.click("#streamBtn")
        page.wait_for_timeout(5000)

    def test_next_button_navigates(self, page: Page):
        """Test next button navigates to next token."""
        self._generate_first(page)

        # Get initial position
        initial_pos = page.locator("#timelinePosition").inner_text()

        # Click next
        page.click("button:has-text('▶')")
        page.wait_for_timeout(500)

        # Position should change
        new_pos = page.locator("#timelinePosition").inner_text()
        assert new_pos != initial_pos, "Position should change after clicking next"

    def test_prev_button_navigates(self, page: Page):
        """Test prev button navigates to previous token."""
        self._generate_first(page)

        # Go to end first
        page.click("button:has-text('⏭')")
        page.wait_for_timeout(500)

        end_pos = page.locator("#timelinePosition").inner_text()

        # Click prev
        page.click("button:has-text('◀')")
        page.wait_for_timeout(500)

        new_pos = page.locator("#timelinePosition").inner_text()
        assert new_pos != end_pos, "Position should change after clicking prev"

    def test_slider_navigates(self, page: Page):
        """Test slider navigates through tokens."""
        self._generate_first(page)

        slider = page.locator("#timeline-slider")

        # Move slider to middle
        slider.evaluate("el => { el.value = Math.floor(el.max / 2); el.dispatchEvent(new Event('input')); }")
        page.wait_for_timeout(500)

        # Check position updated
        pos_text = page.locator("#timelinePosition").inner_text()
        assert "/ " in pos_text, "Position should show format 'X / Y'"

    def test_start_end_buttons(self, page: Page):
        """Test start/end navigation buttons."""
        self._generate_first(page)

        # Go to end
        page.click("button:has-text('⏭')")
        page.wait_for_timeout(500)

        end_pos = page.locator("#timelinePosition").inner_text()
        total = end_pos.split(" / ")[1]
        assert end_pos.startswith(total), "Should be at end position"

        # Go to start
        page.click("button:has-text('⏮')")
        page.wait_for_timeout(500)

        start_pos = page.locator("#timelinePosition").inner_text()
        assert start_pos.startswith("1 "), "Should be at position 1"

    def test_arrow_key_navigation(self, page: Page):
        """Test arrow keys navigate through tokens."""
        self._generate_first(page)

        # Focus on canvas area
        page.locator("#canvas-container").click()

        initial_pos = page.locator("#timelinePosition").inner_text()

        # Press right arrow
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(500)

        new_pos = page.locator("#timelinePosition").inner_text()
        assert new_pos != initial_pos, "Arrow right should navigate"


class TestWordInteraction:
    """Test word clicking and highlighting."""

    def _generate_first(self, page: Page):
        """Helper to generate content first."""
        page.wait_for_selector("#statusDot.connected", timeout=5000)
        page.fill("#inputText", "Explain transformers briefly.")
        page.click("#streamBtn")
        page.wait_for_timeout(5000)

    def test_click_word_highlights(self, page: Page):
        """Test clicking a word highlights it."""
        self._generate_first(page)

        # Click first word
        first_word = page.locator(".text-word").first
        first_word.click()
        page.wait_for_timeout(500)

        # Check it has viewing class
        expect(first_word).to_have_class(/viewing/)

    def test_click_word_updates_sidebar(self, page: Page):
        """Test clicking a word updates selected token sidebar."""
        self._generate_first(page)

        # Click a word
        page.locator(".text-word").first.click()
        page.wait_for_timeout(500)

        # Sidebar should show selection
        expect(page.locator("#tokenDetailSection")).to_be_visible()

    def test_click_top_attention_word(self, page: Page):
        """Test clicking top attention word navigates."""
        self._generate_first(page)

        # Click first top word
        top_word = page.locator(".top-word-item").first
        if top_word.count() > 0:
            top_word.click()
            page.wait_for_timeout(500)

            # Should update current token display
            expect(page.locator("#currentTokenText")).not_to_have_text("No token selected")


class TestViewControls:
    """Test view switching controls."""

    def _generate_first(self, page: Page):
        """Helper to generate content first."""
        page.wait_for_selector("#statusDot.connected", timeout=5000)
        page.fill("#inputText", "What is AI?")
        page.click("#streamBtn")
        page.wait_for_timeout(4000)

    def test_view_buttons_work(self, page: Page):
        """Test view preset buttons."""
        self._generate_first(page)

        # Test each view button
        for view in ["Tree", "Top", "Side", "Front"]:
            btn = page.locator(f"button:has-text('{view}')")
            btn.click()
            page.wait_for_timeout(300)
            # Button should become active
            expect(btn).to_have_class(/active/)

    def test_keyboard_view_shortcuts(self, page: Page):
        """Test keyboard shortcuts for views."""
        self._generate_first(page)

        # Focus canvas
        page.locator("#canvas-container").click()

        # Test key 1-4
        for key in ["1", "2", "3", "4"]:
            page.keyboard.press(key)
            page.wait_for_timeout(300)

    def test_toggle_edges(self, page: Page):
        """Test edge toggle button."""
        self._generate_first(page)

        # Click edges button
        page.click("button:has-text('Edges')")
        page.wait_for_timeout(300)

        # Click again to toggle back
        page.click("button:has-text('Edges')")
        page.wait_for_timeout(300)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_clear_and_regenerate(self, page: Page):
        """Test clearing and regenerating."""
        page.wait_for_selector("#statusDot.connected", timeout=5000)

        # Generate first
        page.fill("#inputText", "Hello")
        page.click("#streamBtn")
        page.wait_for_timeout(3000)

        # Clear
        page.click("button:has-text('Clear')")
        page.wait_for_timeout(500)

        # Token count should be 0
        expect(page.locator("#tokenCount")).to_have_text("0")

        # Generate again
        page.fill("#inputText", "World")
        page.click("#streamBtn")
        page.wait_for_timeout(3000)

        # Should have tokens again
        token_count = page.locator("#tokenCount").inner_text()
        assert int(token_count) > 0

    def test_empty_prompt_handling(self, page: Page):
        """Test handling of empty prompt."""
        page.wait_for_selector("#statusDot.connected", timeout=5000)

        # Clear input
        page.fill("#inputText", "")

        # Click generate - should not crash
        page.click("#streamBtn")
        page.wait_for_timeout(1000)

    def test_long_prompt(self, page: Page):
        """Test handling of long prompt."""
        page.wait_for_selector("#statusDot.connected", timeout=5000)

        long_prompt = "Explain in detail: " + "AI and machine learning " * 10
        page.fill("#inputText", long_prompt)
        page.click("#streamBtn")
        page.wait_for_timeout(6000)

        # Should still work
        token_count = page.locator("#tokenCount").inner_text()
        assert int(token_count) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--headed"])
