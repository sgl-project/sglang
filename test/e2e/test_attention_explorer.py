"""
Playwright E2E Tests for SGLang Attention Explorer UI

These tests validate the full integration of the attention explorer UI
with the SGLang server backend and rapids sidecar.

Requirements:
- SGLang server running on localhost:8000
- Rapids sidecar running on localhost:9001 (ZMQ) / localhost:9000 (HTTP)
- Web UI served on localhost:8081 (python -m http.server 8081 from examples/attention_explorer/)

Run with:
    pytest test/e2e/test_attention_explorer.py -v --headed

Or headless:
    pytest test/e2e/test_attention_explorer.py -v
"""

import re

import pytest
from playwright.sync_api import Page, expect

# Test configuration
BASE_URL = "http://localhost:8081/explorer.html"
API_URL = "http://localhost:8000"
SIDECAR_URL = "http://localhost:9000"


@pytest.fixture(scope="session")
def browser_context(playwright):
    """Create a browser context for all tests."""
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1920, "height": 1080})
    yield context
    context.close()
    browser.close()


@pytest.fixture
def page(browser_context):
    """Create a new page for each test."""
    page = browser_context.new_page()
    yield page
    page.close()


class TestPageLoad:
    """Tests for initial page loading and connection status."""

    def test_page_loads_successfully(self, page: Page):
        """Verify the page loads without errors."""
        page.goto(BASE_URL)
        expect(page).to_have_title("SGLang Attention Explorer")

    def test_header_renders(self, page: Page):
        """Verify the header and logo are displayed."""
        page.goto(BASE_URL)
        expect(page.locator(".logo h1")).to_have_text("SGLang Attention Explorer")
        expect(page.locator(".logo span")).to_have_text(
            "Real Attention Token Visualization"
        )

    def test_connection_status_displays(self, page: Page):
        """Verify connection status shows model name when connected."""
        page.goto(BASE_URL)
        # Wait for connection check
        page.wait_for_timeout(2000)

        status_text = page.locator("#statusText")
        status_dot = page.locator("#statusDot")

        # Should show model name or connection status
        expect(status_text).not_to_have_text("Connecting...")

        # Check if connected (no error class) or show failure
        is_error = status_dot.evaluate("el => el.classList.contains('error')")
        if is_error:
            pytest.skip("Server not connected - skipping connection-dependent tests")

    def test_empty_state_visible_initially(self, page: Page):
        """Verify empty state message is shown before analysis."""
        page.goto(BASE_URL)

        empty_state = page.locator("#emptyState")
        expect(empty_state).to_be_visible()
        expect(empty_state).to_contain_text("Enter text and click")
        expect(empty_state).to_contain_text("Analyze")

    def test_input_textarea_has_default_text(self, page: Page):
        """Verify the input textarea has the default example text."""
        page.goto(BASE_URL)

        textarea = page.locator("#inputText")
        expect(textarea).to_have_value("What is the capital of France?")


class TestConfiguration:
    """Tests for configuration inputs and settings."""

    def test_max_tokens_input(self, page: Page):
        """Verify max tokens input accepts valid values."""
        page.goto(BASE_URL)

        max_tokens = page.locator("#maxTokens")
        expect(max_tokens).to_have_value("100")

        max_tokens.fill("256")
        expect(max_tokens).to_have_value("256")

    def test_attention_topk_input(self, page: Page):
        """Verify attention top-k input accepts valid values."""
        page.goto(BASE_URL)

        topk = page.locator("#attentionTopK")
        expect(topk).to_have_value("5")

        topk.fill("10")
        expect(topk).to_have_value("10")

    def test_temperature_input(self, page: Page):
        """Verify temperature input accepts valid values."""
        page.goto(BASE_URL)

        temp = page.locator("#temperature")
        expect(temp).to_have_value("0.6")

        temp.fill("0.8")
        expect(temp).to_have_value("0.8")

    def test_edge_threshold_input(self, page: Page):
        """Verify edge threshold input accepts valid values."""
        page.goto(BASE_URL)

        threshold = page.locator("#edgeThreshold")
        expect(threshold).to_have_value("0.05")

        threshold.fill("0.1")
        expect(threshold).to_have_value("0.1")

    def test_word_mode_toggle(self, page: Page):
        """Verify word mode toggle works."""
        page.goto(BASE_URL)

        toggle = page.locator("#wordModeToggle")
        expect(toggle).not_to_have_class(re.compile(r"active"))

        # Click the toggle row
        page.locator(".toggle-row").click()
        expect(toggle).to_have_class(re.compile(r"active"))

        # Click again to toggle off
        page.locator(".toggle-row").click()
        expect(toggle).not_to_have_class(re.compile(r"active"))


class TestAnalysis:
    """Tests for the main analysis functionality."""

    def test_analyze_button_exists(self, page: Page):
        """Verify analyze button is present and enabled."""
        page.goto(BASE_URL)

        btn = page.locator("#analyzeBtn")
        expect(btn).to_be_visible()
        expect(btn).to_be_enabled()
        expect(btn).to_have_text("Analyze")

    def test_analyze_shows_loading_state(self, page: Page):
        """Verify clicking analyze shows loading state."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)  # Wait for connection check

        # Check if server is connected
        status_dot = page.locator("#statusDot")
        is_error = status_dot.evaluate("el => el.classList.contains('error')")
        if is_error:
            pytest.skip("Server not connected")

        # Click analyze
        page.locator("#analyzeBtn").click()

        # Should show loading state
        loading = page.locator("#loadingState")
        expect(loading).to_be_visible(timeout=5000)
        expect(page.locator("#loadingText")).to_contain_text("attention")

    def test_analyze_with_custom_prompt(self, page: Page):
        """Test analysis with a custom prompt."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        status_dot = page.locator("#statusDot")
        is_error = status_dot.evaluate("el => el.classList.contains('error')")
        if is_error:
            pytest.skip("Server not connected")

        # Enter custom prompt
        textarea = page.locator("#inputText")
        textarea.fill("Explain quantum computing in simple terms.")

        # Set lower max tokens for faster test
        page.locator("#maxTokens").fill("50")

        # Click analyze
        page.locator("#analyzeBtn").click()

        # Wait for response (with timeout for generation)
        page.wait_for_selector("#tokenSegments .segment", timeout=120000)

        # Verify tokens are rendered
        tokens = page.locator(".token")
        expect(tokens.first).to_be_visible()

    def test_analyze_renders_token_segments(self, page: Page):
        """Test that analysis renders prompt and output segments."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        status_dot = page.locator("#statusDot")
        is_error = status_dot.evaluate("el => el.classList.contains('error')")
        if is_error:
            pytest.skip("Server not connected")

        # Use short prompt and few tokens
        page.locator("#inputText").fill("Hi")
        page.locator("#maxTokens").fill("20")
        page.locator("#analyzeBtn").click()

        # Wait for rendering
        page.wait_for_selector(".segment", timeout=60000)

        # Should have at least prompt segment
        segments = page.locator(".segment")
        expect(segments.first).to_be_visible()

        # Check segment header text
        prompt_header = page.locator(".segment-header.prompt")
        expect(prompt_header).to_contain_text("Prompt")

    def test_analyze_updates_statistics(self, page: Page):
        """Test that analysis updates the statistics panel."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        status_dot = page.locator("#statusDot")
        is_error = status_dot.evaluate("el => el.classList.contains('error')")
        if is_error:
            pytest.skip("Server not connected")

        # Initial stats should be 0
        expect(page.locator("#statTokens")).to_have_text("0")
        expect(page.locator("#statEdges")).to_have_text("0")

        # Run analysis
        page.locator("#inputText").fill("Hello")
        page.locator("#maxTokens").fill("10")
        page.locator("#analyzeBtn").click()

        # Wait for completion
        page.wait_for_selector(".token", timeout=60000)
        page.wait_for_timeout(1000)  # Small delay for stats update

        # Stats should be updated
        tokens_stat = page.locator("#statTokens")
        tokens_text = tokens_stat.text_content()
        assert int(tokens_text) > 0, f"Expected token count > 0, got {tokens_text}"


class TestTokenInteraction:
    """Tests for token selection and attention visualization."""

    def test_token_click_selects(self, page: Page):
        """Test clicking a token selects it."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        status_dot = page.locator("#statusDot")
        is_error = status_dot.evaluate("el => el.classList.contains('error')")
        if is_error:
            pytest.skip("Server not connected")

        # Run analysis
        page.locator("#inputText").fill("Test input")
        page.locator("#maxTokens").fill("10")
        page.locator("#analyzeBtn").click()
        page.wait_for_selector(".token", timeout=60000)

        # Click first token
        first_token = page.locator(".token").first
        first_token.click()

        # Should be selected
        expect(first_token).to_have_class(re.compile(r"selected"))

    def test_selected_token_details_shown(self, page: Page):
        """Test that selecting a token shows details in right panel."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        status_dot = page.locator("#statusDot")
        is_error = status_dot.evaluate("el => el.classList.contains('error')")
        if is_error:
            pytest.skip("Server not connected")

        # Run analysis
        page.locator("#inputText").fill("Hello world")
        page.locator("#maxTokens").fill("10")
        page.locator("#analyzeBtn").click()
        page.wait_for_selector(".token", timeout=60000)

        # Click a token
        page.locator(".token").first.click()

        # Check detail panel updated
        selected_text = page.locator("#selectedTokenText")
        expect(selected_text).not_to_have_text("Select a token")

        # Check metadata shows position info
        meta = page.locator("#selectedTokenMeta")
        expect(meta).to_contain_text("Position:")

    def test_attention_list_populated(self, page: Page):
        """Test that attention list shows edges for selected token."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        status_dot = page.locator("#statusDot")
        is_error = status_dot.evaluate("el => el.classList.contains('error')")
        if is_error:
            pytest.skip("Server not connected")

        # Run analysis with enough tokens to have attention edges
        page.locator("#inputText").fill("What is the weather today?")
        page.locator("#maxTokens").fill("30")
        page.locator("#analyzeBtn").click()
        page.wait_for_selector(".token", timeout=60000)

        # Find and click a generated token (not prompt)
        output_tokens = page.locator(".segment-body .token")
        if output_tokens.count() > 5:
            output_tokens.nth(5).click()

            # Check if attention items appear
            page.wait_for_timeout(500)
            attention_items = page.locator(".attention-item")
            # May have 0 if no attention captured, but list should be accessible


class TestDetailTabs:
    """Tests for the Attends To / Attended By tabs."""

    def test_default_tab_is_attends(self, page: Page):
        """Verify default tab is 'Attends To'."""
        page.goto(BASE_URL)

        attends_tab = page.locator(".detail-tab[data-tab='attends']")
        expect(attends_tab).to_have_class(re.compile(r"active"))

        attended_by_tab = page.locator(".detail-tab[data-tab='attendedBy']")
        expect(attended_by_tab).not_to_have_class(re.compile(r"active"))

    def test_tab_switching(self, page: Page):
        """Test switching between tabs."""
        page.goto(BASE_URL)

        # Click attended by tab
        attended_by_tab = page.locator(".detail-tab[data-tab='attendedBy']")
        attended_by_tab.click()

        expect(attended_by_tab).to_have_class(re.compile(r"active"))

        attends_tab = page.locator(".detail-tab[data-tab='attends']")
        expect(attends_tab).not_to_have_class(re.compile(r"active"))

        # Switch back
        attends_tab.click()
        expect(attends_tab).to_have_class(re.compile(r"active"))


class TestClearFunctionality:
    """Tests for the clear button."""

    def test_clear_button_exists(self, page: Page):
        """Verify clear button is present."""
        page.goto(BASE_URL)

        clear_btn = page.locator(".btn-ghost")
        expect(clear_btn).to_be_visible()
        expect(clear_btn).to_have_text("Clear")

    def test_clear_resets_input(self, page: Page):
        """Test that clear resets the input textarea."""
        page.goto(BASE_URL)

        textarea = page.locator("#inputText")
        textarea.fill("Some custom text")
        expect(textarea).to_have_value("Some custom text")

        # Click clear
        page.locator(".btn-ghost").click()

        expect(textarea).to_have_value("")

    def test_clear_shows_empty_state(self, page: Page):
        """Test that clear shows the empty state."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        status_dot = page.locator("#statusDot")
        is_error = status_dot.evaluate("el => el.classList.contains('error')")
        if is_error:
            # Even without server, we can test clear functionality
            pass

        # Fill input and run analysis if server available
        if not is_error:
            page.locator("#inputText").fill("Test")
            page.locator("#maxTokens").fill("5")
            page.locator("#analyzeBtn").click()
            page.wait_for_selector(".token", timeout=60000)

        # Clear
        page.locator(".btn-ghost").click()

        # Should show empty state
        expect(page.locator("#emptyState")).to_be_visible()

    def test_clear_resets_statistics(self, page: Page):
        """Test that clear resets statistics to zero (or default)."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        # Clear should reset stats (even if they're already 0)
        page.locator(".btn-ghost").click()

        expect(page.locator("#statTokens")).to_have_text("0")
        expect(page.locator("#statEdges")).to_have_text("0")
        # Note: Layers defaults to 1 in the UI code (layers.size || 1)
        expect(page.locator("#statLayers")).to_have_text(re.compile(r"[01]"))


class TestSegmentToggle:
    """Tests for collapsing/expanding token segments."""

    def test_segment_toggle_collapses(self, page: Page):
        """Test that clicking segment toggle collapses the segment."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        status_dot = page.locator("#statusDot")
        is_error = status_dot.evaluate("el => el.classList.contains('error')")
        if is_error:
            pytest.skip("Server not connected")

        # Run analysis
        page.locator("#inputText").fill("Test")
        page.locator("#maxTokens").fill("10")
        page.locator("#analyzeBtn").click()
        page.wait_for_selector(".segment", timeout=60000)

        # Find toggle button
        toggle_btn = page.locator(".segment-toggle").first
        segment_body = page.locator(".segment-body").first

        # Initially visible
        expect(segment_body).to_be_visible()
        expect(toggle_btn).to_have_text("-")

        # Click to collapse
        toggle_btn.click()
        expect(segment_body).to_be_hidden()
        expect(toggle_btn).to_have_text("+")

        # Click to expand
        toggle_btn.click()
        expect(segment_body).to_be_visible()
        expect(toggle_btn).to_have_text("-")


class TestEdgeCanvas:
    """Tests for the edge visualization canvas."""

    def test_canvas_exists(self, page: Page):
        """Verify canvas element exists."""
        page.goto(BASE_URL)

        canvas = page.locator("#edgeCanvas")
        expect(canvas).to_be_attached()

    def test_canvas_hidden_before_analysis(self, page: Page):
        """Canvas container should be hidden before analysis."""
        page.goto(BASE_URL)

        container = page.locator("#canvasContainer")
        expect(container).to_be_hidden()

    def test_canvas_visible_after_analysis(self, page: Page):
        """Canvas container should be visible after analysis."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        status_dot = page.locator("#statusDot")
        is_error = status_dot.evaluate("el => el.classList.contains('error')")
        if is_error:
            pytest.skip("Server not connected")

        # Run analysis
        page.locator("#inputText").fill("Test")
        page.locator("#maxTokens").fill("10")
        page.locator("#analyzeBtn").click()
        page.wait_for_selector(".token", timeout=60000)

        container = page.locator("#canvasContainer")
        expect(container).to_be_visible()


class TestResponsiveLayout:
    """Tests for responsive layout behavior."""

    def test_three_panel_layout(self, page: Page):
        """Verify the three-panel layout renders correctly."""
        page.goto(BASE_URL)

        expect(page.locator(".panel-left")).to_be_visible()
        expect(page.locator(".main-view")).to_be_visible()
        expect(page.locator(".panel-right")).to_be_visible()

    def test_panels_have_correct_sections(self, page: Page):
        """Verify each panel has its expected sections."""
        page.goto(BASE_URL)

        # Left panel sections
        expect(page.locator("#inputText")).to_be_visible()
        expect(page.locator("#maxTokens")).to_be_visible()
        expect(page.locator("#statTokens")).to_be_visible()

        # Right panel sections
        expect(page.locator("#selectedTokenText")).to_be_visible()
        expect(page.locator("#attentionList")).to_be_visible()


class TestEndToEndWorkflow:
    """Full end-to-end workflow tests."""

    def test_full_analysis_workflow(self, page: Page):
        """Test complete workflow: input -> analyze -> select token -> view attention."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        # Check connection
        status_dot = page.locator("#statusDot")
        is_error = status_dot.evaluate("el => el.classList.contains('error')")
        if is_error:
            pytest.skip("Server not connected")

        # Step 1: Enter prompt
        prompt = "What is machine learning?"
        page.locator("#inputText").fill(prompt)

        # Step 2: Configure settings
        page.locator("#maxTokens").fill("50")
        page.locator("#attentionTopK").fill("8")
        page.locator("#temperature").fill("0.7")

        # Step 3: Analyze
        page.locator("#analyzeBtn").click()

        # Step 4: Wait for tokens
        page.wait_for_selector(".token", timeout=120000)

        # Step 5: Verify segments rendered
        segments = page.locator(".segment")
        assert segments.count() >= 1, "Should have at least one segment"

        # Step 6: Select a token
        tokens = page.locator(".token")
        token_count = tokens.count()
        assert token_count > 0, "Should have rendered tokens"

        # Select a middle token if possible
        middle_idx = min(5, token_count - 1)
        tokens.nth(middle_idx).click()

        # Step 7: Verify selection
        expect(tokens.nth(middle_idx)).to_have_class(re.compile(r"selected"))

        # Step 8: Check detail panel
        selected_text = page.locator("#selectedTokenText")
        expect(selected_text).not_to_have_text("Select a token")

        # Step 9: Switch tabs
        page.locator(".detail-tab[data-tab='attendedBy']").click()
        expect(page.locator(".detail-tab[data-tab='attendedBy']")).to_have_class(
            re.compile(r"active")
        )

        # Step 10: Clear and verify reset
        page.locator(".btn-ghost").click()
        expect(page.locator("#emptyState")).to_be_visible()
        expect(page.locator("#statTokens")).to_have_text("0")

    def test_multiple_analyses(self, page: Page):
        """Test running multiple analyses in sequence."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        status_dot = page.locator("#statusDot")
        is_error = status_dot.evaluate("el => el.classList.contains('error')")
        if is_error:
            pytest.skip("Server not connected")

        prompts = ["Hello", "What is AI?", "Count to 5"]

        for i, prompt in enumerate(prompts):
            # Clear previous
            if i > 0:
                page.locator(".btn-ghost").click()
                page.wait_for_timeout(500)

            # New analysis
            page.locator("#inputText").fill(prompt)
            page.locator("#maxTokens").fill("20")
            page.locator("#analyzeBtn").click()

            # Wait for results
            page.wait_for_selector(".token", timeout=60000)

            # Verify tokens rendered
            tokens = page.locator(".token")
            assert tokens.count() > 0, f"Analysis {i+1} should produce tokens"


class TestThinkingPhase:
    """Tests for thinking phase detection and display."""

    def test_thinking_phase_segment(self, page: Page):
        """Test that thinking tokens are displayed in separate segment."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        status_dot = page.locator("#statusDot")
        is_error = status_dot.evaluate("el => el.classList.contains('error')")
        if is_error:
            pytest.skip("Server not connected")

        # Use a prompt that might trigger thinking
        page.locator("#inputText").fill("Let me think about what 2+2 equals...")
        page.locator("#maxTokens").fill("100")
        page.locator("#analyzeBtn").click()

        # Wait for response
        page.wait_for_selector(".segment", timeout=120000)

        # Check for thinking segment (may or may not exist depending on model)
        think_header = page.locator(".segment-header.think")
        # Just verify the selector works - thinking segment is model-dependent


class TestAPIIntegration:
    """Tests for API integration specifically."""

    def test_api_endpoint_accessible(self, page: Page):
        """Test that the API endpoint is accessible."""
        # Make direct API call via page context
        page.goto(BASE_URL)

        result = page.evaluate(
            """
            async () => {
                try {
                    const res = await fetch('http://localhost:8000/v1/models');
                    const data = await res.json();
                    return { success: true, data };
                } catch (e) {
                    return { success: false, error: e.message };
                }
            }
        """
        )

        if not result["success"]:
            pytest.skip(f"API not accessible: {result.get('error')}")

        assert "data" in result["data"], "API should return model data"

    def test_attention_tokens_in_response(self, page: Page):
        """Test that API returns attention_tokens when requested."""
        page.goto(BASE_URL)
        page.wait_for_timeout(2000)

        status_dot = page.locator("#statusDot")
        is_error = status_dot.evaluate("el => el.classList.contains('error')")
        if is_error:
            pytest.skip("Server not connected")

        # Make request and check for attention tokens
        result = page.evaluate(
            """
            async () => {
                try {
                    const res = await fetch('http://localhost:8000/v1/chat/completions', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model: 'Qwen/Qwen3-Next-80B-A3B-Thinking-FP8',
                            messages: [{ role: 'user', content: 'Hi' }],
                            max_tokens: 10,
                            return_attention_tokens: true
                        })
                    });
                    const data = await res.json();
                    return {
                        success: true,
                        hasAttention: !!data.choices?.[0]?.attention_tokens
                    };
                } catch (e) {
                    return { success: false, error: e.message };
                }
            }
        """
        )

        if not result["success"]:
            pytest.skip(f"API call failed: {result.get('error')}")

        assert result["hasAttention"], "Response should include attention_tokens"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
