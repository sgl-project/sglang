#!/usr/bin/env python3
"""
Take comprehensive screenshots of the Attention Explorer UI for documentation.
"""

import time
from pathlib import Path

from playwright.sync_api import sync_playwright, Page


def setup_output_dir():
    """Create output directory for screenshots."""
    output_dir = Path("/media/thread/pyth/agentic/attentio/sglang/results/ui_screenshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def take_screenshot(page: Page, name: str, output_dir: Path):
    """Take a screenshot with a descriptive name."""
    filepath = output_dir / f"{name}.png"
    page.screenshot(path=str(filepath), full_page=False)
    print(f"Saved: {filepath}")
    return filepath


def run_analysis(page: Page, prompt: str, max_tokens: int = 100):
    """Run analysis and wait for results."""
    # Set the prompt
    textarea = page.locator("#inputText")
    textarea.fill(prompt)

    # Set max tokens
    max_tokens_input = page.locator("#maxTokens")
    if max_tokens_input.is_visible():
        max_tokens_input.fill(str(max_tokens))

    # Click analyze button
    analyze_btn = page.locator("#analyzeBtn")
    analyze_btn.click()

    # Wait for tokens to appear
    time.sleep(15)
    return True


def main():
    output_dir = setup_output_dir()
    base_url = "http://localhost:8082/explorer.html"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )
        page = context.new_page()

        # Capture console messages
        console_messages = []
        page.on("console", lambda msg: console_messages.append(f"[{msg.type}] {msg.text}"))

        print("Loading UI...")
        page.goto(base_url)
        time.sleep(2)

        # Screenshot 1: Initial empty state
        take_screenshot(page, "01_empty_state", output_dir)

        # Run analysis 1: Simple math question
        print("\n=== Analysis 1: Math Question ===")
        run_analysis(page, "What is 2 + 2?", max_tokens=50)
        take_screenshot(page, "02_math_overview", output_dir)

        # Click on a token to show attention details
        tokens = page.locator("#tokenGrid .token")
        if tokens.count() > 20:
            tokens.nth(20).click()
            time.sleep(1)
            take_screenshot(page, "03_math_token_selected", output_dir)

            # Click "Attended By" tab
            attended_by = page.locator("text=ATTENDED BY")
            if attended_by.is_visible():
                attended_by.click()
                time.sleep(0.5)
                take_screenshot(page, "04_math_attended_by", output_dir)

        # Run analysis 2: Code generation
        print("\n=== Analysis 2: Code Generation ===")
        run_analysis(page, "Write a Python function to check if a number is prime", max_tokens=150)
        take_screenshot(page, "05_code_overview", output_dir)

        # Click on a code token
        tokens = page.locator("#tokenGrid .token")
        if tokens.count() > 50:
            tokens.nth(50).click()
            time.sleep(1)
            take_screenshot(page, "06_code_token_selected", output_dir)

        # Run analysis 3: Reasoning question
        print("\n=== Analysis 3: Reasoning Question ===")
        run_analysis(page, "If Alice is taller than Bob, and Bob is taller than Carol, who is the shortest?", max_tokens=80)
        take_screenshot(page, "07_reasoning_overview", output_dir)

        # Select a reasoning token
        tokens = page.locator("#tokenGrid .token")
        if tokens.count() > 30:
            tokens.nth(30).click()
            time.sleep(1)
            take_screenshot(page, "08_reasoning_token_detail", output_dir)

        # Run analysis 4: Creative writing
        print("\n=== Analysis 4: Creative Writing ===")
        run_analysis(page, "Write a haiku about programming", max_tokens=60)
        take_screenshot(page, "09_creative_overview", output_dir)

        # Run analysis 5: Factual question (potential hallucination case)
        print("\n=== Analysis 5: Factual Question ===")
        run_analysis(page, "What is the population of Tokyo?", max_tokens=80)
        take_screenshot(page, "10_factual_overview", output_dir)

        # Print final console messages
        print("\n--- Console Messages ---")
        for msg in console_messages[-20:]:
            print(msg)

        browser.close()
        print(f"\nAll screenshots saved to: {output_dir}")
        print(f"Total screenshots: {len(list(output_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
