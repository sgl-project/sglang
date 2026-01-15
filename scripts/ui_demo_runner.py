#!/usr/bin/env python3
"""
UI Demo Runner - Takes screenshots of the Attention Explorer UIs in action.

This script demonstrates the full workflow:
1. Explorer.html - Attention visualization with real query
2. RAG Explorer - Document ranking with attention fingerprints

Saves screenshots to demos/ directory.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Please install playwright: pip install playwright && playwright install")
    sys.exit(1)


def run_demo():
    """Run full UI demo with screenshots."""

    # Setup - file server runs from examples/attention_explorer/
    base_url = "http://localhost:8082"
    output_dir = Path("demos") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== UI DEMO RUNNER ===")
    print(f"Output: {output_dir}")
    print()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})

        # ============================================
        # DEMO 1: Explorer.html
        # ============================================
        print("--- EXPLORER.HTML DEMO ---")

        # Step 1: Load page
        print("1. Loading explorer.html...")
        page.goto(f"{base_url}/explorer.html")
        page.wait_for_load_state("networkidle")
        time.sleep(1)
        page.screenshot(path=str(output_dir / "01_explorer_loaded.png"))
        print(f"   Screenshot: 01_explorer_loaded.png")

        # Step 2: Show help modal
        print("2. Opening help modal...")
        help_btn = page.locator(".help-btn")
        if help_btn.count() > 0:
            help_btn.click()
            time.sleep(0.5)
            page.screenshot(path=str(output_dir / "02_explorer_help_intro.png"))
            print(f"   Screenshot: 02_explorer_help_intro.png")

            # Scroll to show layer diagram
            modal_content = page.locator(".modal-content")
            if modal_content.count() > 0:
                modal_content.evaluate("el => el.scrollTop = 800")
                time.sleep(0.3)
                page.screenshot(path=str(output_dir / "02b_explorer_help_layers.png"))
                print(f"   Screenshot: 02b_explorer_help_layers.png")

                # Scroll more to show attention flow
                modal_content.evaluate("el => el.scrollTop = 1400")
                time.sleep(0.3)
                page.screenshot(path=str(output_dir / "02c_explorer_help_attention_flow.png"))
                print(f"   Screenshot: 02c_explorer_help_attention_flow.png")

            # Close modal
            page.locator(".modal-close").click()
            time.sleep(0.3)
        else:
            print("   Help button not found, skipping...")

        # Step 3: Enter a query
        print("3. Entering query...")
        page.fill("#inputText", "What is the capital of France and why is it important?")
        page.screenshot(path=str(output_dir / "03_explorer_query_entered.png"))
        print(f"   Screenshot: 03_explorer_query_entered.png")

        # Step 4: Click analyze and wait for response
        print("4. Running analysis (this may take a moment)...")
        page.click("text=Analyze")

        # Wait for response (with timeout)
        try:
            page.wait_for_selector(".token", timeout=60000)
            time.sleep(2)  # Let visualization render
            page.screenshot(path=str(output_dir / "04_explorer_results.png"))
            print(f"   Screenshot: 04_explorer_results.png")

            # Step 5: Click on a token to see attention
            print("5. Clicking on a token to show attention...")
            tokens = page.locator(".token")
            if tokens.count() > 10:
                tokens.nth(10).click()
                time.sleep(0.5)
                page.screenshot(path=str(output_dir / "05_explorer_token_selected.png"))
                print(f"   Screenshot: 05_explorer_token_selected.png")

            # Step 6: Show the detail panel
            print("6. Showing attention details...")
            page.screenshot(path=str(output_dir / "06_explorer_attention_detail.png"), full_page=True)
            print(f"   Screenshot: 06_explorer_attention_detail.png")

        except Exception as e:
            print(f"   Warning: Analysis timed out or failed: {e}")
            page.screenshot(path=str(output_dir / "04_explorer_timeout.png"))

        # ============================================
        # DEMO 2: RAG Explorer
        # ============================================
        print()
        print("--- RAG_EXPLORER.HTML DEMO ---")

        # Step 1: Load page
        print("1. Loading rag_explorer.html...")
        page.goto(f"{base_url}/rag_explorer.html")
        page.wait_for_load_state("networkidle")
        time.sleep(1)
        page.screenshot(path=str(output_dir / "07_rag_loaded.png"))
        print(f"   Screenshot: 07_rag_loaded.png")

        # Step 2: Show help modal
        print("2. Opening help modal...")
        help_btn = page.locator(".help-btn, button:has-text('Glossary')")
        if help_btn.count() > 0:
            help_btn.first.click()
            time.sleep(0.5)
            page.screenshot(path=str(output_dir / "08_rag_help_intro.png"))
            print(f"   Screenshot: 08_rag_help_intro.png")

            # Scroll down in modal to show RAG pipeline diagram
            modal = page.locator(".modal-body")
            if modal.count() > 0:
                modal.evaluate("el => el.scrollTop = 600")
                time.sleep(0.3)
                page.screenshot(path=str(output_dir / "08b_rag_help_pipeline.png"))
                print(f"   Screenshot: 08b_rag_help_pipeline.png")

                # Scroll more to show layer explanation
                modal.evaluate("el => el.scrollTop = 1200")
                time.sleep(0.3)
                page.screenshot(path=str(output_dir / "08c_rag_help_layers.png"))
                print(f"   Screenshot: 08c_rag_help_layers.png")

            # Close modal
            close_btn = page.locator(".modal-close")
            if close_btn.count() > 0:
                close_btn.click()
                time.sleep(0.3)
        else:
            print("   Help button not found, skipping...")

        # Step 3: Click sample query
        print("3. Clicking sample query...")
        sample_queries = page.locator(".sample-query")
        if sample_queries.count() > 0:
            sample_queries.first.click()
            time.sleep(0.3)
            page.screenshot(path=str(output_dir / "10_rag_sample_query.png"))
            print(f"   Screenshot: 10_rag_sample_query.png")

        # Step 4: Click Analyze Query
        print("4. Analyzing query...")
        page.click("text=Analyze Query")
        time.sleep(1)
        page.screenshot(path=str(output_dir / "11_rag_query_analyzed.png"))
        print(f"   Screenshot: 11_rag_query_analyzed.png")

        # Step 5: Show document ranking
        print("5. Showing document ranking...")
        page.screenshot(path=str(output_dir / "12_rag_documents.png"), full_page=True)
        print(f"   Screenshot: 12_rag_documents.png")

        # Step 6: Hover over tooltip to show it
        print("6. Showing tooltip example...")
        info_icons = page.locator(".info-icon")
        if info_icons.count() > 0:
            info_icons.first.hover()
            time.sleep(0.5)
            page.screenshot(path=str(output_dir / "13_rag_tooltip.png"))
            print(f"   Screenshot: 13_rag_tooltip.png")

        browser.close()

    print()
    print(f"=== DEMO COMPLETE ===")
    print(f"Screenshots saved to: {output_dir}")
    print(f"Total screenshots: {len(list(output_dir.glob('*.png')))}")

    return output_dir


if __name__ == "__main__":
    output_dir = run_demo()
