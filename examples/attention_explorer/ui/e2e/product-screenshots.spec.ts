import { test, expect } from '@playwright/test';

/**
 * Product Screenshots E2E Test
 *
 * Captures screenshots of the Attention Explorer UI for product presentation.
 *
 * Requirements:
 * - SGLang server running at localhost:30000 with Qwen3-Next model
 * - Run with: npx playwright test product-screenshots --headed
 *
 * Screenshots are saved to: e2e/screenshots/
 */

const SCREENSHOT_DIR = 'e2e/screenshots';
const SGLANG_URL = 'http://localhost:8000';

// Extended timeout for LLM responses
test.setTimeout(300000);

test.describe('Product Screenshots - Attention Explorer', () => {
  test.beforeEach(async ({ page }) => {
    // Set viewport for consistent screenshots
    await page.setViewportSize({ width: 1920, height: 1080 });

    // Log console messages for debugging
    page.on('console', msg => {
      const text = msg.text();
      if (text.includes('[Client]') || text.includes('[Hook]') || text.includes('[LinksTab]') || text.includes('ATTENTION')) {
        console.log(`[Browser] ${text}`);
      }
    });
  });

  test('capture complete UI showcase', async ({ page }) => {
    // =========================================================================
    // 1. INITIAL LOAD - Empty State
    // =========================================================================
    console.log('📸 1. Capturing initial load...');
    await page.goto('/');
    await page.waitForTimeout(2000);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-initial-load.png`,
      fullPage: false,
    });
    console.log('✓ Initial load captured');

    // =========================================================================
    // 2. CONNECTION STATUS
    // =========================================================================
    console.log('📸 2. Waiting for connection...');
    const connectionPill = page.locator('.pill').first();

    try {
      await expect(connectionPill).toContainText('Qwen', { timeout: 15000 });
      const dot = page.locator('.dot');
      await expect(dot).toHaveClass(/connected/, { timeout: 5000 });
      console.log('✓ Connected to Qwen model');
    } catch {
      console.log('⚠ Server not connected - continuing with available UI');
    }

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-connected.png`,
      fullPage: false,
    });

    // =========================================================================
    // 3. SEND FIRST MESSAGE
    // =========================================================================
    console.log('📸 3. Sending first message...');

    // Switch to debug mode for raw attention data (with layer info)
    const programSelectEarly = page.locator('.program-select');
    if (await programSelectEarly.isVisible()) {
      await programSelectEarly.selectOption('debug');
      await page.waitForTimeout(300);
      console.log('Switched to debug mode');
    }

    const input = page.locator('.input-textarea');
    // Use a prompt that generates a moderate response with interesting attention patterns
    await input.fill('What are the three primary colors? Answer in one sentence.');

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/03-message-typed.png`,
      fullPage: false,
    });

    const sendButton = page.locator('.btn-primary');
    await sendButton.click();

    // Wait for user message to appear
    await page.waitForTimeout(1000);
    await page.screenshot({
      path: `${SCREENSHOT_DIR}/04-message-sent.png`,
      fullPage: false,
    });

    // =========================================================================
    // 4. STREAMING RESPONSE
    // =========================================================================
    console.log('📸 4. Capturing streaming response...');

    try {
      const assistantBubble = page.locator('.bubble.assistant').first();
      await expect(assistantBubble).toBeVisible({ timeout: 60000 });

      // Capture mid-stream
      await page.waitForTimeout(3000);
      await page.screenshot({
        path: `${SCREENSHOT_DIR}/05-streaming.png`,
        fullPage: false,
      });

      // Wait for stream to complete - check for [DONE] received in localStorage
      // or wait for stop button to disappear (indicating stream finished)
      console.log('Waiting for stream to complete...');
      await page.waitForFunction(() => {
        return localStorage.getItem('__done_received__') === 'true';
      }, { timeout: 120000 });
      console.log('Stream completed!');

      // Give a bit more time for attention processing
      await page.waitForTimeout(1000);

      await page.screenshot({
        path: `${SCREENSHOT_DIR}/06-response-complete.png`,
        fullPage: false,
      });
      console.log('✓ Response captured');
    } catch (e) {
      console.log('⚠ Stream did not complete in time or error:', e);
      // Still take screenshot
      await page.screenshot({
        path: `${SCREENSHOT_DIR}/06-response-complete.png`,
        fullPage: false,
      });
    }

    // =========================================================================
    // 5. TOKEN INTERACTION - HOVER
    // =========================================================================
    console.log('📸 5. Capturing token hover...');
    const outputTokens = page.locator('.bubble.assistant .tok');
    const tokenCount = await outputTokens.count();

    if (tokenCount > 5) {
      await outputTokens.nth(5).hover();
      await page.waitForTimeout(400);
      await page.screenshot({
        path: `${SCREENSHOT_DIR}/07-token-hover.png`,
        fullPage: false,
      });
      console.log('✓ Token hover captured');
    }

    // =========================================================================
    // 6. TOKEN LENS DRAWER - PINNED
    // =========================================================================
    console.log('📸 6. Capturing Token Lens Drawer...');

    // Debug: Check what attention data is available
    const attentionDebug = await page.evaluate(() => {
      const store = (window as any).__SESSION_STORE__?.get?.();
      const attentionCountFromLS = localStorage.getItem('__attention_count__');
      if (!store) {
        return { error: 'Store not accessible', attentionCountFromLS };
      }
      const lastMsg = store.messages?.filter((m: any) => m.role === 'assistant').pop();
      const attentionIndices: number[] = [];
      if (lastMsg?.attention) {
        for (let i = 0; i < 20; i++) {
          if (lastMsg.attention[i]) attentionIndices.push(i);
        }
      }
      return {
        uiTokenCount: lastMsg?.tokens?.length,
        attentionIndices,
        attentionArrayLength: lastMsg?.attention?.length,
        currentAttentionSize: store.currentAttention?.size,
        firstFewTokens: lastMsg?.tokens?.slice(0, 5),
        attentionCountFromLS,
        attentionFound: localStorage.getItem('__attention_found__'),
        chunkCount: localStorage.getItem('__chunk_count__'),
        doneReceived: localStorage.getItem('__done_received__'),
      };
    });
    console.log('📊 Attention debug:', JSON.stringify(attentionDebug, null, 2));

    if (tokenCount > 8) {
      // Try clicking token #1 first (more likely to have attention data)
      await outputTokens.nth(1).click();
      await page.waitForTimeout(500);
      await page.screenshot({
        path: `${SCREENSHOT_DIR}/08-drawer-pinned.png`,
        fullPage: false,
      });

      // Switch to Signal tab
      const signalTab = page.locator('.drawer-tab:has-text("Signal")');
      if (await signalTab.isVisible()) {
        await signalTab.click();
        await page.waitForTimeout(300);
        await page.screenshot({
          path: `${SCREENSHOT_DIR}/09-drawer-signal-tab.png`,
          fullPage: false,
        });
      }

      // Switch to MoE tab
      const moeTab = page.locator('.drawer-tab:has-text("MoE")');
      if (await moeTab.isVisible()) {
        await moeTab.click();
        await page.waitForTimeout(300);
        await page.screenshot({
          path: `${SCREENSHOT_DIR}/10-drawer-moe-tab.png`,
          fullPage: false,
        });
      }
      console.log('✓ Drawer tabs captured');
    }

    // =========================================================================
    // 7. INSPECT VIEW
    // =========================================================================
    console.log('📸 7. Capturing Inspect View...');
    await page.click('.tab:has-text("Inspect")');
    await page.waitForTimeout(1000);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/11-inspect-view.png`,
      fullPage: false,
    });

    // Try selecting a layer
    const layerTab = page.locator('.seg-tab').first();
    if (await layerTab.isVisible()) {
      await layerTab.click();
      await page.waitForTimeout(500);
      await page.screenshot({
        path: `${SCREENSHOT_DIR}/12-inspect-layer-selected.png`,
        fullPage: false,
      });
    }
    console.log('✓ Inspect View captured');

    // =========================================================================
    // 8. MANIFOLD VIEW
    // =========================================================================
    console.log('📸 8. Capturing Manifold View...');
    await page.click('.tab:has-text("Manifold")');
    await page.waitForTimeout(1000);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/13-manifold-view.png`,
      fullPage: false,
    });

    // Try different scopes
    const recentBtn = page.locator('.scope-btn:has-text("Recent")');
    if (await recentBtn.isVisible()) {
      await recentBtn.click();
      await page.waitForTimeout(500);
      await page.screenshot({
        path: `${SCREENSHOT_DIR}/14-manifold-recent.png`,
        fullPage: false,
      });
    }

    const allBtn = page.locator('.scope-btn:has-text("All")');
    if (await allBtn.isVisible()) {
      await allBtn.click();
      await page.waitForTimeout(500);
    }
    console.log('✓ Manifold View captured');

    // =========================================================================
    // 9. ROUTER VIEW
    // =========================================================================
    console.log('📸 9. Capturing Router View...');
    await page.click('.tab:has-text("Router")');
    await page.waitForTimeout(1000);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/15-router-view.png`,
      fullPage: false,
    });

    // Capture the Apply button interaction
    const applyBtn = page.locator('.apply-btn');
    if (await applyBtn.isVisible() && await applyBtn.isEnabled()) {
      await applyBtn.click();
      await page.waitForTimeout(500);
      await page.screenshot({
        path: `${SCREENSHOT_DIR}/16-router-applied.png`,
        fullPage: false,
      });
    }
    console.log('✓ Router View captured');

    // =========================================================================
    // 10. THINK SECTION (if model produces it)
    // =========================================================================
    console.log('📸 10. Testing think section...');
    await page.click('.tab:has-text("Chat")');
    await page.waitForTimeout(500);

    // Send a reasoning prompt
    await input.fill('Think step by step: What is 15 * 17?');
    await sendButton.click();

    try {
      await page.waitForTimeout(20000);

      // Check for think section
      const thinkSection = page.locator('.segment-think, .think-badge');
      if (await thinkSection.first().isVisible()) {
        await page.screenshot({
          path: `${SCREENSHOT_DIR}/17-think-section.png`,
          fullPage: false,
        });
        console.log('✓ Think section captured');

        // Try collapsing
        const collapseBtn = page.locator('.segment-header .collapse-btn').first();
        if (await collapseBtn.isVisible()) {
          await collapseBtn.click();
          await page.waitForTimeout(300);
          await page.screenshot({
            path: `${SCREENSHOT_DIR}/18-think-collapsed.png`,
            fullPage: false,
          });
        }
      }
    } catch {
      console.log('⚠ Think section not available');
    }

    // =========================================================================
    // 11. PROGRAM SELECTOR
    // =========================================================================
    console.log('📸 11. Capturing Program selector...');
    const programSelect = page.locator('.program-select');
    if (await programSelect.isVisible()) {
      await programSelect.selectOption('debug');
      await page.waitForTimeout(500);
      await page.screenshot({
        path: `${SCREENSHOT_DIR}/19-program-debug.png`,
        fullPage: false,
      });

      await programSelect.selectOption('prod');
      await page.waitForTimeout(500);
      await page.screenshot({
        path: `${SCREENSHOT_DIR}/20-program-prod.png`,
        fullPage: false,
      });

      await programSelect.selectOption('discovery');
    }
    console.log('✓ Program selector captured');

    // =========================================================================
    // 12. FINAL OVERVIEW
    // =========================================================================
    console.log('📸 12. Capturing final overview...');

    // Back to chat with full conversation
    await page.click('.tab:has-text("Chat")');
    await page.waitForTimeout(500);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/21-final-chat.png`,
      fullPage: false,
    });

    // Scroll to show full conversation if needed
    const messageList = page.locator('.message-list');
    if (await messageList.isVisible()) {
      await messageList.evaluate(el => el.scrollTop = 0);
      await page.waitForTimeout(300);
      await page.screenshot({
        path: `${SCREENSHOT_DIR}/22-chat-scrolled-top.png`,
        fullPage: false,
      });
    }

    console.log('\n========================================');
    console.log('📸 ALL SCREENSHOTS CAPTURED!');
    console.log(`📁 Location: ${SCREENSHOT_DIR}/`);
    console.log('========================================');
  });
});
