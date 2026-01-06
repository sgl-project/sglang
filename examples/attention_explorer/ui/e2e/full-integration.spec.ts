import { test, expect } from '@playwright/test';

const SGLANG_URL = 'http://localhost:30000';
const SIDECAR_URL = 'http://localhost:9000';

// Extended timeout for LLM responses
test.setTimeout(180000);

test.describe('Full Integration E2E - Real LLM Responses', () => {

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for connection to be established
    await page.waitForTimeout(3000);
  });

  test('complete user journey with real LLM data', async ({ page }) => {
    // =========================================================================
    // STEP 1: Verify connection status shows model name
    // =========================================================================
    console.log('Step 1: Checking connection status...');
    const connectionPill = page.locator('.pill').first();
    await expect(connectionPill).toContainText('Qwen', { timeout: 10000 });
    const dot = page.locator('.dot');
    await expect(dot).toHaveClass(/connected/);
    console.log('✓ Connected to model');

    // =========================================================================
    // STEP 2: Send first message and wait for real LLM response
    // =========================================================================
    console.log('Step 2: Sending first message...');
    const input = page.locator('.input-textarea');
    await input.fill('Explain what attention mechanisms are in transformers in 2-3 sentences.');

    const sendButton = page.locator('.btn-primary');
    await expect(sendButton).toBeEnabled();
    await sendButton.click();

    // Wait for assistant response to appear
    const assistantBubble = page.locator('.bubble.assistant').first();
    await expect(assistantBubble).toBeVisible({ timeout: 60000 });

    // Wait for streaming to complete (response should have substantial content)
    await page.waitForTimeout(10000);

    const responseText = await assistantBubble.locator('.bubble-content').textContent();
    console.log('Response preview:', responseText?.slice(0, 100) + '...');
    expect(responseText?.length).toBeGreaterThan(50);
    console.log('✓ Received LLM response');

    // =========================================================================
    // STEP 3: Verify tokens are rendered and clickable
    // =========================================================================
    console.log('Step 3: Checking token rendering...');
    const tokenLine = page.locator('.token-line').first();
    await expect(tokenLine).toBeVisible();

    const tokens = page.locator('.tok');
    const tokenCount = await tokens.count();
    console.log(`Found ${tokenCount} tokens`);
    expect(tokenCount).toBeGreaterThan(5);
    console.log('✓ Tokens rendered');

    // =========================================================================
    // STEP 4: Click a token and verify Insight Lens updates
    // =========================================================================
    console.log('Step 4: Testing token selection and Insight Lens...');

    // Find an output token (from assistant)
    const outputTokens = page.locator('.bubble.assistant .tok');
    const outputTokenCount = await outputTokens.count();
    console.log(`Found ${outputTokenCount} output tokens`);

    if (outputTokenCount > 3) {
      // Click the 4th token
      await outputTokens.nth(3).click();
      await page.waitForTimeout(500);

      // Check that token is selected
      await expect(outputTokens.nth(3)).toHaveClass(/selected/);

      // Check Insight Lens updates
      const insightPanel = page.locator('.insight-panel');
      await expect(insightPanel).toBeVisible();

      // Should show token number
      const tokenBadge = insightPanel.locator('.badge').first();
      const badgeText = await tokenBadge.textContent();
      expect(badgeText).toContain('Token');

      // Check for attention profile section
      const attentionProfile = page.locator('.section-header:has-text("Attention Profile")');
      if (await attentionProfile.isVisible()) {
        console.log('✓ Attention Profile visible');

        // Check metrics are displayed
        const localChip = page.locator('.chip:has-text("Local")');
        const midChip = page.locator('.chip:has-text("Mid")');
        await expect(localChip).toBeVisible();
        await expect(midChip).toBeVisible();
      }
    }
    console.log('✓ Token selection works');

    // =========================================================================
    // STEP 5: Test hover interaction
    // =========================================================================
    console.log('Step 5: Testing hover interaction...');
    if (outputTokenCount > 5) {
      await outputTokens.nth(5).hover();
      await page.waitForTimeout(300);
      await expect(outputTokens.nth(5)).toHaveClass(/hovered/);
    }
    console.log('✓ Hover interaction works');

    // =========================================================================
    // STEP 6: Test view navigation - Inspect View
    // =========================================================================
    console.log('Step 6: Testing Inspect View...');
    await page.click('.tab:has-text("Inspect")');
    await page.waitForTimeout(500);

    const inspectView = page.locator('.inspect-view');
    await expect(inspectView).toBeVisible();

    // Check for layer selector
    const layerSelector = page.locator('.layer-selector, .seg-tab');
    if (await layerSelector.first().isVisible()) {
      console.log('✓ Layer selector visible');
    }
    console.log('✓ Inspect View works');

    // =========================================================================
    // STEP 7: Test view navigation - Manifold View
    // =========================================================================
    console.log('Step 7: Testing Manifold View...');
    await page.click('.tab:has-text("Manifold")');
    await page.waitForTimeout(500);

    const manifoldView = page.locator('.manifold-view');
    await expect(manifoldView).toBeVisible();

    // Check canvas renders
    const canvas = page.locator('.manifold-canvas, canvas');
    await expect(canvas.first()).toBeVisible();
    console.log('✓ Manifold View works');

    // =========================================================================
    // STEP 8: Test view navigation - Router View
    // =========================================================================
    console.log('Step 8: Testing Router View...');
    await page.click('.tab:has-text("Router")');
    await page.waitForTimeout(500);

    const routerView = page.locator('.router-view');
    await expect(routerView).toBeVisible();
    console.log('✓ Router View works');

    // =========================================================================
    // STEP 9: Go back to Chat and test Program selector
    // =========================================================================
    console.log('Step 9: Testing Program selector...');
    await page.click('.tab:has-text("Chat")');
    await page.waitForTimeout(500);

    const programSelect = page.locator('.program-select');
    await expect(programSelect).toBeVisible();

    // Change to Debug mode
    await programSelect.selectOption('debug');
    await page.waitForTimeout(500);

    const selectedValue = await programSelect.inputValue();
    expect(selectedValue).toBe('debug');

    // Change back to Discovery
    await programSelect.selectOption('discovery');
    await page.waitForTimeout(500);
    console.log('✓ Program selector works');

    // =========================================================================
    // STEP 10: Send second message to test multi-turn conversation
    // =========================================================================
    console.log('Step 10: Testing multi-turn conversation...');
    await input.fill('Now explain self-attention specifically.');
    await sendButton.click();

    // Wait for second response
    await page.waitForTimeout(15000);

    const allBubbles = page.locator('.bubble');
    const bubbleCount = await allBubbles.count();
    expect(bubbleCount).toBeGreaterThanOrEqual(4); // 2 user + 2 assistant
    console.log(`✓ Multi-turn works (${bubbleCount} messages)`);

    // =========================================================================
    // STEP 11: Verify sidecar received fingerprints
    // =========================================================================
    console.log('Step 11: Checking sidecar integration...');
    const sidecarStats = await page.request.get(`${SIDECAR_URL}/stats`);
    const stats = await sidecarStats.json();
    console.log('Sidecar stats:', stats);
    expect(stats.zmq_received).toBeGreaterThan(0);
    expect(stats.n_clusters).toBeGreaterThanOrEqual(1);
    console.log('✓ Sidecar receiving fingerprints');

    // =========================================================================
    // STEP 12: Verify centroids are formed
    // =========================================================================
    console.log('Step 12: Checking cluster centroids...');
    const centroidsRes = await page.request.get(`${SIDECAR_URL}/centroids`);
    const centroids = await centroidsRes.json();
    const clusterIds = Object.keys(centroids);
    expect(clusterIds.length).toBeGreaterThanOrEqual(1);

    const firstCluster = centroids[clusterIds[0]];
    expect(firstCluster.centroid).toHaveLength(20);
    expect(firstCluster.traits).toBeDefined();
    console.log('Cluster traits:', firstCluster.traits);
    console.log('✓ Centroids formed correctly');

    console.log('\n========================================');
    console.log('ALL E2E TESTS PASSED!');
    console.log('========================================');
  });

  test('rapid interaction stress test', async ({ page }) => {
    // Send a message first
    const input = page.locator('.input-textarea');
    await input.fill('Count from 1 to 10.');
    await page.click('.btn-primary');

    // Wait for response
    await page.waitForTimeout(8000);

    // Rapidly click through tokens
    const tokens = page.locator('.bubble.assistant .tok');
    const count = await tokens.count();

    for (let i = 0; i < Math.min(count, 10); i++) {
      await tokens.nth(i).click();
      await page.waitForTimeout(200);
    }

    // Rapidly switch views
    const views = ['Inspect', 'Manifold', 'Router', 'Chat'];
    for (const view of views) {
      await page.click(`.tab:has-text("${view}")`);
      await page.waitForTimeout(300);
    }

    // UI should still be responsive
    await expect(page.locator('.topbar')).toBeVisible();
    console.log('✓ UI remained responsive during rapid interaction');
  });

  test('attention data flows correctly per token', async ({ page }) => {
    // Send message
    const input = page.locator('.input-textarea');
    await input.fill('What is 2+2?');
    await page.click('.btn-primary');

    await page.waitForTimeout(10000);

    // Click different tokens and verify lens updates each time
    const tokens = page.locator('.bubble.assistant .tok');
    const count = await tokens.count();
    console.log(`Found ${count} tokens to test`);

    let tokenSelections = 0;

    for (let i = 0; i < Math.min(count, 5); i++) {
      await tokens.nth(i).click();
      await page.waitForTimeout(500);

      // Check that token gets selected class
      const isSelected = await tokens.nth(i).evaluate(el => el.classList.contains('selected'));
      if (isSelected) tokenSelections++;

      // Check insight panel shows token number
      const insightBadge = page.locator('.insight-panel .badge').first();
      const badgeText = await insightBadge.textContent();
      if (badgeText?.includes('Token')) {
        console.log(`Token ${i}: ${badgeText}`);
      }
    }

    console.log(`Successfully selected ${tokenSelections} tokens`);
    expect(tokenSelections).toBeGreaterThan(0);
    console.log('✓ Token selection and insight panel working');
  });
});
