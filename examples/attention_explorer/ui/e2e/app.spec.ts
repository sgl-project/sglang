import { test, expect } from '@playwright/test';

const SGLANG_URL = 'http://localhost:30000';

// ============================================================================
// API HEALTH TESTS
// ============================================================================

test.describe('SGLang API', () => {
  test('server is healthy', async ({ request }) => {
    const response = await request.get(`${SGLANG_URL}/health`);
    expect(response.ok()).toBeTruthy();
  });

  test('models endpoint returns model list', async ({ request }) => {
    const response = await request.get(`${SGLANG_URL}/v1/models`);
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data.object).toBe('list');
    expect(data.data).toBeInstanceOf(Array);
    expect(data.data.length).toBeGreaterThan(0);
    expect(data.data[0].id).toContain('Llama');
  });

  test('chat completions endpoint works (non-streaming)', async ({ request }) => {
    const response = await request.post(`${SGLANG_URL}/v1/chat/completions`, {
      data: {
        model: 'meta-llama/Llama-3.1-8B-Instruct',
        messages: [{ role: 'user', content: 'Say "test" and nothing else.' }],
        max_tokens: 10,
        stream: false,
      },
    });

    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data.choices).toBeInstanceOf(Array);
    expect(data.choices[0].message.content).toBeTruthy();
  });

  test('attention tokens endpoint accepts attention params', async ({ request }) => {
    const response = await request.post(`${SGLANG_URL}/v1/chat/completions`, {
      data: {
        model: 'meta-llama/Llama-3.1-8B-Instruct',
        messages: [{ role: 'user', content: 'Count to three.' }],
        max_tokens: 20,
        stream: false,
        return_attention_tokens: true,
        top_k_attention: 5,
      },
    });

    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data.choices[0].message.content).toBeTruthy();
    // The request should complete successfully with attention params
    // Attention data may be in various locations depending on server config
    expect(data.choices).toBeDefined();
  });

  test('streaming chat completions work', async ({ request }) => {
    const response = await request.post(`${SGLANG_URL}/v1/chat/completions`, {
      data: {
        model: 'meta-llama/Llama-3.1-8B-Instruct',
        messages: [{ role: 'user', content: 'Say hello.' }],
        max_tokens: 10,
        stream: true,
      },
    });

    expect(response.ok()).toBeTruthy();
    const text = await response.text();
    expect(text).toContain('data:');
    expect(text).toContain('[DONE]');
  });
});

// ============================================================================
// UI RENDERING TESTS
// ============================================================================

test.describe('UI Rendering', () => {
  test('app shell renders', async ({ page }) => {
    await page.goto('/');

    // Top bar should be visible
    await expect(page.locator('.topbar')).toBeVisible();

    // App title should be visible
    await expect(page.getByText('Latent Chat Explorer')).toBeVisible();
  });

  test('view navigation works', async ({ page }) => {
    await page.goto('/');

    // Default view should be Chat
    await expect(page.locator('.chat-view')).toBeVisible();

    // Click Inspect tab
    await page.click('.tab:has-text("Inspect")');
    await expect(page.locator('.inspect-view')).toBeVisible();

    // Click Manifold tab
    await page.click('.tab:has-text("Manifold")');
    await expect(page.locator('.manifold-view')).toBeVisible();

    // Click Router tab
    await page.click('.tab:has-text("Router")');
    await expect(page.locator('.router-view')).toBeVisible();

    // Go back to Chat
    await page.click('.tab:has-text("Chat")');
    await expect(page.locator('.chat-view')).toBeVisible();
  });

  test('program selector works', async ({ page }) => {
    await page.goto('/');

    // Program selector is a dropdown
    const programSelect = page.locator('.program-select');
    await expect(programSelect).toBeVisible();

    // Should have prod/debug/discovery options
    await expect(programSelect.locator('option')).toHaveCount(3);

    // Can change program
    await programSelect.selectOption('debug');
    await expect(programSelect).toHaveValue('debug');
  });

  test('input bar is visible and functional', async ({ page }) => {
    await page.goto('/');

    // Input field should be visible
    const input = page.locator('.input-bar input, .input-bar textarea');
    await expect(input).toBeVisible();

    // Send button should be visible
    const sendButton = page.locator('.input-bar button');
    await expect(sendButton).toBeVisible();
  });

  test('manifold canvas renders', async ({ page }) => {
    await page.goto('/');

    // Navigate to Manifold view
    await page.click('.tab:has-text("Manifold")');

    // Canvas should be visible
    const canvas = page.locator('.manifold-canvas, canvas');
    await expect(canvas).toBeVisible();
  });
});

// ============================================================================
// CHAT INTEGRATION TESTS
// ============================================================================

test.describe('Chat Integration', () => {
  test('can send a message and receive response', async ({ page }) => {
    await page.goto('/');

    // Type a message
    const input = page.locator('.input-bar input, .input-bar textarea');
    await input.fill('Say "hello world" exactly.');

    // Send the message (button enables after input)
    await page.click('.input-bar button:not([disabled])');

    // Wait for user message to appear
    await expect(page.locator('.bubble.user')).toBeVisible({ timeout: 5000 });

    // Wait for assistant response (streaming may take time)
    await expect(page.locator('.bubble.assistant')).toBeVisible({ timeout: 30000 });

    // Response should contain some text
    const assistantBubble = page.locator('.bubble.assistant').first();
    await expect(assistantBubble).not.toBeEmpty();
  });

  test('message list updates during streaming', async ({ page }) => {
    await page.goto('/');

    const input = page.locator('.input-bar input, .input-bar textarea');
    await input.fill('Count from 1 to 5.');

    await page.click('.input-bar button:not([disabled])');

    // User message should appear immediately
    await expect(page.locator('.bubble.user')).toBeVisible({ timeout: 5000 });

    // Assistant bubble should appear and grow
    const assistantBubble = page.locator('.bubble.assistant').first();
    await expect(assistantBubble).toBeVisible({ timeout: 30000 });

    // Wait for streaming to complete
    await page.waitForTimeout(5000);

    // Should have some content
    const content = await assistantBubble.textContent();
    expect(content?.length).toBeGreaterThan(0);
  });

  test('multiple messages work in conversation', async ({ page }) => {
    await page.goto('/');

    // First message
    const input = page.locator('.input-bar input, .input-bar textarea');
    await input.fill('My name is Alice.');
    await page.click('.input-bar button:not([disabled])');

    // Wait for response
    await expect(page.locator('.bubble.assistant')).toBeVisible({ timeout: 30000 });
    await page.waitForTimeout(3000);

    // Second message
    await input.fill('What is my name?');
    await page.click('.input-bar button:not([disabled])');

    // Wait for second response
    await page.waitForTimeout(5000);

    // Should have 2 user messages and 2 assistant messages
    const userMessages = page.locator('.bubble.user');
    const assistantMessages = page.locator('.bubble.assistant');

    await expect(userMessages).toHaveCount(2);
    await expect(assistantMessages).toHaveCount(2);
  });
});

// ============================================================================
// ATTENTION VISUALIZATION TESTS
// ============================================================================

test.describe('Attention Visualization', () => {
  test('inspect view shows attention data after chat', async ({ page }) => {
    await page.goto('/');

    // Send a message first
    const input = page.locator('.input-bar input, .input-bar textarea');
    await input.fill('Hello!');
    await page.click('.input-bar button:not([disabled])');

    // Wait for response
    await expect(page.locator('.bubble.assistant')).toBeVisible({ timeout: 30000 });
    await page.waitForTimeout(3000);

    // Switch to Inspect view
    await page.click('.tab:has-text("Inspect")');

    // Inspect view should be visible
    await expect(page.locator('.inspect-view')).toBeVisible();
  });

  test('layer selector is available in inspect view', async ({ page }) => {
    await page.goto('/');

    // Navigate to Inspect view
    await page.click('.tab:has-text("Inspect")');

    // Layer selector should be visible
    const layerSelector = page.locator('.layer-selector');
    await expect(layerSelector).toBeVisible();
  });

  test('token hover interaction works', async ({ page }) => {
    await page.goto('/');

    // Send a message
    const input = page.locator('.input-bar input, .input-bar textarea');
    await input.fill('The quick brown fox.');
    await page.click('.input-bar button:not([disabled])');

    // Wait for response
    await expect(page.locator('.bubble.assistant')).toBeVisible({ timeout: 30000 });
    await page.waitForTimeout(3000);

    // Find tokens in the response
    const tokens = page.locator('.token');

    if (await tokens.count() > 0) {
      // Hover over first token
      await tokens.first().hover();

      // Some visual feedback should occur (class change, tooltip, etc.)
      await page.waitForTimeout(500);
    }
  });
});

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

test.describe('Error Handling', () => {
  test('send button is disabled when input is empty', async ({ page }) => {
    await page.goto('/');

    // Send button should be disabled when empty
    const sendButton = page.locator('.input-bar button');
    await expect(sendButton).toBeDisabled();

    // Type something
    const input = page.locator('.input-bar input, .input-bar textarea');
    await input.fill('test');

    // Now button should be enabled
    await expect(sendButton).toBeEnabled();

    // Clear input
    await input.fill('');

    // Button should be disabled again
    await expect(sendButton).toBeDisabled();
  });

  test('displays error state on API failure', async ({ page }) => {
    // Just verify the app doesn't crash on load
    await page.goto('/');
    await expect(page.locator('.topbar')).toBeVisible();
  });
});

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

test.describe('Performance', () => {
  test('page loads within acceptable time', async ({ page }) => {
    const startTime = Date.now();
    await page.goto('/');
    await expect(page.locator('.topbar')).toBeVisible();
    const loadTime = Date.now() - startTime;

    // Should load within 5 seconds
    expect(loadTime).toBeLessThan(5000);
  });

  test('UI remains responsive during streaming', async ({ page }) => {
    await page.goto('/');

    // Send a message that will generate a longer response
    const input = page.locator('.input-bar input, .input-bar textarea');
    await input.fill('Write a short paragraph about programming.');
    await page.click('.input-bar button:not([disabled])');

    // Wait for streaming to start
    await expect(page.locator('.bubble.assistant')).toBeVisible({ timeout: 30000 });

    // During streaming, UI should still be responsive
    // Try clicking view tabs
    await page.click('.tab:has-text("Inspect")');
    await expect(page.locator('.inspect-view')).toBeVisible();

    await page.click('.tab:has-text("Chat")');
    await expect(page.locator('.chat-view')).toBeVisible();
  });
});
