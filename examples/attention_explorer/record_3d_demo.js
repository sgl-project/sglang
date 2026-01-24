#!/usr/bin/env node
/**
 * 3D Attention Visualization Demo Recording Script
 *
 * Records a video demo of the SGLang 3D attention tree explorer.
 *
 * Prerequisites:
 *   - SGLang server running on localhost:30000 with --return-attention-tokens
 *   - WebSocket server: python attention_ws_server.py --sglang-url http://localhost:30000
 *
 * Usage:
 *   node record_3d_demo.js
 *
 * Output:
 *   demo_recordings/sglang_3d_attention_demo_<timestamp>.webm
 */

const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const CONFIG = {
  explorerUrl: `file://${path.join(__dirname, 'explorer_3d.html')}`,
  wsServerUrl: 'http://localhost:8765',
  outputDir: path.join(__dirname, 'demo_recordings'),
  viewportWidth: 1920,
  viewportHeight: 1080,
  prompt: 'Explain how attention mechanisms in transformers help the model understand context and relationships between words in a sentence.',
  maxTokens: 80,
  topK: 10,
};

async function waitForWebSocket(page, timeout = 10000) {
  console.log('Waiting for WebSocket connection...');
  const start = Date.now();
  while (Date.now() - start < timeout) {
    const connected = await page.evaluate(() => {
      const dot = document.querySelector('#statusDot');
      return dot && dot.classList.contains('connected');
    });
    if (connected) {
      console.log('WebSocket connected!');
      return true;
    }
    await page.waitForTimeout(500);
  }
  throw new Error('WebSocket connection timeout');
}

async function main() {
  // Ensure output directory exists
  if (!fs.existsSync(CONFIG.outputDir)) {
    fs.mkdirSync(CONFIG.outputDir, { recursive: true });
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const videoPath = path.join(CONFIG.outputDir, `sglang_3d_attention_demo_${timestamp}.webm`);

  console.log('='.repeat(60));
  console.log('SGLang 3D Attention Visualization Demo Recording');
  console.log('='.repeat(60));
  console.log(`Output: ${videoPath}`);
  console.log(`Prompt: "${CONFIG.prompt}"`);
  console.log('');

  const browser = await chromium.launch({
    headless: true,  // Headless for CI/server environments
    args: ['--window-size=1920,1080', '--no-sandbox'],
  });

  const context = await browser.newContext({
    viewport: { width: CONFIG.viewportWidth, height: CONFIG.viewportHeight },
    recordVideo: {
      dir: CONFIG.outputDir,
      size: { width: CONFIG.viewportWidth, height: CONFIG.viewportHeight },
    },
  });

  const page = await context.newPage();

  try {
    // Load the 3D explorer
    console.log('Loading 3D explorer...');
    await page.goto(CONFIG.explorerUrl);
    await page.waitForTimeout(2000);

    // Wait for WebSocket connection
    await waitForWebSocket(page);

    // Take initial screenshot
    await page.screenshot({ path: path.join(CONFIG.outputDir, 'demo_01_initial.png') });
    console.log('Screenshot: Initial state');

    // Configure generation settings
    console.log('Configuring generation settings...');
    await page.fill('#maxTokens', String(CONFIG.maxTokens));
    await page.fill('#topK', String(CONFIG.topK));

    // Enter the prompt
    console.log('Entering prompt...');
    await page.fill('#inputText', CONFIG.prompt);
    await page.waitForTimeout(1000);

    // Take screenshot with prompt
    await page.screenshot({ path: path.join(CONFIG.outputDir, 'demo_02_prompt.png') });
    console.log('Screenshot: Prompt entered');

    // Start generation
    console.log('Starting generation...');
    await page.click('#streamBtn');

    // Wait for streaming to start
    await page.waitForTimeout(2000);
    await page.screenshot({ path: path.join(CONFIG.outputDir, 'demo_03_streaming.png') });
    console.log('Screenshot: Streaming in progress');

    // Wait for tokens to appear and tree to build
    console.log('Waiting for 3D tree to build...');
    let lastTokenCount = 0;
    let stableCount = 0;

    for (let i = 0; i < 120; i++) {  // Max 2 minutes
      await page.waitForTimeout(1000);

      const tokenCount = await page.evaluate(() => {
        const el = document.querySelector('#tokenCount');
        return el ? parseInt(el.textContent) : 0;
      });

      if (tokenCount > 0) {
        console.log(`  Tokens: ${tokenCount}`);
      }

      // Check if generation is complete (token count stable)
      if (tokenCount === lastTokenCount && tokenCount > 0) {
        stableCount++;
        if (stableCount >= 3) {
          console.log('Generation complete!');
          break;
        }
      } else {
        stableCount = 0;
      }
      lastTokenCount = tokenCount;

      // Take periodic screenshots during generation
      if (i === 5) {
        await page.screenshot({ path: path.join(CONFIG.outputDir, 'demo_04_early_tree.png') });
        console.log('Screenshot: Early tree');
      }
      if (i === 15) {
        await page.screenshot({ path: path.join(CONFIG.outputDir, 'demo_05_mid_tree.png') });
        console.log('Screenshot: Mid tree');
      }
    }

    // Wait for final render
    await page.waitForTimeout(3000);
    await page.screenshot({ path: path.join(CONFIG.outputDir, 'demo_06_complete.png') });
    console.log('Screenshot: Complete tree');

    // Demo: Switch camera views
    console.log('Demonstrating camera views...');

    // Tree view (default)
    await page.waitForTimeout(1500);

    // Top view
    console.log('  View: Top');
    await page.click('button[data-view="top"]');
    await page.waitForTimeout(2000);
    await page.screenshot({ path: path.join(CONFIG.outputDir, 'demo_07_top_view.png') });

    // Side view
    console.log('  View: Side');
    await page.click('button[data-view="side"]');
    await page.waitForTimeout(2000);
    await page.screenshot({ path: path.join(CONFIG.outputDir, 'demo_08_side_view.png') });

    // Front view
    console.log('  View: Front');
    await page.click('button[data-view="front"]');
    await page.waitForTimeout(2000);
    await page.screenshot({ path: path.join(CONFIG.outputDir, 'demo_09_front_view.png') });

    // Back to tree view
    console.log('  View: Tree');
    await page.click('button[data-view="tree"]');
    await page.waitForTimeout(2000);

    // Demo: Token navigation
    console.log('Demonstrating token navigation...');

    // Navigate through tokens using slider
    const tokenCount = await page.evaluate(() => {
      const el = document.querySelector('#tokenCount');
      return el ? parseInt(el.textContent) : 0;
    });

    if (tokenCount > 10) {
      // Navigate to different positions
      for (const pos of [10, 20, Math.floor(tokenCount / 2), tokenCount - 5]) {
        if (pos < tokenCount) {
          console.log(`  Navigating to token ${pos}`);
          await page.evaluate((p) => navToIndex(p), pos);
          await page.waitForTimeout(1500);
        }
      }
    }

    await page.screenshot({ path: path.join(CONFIG.outputDir, 'demo_10_navigation.png') });
    console.log('Screenshot: Token navigation');

    // Demo: Toggle edges
    console.log('Toggling attention edges...');
    await page.click('button:has-text("Edges")');
    await page.waitForTimeout(1500);
    await page.screenshot({ path: path.join(CONFIG.outputDir, 'demo_11_no_edges.png') });

    // Toggle back
    await page.click('button:has-text("Edges")');
    await page.waitForTimeout(1500);

    // Final cinematic pause
    console.log('Final view...');
    await page.click('button[data-view="tree"]');
    await page.waitForTimeout(3000);
    await page.screenshot({ path: path.join(CONFIG.outputDir, 'demo_12_final.png') });
    console.log('Screenshot: Final');

  } catch (error) {
    console.error('Error during demo:', error);
    await page.screenshot({ path: path.join(CONFIG.outputDir, 'demo_error.png') });
  } finally {
    // Close context to finalize video
    await context.close();
    await browser.close();

    // Find the recorded video and rename it
    const files = fs.readdirSync(CONFIG.outputDir);
    const webmFiles = files.filter(f => f.endsWith('.webm') && !f.includes('sglang_3d'));

    if (webmFiles.length > 0) {
      const latestVideo = webmFiles.sort().pop();
      const sourcePath = path.join(CONFIG.outputDir, latestVideo);
      fs.renameSync(sourcePath, videoPath);
      console.log('');
      console.log('='.repeat(60));
      console.log('Demo recording complete!');
      console.log('='.repeat(60));
      console.log(`Video: ${videoPath}`);
      console.log(`Screenshots: ${CONFIG.outputDir}/demo_*.png`);
    } else {
      console.log('Video file not found - check demo_recordings folder');
    }
  }
}

main().catch(console.error);
