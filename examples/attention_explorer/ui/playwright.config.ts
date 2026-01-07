import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for the Attention Explorer UI.
 *
 * Test modes:
 * - Default: Runs all tests except full-integration (requires real LLM)
 * - Integration: Set RUN_INTEGRATION_E2E=1 to include full-integration tests
 *
 * Usage:
 *   npm test                           # Run unit/UI tests only
 *   RUN_INTEGRATION_E2E=1 npm test     # Run all tests including integration
 *   npm test -- --project=integration  # Run only integration tests
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: process.env.CI ? 'github' : 'html',
  timeout: 60000,

  use: {
    baseURL: 'http://localhost:3001',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },

  projects: [
    {
      name: 'unit',
      testMatch: /app\.spec\.ts/,
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'integration',
      testMatch: /full-integration\.spec\.ts/,
      use: { ...devices['Desktop Chrome'] },
      // Only run in CI if explicitly enabled
      ...(process.env.CI && !process.env.RUN_INTEGRATION_E2E ? {
        testIgnore: /.*/
      } : {}),
    },
    {
      name: 'screenshots',
      testMatch: /product-screenshots\.spec\.ts/,
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1920, height: 1080 },
      },
    },
  ],

  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3001',
    reuseExistingServer: true,
    timeout: 30000,
  },
});
