// Run after verify_capture_monitoring.py, while its Prometheus history is retained.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { parseArgs } = require('node:util');
const { chromium } = require('playwright');

const { values: args } = parseArgs({ options: {
  'runtime-dir': { type: 'string' },
  'output-dir': { type: 'string' },
  dashboard: { type: 'string', default: path.join(__dirname, '../../examples/monitoring/grafana/dashboards/json/training-capture-dashboard.json') },
  grafana: { type: 'string', default: 'http://127.0.0.1:13000' },
  prometheus: { type: 'string', default: 'http://127.0.0.1:19090' },
  datasource: { type: 'string', default: 'capture-prometheus' },
  model: { type: 'string', default: 'capture-monitoring-qwen3' },
  instance: { type: 'string', default: '127.0.0.1:18081' },
} });
assert(args['runtime-dir'] && args['output-dir'], '--runtime-dir and --output-dir are required');
fs.mkdirSync(args['output-dir'], { recursive: false });
const output = name => path.join(args['output-dir'], name);
const write = (name, value) => fs.writeFileSync(output(name), JSON.stringify(value, null, 2) + '\n');
const report = { status: 'running', config: args, queries: [], phases: [], views: [] };
const sha256 = data => crypto.createHash('sha256').update(data).digest('hex');
const metric = suffix => `sglang:training_capture_${suffix}`;
const expectsEmpty = expr => !report.cohort && expr.includes(metric('routing_events_total'));

async function json(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(30000) });
  assert(response.ok, `${response.status}: ${url}`);
  return response.json();
}

async function query(expr, start, end, filename) {
  const url = new URL('/api/v1/query_range', args.prometheus);
  for (const [key, value] of Object.entries({ query: expr, start, end, step: 2 })) url.searchParams.set(key, value);
  const data = await json(url);
  if (filename) write(filename, data);
  assert.equal(data.status, 'success', JSON.stringify(data));
  assert.equal(data.data.resultType, 'matrix');
  assert(!data.warnings?.length, JSON.stringify(data.warnings));
  return data.data.result;
}

const finiteValues = rows => rows.flatMap(row => row.values.map(([, value]) => Number(value))).filter(Number.isFinite);
const regex = text => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const quoted = text => JSON.stringify(text).slice(1, -1);
const substitute = (expr, selected) => expr
  .replaceAll('$model_name', selected ? quoted(regex(args.model)) : '.*')
  .replaceAll('$instance', selected ? quoted(regex(args.instance)) : '.*')
  .replaceAll('$__rate_interval', '8s');

async function checkView(browser, dashboard, start, end, name, viewport, selected) {
  const context = await browser.newContext({ locale: 'en-US', timezoneId: 'UTC', viewport });
  const page = await context.newPage();
  const view = { name, viewport, selected, errors: [], consoleErrors: [], responses: [], queryResults: [], panels: [] };
  report.views.push(view);
  const pending = [];
  page.on('pageerror', error => view.errors.push(String(error)));
  page.on('console', message => { if (message.type() === 'error') view.consoleErrors.push(message.text()); });
  page.on('response', response => {
    const url = response.url();
    if (response.status() >= 400) view.responses.push({ url, status: response.status() });
    if (new URL(url).pathname === '/api/ds/query') pending.push((async () => {
      try {
        const body = await response.json();
        const request = response.request().postDataJSON();
        for (const item of request.queries) {
          const result = body.results?.[item.refId];
          assert(result && !result.error && (result.status ?? 200) < 400, JSON.stringify(body));
          assert(!item.expr.replaceAll('$__rate_interval', '').includes('$'), `Unresolved dashboard variable: ${item.expr}`);
          // Grafana expands its rate macro in the backend, after this request.
          const executed = [...new Set((result.frames || []).map(frame => frame.schema.meta?.executedQueryString).filter(Boolean))];
          assert((executed.length || expectsEmpty(item.expr)) && executed.every(expr => !expr.includes('$')), 'Backend did not resolve query macros');
          const numeric = (result.frames || []).flatMap(frame => frame.schema.fields.flatMap((field, index) =>
            field.type === 'number' ? frame.data.values[index].filter(Number.isFinite) : []));
          if (expectsEmpty(item.expr)) assert.equal(numeric.length, 0, 'Single-rank producer unexpectedly exports cohort routing');
          view.queryResults.push({ expr: item.expr, executed, refId: item.refId, finiteValues: numeric.length, expectedEmpty: expectsEmpty(item.expr) });
        }
      } catch (error) { view.errors.push(String(error)); }
    })());
  });
  try {
    const url = new URL(`/d/${dashboard.uid}`, args.grafana);
    for (const [key, value] of Object.entries({
      orgId: 1, from: Math.floor(start * 1000), to: Math.floor(end * 1000),
      'var-datasource': args.datasource, 'var-model_name': selected ? args.model : '$__all',
      'var-instance': selected ? args.instance : '$__all',
    })) url.searchParams.set(key, value);
    await page.goto(url.href, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(title => document.title.includes(title), dashboard.title, { timeout: 30000 });
    for (const panel of dashboard.panels) {
      const element = page.locator(`[data-testid="${panel.type}-panel-${panel.id}"]`);
      await element.scrollIntoViewIfNeeded();
      await page.waitForTimeout(1500);
      await Promise.all(pending);
      const pixels = await element.locator('canvas').evaluateAll(canvases => canvases.map(canvas => {
        const ctx = canvas.getContext('2d');
        const { width, height } = canvas;
        if (!ctx || !width || !height) return { width, height, colors: 0 };
        const data = ctx.getImageData(0, 0, width, height).data;
        const colors = new Set();
        for (let i = 0; i < data.length; i += 16) if (data[i + 3]) colors.add(`${data[i]},${data[i + 1]},${data[i + 2]},${data[i + 3]}`);
        return { width, height, colors: colors.size };
      }));
      const expectedEmpty = panel.targets.every(target => expectsEmpty(target.expr));
      if (panel.type === 'timeseries' && !expectedEmpty) assert(pixels.some(p => p.colors >= 4), `${name}: blank panel ${panel.id}`);
      const box = await element.boundingBox();
      assert(box && box.width > 150 && box.height > 100, `Invalid panel dimensions: ${panel.id}`);
      assert(box.x >= -1 && box.x + box.width <= viewport.width + 1, `Panel overflows viewport: ${panel.id}`);
      const text = await element.innerText();
      assert(!/Query error|Panel plugin not found/i.test(text), `${panel.id}: ${text}`);
      assert.equal(/No data/i.test(text), expectedEmpty, `${panel.id}: ${text}`);
      await element.screenshot({ path: output(`${name}-panel-${panel.id}.png`) });
      view.panels.push({ id: panel.id, box, pixels, text, expectedEmpty });
    }
    await page.locator(`[data-testid="${dashboard.panels[0].type}-panel-${dashboard.panels[0].id}"]`).scrollIntoViewIfNeeded();
    await page.screenshot({ path: output(`${name}.png`) });
    view.url = page.url();
    view.document = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scrollWidth: document.documentElement.scrollWidth }));
    assert(view.document.scrollWidth <= view.document.width + 1, 'Horizontal page overflow');
    await Promise.all(pending);
    assert.equal(view.errors.length, 0, JSON.stringify(view.errors));
    // Grafana's anonymous Viewer requests an authenticated-only stars endpoint.
    const unexpected = view.responses.filter(r => !(new URL(r.url).pathname === '/api/user/stars' && r.status === 401));
    assert.equal(unexpected.length, 0, JSON.stringify(unexpected));
    assert(view.queryResults.length >= dashboard.panels.reduce((count, p) => count + p.targets.length, 0));
    assert(view.queryResults.every(q => q.expectedEmpty || q.finiteValues > 0), 'Grafana returned an empty query result');
    console.log(JSON.stringify({ view: name, panels: view.panels.length, queries: view.queryResults.length }));
  } catch (error) {
    view.failedTitle = await page.title();
    view.failedBody = (await page.locator('body').innerText()).slice(0, 8000);
    await page.screenshot({ path: output(`${name}-failed.png`) });
    throw error;
  } finally { await context.close(); }
}

async function main() {
  const runtimeBytes = fs.readFileSync(path.join(args['runtime-dir'], 'report.json'));
  const runtime = JSON.parse(runtimeBytes);
  assert.equal(runtime.status, 'completed', 'Producer runtime must pass first');
  const dashboardBytes = fs.readFileSync(args.dashboard);
  const dashboard = JSON.parse(dashboardBytes);
  assert.equal(sha256(dashboardBytes), runtime.source_sha256['examples/monitoring/grafana/dashboards/json/training-capture-dashboard.json']);
  report.sourceSha256 = { dashboard: sha256(dashboardBytes), runtimeReport: sha256(runtimeBytes), verifier: sha256(fs.readFileSync(__filename)) };
  const observations = runtime.observations;
  report.cohort = observations.some(row => Object.hasOwn(row.state, 'request_router'));
  const start = observations[0].unix + 4;
  const end = observations.at(-1).unix;
  report.window = { start, end };
  const provisioned = await json(new URL(`/api/dashboards/uid/${dashboard.uid}`, args.grafana));
  assert.deepEqual(provisioned.dashboard.panels, dashboard.panels, 'Provisioned panels differ from source');
  const datasource = await json(new URL(`/api/datasources/uid/${args.datasource}`, args.grafana));
  assert.equal(datasource.type, 'prometheus');
  report.versions = {
    grafana: await json(new URL('/api/health', args.grafana)),
    prometheus: await json(new URL('/api/v1/status/buildinfo', args.prometheus)),
  };
  const up = finiteValues(await query(`up{instance="${args.instance}"}`, start, end, 'up.json'));
  assert(up.length > (end - start) / 3 && up.every(value => value === 1), 'Missing or failing scrapes');
  for (const panel of dashboard.panels) for (const target of panel.targets) {
    for (const selected of [false, true]) {
      const expr = substitute(target.expr, selected);
      const rows = await query(expr, start, end, `query-${panel.id}-${target.refId}-${selected ? 'selected' : 'all'}.json`);
      const values = finiteValues(rows);
      if (expectsEmpty(expr)) assert.equal(rows.length, 0, 'Single-rank producer unexpectedly exports cohort routing');
      else assert(values.length > 0, `No finite data for panel ${panel.id}/${target.refId}`);
      report.queries.push({ panel: panel.id, refId: target.refId, selected, expr, series: rows.length, finiteValues: values.length, expectedEmpty: expectsEmpty(expr) });
    }
  }
  const noMatch = await query(substitute(dashboard.panels[0].targets[0].expr, true).replace(quoted(regex(args.instance)), 'no-such-capture-instance'), start, end);
  assert.equal(noMatch.length, 0, 'Instance filter is ineffective');
  for (const phase of ['capture', 'paused', 'resumed']) {
    const rows = observations.filter(row => row.phase === phase);
    assert(rows.length >= 2, `Insufficient ${phase} observations`);
    const from = rows[0].unix + 4, to = rows.at(-1).unix;
    const paused = finiteValues(await query(metric('admission_paused'), from, to));
    assert(paused.length > 0 && paused.every(v => v === Number(phase === 'paused')), `${phase} pause state mismatch`);
    const ready = finiteValues(await query(`${metric('events_total')}{event="ready",tp_rank="0",pp_rank="0"}`, from, to));
    const host = finiteValues(await query(`${metric('kv_export_enqueued_bytes_total')}{destination="host",tp_rank="0",pp_rank="0"}`, from, to));
    assert(ready.length > 1 && host.length > 1);
    if (phase === 'paused') {
      assert(ready.every(v => v === rows[0].state.counters.ready));
      assert(host.every(v => v === rows[0].metrics['kv_export_enqueued_bytes_total:host']));
    } else {
      assert(ready.at(-1) > ready[0] && host.at(-1) > host[0], `${phase} publication/export did not progress`);
    }
    report.phases.push({ phase, start: from, end: to, paused: [...new Set(paused)], ready: [ready[0], ready.at(-1)], hostBytes: [host[0], host.at(-1)] });
  }
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
  report.browser = browser.version();
  try {
    await checkView(browser, dashboard, start, end, 'desktop-all', { width: 1440, height: 1000 }, false);
    await checkView(browser, dashboard, start, end, 'desktop-selected', { width: 1440, height: 1000 }, true);
    await checkView(browser, dashboard, start, end, 'mobile-selected', { width: 390, height: 844 }, true);
  } finally { await browser.close(); }
  report.status = 'completed';
}

main().catch(error => { report.status = 'failed'; report.error = String(error.stack || error); process.exitCode = 1; })
  .finally(() => { write('report.json', report); console.log(JSON.stringify({ status: report.status, error: report.error, output: args['output-dir'] })); });
