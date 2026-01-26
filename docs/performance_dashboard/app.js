// SGLang Performance Dashboard Application

const GITHUB_REPO = 'sgl-project/sglang';
const WORKFLOW_NAME = 'nightly-test-nvidia.yml';
const ARTIFACT_PREFIX = 'consolidated-metrics-';

// Chart instances (array for batch-separated charts)
let activeCharts = [];

// Data storage
let allMetricsData = [];
let currentModel = null;
let currentMetricType = 'throughput'; // throughput, latency, ttft, inputThroughput

// Metric type definitions
const metricTypes = {
    throughput: { label: 'Overall Throughput', unit: 'tokens/sec', field: 'throughput' },
    latency: { label: 'Latency', unit: 'ms', field: 'latency' },
    ttft: { label: 'Time to First Token', unit: 'ms', field: 'ttft' },
    inputThroughput: { label: 'Input Throughput', unit: 'tokens/sec', field: 'inputThroughput' }
};

// Chart.js default configuration for dark theme
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';

const chartColors = [
    '#58a6ff', '#3fb950', '#d29922', '#f85149', '#a371f7',
    '#79c0ff', '#56d364', '#e3b341', '#ff7b72', '#bc8cff'
];

// Initialize the dashboard
async function init() {
    try {
        await loadData();
        document.getElementById('loading').style.display = 'none';
        document.getElementById('content').style.display = 'block';
        populateFilters();
        updateStats();
        updateCharts();
        updateRunsTable();
    } catch (error) {
        console.error('Failed to initialize dashboard:', error);
        document.getElementById('loading').style.display = 'none';
        document.getElementById('error').style.display = 'block';
        document.getElementById('error-message').textContent = error.message;
    }
}

// Load data from local server API or GitHub
async function loadData() {
    // Try local server API first (if running server.py)
    try {
        const response = await fetch('/api/metrics');
        if (response.ok) {
            const data = await response.json();
            if (data.length > 0 && data[0].results && data[0].results.length > 0) {
                allMetricsData = data;
                console.log(`Loaded ${data.length} records from local API`);
                allMetricsData.sort((a, b) => new Date(b.run_date) - new Date(a.run_date));
                return;
            }
        }
    } catch (error) {
        console.log('Local API not available, trying GitHub API');
    }

    // Try to load from GitHub API
    const runs = await fetchWorkflowRuns();
    const metricsPromises = runs.map(run => fetchMetricsForRun(run));
    const results = await Promise.allSettled(metricsPromises);

    allMetricsData = results
        .filter(r => r.status === 'fulfilled' && r.value !== null)
        .map(r => r.value);

    if (allMetricsData.length === 0) {
        throw new Error('No metrics data available. Please run the server.py with --fetch-on-start to fetch data from GitHub.');
    }

    // Sort by date descending
    allMetricsData.sort((a, b) => new Date(b.run_date) - new Date(a.run_date));
}

// Fetch workflow runs from GitHub API
async function fetchWorkflowRuns() {
    const response = await fetch(
        `https://api.github.com/repos/${GITHUB_REPO}/actions/workflows/${WORKFLOW_NAME}/runs?status=completed&per_page=30`,
        {
            headers: {
                'Accept': 'application/vnd.github.v3+json'
            }
        }
    );

    if (!response.ok) {
        throw new Error(`GitHub API error: ${response.status}`);
    }

    const data = await response.json();
    return data.workflow_runs || [];
}

// Fetch metrics artifact for a specific run
async function fetchMetricsForRun(run) {
    try {
        // Get artifacts for this run
        const artifactsResponse = await fetch(
            `https://api.github.com/repos/${GITHUB_REPO}/actions/runs/${run.id}/artifacts`,
            {
                headers: {
                    'Accept': 'application/vnd.github.v3+json'
                }
            }
        );

        if (!artifactsResponse.ok) return null;

        const artifactsData = await artifactsResponse.json();
        const metricsArtifact = artifactsData.artifacts.find(
            a => a.name.startsWith(ARTIFACT_PREFIX)
        );

        if (!metricsArtifact) return null;

        // Note: GitHub API doesn't allow direct artifact download without authentication
        // For public access, we would need to use a proxy or pre-process the data
        // For now, return run metadata - in production, use a backend to fetch artifacts
        return {
            run_id: run.id.toString(),
            run_date: run.created_at,
            commit_sha: run.head_sha,
            branch: run.head_branch,
            artifact_id: metricsArtifact.id,
            results: [] // Would be populated from artifact content
        };
    } catch (error) {
        console.warn(`Failed to fetch metrics for run ${run.id}:`, error);
        return null;
    }
}

// Populate filter dropdowns
function populateFilters() {
    const gpuConfigs = new Set();
    const models = new Set();
    const batchSizes = new Set();

    allMetricsData.forEach(run => {
        run.results.forEach(result => {
            gpuConfigs.add(result.gpu_config);
            models.add(result.model);
            result.benchmarks.forEach(bench => {
                batchSizes.add(bench.batch_size);
            });
        });
    });

    // No "all" option for GPU and Model - populate with first value selected
    const gpuArray = Array.from(gpuConfigs).sort();
    const modelArray = Array.from(models).sort();

    populateSelectNoAll('gpu-filter', gpuArray);
    populateSelectNoAll('model-filter', modelArray);
    populateSelect('batch-filter', Array.from(batchSizes).sort((a, b) => a - b));

    // Set initial values (first option)
    if (gpuArray.length > 0) {
        document.getElementById('gpu-filter').value = gpuArray[0];
    }
    if (modelArray.length > 0) {
        document.getElementById('model-filter').value = modelArray[0];
        currentModel = modelArray[0];
    }

    // Update variants based on selected model
    updateVariantFilter();

    // Create metric type tabs
    createMetricTabs();
}

// Update variant filter based on selected GPU and model
function updateVariantFilter() {
    const gpuFilter = document.getElementById('gpu-filter').value;
    const modelFilter = document.getElementById('model-filter').value;

    const variants = new Set();

    allMetricsData.forEach(run => {
        run.results.forEach(result => {
            if (result.gpu_config === gpuFilter && result.model === modelFilter) {
                variants.add(result.variant);
            }
        });
    });

    const variantArray = Array.from(variants).sort();
    const variantSelect = document.getElementById('variant-filter');
    const currentVariant = variantSelect.value;

    // Clear and repopulate
    variantSelect.innerHTML = '<option value="all">All Variants</option>';
    variantArray.forEach(variant => {
        const opt = document.createElement('option');
        opt.value = variant;
        opt.textContent = variant;
        variantSelect.appendChild(opt);
    });

    // Try to restore previous selection if still valid
    if (variantArray.includes(currentVariant)) {
        variantSelect.value = currentVariant;
    } else {
        variantSelect.value = 'all';
    }
}

function populateSelect(selectId, options) {
    const select = document.getElementById(selectId);
    options.forEach(option => {
        const opt = document.createElement('option');
        opt.value = option;
        opt.textContent = option;
        select.appendChild(opt);
    });
}

function populateSelectNoAll(selectId, options) {
    const select = document.getElementById(selectId);
    // Remove the "all" option if present
    while (select.options.length > 0) {
        select.remove(0);
    }
    options.forEach(option => {
        const opt = document.createElement('option');
        opt.value = option;
        opt.textContent = option;
        select.appendChild(opt);
    });
}

function createMetricTabs() {
    const tabsContainer = document.getElementById('metric-tabs');
    tabsContainer.innerHTML = '';

    Object.entries(metricTypes).forEach(([key, metric], index) => {
        const tab = document.createElement('div');
        tab.className = index === 0 ? 'tab active' : 'tab';
        tab.textContent = metric.label;
        tab.dataset.metric = key;
        tab.onclick = () => selectMetricTab(key, tab);
        tabsContainer.appendChild(tab);
    });
}

function selectMetricTab(metricKey, tabElement) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tabElement.classList.add('active');
    currentMetricType = metricKey;

    // Update chart title
    const metric = metricTypes[metricKey];
    document.getElementById('metric-title').textContent = `${metric.label} (${metric.unit})`;

    updateCharts();
}

// Handle model filter dropdown change
function handleModelFilterChange(model) {
    currentModel = model;
    // Update variant filter based on new model selection
    updateVariantFilter();
    updateCharts();
}

// Handle GPU filter change
function handleGpuFilterChange() {
    // Update variant filter based on new GPU selection
    updateVariantFilter();
    updateCharts();
}

// Update summary stats
function updateStats() {
    const statsRow = document.getElementById('stats-row');
    const latestRun = allMetricsData[0];

    if (!latestRun) {
        statsRow.innerHTML = '';
        const noDataDiv = document.createElement('div');
        noDataDiv.className = 'no-data';
        noDataDiv.textContent = 'No data available';
        statsRow.appendChild(noDataDiv);
        return;
    }

    const totalModels = new Set(latestRun.results.map(r => r.model)).size;
    const totalBenchmarks = latestRun.results.reduce((sum, r) => sum + r.benchmarks.length, 0);

    // Calculate best throughput
    let maxThroughput = 0;
    let maxThroughputModel = '';
    latestRun.results.forEach(result => {
        result.benchmarks.forEach(bench => {
            if (bench.overall_throughput > maxThroughput) {
                maxThroughput = bench.overall_throughput;
                maxThroughputModel = result.model.split('/').pop();
            }
        });
    });

    statsRow.innerHTML = ''; // Clear previous stats

    const addStat = (label, value, change) => {
        const card = document.createElement('div');
        card.className = 'stat-card';
        const labelEl = document.createElement('div');
        labelEl.className = 'label';
        labelEl.textContent = label;
        const valueEl = document.createElement('div');
        valueEl.className = 'value';
        valueEl.textContent = value;
        card.appendChild(labelEl);
        card.appendChild(valueEl);
        if (change) {
            const changeEl = document.createElement('div');
            changeEl.className = 'change';
            changeEl.textContent = change;
            card.appendChild(changeEl);
        }
        statsRow.appendChild(card);
    };

    addStat('Total Runs', allMetricsData.length);
    addStat('Models Tested', totalModels);
    addStat('Benchmarks', totalBenchmarks);
    addStat('Peak Throughput', formatNumber(maxThroughput), maxThroughputModel);
}

// Update charts based on current filters and selected metric type
function updateCharts() {
    const gpuFilter = document.getElementById('gpu-filter').value;
    const modelFilter = currentModel;
    const variantFilter = document.getElementById('variant-filter').value;
    const batchFilter = document.getElementById('batch-filter').value;

    // Prepare data for charts - grouped by batch size
    const chartDataByBatch = prepareChartDataByBatch(gpuFilter, modelFilter, variantFilter, batchFilter);

    // Update chart for the selected metric type
    updateMetricChart(chartDataByBatch, currentMetricType);
}

function prepareChartData(gpuFilter, modelFilter, variantFilter, batchFilter) {
    const seriesMap = new Map();

    allMetricsData.forEach(run => {
        const runDate = new Date(run.run_date);

        run.results.forEach(result => {
            // Apply filters
            if (result.gpu_config !== gpuFilter) return;
            if (result.model !== modelFilter) return;
            if (variantFilter !== 'all' && result.variant !== variantFilter) return;

            result.benchmarks.forEach(bench => {
                if (batchFilter !== 'all' && bench.batch_size !== parseInt(batchFilter)) return;

                const seriesKey = `${result.model.split('/').pop()} (${result.variant}, BS=${bench.batch_size})`;

                if (!seriesMap.has(seriesKey)) {
                    seriesMap.set(seriesKey, {
                        label: seriesKey,
                        data: [],
                        model: result.model,
                        variant: result.variant,
                        batchSize: bench.batch_size
                    });
                }

                seriesMap.get(seriesKey).data.push({
                    x: runDate,
                    throughput: bench.overall_throughput,
                    latency: bench.latency_ms,
                    ttft: bench.ttft_ms,
                    inputThroughput: bench.input_throughput,
                    runId: run.run_id
                });
            });
        });
    });

    // Sort data points by date
    seriesMap.forEach(series => {
        series.data.sort((a, b) => a.x - b.x);
    });

    return Array.from(seriesMap.values());
}

// Prepare chart data grouped by batch size - each batch size is a separate series
function prepareChartDataByBatch(gpuFilter, modelFilter, variantFilter, batchFilter) {
    const batchDataMap = new Map(); // batch_size -> Map of variant -> data

    allMetricsData.forEach(run => {
        const runDate = new Date(run.run_date);

        run.results.forEach(result => {
            // Apply filters - GPU and Model are required (no "all" option)
            if (result.gpu_config !== gpuFilter) return;
            if (result.model !== modelFilter) return;
            if (variantFilter !== 'all' && result.variant !== variantFilter) return;

            result.benchmarks.forEach(bench => {
                if (batchFilter !== 'all' && bench.batch_size !== parseInt(batchFilter)) return;

                const batchSize = bench.batch_size;
                const seriesKey = result.variant;

                if (!batchDataMap.has(batchSize)) {
                    batchDataMap.set(batchSize, new Map());
                }

                const variantMap = batchDataMap.get(batchSize);
                if (!variantMap.has(seriesKey)) {
                    variantMap.set(seriesKey, {
                        label: seriesKey,
                        data: [],
                        model: result.model,
                        variant: result.variant,
                        batchSize: batchSize
                    });
                }

                variantMap.get(seriesKey).data.push({
                    x: runDate,
                    throughput: bench.overall_throughput,
                    latency: bench.latency_ms,
                    ttft: bench.ttft_ms,
                    inputThroughput: bench.input_throughput,
                    runId: run.run_id
                });
            });
        });
    });

    // Sort data points by date and convert to array format
    const result = {};
    batchDataMap.forEach((variantMap, batchSize) => {
        variantMap.forEach(series => {
            series.data.sort((a, b) => a.x - b.x);
        });
        result[batchSize] = Array.from(variantMap.values());
    });

    return result;
}

// Unified chart update function for any metric type
function updateMetricChart(chartDataByBatch, metricType) {
    const container = document.getElementById('charts-container');
    container.innerHTML = '';

    // Destroy existing charts
    activeCharts.forEach(chart => chart.destroy());
    activeCharts = [];

    const metric = metricTypes[metricType];
    const batchSizes = Object.keys(chartDataByBatch).sort((a, b) => parseInt(a) - parseInt(b));

    if (batchSizes.length === 0) {
        container.innerHTML = '<div class="no-data">No data available for the selected filters</div>';
        return;
    }

    batchSizes.forEach(batchSize => {
        const chartData = chartDataByBatch[batchSize];

        const chartWrapper = document.createElement('div');
        chartWrapper.className = 'batch-chart-wrapper';

        const title = document.createElement('div');
        title.className = 'batch-chart-title';
        title.textContent = `Batch Size: ${batchSize}`;
        chartWrapper.appendChild(title);

        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        const canvas = document.createElement('canvas');
        chartContainer.appendChild(canvas);
        chartWrapper.appendChild(chartContainer);
        container.appendChild(chartWrapper);

        const ctx = canvas.getContext('2d');
        const datasets = chartData.map((series, index) => ({
            label: series.label,
            data: series.data.map(d => ({ x: d.x, y: d[metric.field] })),
            borderColor: chartColors[index % chartColors.length],
            backgroundColor: chartColors[index % chartColors.length] + '20',
            tension: 0.1,
            fill: false
        }));

        const chart = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: getChartOptions(metric.unit)
        });
        activeCharts.push(chart);
    });
}

function getChartOptions(yAxisLabel) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false
        },
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    boxWidth: 12,
                    padding: 10,
                    font: { size: 11 }
                }
            },
            tooltip: {
                backgroundColor: '#21262d',
                borderColor: '#30363d',
                borderWidth: 1,
                titleFont: { size: 13 },
                bodyFont: { size: 12 },
                padding: 12
            }
        },
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'day',
                    displayFormats: {
                        day: 'MMM d'
                    }
                },
                grid: {
                    color: '#21262d'
                }
            },
            y: {
                title: {
                    display: true,
                    text: yAxisLabel
                },
                grid: {
                    color: '#21262d'
                }
            }
        }
    };
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Update runs table
function updateRunsTable() {
    const tbody = document.getElementById('runs-table-body');
    tbody.innerHTML = '';

    allMetricsData.slice(0, 10).forEach(run => {
        const models = new Set(run.results.map(r => r.model.split('/').pop()));
        const date = new Date(run.run_date);

        const row = document.createElement('tr');

        // Create cells safely to prevent XSS
        const dateCell = document.createElement('td');
        dateCell.textContent = `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;

        const runIdCell = document.createElement('td');
        const runLink = document.createElement('a');
        runLink.href = `https://github.com/${GITHUB_REPO}/actions/runs/${encodeURIComponent(run.run_id)}`;
        runLink.target = '_blank';
        runLink.className = 'run-link';
        runLink.textContent = run.run_id;
        runIdCell.appendChild(runLink);

        const commitCell = document.createElement('td');
        const commitCode = document.createElement('code');
        commitCode.textContent = run.commit_sha.substring(0, 7);
        commitCell.appendChild(commitCode);

        const branchCell = document.createElement('td');
        branchCell.textContent = run.branch;

        const modelsCell = document.createElement('td');
        Array.from(models).forEach((model, index) => {
            if (index > 0) modelsCell.appendChild(document.createTextNode(' '));
            const badge = document.createElement('span');
            badge.className = 'model-badge';
            badge.textContent = model;
            modelsCell.appendChild(badge);
        });

        row.appendChild(dateCell);
        row.appendChild(runIdCell);
        row.appendChild(commitCell);
        row.appendChild(branchCell);
        row.appendChild(modelsCell);

        tbody.appendChild(row);
    });
}

// Refresh data
async function refreshData() {
    document.getElementById('content').style.display = 'none';
    document.getElementById('loading').style.display = 'flex';
    await init();
}

// Format numbers for display
function formatNumber(num) {
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'k';
    }
    return num.toFixed(1);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);
