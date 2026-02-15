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
    outputThroughput: { label: 'Output Throughput', unit: 'tokens/sec', field: 'outputThroughput' },
    inputThroughput: { label: 'Input Throughput', unit: 'tokens/sec', field: 'inputThroughput' },
    latency: { label: 'Latency', unit: 'ms', field: 'latency' },
    ttft: { label: 'Time to First Token', unit: 'ms', field: 'ttft' },
    accLength: { label: 'Accept Length', unit: 'tokens', field: 'accLength', filterInvalid: true }
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
    const ioLengths = new Set();

    allMetricsData.forEach(run => {
        run.results.forEach(result => {
            gpuConfigs.add(result.gpu_config);
            models.add(result.model);
            // Try new structure first (benchmarks_by_io_len), fall back to flat benchmarks
            if (result.benchmarks_by_io_len) {
                Object.entries(result.benchmarks_by_io_len).forEach(([ioKey, ioData]) => {
                    ioLengths.add(ioKey);
                    ioData.benchmarks.forEach(bench => {
                        batchSizes.add(bench.batch_size);
                    });
                });
            } else if (result.benchmarks) {
                result.benchmarks.forEach(bench => {
                    batchSizes.add(bench.batch_size);
                    if (bench.input_len && bench.output_len) {
                        ioLengths.add(`${bench.input_len}_${bench.output_len}`);
                    }
                });
            }
        });
    });

    // No "all" option for GPU and Model - populate with first value selected
    const gpuArray = Array.from(gpuConfigs).sort();
    const modelArray = Array.from(models).sort();

    populateSelectNoAll('gpu-filter', gpuArray);
    populateSelectNoAll('model-filter', modelArray);
    populateSelect('batch-filter', Array.from(batchSizes).sort((a, b) => a - b));
    populateSelectWithLabels('io-len-filter', sortIoLengths(Array.from(ioLengths)), formatIoLenLabel);

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
    // Update IO length filter based on selected GPU/model
    updateIoLenFilter();

    // Create metric type tabs
    createMetricTabs();
}

// Format input/output length key for display
function formatIoLenLabel(ioKey) {
    if (!ioKey) return 'Unknown';
    const parts = ioKey.split('_');
    if (parts.length === 2) {
        return `In: ${parts[0]}, Out: ${parts[1]}`;
    }
    return ioKey;
}

// Sort IO length keys numerically (by input length, then output length)
function sortIoLengths(ioLengths) {
    return ioLengths.filter(key => key && key.includes('_')).sort((a, b) => {
        const [aIn, aOut] = a.split('_').map(Number);
        const [bIn, bOut] = b.split('_').map(Number);
        if (isNaN(aIn) || isNaN(bIn)) return 0;
        return (aIn - bIn) || (aOut - bOut);
    });
}

// Populate select with custom label formatting
function populateSelectWithLabels(selectId, options, labelFormatter) {
    const select = document.getElementById(selectId);
    options.forEach(option => {
        const opt = document.createElement('option');
        opt.value = option;
        opt.textContent = labelFormatter ? labelFormatter(option) : option;
        select.appendChild(opt);
    });
}

// Update IO length filter based on selected GPU and model
function updateIoLenFilter() {
    const gpuFilterEl = document.getElementById('gpu-filter');
    const modelFilterEl = document.getElementById('model-filter');
    const ioLenSelect = document.getElementById('io-len-filter');
    if (!gpuFilterEl || !modelFilterEl || !ioLenSelect) return;

    const gpuFilter = gpuFilterEl.value;
    const modelFilter = modelFilterEl.value;

    const ioLengths = new Set();

    allMetricsData.forEach(run => {
        run.results.forEach(result => {
            if (result.gpu_config === gpuFilter && result.model === modelFilter) {
                if (result.benchmarks_by_io_len) {
                    Object.keys(result.benchmarks_by_io_len).forEach(ioKey => {
                        ioLengths.add(ioKey);
                    });
                } else if (result.benchmarks) {
                    result.benchmarks.forEach(bench => {
                        if (bench.input_len && bench.output_len) {
                            ioLengths.add(`${bench.input_len}_${bench.output_len}`);
                        }
                    });
                }
            }
        });
    });

    const ioLenArray = sortIoLengths(Array.from(ioLengths));
    const currentIoLen = ioLenSelect.value;

    // Clear and repopulate
    ioLenSelect.innerHTML = '<option value="all">All Lengths</option>';
    ioLenArray.forEach(ioLen => {
        const opt = document.createElement('option');
        opt.value = ioLen;
        opt.textContent = formatIoLenLabel(ioLen);
        ioLenSelect.appendChild(opt);
    });

    // Try to restore previous selection if still valid
    if (ioLenArray.includes(currentIoLen)) {
        ioLenSelect.value = currentIoLen;
    } else {
        ioLenSelect.value = 'all';
    }
}

// Update variant filter based on selected GPU and model
function updateVariantFilter() {
    const gpuFilter = document.getElementById('gpu-filter').value;
    const modelFilter = document.getElementById('model-filter').value;

    const variants = new Set();

    allMetricsData.forEach(run => {
        run.results.forEach(result => {
            if (result.gpu_config === gpuFilter && result.model === modelFilter) {
                // Use 'default' for null/undefined variants
                variants.add(result.variant || 'default');
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
    // Update IO length filter based on new model selection
    updateIoLenFilter();
    updateCharts();
}

// Handle GPU filter change
function handleGpuFilterChange() {
    // Update variant filter based on new GPU selection
    updateVariantFilter();
    // Update IO length filter based on new GPU selection
    updateIoLenFilter();
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
    const totalBenchmarks = latestRun.results.reduce((sum, r) => {
        // Count benchmarks from either structure
        if (r.benchmarks_by_io_len) {
            return sum + Object.values(r.benchmarks_by_io_len).reduce(
                (ioSum, ioData) => ioSum + ioData.benchmarks.length, 0
            );
        }
        return sum + (r.benchmarks ? r.benchmarks.length : 0);
    }, 0);

    statsRow.innerHTML = ''; // Clear previous stats

    const addStat = (label, value) => {
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
        statsRow.appendChild(card);
    };

    addStat('Total Runs', allMetricsData.length);
    addStat('Models Tested', totalModels);
    addStat('Benchmarks', totalBenchmarks);
}

// Update charts based on current filters and selected metric type
function updateCharts() {
    const gpuFilter = document.getElementById('gpu-filter').value;
    const modelFilter = currentModel;
    const variantFilter = document.getElementById('variant-filter').value;
    const ioLenFilter = document.getElementById('io-len-filter').value;
    const batchFilter = document.getElementById('batch-filter').value;

    // Prepare data for charts - grouped by batch size
    const chartDataByBatch = prepareChartDataByBatch(gpuFilter, modelFilter, variantFilter, ioLenFilter, batchFilter);

    // Update chart for the selected metric type
    updateMetricChart(chartDataByBatch, currentMetricType);
}

function prepareChartData(gpuFilter, modelFilter, variantFilter, ioLenFilter, batchFilter) {
    const seriesMap = new Map();

    allMetricsData.forEach(run => {
        const runDate = new Date(run.run_date);

        run.results.forEach(result => {
            // Apply filters
            if (result.gpu_config !== gpuFilter) return;
            if (result.model !== modelFilter) return;
            if (variantFilter !== 'all' && result.variant !== variantFilter) return;

            // Helper function to process a benchmark entry
            const processBenchmark = (bench, ioKey) => {
                if (batchFilter !== 'all' && bench.batch_size !== parseInt(batchFilter)) return;

                const ioLabel = ioKey ? `, ${formatIoLenLabel(ioKey)}` : '';
                const seriesKey = `${result.model.split('/').pop()} (${result.variant}, BS=${bench.batch_size}${ioLabel})`;

                if (!seriesMap.has(seriesKey)) {
                    seriesMap.set(seriesKey, {
                        label: seriesKey,
                        data: [],
                        model: result.model,
                        variant: result.variant,
                        batchSize: bench.batch_size,
                        ioKey: ioKey
                    });
                }

                seriesMap.get(seriesKey).data.push({
                    x: runDate,
                    throughput: bench.overall_throughput,
                    outputThroughput: bench.output_throughput,
                    latency: bench.latency_ms,
                    ttft: bench.ttft_ms,
                    inputThroughput: bench.input_throughput,
                    accLength: bench.acc_length,
                    runId: run.run_id
                });
            };

            // Use benchmarks_by_io_len if available
            if (result.benchmarks_by_io_len) {
                Object.entries(result.benchmarks_by_io_len).forEach(([ioKey, ioData]) => {
                    if (ioLenFilter !== 'all' && ioKey !== ioLenFilter) return;
                    ioData.benchmarks.forEach(bench => processBenchmark(bench, ioKey));
                });
            } else if (result.benchmarks) {
                result.benchmarks.forEach(bench => {
                    const benchIoKey = bench.input_len && bench.output_len
                        ? `${bench.input_len}_${bench.output_len}`
                        : null;
                    if (ioLenFilter !== 'all' && benchIoKey !== ioLenFilter) return;
                    processBenchmark(bench, benchIoKey);
                });
            }
        });
    });

    // Sort data points by date
    seriesMap.forEach(series => {
        series.data.sort((a, b) => a.x - b.x);
    });

    return Array.from(seriesMap.values());
}

// Prepare chart data grouped by batch size - each batch size is a separate series
function prepareChartDataByBatch(gpuFilter, modelFilter, variantFilter, ioLenFilter, batchFilter) {
    const batchDataMap = new Map(); // batch_size -> Map of variant -> data

    allMetricsData.forEach(run => {
        const runDate = new Date(run.run_date);

        run.results.forEach(result => {
            // Apply filters - GPU and Model are required (no "all" option)
            if (result.gpu_config !== gpuFilter) return;
            if (result.model !== modelFilter) return;
            if (variantFilter !== 'all' && result.variant !== variantFilter) return;

            // Use benchmarks_by_io_len if available, otherwise fall back to flat benchmarks
            if (result.benchmarks_by_io_len) {
                Object.entries(result.benchmarks_by_io_len).forEach(([ioKey, ioData]) => {
                    // Apply IO length filter
                    if (ioLenFilter !== 'all' && ioKey !== ioLenFilter) return;

                    ioData.benchmarks.forEach(bench => {
                        if (batchFilter !== 'all' && bench.batch_size !== parseInt(batchFilter)) return;

                        const batchSize = bench.batch_size;
                        const variantLabel = result.variant || 'default';
                        // Include IO length in series key when showing all lengths
                        const seriesKey = ioLenFilter === 'all'
                            ? `${variantLabel} (${formatIoLenLabel(ioKey)})`
                            : variantLabel;

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
                                batchSize: batchSize,
                                ioKey: ioKey
                            });
                        }

                        variantMap.get(seriesKey).data.push({
                            x: runDate,
                            throughput: bench.overall_throughput,
                            outputThroughput: bench.output_throughput,
                            latency: bench.latency_ms,
                            ttft: bench.ttft_ms,
                            inputThroughput: bench.input_throughput,
                            accLength: bench.acc_length,
                            runId: run.run_id
                        });
                    });
                });
            } else if (result.benchmarks) {
                // Fall back to flat benchmarks for backward compatibility
                result.benchmarks.forEach(bench => {
                    // Apply IO length filter using flat structure
                    const benchIoKey = bench.input_len && bench.output_len
                        ? `${bench.input_len}_${bench.output_len}`
                        : null;
                    if (ioLenFilter !== 'all' && benchIoKey !== ioLenFilter) return;
                    if (batchFilter !== 'all' && bench.batch_size !== parseInt(batchFilter)) return;

                    const batchSize = bench.batch_size;
                    const variantLabel = result.variant || 'default';
                    // Include IO length in series key when showing all lengths
                    const seriesKey = ioLenFilter === 'all' && benchIoKey
                        ? `${variantLabel} (${formatIoLenLabel(benchIoKey)})`
                        : variantLabel;

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
                            batchSize: batchSize,
                            ioKey: benchIoKey
                        });
                    }

                    variantMap.get(seriesKey).data.push({
                        x: runDate,
                        throughput: bench.overall_throughput,
                        outputThroughput: bench.output_throughput,
                        latency: bench.latency_ms,
                        ttft: bench.ttft_ms,
                        inputThroughput: bench.input_throughput,
                        accLength: bench.acc_length,
                        runId: run.run_id
                    });
                });
            }
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

    let hasAnyData = false;

    batchSizes.forEach(batchSize => {
        const chartData = chartDataByBatch[batchSize];

        const ctx_datasets = chartData.map((series, index) => {
            // Filter data points - for metrics like accLength, exclude invalid values (-1 or null)
            let dataPoints = series.data.map(d => ({ x: d.x, y: d[metric.field] }));
            if (metric.filterInvalid) {
                dataPoints = dataPoints.filter(d => d.y != null && d.y !== -1 && d.y > 0);
            }
            return {
                label: series.label,
                data: dataPoints,
                borderColor: chartColors[index % chartColors.length],
                backgroundColor: chartColors[index % chartColors.length] + '20',
                tension: 0.1,
                fill: false
            };
        }).filter(dataset => dataset.data.length > 0); // Remove empty datasets

        // Skip this batch size if no valid data
        if (ctx_datasets.length === 0) {
            return;
        }

        hasAnyData = true;

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

        const chart = new Chart(ctx, {
            type: 'line',
            data: { datasets: ctx_datasets },
            options: getChartOptions(metric.unit)
        });
        activeCharts.push(chart);
    });

    // Show message if no valid data for this metric
    if (!hasAnyData) {
        container.innerHTML = `<div class="no-data">No valid ${metric.label.toLowerCase()} data available for the selected filters</div>`;
    }
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
