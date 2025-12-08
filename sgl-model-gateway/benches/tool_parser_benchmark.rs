//! Comprehensive tool parser benchmark for measuring performance under various scenarios
//!
//! This benchmark tests:
//! - Single parser parsing performance
//! - Registry creation overhead
//! - Concurrent parsing with shared parsers
//! - Streaming vs complete parsing
//! - Different model formats (JSON, Mistral, Qwen, Pythonic, etc.)

use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use criterion::{black_box, criterion_group, BenchmarkId, Criterion, Throughput};
use serde_json::json;
use sgl_model_gateway::{
    protocols::common::{Function, Tool},
    tool_parser::{JsonParser, ParserFactory as ToolParserFactory, ToolParser},
};
use tokio::runtime::Runtime;

// Test data for different parser formats - realistic complex examples
const JSON_SIMPLE: &str = r#"{"name": "code_interpreter", "arguments": "{\"language\": \"python\", \"code\": \"import numpy as np\\nimport matplotlib.pyplot as plt\\n\\n# Generate sample data\\nx = np.linspace(0, 10, 100)\\ny = np.sin(x) * np.exp(-x/10)\\n\\n# Create the plot\\nplt.figure(figsize=(10, 6))\\nplt.plot(x, y, 'b-', linewidth=2)\\nplt.grid(True)\\nplt.xlabel('Time (s)')\\nplt.ylabel('Amplitude')\\nplt.title('Damped Oscillation')\\nplt.show()\"}"}"#;

const JSON_ARRAY: &str = r#"[{"name": "web_search", "arguments": "{\"query\": \"latest developments in quantum computing 2024\", \"num_results\": 10, \"search_type\": \"news\", \"date_range\": \"2024-01-01:2024-12-31\", \"exclude_domains\": [\"reddit.com\", \"facebook.com\"], \"language\": \"en\"}"}, {"name": "analyze_sentiment", "arguments": "{\"text\": \"The breakthrough in quantum error correction represents a significant milestone. Researchers are optimistic about practical applications within the next decade.\", \"granularity\": \"sentence\", \"aspects\": [\"technology\", \"timeline\", \"impact\"], \"confidence_threshold\": 0.85}"}, {"name": "create_summary", "arguments": "{\"content_ids\": [\"doc_1234\", \"doc_5678\", \"doc_9012\"], \"max_length\": 500, \"style\": \"technical\", \"include_citations\": true}"}]"#;

const JSON_WITH_PARAMS: &str = r#"{"name": "database_query", "parameters": {"connection_string": "postgresql://user:pass@localhost:5432/analytics", "query": "SELECT customer_id, COUNT(*) as order_count, SUM(total_amount) as lifetime_value, AVG(order_amount) as avg_order_value FROM orders WHERE created_at >= '2024-01-01' GROUP BY customer_id HAVING COUNT(*) > 5 ORDER BY lifetime_value DESC LIMIT 100", "timeout_ms": 30000, "read_consistency": "strong", "partition_key": "customer_id"}}"#;

const MISTRAL_FORMAT: &str = r#"I'll help you analyze the sales data and create visualizations. Let me start by querying the database and then create some charts.

[TOOL_CALLS] [{"name": "sql_query", "arguments": {"database": "sales_analytics", "query": "WITH monthly_sales AS (SELECT DATE_TRUNC('month', order_date) as month, SUM(total_amount) as revenue, COUNT(DISTINCT customer_id) as unique_customers, COUNT(*) as total_orders FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL '12 months' GROUP BY DATE_TRUNC('month', order_date)) SELECT month, revenue, unique_customers, total_orders, LAG(revenue) OVER (ORDER BY month) as prev_month_revenue, (revenue - LAG(revenue) OVER (ORDER BY month)) / LAG(revenue) OVER (ORDER BY month) * 100 as growth_rate FROM monthly_sales ORDER BY month DESC", "format": "json", "timeout": 60000}}]

Based on the query results, I can see interesting trends in your sales data."#;

const MISTRAL_MULTI: &str = r#"Let me help you with a comprehensive analysis of your application's performance.

[TOOL_CALLS] [{"name": "get_metrics", "arguments": {"service": "api-gateway", "metrics": ["latency_p50", "latency_p95", "latency_p99", "error_rate", "requests_per_second"], "start_time": "2024-01-01T00:00:00Z", "end_time": "2024-01-01T23:59:59Z", "aggregation": "5m", "filters": {"environment": "production", "region": "us-east-1"}}}, {"name": "analyze_logs", "arguments": {"log_group": "/aws/lambda/process-orders", "query": "fields @timestamp, @message, @requestId, duration | filter @message like /ERROR/ | stats count() by bin(@timestamp, 5m) as time_window", "start_time": "2024-01-01T00:00:00Z", "end_time": "2024-01-01T23:59:59Z", "limit": 1000}}, {"name": "get_traces", "arguments": {"service": "order-processing", "operation": "ProcessOrder", "min_duration_ms": 1000, "max_results": 100, "include_downstream": true}}]

Now let me create a comprehensive report based on this data."#;

const QWEN_FORMAT: &str = r#"Let me search for information about machine learning frameworks and their performance benchmarks.

<tool_call>
{"name": "academic_search", "arguments": {"query": "transformer architecture optimization techniques GPU inference latency reduction", "databases": ["arxiv", "ieee", "acm"], "year_range": [2020, 2024], "citation_count_min": 10, "include_code": true, "page_size": 25, "sort_by": "relevance"}}
</tool_call>

I found several interesting papers on optimization techniques."#;

const QWEN_MULTI: &str = r#"I'll help you set up a complete data pipeline for your analytics system.

<tool_call>
{"name": "create_data_pipeline", "arguments": {"name": "customer_analytics_etl", "source": {"type": "kafka", "config": {"bootstrap_servers": "kafka1:9092,kafka2:9092", "topic": "customer_events", "consumer_group": "analytics_consumer", "auto_offset_reset": "earliest"}}, "transformations": [{"type": "filter", "condition": "event_type IN ('purchase', 'signup', 'churn')"}, {"type": "aggregate", "window": "1h", "group_by": ["customer_id", "event_type"], "metrics": ["count", "sum(amount)"]}], "destination": {"type": "bigquery", "dataset": "analytics", "table": "customer_metrics", "write_mode": "append"}}}
</tool_call>
<tool_call>
{"name": "schedule_job", "arguments": {"job_id": "customer_analytics_etl", "schedule": "0 */4 * * *", "timezone": "UTC", "retry_policy": {"max_attempts": 3, "backoff_multiplier": 2, "max_backoff": 3600}, "notifications": {"on_failure": ["ops-team@company.com"], "on_success": null}, "monitoring": {"sla_minutes": 30, "alert_threshold": 0.95}}}
</tool_call>
<tool_call>
{"name": "create_dashboard", "arguments": {"title": "Customer Analytics Dashboard", "widgets": [{"type": "time_series", "title": "Customer Acquisition", "query": "SELECT DATE(timestamp) as date, COUNT(DISTINCT customer_id) as new_customers FROM analytics.customer_metrics WHERE event_type = 'signup' GROUP BY date ORDER BY date", "visualization": "line"}, {"type": "metric", "title": "Total Revenue", "query": "SELECT SUM(amount) as total FROM analytics.customer_metrics WHERE event_type = 'purchase' AND DATE(timestamp) = CURRENT_DATE()", "format": "currency"}, {"type": "table", "title": "Top Customers", "query": "SELECT customer_id, COUNT(*) as purchases, SUM(amount) as total_spent FROM analytics.customer_metrics WHERE event_type = 'purchase' GROUP BY customer_id ORDER BY total_spent DESC LIMIT 10"}], "refresh_interval": 300}}
</tool_call>

The data pipeline has been configured and the dashboard is ready."#;

const LLAMA_FORMAT: &str = r#"<|python_tag|>{"name": "execute_code", "arguments": "{\"code\": \"import pandas as pd\\nimport numpy as np\\nfrom sklearn.model_selection import train_test_split\\nfrom sklearn.ensemble import RandomForestClassifier\\nfrom sklearn.metrics import classification_report, confusion_matrix\\nimport joblib\\n\\n# Load and preprocess data\\ndf = pd.read_csv('/data/customer_churn.csv')\\nprint(f'Dataset shape: {df.shape}')\\nprint(f'Missing values: {df.isnull().sum().sum()}')\\n\\n# Feature engineering\\ndf['tenure_months'] = pd.to_datetime('today') - pd.to_datetime(df['signup_date'])\\ndf['tenure_months'] = df['tenure_months'].dt.days // 30\\ndf['avg_monthly_spend'] = df['total_spend'] / df['tenure_months'].clip(lower=1)\\n\\n# Prepare features and target\\nfeature_cols = ['tenure_months', 'avg_monthly_spend', 'support_tickets', 'product_usage_hours', 'feature_adoption_score']\\nX = df[feature_cols]\\ny = df['churned']\\n\\n# Split and train\\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\\nrf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)\\nrf_model.fit(X_train, y_train)\\n\\n# Evaluate\\ny_pred = rf_model.predict(X_test)\\nprint('Classification Report:')\\nprint(classification_report(y_test, y_pred))\\n\\n# Save model\\njoblib.dump(rf_model, '/models/churn_predictor_v1.pkl')\\nprint('Model saved successfully!')\"}"}"#;

const PYTHONIC_FORMAT: &str = r#"[retrieve_context(query="How do transformer models handle long-range dependencies in natural language processing tasks?", index="ml_knowledge_base", top_k=5, similarity_threshold=0.75, rerank=True, include_metadata=True, filters={"category": "deep_learning", "year": {"$gte": 2020}})]"#;

const PYTHONIC_MULTI: &str = r#"[fetch_api_data(endpoint="https://api.weather.com/v1/forecast", params={"lat": 37.7749, "lon": -122.4194, "units": "metric", "days": 7, "hourly": True}, headers={"API-Key": "${WEATHER_API_KEY}"}, timeout=30, retry_count=3), process_weather_data(data="${response}", extract_fields=["temperature", "humidity", "precipitation", "wind_speed", "uv_index"], aggregation="daily", calculate_trends=True), generate_report(data="${processed_data}", template="weather_forecast", format="html", include_charts=True, language="en")]"#;

const DEEPSEEK_FORMAT: &str = r#"I'll analyze your codebase and identify potential security vulnerabilities.

🤔[{"name": "scan_repository", "arguments": {"repo_path": "/src/application", "scan_types": ["security", "dependencies", "secrets", "code_quality"], "file_patterns": ["*.py", "*.js", "*.java", "*.go"], "exclude_dirs": ["node_modules", ".git", "vendor", "build"], "vulnerability_databases": ["cve", "nvd", "ghsa"], "min_severity": "medium", "check_dependencies": true, "deep_scan": true, "parallel_workers": 8}}]

Let me examine the scan results and provide recommendations."#;

const KIMIK2_FORMAT: &str = r#"⍼validate_and_deploy⍁{"deployment_config": {"application": "payment-service", "version": "2.3.1", "environment": "staging", "region": "us-west-2", "deployment_strategy": "blue_green", "health_check": {"endpoint": "/health", "interval": 30, "timeout": 5, "healthy_threshold": 2, "unhealthy_threshold": 3}, "rollback_on_failure": true, "canary_config": {"percentage": 10, "duration_minutes": 30, "metrics": ["error_rate", "latency_p99", "success_rate"], "thresholds": {"error_rate": 0.01, "latency_p99": 500, "success_rate": 0.99}}, "pre_deployment_hooks": ["run_tests", "security_scan", "backup_database"], "post_deployment_hooks": ["smoke_tests", "notify_team", "update_documentation"]}}"#;

const GLM4_FORMAT: &str = r#"<tool>
analyze_customer_behavior
<parameter>dataset_id=customer_interactions_2024</parameter>
<parameter>analysis_type=cohort_retention</parameter>
<parameter>cohort_definition=signup_month</parameter>
<parameter>retention_periods=[1, 7, 14, 30, 60, 90, 180, 365]</parameter>
<parameter>segment_by=["acquisition_channel", "pricing_tier", "industry", "company_size"]</parameter>
<parameter>metrics=["active_users", "revenue", "feature_usage", "engagement_score"]</parameter>
<parameter>statistical_tests=["chi_square", "anova", "trend_analysis"]</parameter>
<parameter>visualization_types=["heatmap", "line_chart", "funnel", "sankey"]</parameter>
<parameter>export_format=dashboard</parameter>
<parameter>confidence_level=0.95</parameter>
</tool>"#;

const STEP3_FORMAT: &str = r#"<step.tML version="0.1">
<call>
<name>orchestrate_ml_pipeline</name>
<parameters>
<parameter name="pipeline_name">fraud_detection_model_v3</parameter>
<parameter name="data_source">s3://ml-datasets/transactions/2024/</parameter>
<parameter name="preprocessing_steps">
  <step order="1" type="clean">{"remove_duplicates": true, "handle_missing": "interpolate", "outlier_method": "isolation_forest"}</step>
  <step order="2" type="feature_engineering">{"create_ratios": true, "time_features": ["hour", "day_of_week", "month"], "aggregations": ["mean", "std", "max"]}</step>
  <step order="3" type="normalize">{"method": "robust_scaler", "clip_outliers": true}</step>
</parameter>
<parameter name="model_config">{"algorithm": "xgboost", "hyperparameters": {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.01, "subsample": 0.8}, "cross_validation": {"method": "stratified_kfold", "n_splits": 5}}</parameter>
<parameter name="evaluation_metrics">["auc_roc", "precision_recall", "f1", "confusion_matrix"]</parameter>
<parameter name="deployment_target">sagemaker_endpoint</parameter>
<parameter name="monitoring_config">{"drift_detection": true, "performance_threshold": 0.92, "alert_emails": ["ml-team@company.com"]}</parameter>
</parameters>
</call>
</step.tML>"#;

const GPT_OSS_FORMAT: &str = r#"<Channel.vector_search>{"collection": "technical_documentation", "query_embedding": [0.0234, -0.1456, 0.0891, 0.2341, -0.0567, 0.1234, 0.0456, -0.0789, 0.1567, 0.0234, -0.1123, 0.0678, 0.2345, -0.0456, 0.0891, 0.1234, -0.0567, 0.0789, 0.1456, -0.0234, 0.0891, 0.1567, -0.0678, 0.0345, 0.1234, -0.0456, 0.0789, 0.1891, -0.0234, 0.0567, 0.1345, -0.0891], "top_k": 10, "similarity_metric": "cosine", "filters": {"language": "en", "last_updated": {"$gte": "2023-01-01"}, "categories": {"$in": ["api", "sdk", "integration"]}}, "include_metadata": true, "rerank_with_cross_encoder": true}</Channel.vector_search>"#;

// Create test tools for parsers that need them
fn create_test_tools() -> Vec<Tool> {
    vec![
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "search".to_string(),
                description: Some("Search for information".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "number"}
                    }
                }),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "code_interpreter".to_string(),
                description: Some("Execute code".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "language": {"type": "string"},
                        "code": {"type": "string"}
                    }
                }),
                strict: None,
            },
        },
    ]
}

// Large test data for stress testing
fn generate_large_json(num_tools: usize) -> String {
    let mut tools = Vec::new();
    for i in 0..num_tools {
        tools.push(format!(
            r#"{{"name": "tool_{}", "arguments": {{"param1": "value{}", "param2": {}, "param3": true}}}}"#,
            i, i, i
        ));
    }
    format!("[{}]", tools.join(", "))
}

// Global results storage
lazy_static::lazy_static! {
    static ref BENCHMARK_RESULTS: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());
}

fn add_result(category: &str, result: String) {
    let mut results = BENCHMARK_RESULTS.lock().unwrap();
    let index = results.len();
    results.insert(format!("{:03}_{}", index, category), result);
}

fn bench_registry_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("registry_creation");

    let printed = Arc::new(AtomicBool::new(false));
    group.bench_function("new_registry", |b| {
        let printed_clone = printed.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let registry = black_box(ToolParserFactory::new());
                // Force evaluation to prevent optimization
                black_box(registry.list_parsers());
            }
            let duration = start.elapsed();

            if !printed_clone.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let time_per_op = duration.as_micros() as f64 / iters as f64;

                let result = format!(
                    "{:<25} | {:>12.0} | {:>12.1}µs | {:>15}",
                    "Registry Creation", ops_per_sec, time_per_op, "N/A"
                );
                add_result("registry", result);

                printed_clone.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

fn bench_parser_lookup(c: &mut Criterion) {
    let registry = Arc::new(ToolParserFactory::new());
    let models = vec![
        "gpt-4",
        "mistral-large",
        "qwen-72b",
        "llama-3.2",
        "deepseek-v3",
        "unknown-model",
    ];

    let mut group = c.benchmark_group("parser_lookup");

    for model in models {
        let printed = Arc::new(AtomicBool::new(false));
        let registry_clone = registry.clone();

        group.bench_function(model, |b| {
            let printed_clone = printed.clone();
            let registry = registry_clone.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let parser = black_box(registry.get_parser(model));
                    // Force evaluation
                    black_box(parser.is_some());
                }
                let duration = start.elapsed();

                if !printed_clone.load(Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let time_per_op = duration.as_nanos() as f64 / iters as f64;

                    let result = format!(
                        "{:<25} | {:>12.0} | {:>12.1}ns | {:>15}",
                        format!("Lookup {}", model),
                        ops_per_sec,
                        time_per_op,
                        if registry.get_parser(model).is_some() {
                            "Found"
                        } else {
                            "Fallback"
                        }
                    );
                    add_result("lookup", result);

                    printed_clone.store(true, Ordering::Relaxed);
                }

                duration
            });
        });
    }

    group.finish();
}

fn bench_complete_parsing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let registry = Arc::new(ToolParserFactory::new());

    let test_cases = vec![
        ("json_simple", "json", JSON_SIMPLE),
        ("json_array", "json", JSON_ARRAY),
        ("json_params", "json", JSON_WITH_PARAMS),
        ("mistral_single", "mistral", MISTRAL_FORMAT),
        ("mistral_multi", "mistral", MISTRAL_MULTI),
        ("qwen_single", "qwen", QWEN_FORMAT),
        ("qwen_multi", "qwen", QWEN_MULTI),
        ("llama", "llama", LLAMA_FORMAT),
        ("pythonic_single", "pythonic", PYTHONIC_FORMAT),
        ("pythonic_multi", "pythonic", PYTHONIC_MULTI),
        ("deepseek", "deepseek", DEEPSEEK_FORMAT),
        ("kimik2", "kimik2", KIMIK2_FORMAT),
        ("glm4", "glm4_moe", GLM4_FORMAT),
        ("step3", "step3", STEP3_FORMAT),
        ("gpt_oss", "gpt_oss", GPT_OSS_FORMAT),
    ];

    let mut group = c.benchmark_group("complete_parsing");

    for (name, parser_name, input) in test_cases {
        let printed = Arc::new(AtomicBool::new(false));
        let registry_clone = registry.clone();
        let input_len = input.len();

        group.throughput(Throughput::Bytes(input_len as u64));
        group.bench_function(name, |b| {
            let printed_clone = printed.clone();
            let registry = registry_clone.clone();
            let rt = rt.handle().clone();

            b.iter_custom(|iters| {
                let parser = registry.get_parser(parser_name).expect("Parser not found");

                let start = Instant::now();
                for _ in 0..iters {
                    let parser = parser.clone();
                    let result = rt.block_on(async { parser.parse_complete(input).await });
                    black_box(result.unwrap());
                }
                let duration = start.elapsed();

                if !printed_clone.load(Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let bytes_per_sec = (iters as f64 * input_len as f64) / duration.as_secs_f64();
                    let time_per_op = duration.as_micros() as f64 / iters as f64;

                    let result = format!(
                        "{:<25} | {:>10} | {:>12.0} | {:>12.0} | {:>10.1}µs",
                        name, input_len, ops_per_sec, bytes_per_sec, time_per_op
                    );
                    add_result("complete", result);

                    printed_clone.store(true, Ordering::Relaxed);
                }

                duration
            });
        });
    }

    group.finish();
}

fn bench_streaming_parsing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Streaming test with chunked input
    let chunks = vec![
        r#"{"na"#,
        r#"me": "sear"#,
        r#"ch", "argu"#,
        r#"ments": {"qu"#,
        r#"ery": "rust prog"#,
        r#"ramming", "li"#,
        r#"mit": 10, "off"#,
        r#"set": 0}"#,
        r#"}"#,
    ];

    let mut group = c.benchmark_group("streaming_parsing");

    let printed = Arc::new(AtomicBool::new(false));
    group.bench_function("json_streaming", |b| {
        let printed_clone = printed.clone();
        let rt = rt.handle().clone();

        b.iter_custom(|iters| {
            let tools = create_test_tools();

            let start = Instant::now();
            for _ in 0..iters {
                let mut parser = JsonParser::new();
                let mut complete_tools = Vec::new();

                rt.block_on(async {
                    for chunk in &chunks {
                        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
                        if !result.calls.is_empty() {
                            complete_tools.extend(result.calls);
                        }
                    }
                });

                black_box(complete_tools);
            }
            let duration = start.elapsed();

            if !printed_clone.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let time_per_op = duration.as_micros() as f64 / iters as f64;
                let chunks_per_sec = (iters as f64 * chunks.len() as f64) / duration.as_secs_f64();

                let result = format!(
                    "{:<25} | {:>10} | {:>12.0} | {:>12.0} | {:>10.1}µs",
                    "JSON Streaming",
                    chunks.len(),
                    ops_per_sec,
                    chunks_per_sec,
                    time_per_op
                );
                add_result("streaming", result);

                printed_clone.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

fn bench_concurrent_parsing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let registry = Arc::new(ToolParserFactory::new());
    let parser = registry.get_parser("json").expect("Parser not found");

    let thread_counts = vec![1, 2, 4, 8, 16, 32];
    let operations_per_thread = 100;

    let mut group = c.benchmark_group("concurrent_parsing");
    group.measurement_time(Duration::from_secs(3));

    for num_threads in thread_counts {
        let printed = Arc::new(AtomicBool::new(false));
        let parser_clone = parser.clone();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |b, &threads| {
                let printed_clone = printed.clone();
                let parser = parser_clone.clone();
                let rt = rt.handle().clone();

                b.iter_custom(|_iters| {
                    let total_operations = Arc::new(AtomicU64::new(0));
                    let total_parsed = Arc::new(AtomicU64::new(0));
                    let start = Instant::now();

                    let handles: Vec<_> = (0..threads)
                        .map(|_thread_id| {
                            let parser = parser.clone();
                            let total_ops = total_operations.clone();
                            let total_p = total_parsed.clone();
                            let rt = rt.clone();

                            thread::spawn(move || {
                                let test_inputs = [JSON_SIMPLE, JSON_ARRAY, JSON_WITH_PARAMS];

                                for i in 0..operations_per_thread {
                                    let input = test_inputs[i % test_inputs.len()];
                                    let result =
                                        rt.block_on(async { parser.parse_complete(input).await });

                                    if let Ok((_normal_text, tools)) = result {
                                        total_p.fetch_add(tools.len() as u64, Ordering::Relaxed);
                                    }
                                }

                                total_ops
                                    .fetch_add(operations_per_thread as u64, Ordering::Relaxed);
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    let duration = start.elapsed();

                    if !printed_clone.load(Ordering::Relaxed) {
                        let total_ops = total_operations.load(Ordering::Relaxed);
                        let total_p = total_parsed.load(Ordering::Relaxed);
                        let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                        let tools_per_sec = total_p as f64 / duration.as_secs_f64();

                        let result = format!(
                            "{:<25} | {:>10} | {:>12.0} | {:>12.0} | {:>10}",
                            format!("{}_threads", threads),
                            total_ops,
                            ops_per_sec,
                            tools_per_sec,
                            threads
                        );
                        add_result("concurrent", result);

                        printed_clone.store(true, Ordering::Relaxed);
                    }

                    duration
                });
            },
        );
    }

    group.finish();
}

fn bench_large_payloads(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let registry = Arc::new(ToolParserFactory::new());
    let parser = registry.get_parser("json").expect("Parser not found");

    let sizes = vec![1, 10, 50, 100, 500];

    let mut group = c.benchmark_group("large_payloads");

    for size in sizes {
        let large_json = generate_large_json(size);
        let input_len = large_json.len();
        let printed = Arc::new(AtomicBool::new(false));
        let parser_clone = parser.clone();

        group.throughput(Throughput::Bytes(input_len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &num_tools| {
            let printed_clone = printed.clone();
            let parser = parser_clone.clone();
            let rt = rt.handle().clone();
            let input = &large_json;

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let parser = parser.clone();
                    let result = rt.block_on(async { parser.parse_complete(input).await });
                    black_box(result.unwrap());
                }
                let duration = start.elapsed();

                if !printed_clone.load(Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let bytes_per_sec = (iters as f64 * input_len as f64) / duration.as_secs_f64();
                    let time_per_op = duration.as_millis() as f64 / iters as f64;

                    let result = format!(
                        "{:<25} | {:>10} | {:>10} | {:>12.0} | {:>12.0} | {:>10.1}ms",
                        format!("{}_tools", num_tools),
                        num_tools,
                        input_len,
                        ops_per_sec,
                        bytes_per_sec,
                        time_per_op
                    );
                    add_result("large", result);

                    printed_clone.store(true, Ordering::Relaxed);
                }

                duration
            });
        });
    }

    group.finish();
}

fn bench_parser_reuse(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("parser_reuse");

    // Benchmark creating new registry each time
    let printed_new = Arc::new(AtomicBool::new(false));
    group.bench_function("new_registry_each_time", |b| {
        let printed_clone = printed_new.clone();
        let rt = rt.handle().clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let registry = ToolParserFactory::new();
                let parser = registry.get_parser("json").unwrap();
                let result = rt.block_on(async { parser.parse_complete(JSON_SIMPLE).await });
                black_box(result.unwrap());
            }
            let duration = start.elapsed();

            if !printed_clone.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let time_per_op = duration.as_micros() as f64 / iters as f64;

                let result = format!(
                    "{:<25} | {:>12.0} | {:>12.1}µs | {:>15}",
                    "New Registry Each Time", ops_per_sec, time_per_op, "Baseline"
                );
                add_result("reuse", result);

                printed_clone.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Benchmark reusing registry
    let printed_reuse = Arc::new(AtomicBool::new(false));
    let shared_registry = Arc::new(ToolParserFactory::new());

    group.bench_function("reuse_registry", |b| {
        let printed_clone = printed_reuse.clone();
        let registry = shared_registry.clone();
        let rt = rt.handle().clone();

        b.iter_custom(|iters| {
            let parser = registry.get_parser("json").unwrap();

            let start = Instant::now();
            for _ in 0..iters {
                let parser = parser.clone();
                let result = rt.block_on(async { parser.parse_complete(JSON_SIMPLE).await });
                black_box(result.unwrap());
            }
            let duration = start.elapsed();

            if !printed_clone.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let time_per_op = duration.as_micros() as f64 / iters as f64;

                let result = format!(
                    "{:<25} | {:>12.0} | {:>12.1}µs | {:>15}",
                    "Reuse Registry", ops_per_sec, time_per_op, "Optimized"
                );
                add_result("reuse", result);

                printed_clone.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Benchmark reusing parser
    let printed_parser = Arc::new(AtomicBool::new(false));
    let shared_parser = shared_registry.get_parser("json").unwrap();

    group.bench_function("reuse_parser", |b| {
        let printed_clone = printed_parser.clone();
        let parser = shared_parser.clone();
        let rt = rt.handle().clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let parser = parser.clone();
                let result = rt.block_on(async { parser.parse_complete(JSON_SIMPLE).await });
                black_box(result.unwrap());
            }
            let duration = start.elapsed();

            if !printed_clone.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let time_per_op = duration.as_micros() as f64 / iters as f64;

                let result = format!(
                    "{:<25} | {:>12.0} | {:>12.1}µs | {:>15}",
                    "Reuse Parser", ops_per_sec, time_per_op, "Best"
                );
                add_result("reuse", result);

                printed_clone.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

fn bench_latency_distribution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let registry = Arc::new(ToolParserFactory::new());

    let test_cases = vec![
        ("json", JSON_SIMPLE),
        ("mistral", MISTRAL_FORMAT),
        ("qwen", QWEN_FORMAT),
        ("pythonic", PYTHONIC_FORMAT),
    ];

    let mut group = c.benchmark_group("latency");

    for (parser_name, input) in test_cases {
        let printed = Arc::new(AtomicBool::new(false));
        let registry_clone = registry.clone();

        group.bench_function(parser_name, |b| {
            let printed_clone = printed.clone();
            let registry = registry_clone.clone();
            let rt = rt.handle().clone();

            b.iter_custom(|iters| {
                let parser = registry.get_parser(parser_name).expect("Parser not found");

                let total_duration = if !printed_clone.load(Ordering::Relaxed) {
                    let mut latencies = Vec::new();

                    // Warm up
                    for _ in 0..100 {
                        let parser = parser.clone();
                        rt.block_on(async { parser.parse_complete(input).await })
                            .unwrap();
                    }

                    // Measure for statistics
                    for _ in 0..1000 {
                        let parser = parser.clone();
                        let start = Instant::now();
                        rt.block_on(async { parser.parse_complete(input).await })
                            .unwrap();
                        let latency = start.elapsed();
                        latencies.push(latency);
                    }

                    latencies.sort();
                    let p50 = latencies[latencies.len() / 2];
                    let p95 = latencies[latencies.len() * 95 / 100];
                    let p99 = latencies[latencies.len() * 99 / 100];
                    let max = latencies.last().unwrap();

                    let result = format!(
                        "{:<25} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10}",
                        parser_name,
                        p50.as_micros() as f64,
                        p95.as_micros() as f64,
                        p99.as_micros() as f64,
                        max.as_micros() as f64,
                        1000
                    );
                    add_result("latency", result);

                    printed_clone.store(true, Ordering::Relaxed);

                    // Return median for consistency
                    p50 * iters as u32
                } else {
                    // Regular benchmark iterations
                    let start = Instant::now();
                    for _ in 0..iters {
                        let parser = parser.clone();
                        rt.block_on(async { parser.parse_complete(input).await })
                            .unwrap();
                    }
                    start.elapsed()
                };

                total_duration
            });
        });
    }

    group.finish();
}

// Print final summary table
fn print_summary() {
    println!("\n{}", "=".repeat(120));
    println!("TOOL PARSER BENCHMARK SUMMARY");
    println!("{}", "=".repeat(120));

    let results = BENCHMARK_RESULTS.lock().unwrap();

    let mut current_category = String::new();
    for (key, value) in results.iter() {
        let category = key.split('_').skip(1).collect::<Vec<_>>().join("_");

        if category != current_category {
            current_category = category.clone();

            // Print section header based on category
            println!("\n{}", "-".repeat(120));
            match category.as_str() {
                "registry" => {
                    println!("REGISTRY OPERATIONS");
                    println!(
                        "{:<25} | {:>12} | {:>12} | {:>15}",
                        "Operation", "Ops/sec", "Time/op", "Notes"
                    );
                }
                "lookup" => {
                    println!("PARSER LOOKUP PERFORMANCE");
                    println!(
                        "{:<25} | {:>12} | {:>12} | {:>15}",
                        "Model", "Lookups/sec", "Time/lookup", "Result"
                    );
                }
                "complete" => {
                    println!("COMPLETE PARSING PERFORMANCE");
                    println!(
                        "{:<25} | {:>10} | {:>12} | {:>12} | {:>12}",
                        "Parser Format", "Size(B)", "Ops/sec", "Bytes/sec", "Time/op"
                    );
                }
                "streaming" => {
                    println!("STREAMING PARSING PERFORMANCE");
                    println!(
                        "{:<25} | {:>10} | {:>12} | {:>12} | {:>12}",
                        "Parser", "Chunks", "Ops/sec", "Chunks/sec", "Time/op"
                    );
                }
                "concurrent" => {
                    println!("CONCURRENT PARSING");
                    println!(
                        "{:<25} | {:>10} | {:>12} | {:>12} | {:>10}",
                        "Configuration", "Total Ops", "Ops/sec", "Tools/sec", "Threads"
                    );
                }
                "large" => {
                    println!("LARGE PAYLOAD PARSING");
                    println!(
                        "{:<25} | {:>10} | {:>10} | {:>12} | {:>12} | {:>12}",
                        "Payload", "Tools", "Size(B)", "Ops/sec", "Bytes/sec", "Time/op"
                    );
                }
                "reuse" => {
                    println!("PARSER REUSE COMPARISON");
                    println!(
                        "{:<25} | {:>12} | {:>12} | {:>15}",
                        "Strategy", "Ops/sec", "Time/op", "Performance"
                    );
                }
                "latency" => {
                    println!("LATENCY DISTRIBUTION");
                    println!(
                        "{:<25} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
                        "Parser", "P50(µs)", "P95(µs)", "P99(µs)", "Max(µs)", "Samples"
                    );
                }
                _ => {}
            }
            println!("{}", "-".repeat(120));
        }

        println!("{}", value);
    }

    println!("\n{}", "=".repeat(120));

    // Print performance analysis
    println!("\nPERFORMANCE ANALYSIS:");
    println!("{}", "-".repeat(120));

    // Calculate and display key metrics
    if let Some(new_registry) = results.get("007_reuse") {
        if let Some(reuse_parser) = results.get("009_reuse") {
            // Extract ops/sec values
            let new_ops: f64 = new_registry
                .split('|')
                .nth(1)
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0.0);
            let reuse_ops: f64 = reuse_parser
                .split('|')
                .nth(1)
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0.0);

            if new_ops > 0.0 && reuse_ops > 0.0 {
                let improvement = (reuse_ops / new_ops - 1.0) * 100.0;
                println!("Parser Reuse Improvement: {:.1}% faster", improvement);

                if improvement < 100.0 {
                    println!("⚠️  WARNING: Parser reuse improvement is lower than expected!");
                    println!("   Expected: >100% improvement with singleton pattern");
                    println!("   Actual: {:.1}% improvement", improvement);
                    println!("   Recommendation: Implement global singleton registry");
                }
            }
        }
    }

    println!("{}", "=".repeat(120));
}

fn run_benchmarks(c: &mut Criterion) {
    bench_registry_creation(c);
    bench_parser_lookup(c);
    bench_complete_parsing(c);
    bench_streaming_parsing(c);
    bench_concurrent_parsing(c);
    bench_large_payloads(c);
    bench_parser_reuse(c);
    bench_latency_distribution(c);

    // Print summary at the end
    print_summary();
}

criterion_group!(benches, run_benchmarks);
criterion::criterion_main!(benches);
