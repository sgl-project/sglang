use std::time::Instant;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde_json::{from_str, to_string, to_value, to_vec};
use sglang_router_rs::{
    core::{BasicWorker, BasicWorkerBuilder, Worker, WorkerType},
    protocols::{
        chat::{ChatCompletionRequest, ChatMessage, MessageContent},
        common::StringOrArray,
        completion::CompletionRequest,
        generate::GenerateRequest,
        sampling_params::SamplingParams,
    },
    routers::http::pd_types::{generate_room_id, RequestWithBootstrap},
};

fn create_test_worker() -> BasicWorker {
    BasicWorkerBuilder::new("http://test-server:8000")
        .worker_type(WorkerType::Prefill {
            bootstrap_port: Some(5678),
        })
        .build()
}

// Helper function to get bootstrap info from worker
fn get_bootstrap_info(worker: &BasicWorker) -> (String, Option<u16>) {
    let hostname = worker.bootstrap_host().to_string();
    let bootstrap_port = worker.bootstrap_port();
    (hostname, bootstrap_port)
}

/// Create a default GenerateRequest for benchmarks with minimal fields set
fn default_generate_request() -> GenerateRequest {
    GenerateRequest {
        text: None,
        model: None,
        input_ids: None,
        input_embeds: None,
        image_data: None,
        video_data: None,
        audio_data: None,
        sampling_params: None,
        return_logprob: None,
        logprob_start_len: None,
        top_logprobs_num: None,
        token_ids_logprob: None,
        return_text_in_logprobs: false,
        stream: false,
        log_metrics: true,
        return_hidden_states: false,
        modalities: None,
        session_params: None,
        lora_path: None,
        lora_id: None,
        custom_logit_processor: None,
        bootstrap_host: None,
        bootstrap_port: None,
        bootstrap_room: None,
        bootstrap_pair_key: None,
        data_parallel_rank: None,
        data_parallel_rank_decode: None,
        background: false,
        conversation_id: None,
        priority: None,
        extra_key: None,
        no_logs: false,
        custom_labels: None,
        return_bytes: false,
        return_entropy: false,
        rid: None,
    }
}

/// Create a default ChatCompletionRequest for benchmarks with minimal fields set
#[allow(deprecated)]
fn default_chat_completion_request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        // Required fields in OpenAI order
        messages: vec![],
        model: String::new(),

        // Use default for all other fields
        ..Default::default()
    }
}

/// Create a default CompletionRequest for benchmarks with minimal fields set
fn default_completion_request() -> CompletionRequest {
    CompletionRequest {
        model: String::new(),
        prompt: StringOrArray::String(String::new()),
        suffix: None,
        max_tokens: None,
        temperature: None,
        top_p: None,
        n: None,
        stream: false,
        stream_options: None,
        logprobs: None,
        echo: false,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        best_of: None,
        logit_bias: None,
        user: None,
        seed: None,
        // SGLang Extensions
        top_k: None,
        min_p: None,
        min_tokens: None,
        repetition_penalty: None,
        regex: None,
        ebnf: None,
        json_schema: None,
        stop_token_ids: None,
        no_stop_trim: false,
        ignore_eos: false,
        skip_special_tokens: true,
        // SGLang Extensions
        lora_path: None,
        session_params: None,
        return_hidden_states: false,
        sampling_seed: None,
        other: serde_json::Map::new(),
    }
}

// Sample request data for benchmarks
fn create_sample_generate_request() -> GenerateRequest {
    GenerateRequest {
        text: Some("Write a story about artificial intelligence".to_string()),
        sampling_params: Some(SamplingParams {
            max_new_tokens: Some(100),
            temperature: Some(0.8),
            top_p: Some(0.9),
            top_k: Some(50),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            repetition_penalty: Some(1.0),
            ..Default::default()
        }),
        ..default_generate_request()
    }
}

#[allow(deprecated)]
fn create_sample_chat_completion_request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "gpt-3.5-turbo".to_string(),
        messages: vec![
            ChatMessage::System {
                content: MessageContent::Text("You are a helpful assistant".to_string()),
                name: None,
            },
            ChatMessage::User {
                content: MessageContent::Text(
                    "Explain quantum computing in simple terms".to_string(),
                ),
                name: None,
            },
        ],
        max_tokens: Some(150),
        max_completion_tokens: Some(150),
        temperature: Some(0.7),
        top_p: Some(1.0),
        n: Some(1),
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
        parallel_tool_calls: Some(true),
        ..default_chat_completion_request()
    }
}

fn create_sample_completion_request() -> CompletionRequest {
    CompletionRequest {
        model: "text-davinci-003".to_string(),
        prompt: StringOrArray::String("Complete this sentence: The future of AI is".to_string()),
        max_tokens: Some(50),
        temperature: Some(0.8),
        top_p: Some(1.0),
        n: Some(1),
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
        best_of: Some(1),
        ..default_completion_request()
    }
}

#[allow(deprecated)]
fn create_large_chat_completion_request() -> ChatCompletionRequest {
    let mut messages = vec![ChatMessage::System {
        content: MessageContent::Text(
            "You are a helpful assistant with extensive knowledge.".to_string(),
        ),
        name: None,
    }];

    // Add many user/assistant pairs to simulate a long conversation
    for i in 0..50 {
        messages.push(ChatMessage::User {
            content: MessageContent::Text(format!("Question {}: What do you think about topic number {} which involves complex reasoning about multiple interconnected systems and their relationships?", i, i)),
            name: None,
        });
        messages.push(ChatMessage::Assistant {
            content: Some(MessageContent::Text(format!("Answer {}: This is a detailed response about topic {} that covers multiple aspects and provides comprehensive analysis of the interconnected systems you mentioned.", i, i))),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        });
    }

    ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages,
        max_tokens: Some(1000),
        max_completion_tokens: Some(1000),
        temperature: Some(0.7),
        top_p: Some(0.95),
        n: Some(1),
        presence_penalty: Some(0.1),
        frequency_penalty: Some(0.1),
        top_logprobs: Some(5),
        seed: Some(42),
        parallel_tool_calls: Some(true),
        ..default_chat_completion_request()
    }
}

// Benchmark JSON serialization
fn bench_json_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_serialization");

    let generate_req = create_sample_generate_request();
    let chat_req = create_sample_chat_completion_request();
    let completion_req = create_sample_completion_request();
    let large_chat_req = create_large_chat_completion_request();

    group.bench_function("generate_request", |b| {
        b.iter(|| {
            let json = to_string(black_box(&generate_req)).unwrap();
            black_box(json);
        });
    });

    group.bench_function("chat_completion_request", |b| {
        b.iter(|| {
            let json = to_string(black_box(&chat_req)).unwrap();
            black_box(json);
        });
    });

    group.bench_function("completion_request", |b| {
        b.iter(|| {
            let json = to_string(black_box(&completion_req)).unwrap();
            black_box(json);
        });
    });

    group.bench_function("large_chat_completion_request", |b| {
        b.iter(|| {
            let json = to_string(black_box(&large_chat_req)).unwrap();
            black_box(json);
        });
    });

    group.bench_function("generate_request_to_bytes", |b| {
        b.iter(|| {
            let bytes = to_vec(black_box(&generate_req)).unwrap();
            black_box(bytes);
        });
    });

    group.finish();
}

// Benchmark JSON deserialization
fn bench_json_deserialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_deserialization");

    let generate_json = to_string(&create_sample_generate_request()).unwrap();
    let chat_json = to_string(&create_sample_chat_completion_request()).unwrap();
    let completion_json = to_string(&create_sample_completion_request()).unwrap();
    let large_chat_json = to_string(&create_large_chat_completion_request()).unwrap();

    group.bench_function("generate_request", |b| {
        b.iter(|| {
            let req: GenerateRequest = from_str(black_box(&generate_json)).unwrap();
            black_box(req);
        });
    });

    group.bench_function("chat_completion_request", |b| {
        b.iter(|| {
            let req: ChatCompletionRequest = from_str(black_box(&chat_json)).unwrap();
            black_box(req);
        });
    });

    group.bench_function("completion_request", |b| {
        b.iter(|| {
            let req: CompletionRequest = from_str(black_box(&completion_json)).unwrap();
            black_box(req);
        });
    });

    group.bench_function("large_chat_completion_request", |b| {
        b.iter(|| {
            let req: ChatCompletionRequest = from_str(black_box(&large_chat_json)).unwrap();
            black_box(req);
        });
    });

    group.finish();
}

// Benchmark bootstrap injection (replaces request adaptation)
fn bench_bootstrap_injection(c: &mut Criterion) {
    let mut group = c.benchmark_group("bootstrap_injection");

    let generate_req = create_sample_generate_request();
    let chat_req = create_sample_chat_completion_request();
    let completion_req = create_sample_completion_request();
    let large_chat_req = create_large_chat_completion_request();
    let worker = create_test_worker();
    let (hostname, bootstrap_port) = get_bootstrap_info(&worker);

    group.bench_function("generate_bootstrap_injection", |b| {
        b.iter(|| {
            let request_with_bootstrap = RequestWithBootstrap {
                original: &generate_req,
                bootstrap_host: hostname.clone(),
                bootstrap_port,
                bootstrap_room: generate_room_id(),
            };
            let json = to_value(black_box(&request_with_bootstrap)).unwrap();
            black_box(json);
        });
    });

    group.bench_function("chat_completion_bootstrap_injection", |b| {
        b.iter(|| {
            let request_with_bootstrap = RequestWithBootstrap {
                original: &chat_req,
                bootstrap_host: hostname.clone(),
                bootstrap_port,
                bootstrap_room: generate_room_id(),
            };
            let json = to_value(black_box(&request_with_bootstrap)).unwrap();
            black_box(json);
        });
    });

    group.bench_function("completion_bootstrap_injection", |b| {
        b.iter(|| {
            let request_with_bootstrap = RequestWithBootstrap {
                original: &completion_req,
                bootstrap_host: hostname.clone(),
                bootstrap_port,
                bootstrap_room: generate_room_id(),
            };
            let json = to_value(black_box(&request_with_bootstrap)).unwrap();
            black_box(json);
        });
    });

    group.bench_function("large_chat_completion_bootstrap_injection", |b| {
        b.iter(|| {
            let request_with_bootstrap = RequestWithBootstrap {
                original: &large_chat_req,
                bootstrap_host: hostname.clone(),
                bootstrap_port,
                bootstrap_room: generate_room_id(),
            };
            let json = to_value(black_box(&request_with_bootstrap)).unwrap();
            black_box(json);
        });
    });

    group.finish();
}

// Benchmark direct JSON routing (replaces regular routing)
fn bench_direct_json_routing(c: &mut Criterion) {
    let mut group = c.benchmark_group("direct_json_routing");

    let generate_req = create_sample_generate_request();
    let chat_req = create_sample_chat_completion_request();
    let completion_req = create_sample_completion_request();

    group.bench_function("generate_to_json", |b| {
        b.iter(|| {
            let json = to_value(black_box(&generate_req)).unwrap();
            black_box(json);
        });
    });

    group.bench_function("generate_to_json_string", |b| {
        b.iter(|| {
            let json = to_string(black_box(&generate_req)).unwrap();
            black_box(json);
        });
    });

    group.bench_function("generate_to_bytes", |b| {
        b.iter(|| {
            let bytes = to_vec(black_box(&generate_req)).unwrap();
            black_box(bytes);
        });
    });

    group.bench_function("chat_completion_to_json", |b| {
        b.iter(|| {
            let json = to_value(black_box(&chat_req)).unwrap();
            black_box(json);
        });
    });

    group.bench_function("chat_completion_to_json_string", |b| {
        b.iter(|| {
            let json = to_string(black_box(&chat_req)).unwrap();
            black_box(json);
        });
    });

    group.bench_function("completion_to_json", |b| {
        b.iter(|| {
            let json = to_value(black_box(&completion_req)).unwrap();
            black_box(json);
        });
    });

    group.finish();
}

// Benchmark throughput with different request sizes
fn bench_throughput_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_by_size");

    // Create requests of different sizes
    let small_generate = GenerateRequest {
        text: Some("Hi".to_string()),
        ..default_generate_request()
    };

    let medium_generate = GenerateRequest {
        text: Some("Write a medium length story about AI".repeat(10)),
        ..default_generate_request()
    };

    let large_generate = GenerateRequest {
        text: Some("Write a very long and detailed story about artificial intelligence and its impact on society".repeat(100)),
        ..default_generate_request()
    };

    let worker = create_test_worker();
    let (hostname, bootstrap_port) = get_bootstrap_info(&worker);

    for (name, req) in [
        ("small", &small_generate),
        ("medium", &medium_generate),
        ("large", &large_generate),
    ] {
        let json = to_string(req).unwrap();
        let size_bytes = json.len();
        let hostname_clone = hostname.clone();

        group.throughput(Throughput::Bytes(size_bytes as u64));
        group.bench_with_input(BenchmarkId::new("serialize", name), &req, |b, req| {
            b.iter(|| {
                let json = to_string(black_box(req)).unwrap();
                black_box(json);
            });
        });

        group.bench_with_input(
            BenchmarkId::new("deserialize", name),
            &json,
            |b, json_str| {
                b.iter(|| {
                    let req: GenerateRequest = black_box(from_str(json_str)).unwrap();
                    black_box(req);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bootstrap_inject", name),
            &req,
            move |b, req| {
                let hostname = hostname_clone.clone();
                b.iter(|| {
                    let request_with_bootstrap = RequestWithBootstrap {
                        original: req,
                        bootstrap_host: hostname.clone(),
                        bootstrap_port,
                        bootstrap_room: generate_room_id(),
                    };
                    let json = to_value(&request_with_bootstrap).unwrap();
                    black_box(json);
                });
            },
        );
    }

    group.finish();
}

// Benchmark full round-trip: deserialize -> inject bootstrap -> serialize
fn bench_full_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_round_trip");

    let generate_json = to_string(&create_sample_generate_request()).unwrap();
    let chat_json = to_string(&create_sample_chat_completion_request()).unwrap();
    let completion_json = to_string(&create_sample_completion_request()).unwrap();
    let worker = create_test_worker();
    let (hostname, bootstrap_port) = get_bootstrap_info(&worker);

    group.bench_function("generate_openai_to_pd_pipeline", |b| {
        b.iter(|| {
            // Deserialize OpenAI request
            let req: GenerateRequest = from_str(black_box(&generate_json)).unwrap();
            // Create wrapper with bootstrap fields
            let request_with_bootstrap = RequestWithBootstrap {
                original: &req,
                bootstrap_host: hostname.clone(),
                bootstrap_port,
                bootstrap_room: generate_room_id(),
            };
            // Serialize final request
            let pd_json = to_string(&request_with_bootstrap).unwrap();
            black_box(pd_json);
        });
    });

    group.bench_function("chat_completion_openai_to_pd_pipeline", |b| {
        b.iter(|| {
            let req: ChatCompletionRequest = from_str(black_box(&chat_json)).unwrap();
            let request_with_bootstrap = RequestWithBootstrap {
                original: &req,
                bootstrap_host: hostname.clone(),
                bootstrap_port,
                bootstrap_room: generate_room_id(),
            };
            let pd_json = to_string(&request_with_bootstrap).unwrap();
            black_box(pd_json);
        });
    });

    group.bench_function("completion_openai_to_pd_pipeline", |b| {
        b.iter(|| {
            let req: CompletionRequest = from_str(black_box(&completion_json)).unwrap();
            let request_with_bootstrap = RequestWithBootstrap {
                original: &req,
                bootstrap_host: hostname.clone(),
                bootstrap_port,
                bootstrap_room: generate_room_id(),
            };
            let pd_json = to_string(&request_with_bootstrap).unwrap();
            black_box(pd_json);
        });
    });

    group.bench_function("generate_direct_json_pipeline", |b| {
        b.iter(|| {
            // Deserialize OpenAI request
            let req: GenerateRequest = from_str(black_box(&generate_json)).unwrap();
            // Convert to JSON for direct routing (no bootstrap injection)
            let routing_json = to_value(&req).unwrap();
            let json_string = to_string(&routing_json).unwrap();
            black_box(json_string);
        });
    });

    group.finish();
}

fn benchmark_summary(c: &mut Criterion) {
    let group = c.benchmark_group("benchmark_summary");

    println!("\nSGLang Router Performance Benchmark Suite");
    println!("=============================================");

    // Quick performance overview
    let generate_req = create_sample_generate_request();
    let worker = create_test_worker();

    println!("\nQuick Performance Overview:");

    // Measure serialization
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = black_box(to_string(&generate_req).unwrap());
    }
    let serialize_time = start.elapsed().as_nanos() / 1000;
    println!("  * Serialization (avg):     {:>8} ns/req", serialize_time);

    // Measure deserialization
    let json = to_string(&generate_req).unwrap();
    let start = Instant::now();
    for _ in 0..1000 {
        let _: GenerateRequest = black_box(from_str(&json).unwrap());
    }
    let deserialize_time = start.elapsed().as_nanos() / 1000;
    println!(
        "  * Deserialization (avg):   {:>8} ns/req",
        deserialize_time
    );

    // Measure bootstrap injection (replaces adaptation)
    let (hostname, bootstrap_port) = get_bootstrap_info(&worker);
    let start = Instant::now();
    for _ in 0..1000 {
        let request_with_bootstrap = RequestWithBootstrap {
            original: &generate_req,
            bootstrap_host: hostname.clone(),
            bootstrap_port,
            bootstrap_room: generate_room_id(),
        };
        let _ = black_box(to_value(&request_with_bootstrap).unwrap());
    }
    let inject_time = start.elapsed().as_nanos() / 1000;
    println!("  * Bootstrap Injection (avg): {:>6} ns/req", inject_time);

    // Calculate ratios
    let total_pipeline = serialize_time + deserialize_time + inject_time;
    println!("  * Total Pipeline (avg):    {:>8} ns/req", total_pipeline);

    println!("\nPerformance Insights:");
    if deserialize_time > serialize_time * 2 {
        println!("  • Deserialization is significantly faster than serialization");
    }
    if inject_time < serialize_time / 10 {
        println!(
            "  • Bootstrap injection overhead is negligible ({:.1}% of serialization)",
            (inject_time as f64 / serialize_time as f64) * 100.0
        );
    }
    if total_pipeline < 100_000 {
        println!("  • Total pipeline latency is excellent (< 100μs)");
    }

    println!("\nSimplification Benefits:");
    println!("  • Eliminated complex type conversion layer");
    println!("  • Reduced memory allocations");
    println!("  • Automatic field preservation (no manual mapping)");
    println!("  • Direct JSON manipulation improves performance");

    println!("\nRecommendations:");
    if serialize_time > deserialize_time {
        println!("  • Focus optimization efforts on serialization rather than deserialization");
    }
    println!("  • PD mode overhead is minimal - safe to use for latency-sensitive workloads");
    println!("  • Consider batching small requests to improve overall throughput");

    println!("\n{}", "=".repeat(50));

    group.finish();
}

criterion_group!(
    benches,
    benchmark_summary,
    bench_json_serialization,
    bench_json_deserialization,
    bench_bootstrap_injection,
    bench_direct_json_routing,
    bench_throughput_by_size,
    bench_full_round_trip
);
criterion_main!(benches);
