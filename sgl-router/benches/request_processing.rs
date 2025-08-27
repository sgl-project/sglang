use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
<<<<<<< HEAD
use serde_json::{from_str, to_string, to_vec};
use std::time::Instant;

use sglang_router_rs::openai_api_types::{
    ChatCompletionRequest, ChatMessage, CompletionRequest, GenerateParameters, GenerateRequest,
    SamplingParams, StringOrArray, UserMessageContent,
};
use sglang_router_rs::request_adapter::{RouteableRequest, ToPdRequest};
=======
use serde_json::{from_str, to_string, to_value, to_vec};
use std::time::Instant;

use sglang_router_rs::core::{BasicWorker, Worker, WorkerType};
use sglang_router_rs::protocols::spec::{
    ChatCompletionRequest, ChatMessage, CompletionRequest, GenerateParameters, GenerateRequest,
    SamplingParams, StringOrArray, UserMessageContent,
};
use sglang_router_rs::routers::pd_types::{generate_room_id, get_hostname, RequestWithBootstrap};

fn create_test_worker() -> BasicWorker {
    BasicWorker::new(
        "http://test-server:8000".to_string(),
        WorkerType::Prefill {
            bootstrap_port: Some(5678),
        },
    )
}

// Helper function to get bootstrap info from worker
fn get_bootstrap_info(worker: &BasicWorker) -> (String, Option<u16>) {
    let hostname = get_hostname(worker.url());
    let bootstrap_port = match worker.worker_type() {
        WorkerType::Prefill { bootstrap_port } => bootstrap_port,
        _ => None,
    };
    (hostname, bootstrap_port)
}

/// Create a default GenerateRequest for benchmarks with minimal fields set
fn default_generate_request() -> GenerateRequest {
    GenerateRequest {
        text: None,
        prompt: None,
        input_ids: None,
        stream: false,
        parameters: None,
        sampling_params: None,
        return_logprob: false,
        // SGLang Extensions
        lora_path: None,
        session_params: None,
        return_hidden_states: false,
        rid: None,
    }
}

/// Create a default ChatCompletionRequest for benchmarks with minimal fields set
fn default_chat_completion_request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: String::new(),
        messages: vec![],
        max_tokens: None,
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        n: None,
        stream: false,
        stream_options: None,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        logprobs: false,
        top_logprobs: None,
        user: None,
        response_format: None,
        seed: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        function_call: None,
        functions: None,
        // SGLang Extensions
        top_k: None,
        min_p: None,
        min_tokens: None,
        repetition_penalty: None,
        regex: None,
        ebnf: None,
        stop_token_ids: None,
        no_stop_trim: false,
        ignore_eos: false,
        continue_final_message: false,
        skip_special_tokens: true,
        // SGLang Extensions
        lora_path: None,
        session_params: None,
        separate_reasoning: true,
        stream_reasoning: true,
        return_hidden_states: false,
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
        other: serde_json::Map::new(),
    }
}
>>>>>>> origin/main

// Sample request data for benchmarks
fn create_sample_generate_request() -> GenerateRequest {
    GenerateRequest {
        text: Some("Write a story about artificial intelligence".to_string()),
<<<<<<< HEAD
        input_ids: None,
        prompt: None,
=======
>>>>>>> origin/main
        parameters: Some(GenerateParameters {
            max_new_tokens: Some(100),
            temperature: Some(0.8),
            top_p: Some(0.9),
            top_k: Some(50),
            repetition_penalty: Some(1.0),
            ..Default::default()
        }),
        sampling_params: Some(SamplingParams {
            temperature: Some(0.8),
            top_p: Some(0.9),
            top_k: Some(50),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            repetition_penalty: Some(1.0),
            ..Default::default()
        }),
<<<<<<< HEAD
        stream: false,
        return_logprob: false,
=======
        ..default_generate_request()
>>>>>>> origin/main
    }
}

fn create_sample_chat_completion_request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "gpt-3.5-turbo".to_string(),
        messages: vec![
            ChatMessage::System {
                role: "system".to_string(),
                content: "You are a helpful assistant".to_string(),
                name: None,
            },
            ChatMessage::User {
                role: "user".to_string(),
                content: UserMessageContent::Text(
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
<<<<<<< HEAD
        stream: false,
        stop: None,
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
        logit_bias: None,
        logprobs: false,
        top_logprobs: None,
        user: None,
        response_format: None,
        seed: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: Some(true),
        function_call: None,
        functions: None,
=======
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
        parallel_tool_calls: Some(true),
        ..default_chat_completion_request()
>>>>>>> origin/main
    }
}

fn create_sample_completion_request() -> CompletionRequest {
    CompletionRequest {
        model: "text-davinci-003".to_string(),
        prompt: StringOrArray::String("Complete this sentence: The future of AI is".to_string()),
<<<<<<< HEAD
        suffix: None,
=======
>>>>>>> origin/main
        max_tokens: Some(50),
        temperature: Some(0.8),
        top_p: Some(1.0),
        n: Some(1),
<<<<<<< HEAD
        stream: false,
        logprobs: None,
        echo: false,
        stop: None,
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
        best_of: Some(1),
        logit_bias: None,
        user: None,
        seed: None,
=======
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
        best_of: Some(1),
        ..default_completion_request()
>>>>>>> origin/main
    }
}

fn create_large_chat_completion_request() -> ChatCompletionRequest {
    let mut messages = vec![ChatMessage::System {
        role: "system".to_string(),
        content: "You are a helpful assistant with extensive knowledge.".to_string(),
        name: None,
    }];

    // Add many user/assistant pairs to simulate a long conversation
    for i in 0..50 {
        messages.push(ChatMessage::User {
            role: "user".to_string(),
            content: UserMessageContent::Text(format!("Question {}: What do you think about topic number {} which involves complex reasoning about multiple interconnected systems and their relationships?", i, i)),
            name: None,
        });
        messages.push(ChatMessage::Assistant {
            role: "assistant".to_string(),
            content: Some(format!("Answer {}: This is a detailed response about topic {} that covers multiple aspects and provides comprehensive analysis of the interconnected systems you mentioned.", i, i)),
            name: None,
            tool_calls: None,
            function_call: None,
<<<<<<< HEAD
=======
            reasoning_content: None,
>>>>>>> origin/main
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
<<<<<<< HEAD
        stream: false,
        stop: None,
        presence_penalty: Some(0.1),
        frequency_penalty: Some(0.1),
        logit_bias: None,
        logprobs: false,
        top_logprobs: Some(5),
        user: Some("benchmark_user".to_string()),
        response_format: None,
        seed: Some(42),
        tools: None,
        tool_choice: None,
        parallel_tool_calls: Some(true),
        function_call: None,
        functions: None,
=======
        presence_penalty: Some(0.1),
        frequency_penalty: Some(0.1),
        top_logprobs: Some(5),
        user: Some("benchmark_user".to_string()),
        seed: Some(42),
        parallel_tool_calls: Some(true),
        ..default_chat_completion_request()
>>>>>>> origin/main
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

<<<<<<< HEAD
// Benchmark request adaptation from OpenAI to PD format
fn bench_request_adaptation(c: &mut Criterion) {
    let mut group = c.benchmark_group("request_adaptation");
=======
// Benchmark bootstrap injection (replaces request adaptation)
fn bench_bootstrap_injection(c: &mut Criterion) {
    let mut group = c.benchmark_group("bootstrap_injection");
>>>>>>> origin/main

    let generate_req = create_sample_generate_request();
    let chat_req = create_sample_chat_completion_request();
    let completion_req = create_sample_completion_request();
    let large_chat_req = create_large_chat_completion_request();
<<<<<<< HEAD

    group.bench_function("generate_to_pd", |b| {
        b.iter(|| {
            let pd_req = black_box(generate_req.clone()).to_pd_request();
            black_box(pd_req);
        });
    });

    group.bench_function("chat_completion_to_pd", |b| {
        b.iter(|| {
            let pd_req = black_box(chat_req.clone()).to_pd_request();
            black_box(pd_req);
        });
    });

    group.bench_function("completion_to_pd", |b| {
        b.iter(|| {
            let pd_req = black_box(completion_req.clone()).to_pd_request();
            black_box(pd_req);
        });
    });

    group.bench_function("large_chat_completion_to_pd", |b| {
        b.iter(|| {
            let pd_req = black_box(large_chat_req.clone()).to_pd_request();
            black_box(pd_req);
=======
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
>>>>>>> origin/main
        });
    });

    group.finish();
}

<<<<<<< HEAD
// Benchmark regular routing (RouteableRequest methods)
fn bench_regular_routing(c: &mut Criterion) {
    let mut group = c.benchmark_group("regular_routing");
=======
// Benchmark direct JSON routing (replaces regular routing)
fn bench_direct_json_routing(c: &mut Criterion) {
    let mut group = c.benchmark_group("direct_json_routing");
>>>>>>> origin/main

    let generate_req = create_sample_generate_request();
    let chat_req = create_sample_chat_completion_request();
    let completion_req = create_sample_completion_request();

    group.bench_function("generate_to_json", |b| {
        b.iter(|| {
<<<<<<< HEAD
            let json = black_box(&generate_req).to_json().unwrap();
=======
            let json = to_value(black_box(&generate_req)).unwrap();
            black_box(json);
        });
    });

    group.bench_function("generate_to_json_string", |b| {
        b.iter(|| {
            let json = to_string(black_box(&generate_req)).unwrap();
>>>>>>> origin/main
            black_box(json);
        });
    });

    group.bench_function("generate_to_bytes", |b| {
        b.iter(|| {
<<<<<<< HEAD
            let bytes = black_box(&generate_req).to_bytes().unwrap();
=======
            let bytes = to_vec(black_box(&generate_req)).unwrap();
>>>>>>> origin/main
            black_box(bytes);
        });
    });

    group.bench_function("chat_completion_to_json", |b| {
        b.iter(|| {
<<<<<<< HEAD
            let json = black_box(&chat_req).to_json().unwrap();
=======
            let json = to_value(black_box(&chat_req)).unwrap();
>>>>>>> origin/main
            black_box(json);
        });
    });

<<<<<<< HEAD
    group.bench_function("chat_completion_to_bytes", |b| {
        b.iter(|| {
            let bytes = black_box(&chat_req).to_bytes().unwrap();
            black_box(bytes);
=======
    group.bench_function("chat_completion_to_json_string", |b| {
        b.iter(|| {
            let json = to_string(black_box(&chat_req)).unwrap();
            black_box(json);
>>>>>>> origin/main
        });
    });

    group.bench_function("completion_to_json", |b| {
        b.iter(|| {
<<<<<<< HEAD
            let json = black_box(&completion_req).to_json().unwrap();
=======
            let json = to_value(black_box(&completion_req)).unwrap();
>>>>>>> origin/main
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
<<<<<<< HEAD
        input_ids: None,
        prompt: None,
        parameters: None,
        sampling_params: None,
        stream: false,
        return_logprob: false,
=======
        ..default_generate_request()
>>>>>>> origin/main
    };

    let medium_generate = GenerateRequest {
        text: Some("Write a medium length story about AI".repeat(10)),
<<<<<<< HEAD
        input_ids: None,
        prompt: None,
        parameters: None,
        sampling_params: None,
        stream: false,
        return_logprob: false,
=======
        ..default_generate_request()
>>>>>>> origin/main
    };

    let large_generate = GenerateRequest {
        text: Some("Write a very long and detailed story about artificial intelligence and its impact on society".repeat(100)),
<<<<<<< HEAD
        input_ids: None,
        prompt: None,
        parameters: None,
        sampling_params: None,
        stream: false,
        return_logprob: false,
    };

=======
        ..default_generate_request()
    };

    let worker = create_test_worker();
    let (hostname, bootstrap_port) = get_bootstrap_info(&worker);

>>>>>>> origin/main
    for (name, req) in [
        ("small", &small_generate),
        ("medium", &medium_generate),
        ("large", &large_generate),
    ] {
        let json = to_string(req).unwrap();
        let size_bytes = json.len();
<<<<<<< HEAD
=======
        let hostname_clone = hostname.clone();
>>>>>>> origin/main

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

<<<<<<< HEAD
        group.bench_with_input(BenchmarkId::new("adapt_to_pd", name), &req, |b, req| {
            b.iter(|| {
                let pd_req = (*req).clone().to_pd_request();
                black_box(pd_req);
            });
        });
=======
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
>>>>>>> origin/main
    }

    group.finish();
}

<<<<<<< HEAD
// Benchmark full round-trip: deserialize -> adapt -> serialize
=======
// Benchmark full round-trip: deserialize -> inject bootstrap -> serialize
>>>>>>> origin/main
fn bench_full_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_round_trip");

    let generate_json = to_string(&create_sample_generate_request()).unwrap();
    let chat_json = to_string(&create_sample_chat_completion_request()).unwrap();
    let completion_json = to_string(&create_sample_completion_request()).unwrap();
<<<<<<< HEAD
=======
    let worker = create_test_worker();
    let (hostname, bootstrap_port) = get_bootstrap_info(&worker);
>>>>>>> origin/main

    group.bench_function("generate_openai_to_pd_pipeline", |b| {
        b.iter(|| {
            // Deserialize OpenAI request
            let req: GenerateRequest = from_str(black_box(&generate_json)).unwrap();
<<<<<<< HEAD
            // Adapt to PD format
            let pd_req = req.to_pd_request();
            // Serialize PD request
            let pd_json = to_string(&pd_req).unwrap();
=======
            // Create wrapper with bootstrap fields
            let request_with_bootstrap = RequestWithBootstrap {
                original: &req,
                bootstrap_host: hostname.clone(),
                bootstrap_port,
                bootstrap_room: generate_room_id(),
            };
            // Serialize final request
            let pd_json = to_string(&request_with_bootstrap).unwrap();
>>>>>>> origin/main
            black_box(pd_json);
        });
    });

    group.bench_function("chat_completion_openai_to_pd_pipeline", |b| {
        b.iter(|| {
            let req: ChatCompletionRequest = from_str(black_box(&chat_json)).unwrap();
<<<<<<< HEAD
            let pd_req = req.to_pd_request();
            let pd_json = to_string(&pd_req).unwrap();
=======
            let request_with_bootstrap = RequestWithBootstrap {
                original: &req,
                bootstrap_host: hostname.clone(),
                bootstrap_port,
                bootstrap_room: generate_room_id(),
            };
            let pd_json = to_string(&request_with_bootstrap).unwrap();
>>>>>>> origin/main
            black_box(pd_json);
        });
    });

    group.bench_function("completion_openai_to_pd_pipeline", |b| {
        b.iter(|| {
            let req: CompletionRequest = from_str(black_box(&completion_json)).unwrap();
<<<<<<< HEAD
            let pd_req = req.to_pd_request();
            let pd_json = to_string(&pd_req).unwrap();
=======
            let request_with_bootstrap = RequestWithBootstrap {
                original: &req,
                bootstrap_host: hostname.clone(),
                bootstrap_port,
                bootstrap_room: generate_room_id(),
            };
            let pd_json = to_string(&request_with_bootstrap).unwrap();
>>>>>>> origin/main
            black_box(pd_json);
        });
    });

<<<<<<< HEAD
    group.bench_function("generate_regular_routing_pipeline", |b| {
        b.iter(|| {
            // Deserialize OpenAI request
            let req: GenerateRequest = from_str(black_box(&generate_json)).unwrap();
            // Convert to JSON for regular routing
            let routing_json = req.to_json().unwrap();
            black_box(routing_json);
=======
    group.bench_function("generate_direct_json_pipeline", |b| {
        b.iter(|| {
            // Deserialize OpenAI request
            let req: GenerateRequest = from_str(black_box(&generate_json)).unwrap();
            // Convert to JSON for direct routing (no bootstrap injection)
            let routing_json = to_value(&req).unwrap();
            let json_string = to_string(&routing_json).unwrap();
            black_box(json_string);
>>>>>>> origin/main
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
<<<<<<< HEAD
=======
    let worker = create_test_worker();
>>>>>>> origin/main

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

<<<<<<< HEAD
    // Measure adaptation
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = black_box(generate_req.clone().to_pd_request());
    }
    let adapt_time = start.elapsed().as_nanos() / 1000;
    println!("  * PD Adaptation (avg):     {:>8} ns/req", adapt_time);

    // Calculate ratios
    let total_pipeline = serialize_time + deserialize_time + adapt_time;
=======
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
>>>>>>> origin/main
    println!("  * Total Pipeline (avg):    {:>8} ns/req", total_pipeline);

    println!("\nPerformance Insights:");
    if deserialize_time > serialize_time * 2 {
        println!("  • Deserialization is significantly faster than serialization");
    }
<<<<<<< HEAD
    if adapt_time < serialize_time / 10 {
        println!(
            "  • PD adaptation overhead is negligible ({:.1}% of serialization)",
            (adapt_time as f64 / serialize_time as f64) * 100.0
        );
    }
    if total_pipeline < 10_000 {
        println!("  • Total pipeline latency is excellent (< 10μs)");
    }

=======
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

>>>>>>> origin/main
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
<<<<<<< HEAD
    bench_request_adaptation,
    bench_regular_routing,
=======
    bench_bootstrap_injection,
    bench_direct_json_routing,
>>>>>>> origin/main
    bench_throughput_by_size,
    bench_full_round_trip
);
criterion_main!(benches);
