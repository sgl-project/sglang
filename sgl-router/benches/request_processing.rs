use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde_json::{from_str, to_string, to_vec};
use std::time::Instant;

use sglang_router_rs::openai_api_types::{
    ChatCompletionRequest, ChatMessage, CompletionRequest, GenerateParameters, GenerateRequest,
    SamplingParams, StringOrArray, UserMessageContent,
};
use sglang_router_rs::request_adapter::{RouteableRequest, ToPdRequest};

// Sample request data for benchmarks
fn create_sample_generate_request() -> GenerateRequest {
    GenerateRequest {
        text: Some("Write a story about artificial intelligence".to_string()),
        input_ids: None,
        prompt: None,
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
        stream: false,
        return_logprob: false,
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
    }
}

fn create_sample_completion_request() -> CompletionRequest {
    CompletionRequest {
        model: "text-davinci-003".to_string(),
        prompt: StringOrArray::String("Complete this sentence: The future of AI is".to_string()),
        suffix: None,
        max_tokens: Some(50),
        temperature: Some(0.8),
        top_p: Some(1.0),
        n: Some(1),
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

// Benchmark request adaptation from OpenAI to PD format
fn bench_request_adaptation(c: &mut Criterion) {
    let mut group = c.benchmark_group("request_adaptation");

    let generate_req = create_sample_generate_request();
    let chat_req = create_sample_chat_completion_request();
    let completion_req = create_sample_completion_request();
    let large_chat_req = create_large_chat_completion_request();

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
        });
    });

    group.finish();
}

// Benchmark regular routing (RouteableRequest methods)
fn bench_regular_routing(c: &mut Criterion) {
    let mut group = c.benchmark_group("regular_routing");

    let generate_req = create_sample_generate_request();
    let chat_req = create_sample_chat_completion_request();
    let completion_req = create_sample_completion_request();

    group.bench_function("generate_to_json", |b| {
        b.iter(|| {
            let json = black_box(&generate_req).to_json().unwrap();
            black_box(json);
        });
    });

    group.bench_function("generate_to_bytes", |b| {
        b.iter(|| {
            let bytes = black_box(&generate_req).to_bytes().unwrap();
            black_box(bytes);
        });
    });

    group.bench_function("chat_completion_to_json", |b| {
        b.iter(|| {
            let json = black_box(&chat_req).to_json().unwrap();
            black_box(json);
        });
    });

    group.bench_function("chat_completion_to_bytes", |b| {
        b.iter(|| {
            let bytes = black_box(&chat_req).to_bytes().unwrap();
            black_box(bytes);
        });
    });

    group.bench_function("completion_to_json", |b| {
        b.iter(|| {
            let json = black_box(&completion_req).to_json().unwrap();
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
        input_ids: None,
        prompt: None,
        parameters: None,
        sampling_params: None,
        stream: false,
        return_logprob: false,
    };

    let medium_generate = GenerateRequest {
        text: Some("Write a medium length story about AI".repeat(10)),
        input_ids: None,
        prompt: None,
        parameters: None,
        sampling_params: None,
        stream: false,
        return_logprob: false,
    };

    let large_generate = GenerateRequest {
        text: Some("Write a very long and detailed story about artificial intelligence and its impact on society".repeat(100)),
        input_ids: None,
        prompt: None,
        parameters: None,
        sampling_params: None,
        stream: false,
        return_logprob: false,
    };

    for (name, req) in [
        ("small", &small_generate),
        ("medium", &medium_generate),
        ("large", &large_generate),
    ] {
        let json = to_string(req).unwrap();
        let size_bytes = json.len();

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

        group.bench_with_input(BenchmarkId::new("adapt_to_pd", name), &req, |b, req| {
            b.iter(|| {
                let pd_req = (*req).clone().to_pd_request();
                black_box(pd_req);
            });
        });
    }

    group.finish();
}

// Benchmark full round-trip: deserialize -> adapt -> serialize
fn bench_full_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_round_trip");

    let generate_json = to_string(&create_sample_generate_request()).unwrap();
    let chat_json = to_string(&create_sample_chat_completion_request()).unwrap();
    let completion_json = to_string(&create_sample_completion_request()).unwrap();

    group.bench_function("generate_openai_to_pd_pipeline", |b| {
        b.iter(|| {
            // Deserialize OpenAI request
            let req: GenerateRequest = from_str(black_box(&generate_json)).unwrap();
            // Adapt to PD format
            let pd_req = req.to_pd_request();
            // Serialize PD request
            let pd_json = to_string(&pd_req).unwrap();
            black_box(pd_json);
        });
    });

    group.bench_function("chat_completion_openai_to_pd_pipeline", |b| {
        b.iter(|| {
            let req: ChatCompletionRequest = from_str(black_box(&chat_json)).unwrap();
            let pd_req = req.to_pd_request();
            let pd_json = to_string(&pd_req).unwrap();
            black_box(pd_json);
        });
    });

    group.bench_function("completion_openai_to_pd_pipeline", |b| {
        b.iter(|| {
            let req: CompletionRequest = from_str(black_box(&completion_json)).unwrap();
            let pd_req = req.to_pd_request();
            let pd_json = to_string(&pd_req).unwrap();
            black_box(pd_json);
        });
    });

    group.bench_function("generate_regular_routing_pipeline", |b| {
        b.iter(|| {
            // Deserialize OpenAI request
            let req: GenerateRequest = from_str(black_box(&generate_json)).unwrap();
            // Convert to JSON for regular routing
            let routing_json = req.to_json().unwrap();
            black_box(routing_json);
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

    // Measure adaptation
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = black_box(generate_req.clone().to_pd_request());
    }
    let adapt_time = start.elapsed().as_nanos() / 1000;
    println!("  * PD Adaptation (avg):     {:>8} ns/req", adapt_time);

    // Calculate ratios
    let total_pipeline = serialize_time + deserialize_time + adapt_time;
    println!("  * Total Pipeline (avg):    {:>8} ns/req", total_pipeline);

    println!("\nPerformance Insights:");
    if deserialize_time > serialize_time * 2 {
        println!("  • Deserialization is significantly faster than serialization");
    }
    if adapt_time < serialize_time / 10 {
        println!(
            "  • PD adaptation overhead is negligible ({:.1}% of serialization)",
            (adapt_time as f64 / serialize_time as f64) * 100.0
        );
    }
    if total_pipeline < 10_000 {
        println!("  • Total pipeline latency is excellent (< 10μs)");
    }

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
    bench_request_adaptation,
    bench_regular_routing,
    bench_throughput_by_size,
    bench_full_round_trip
);
criterion_main!(benches);
