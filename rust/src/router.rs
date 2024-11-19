use crate::tree::RadixTree;
use crate::multi_tenant_tree::Tree;
use actix_web::http::header::{HeaderValue, CONTENT_TYPE};
use actix_web::{HttpRequest, HttpResponse};
use bytes::Bytes;
use futures_util::TryStreamExt;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokenizers::tokenizer::Tokenizer;
use std::thread;

const EVICTION_INTERVAL_SECS: u64 = 60;
const MAX_TREE_SIZE: usize = 100000; // 2usize.pow(24);

#[derive(Debug)]
pub enum Router {
    RoundRobin {
        worker_urls: Vec<String>,
        current_index: AtomicUsize,
    },
    Random {
        worker_urls: Vec<String>,
    },
    ApproxTree {
        worker_urls: Vec<String>,
        tree: Arc<Tree>,
        url_to_count: Arc<Mutex<HashMap<String, usize>>>,
        cache_threshold: f32,
        _eviction_thread: Option<thread::JoinHandle<()>>, // Store thread handle
    },
}

// max_total_num_tokens=432247
// 4char/token
// 7B | 80GB = 432247
// 7B | 640 GB = 3957895

// set it as 2^24 for now

pub enum PolicyConfig {
    RandomConfig,
    RoundRobinConfig,
    ApproxTreeConfig {
        cache_threshold: f32,
    },
}


impl Router {
    pub fn new(worker_urls: Vec<String>, policy_config: PolicyConfig) -> Self {
        match policy_config {
            PolicyConfig::RandomConfig => Router::Random { worker_urls },
            PolicyConfig::RoundRobinConfig => Router::RoundRobin {
                worker_urls,
                current_index: std::sync::atomic::AtomicUsize::new(0),
            },
            PolicyConfig::ApproxTreeConfig {
                cache_threshold,
            } => {
                let mut url_to_count = HashMap::new();
                for url in &worker_urls {
                    url_to_count.insert(url.clone(), 0);
                }

                let tree = Arc::new(Tree::new());
                let url_to_count = Arc::new(Mutex::new(url_to_count));

                // Create background eviction thread
                let tree_clone = Arc::clone(&tree);
                let eviction_thread = thread::spawn(move || {
                    loop {
                        // Sleep for the specified interval
                        thread::sleep(Duration::from_secs(EVICTION_INTERVAL_SECS));

                        // Run eviction
                        tree_clone.evict_tenant_data(MAX_TREE_SIZE);
                    }
                });

                Router::ApproxTree {
                    worker_urls,
                    tree,
                    url_to_count,
                    cache_threshold,
                    _eviction_thread: Some(eviction_thread),
                }
            }
        }
    }

    pub fn get_first(&self) -> Option<String> {
        match self {
            Router::RoundRobin { worker_urls, .. }
            | Router::Random { worker_urls }
            | Router::ApproxTree { worker_urls, .. } => {
                if worker_urls.is_empty() {
                    None
                } else {
                    Some(worker_urls[0].clone())
                }
            }
        }
    }

    pub async fn dispatch(
        &self,
        client: &reqwest::Client,
        req: HttpRequest,
        body: Bytes,
    ) -> HttpResponse {
        let start_time = std::time::Instant::now();

        // 1. convert body to json
        let json = serde_json::from_slice::<serde_json::Value>(&body).unwrap();
        let json_parse_time = start_time.elapsed();
        // println!("JSON parsing took: {:?}", json_parse_time);

        // 2. get the text field
        let text = json.get("text").and_then(|t| t.as_str()).unwrap_or("");
        let text_extract_time = start_time.elapsed() - json_parse_time;
        // println!("Text extraction took: {:?}", text_extract_time);

        let router_start = std::time::Instant::now();
        let worker_url = match self {
            Router::RoundRobin {
                worker_urls,
                current_index,
            } => {
                let idx = current_index
                    .fetch_update(
                        std::sync::atomic::Ordering::SeqCst,
                        std::sync::atomic::Ordering::SeqCst,
                        |x| Some((x + 1) % worker_urls.len()),
                    )
                    .unwrap();

                worker_urls[idx].clone()
            }

            Router::Random { worker_urls } => {
                worker_urls[rand::random::<usize>() % worker_urls.len()].clone()
            }

            Router::ApproxTree {
                worker_urls,
                tree,
                url_to_count,
                cache_threshold,
                ..
            } => {
                let tree_match_start = std::time::Instant::now();
                // 1. Find the highest matched worker
                let (matched_text, matched_worker) = tree.prefix_match(text);
                let matched_rate = matched_text.len() as f32 / text.len() as f32;

                // 2. If the rate is higher than the threshold, select the worker. If not, select the worker with the shortest queue
                let selected_url = if matched_rate > *cache_threshold {
                    let worker_url = matched_worker.to_string();

                    let mut locked_url_to_count = url_to_count.lock().unwrap();

                    // write
                    let count = locked_url_to_count.get_mut(&worker_url).unwrap();
                    *count += 1;
                    worker_url

                } else {
                    // pick the shortest queue from url_to_count
                    let mut locked_url_to_count = url_to_count.lock().unwrap();

                    // read
                    let mut min_count = std::usize::MAX;
                    let mut min_count_indices = Vec::new();

                    // First pass: find the minimum count
                    for (i, url) in worker_urls.iter().enumerate() {
                        let count = locked_url_to_count.get(url).unwrap();
                        if *count < min_count {
                            min_count = *count;
                            min_count_indices.clear();
                            min_count_indices.push(i);
                        } else if *count == min_count {
                            min_count_indices.push(i);
                        }
                    }

                    // Randomly select one of the indices with minimum count
                    use rand::seq::SliceRandom;
                    let min_count_id = *min_count_indices.choose(&mut rand::thread_rng()).unwrap();
                    let worker_url = worker_urls[min_count_id].clone();

                    // write
                    let count = locked_url_to_count.get_mut(&worker_url).unwrap();
                    *count += 1;
                    worker_url

                };

                // let tree_clone = Arc::clone(tree);
                // let worker_url_clone = selected_url.clone();
                // let text_clone = text.to_string();
                // thread::spawn(move || {
                //     // Insert text to the tree
                //     tree_clone.insert(&text_clone, &worker_url_clone);
                // });
                tree.insert(&text, &selected_url);

                selected_url
            }
        };
        let routing_time = router_start.elapsed();
        // println!("Total routing took: {:?}", routing_time);

        // let tree_update_start = std::time::Instant::now();
        // if let Router::ApproxTree {
        //     tree,
        //     url_to_count,
        //     ..
        // } = self
        // {
        //     // let tree_clone = Arc::clone(tree);
        //     // let worker_url_clone = worker_url.clone();
        //     // let text_clone = text.to_string();
        //     // thread::spawn(move || {
        //     //     // Insert text to the tree
        //     //     tree_clone.insert(&text_clone, &worker_url_clone);
        //     // });
        //     tree.insert(&text, &worker_url);

        //     let mut locked_url_to_count = url_to_count.lock().unwrap();
        //     let count = locked_url_to_count.get_mut(&worker_url).unwrap();
        //     *count += 1;
        // }
        // let tree_update_time = tree_update_start.elapsed();
        // println!("Tree update took: {:?}", tree_update_time);

        // Check if client requested streaming
        let stream_check_start = std::time::Instant::now();
        let is_stream = serde_json::from_slice::<serde_json::Value>(&body)
            .map(|v| v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false))
            .unwrap_or(false);
        let stream_check_time = stream_check_start.elapsed();
        // println!("Stream check took: {:?}", stream_check_time);


        let request_start = std::time::Instant::now();
        let res = match client
            .post(format!("{}/generate", worker_url))
            .header(
                "Content-Type",
                req.headers()
                    .get("Content-Type")
                    .and_then(|h| h.to_str().ok())
                    .unwrap_or("application/json"),
            )
            .body(body.to_vec())
            .send()
            .await
        {
            Ok(res) => res,
            Err(_) => return HttpResponse::InternalServerError().finish(),
        };
        let request_time = request_start.elapsed();
        // println!("Worker request took: {:?}", request_time);

        // Mock successful response
        // let res = {
        //     // Create a mock response based on whether streaming is requested
        //     if !is_stream {
        //         // For non-streaming requests, return a mock JSON response
        //         let mock_response = serde_json::json!({
        //             "text":" a man left his post in the dust",
        //             "meta_info": {
        //                 "prompt_tokens":6,
        //                 "completion_tokens":128,
        //                 "completion_tokens_wo_jump_forward":128,
        //                 "cached_tokens":5,
        //                 "finish_reason": {"
        //                    type":"length",
        //                    "length":128
        //                 },"
        //                 id":"bbebb829e55a4ac280bf4b0a8e739df1"
        //             },
        //             "index":0}
        //         );


        //         let body = serde_json::to_vec(&mock_response).unwrap_or_default();
        //         let response = bytes::Bytes::from(body);

        //         // sleep for 1~2 sec
        //         thread::sleep(std::time::Duration::from_secs(1));

        //         reqwest::Response::from(
        //             http::Response::builder()
        //                 .status(200)
        //                 .body(response)
        //                 .unwrap()
        //         )

        //     } else {
        //         // For streaming requests, return a mock SSE response
        //         let mock_events = b"data: {\"id\":\"mock_1\",\"choices\":[{\"text\":\"Hello\"}]}\n\ndata: {\"id\":\"mock_2\",\"choices\":[{\"text\":\" world\"}]}\n\ndata: [DONE]\n\n".to_vec();
        //         let response = bytes::Bytes::from(mock_events);

        //         reqwest::Response::from(
        //             http::Response::builder()
        //                 .status(200)
        //                 .header("Content-Type", "text/event-stream")
        //                 .body(response)
        //                 .unwrap()
        //         )
        //     }
        // };

        // let request_time = request_start.elapsed();
        // println!("Mock request took: {:?}", request_time);

        let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

        let response_start = std::time::Instant::now();
        if !is_stream {
            if let Router::ApproxTree { url_to_count, .. } = self {
                let mut locked_url_to_count = url_to_count.lock().unwrap();
                let count = locked_url_to_count.get_mut(&worker_url).unwrap();
                *count -= 1;

                // print url to count
                // println!("{:?}", locked_url_to_count);
            }

            let response = match res.bytes().await {
                Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                Err(_) => HttpResponse::InternalServerError().finish(),
            };
            let response_time = response_start.elapsed();
            // println!("Response processing took: {:?}", response_time);

            let total_time = start_time.elapsed();
            // println!("Total request processing took: {:?}", total_time);

            response
        } else {
            // TODO: decrement ref count after the streaming is closed
            let response = HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .streaming(res.bytes_stream().map_err(|_| {
                    actix_web::error::ErrorInternalServerError("Failed to read string")
                }));

            println!("alohaaaaaaaa!");

            let response_time = response_start.elapsed();
            // println!("Stream response setup took: {:?}", response_time);

            let total_time = start_time.elapsed();
            // println!("Total request processing took: {:?}", total_time);

            response
        }
    }
}
