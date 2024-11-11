use crate::tree::RadixTree;
use actix_web::http::header::{HeaderValue, CONTENT_TYPE};
use actix_web::{HttpRequest, HttpResponse};
use bytes::Bytes;
use futures_util::TryStreamExt;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use tokenizers::tokenizer::Tokenizer;

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
        // TODO: don't lock the whole tree
        url_to_tree: Arc<Mutex<HashMap<String, RadixTree>>>,
        tokenizer: Tokenizer,
        url_to_count: Arc<Mutex<HashMap<String, usize>>>,
        cache_threshold: f32,
    },
}

pub enum PolicyConfig {
    RandomConfig,
    RoundRobinConfig,
    ApproxTreeConfig {
        tokenizer_path: String,
        cache_threshold: f32,
    },
}

fn get_token_ids_from_request(body: &Bytes, tokenizer: &Tokenizer) -> Vec<u32> {
    // 1. convert body to json
    let json = serde_json::from_slice::<serde_json::Value>(body).unwrap();
    // 2. get the text field
    let text = json.get("text").and_then(|t| t.as_str()).unwrap_or("");
    // 3. tokenize the text field
    let tokens = tokenizer.encode(text, false).unwrap();

    tokens.get_ids().to_vec()
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
                tokenizer_path,
                cache_threshold,
            } => {
                let mut url_to_tree = HashMap::new();
                let mut url_to_count = HashMap::new();

                for url in &worker_urls {
                    url_to_tree.insert(url.clone(), RadixTree::new());
                    url_to_count.insert(url.clone(), 0);
                }

                Router::ApproxTree {
                    worker_urls,
                    url_to_tree: Arc::new(Mutex::new(url_to_tree)),
                    // TODO: rust ::from_pretrained cannot load from local file, so use ::from_file to load local file
                    tokenizer: Tokenizer::from_file(tokenizer_path).unwrap(),
                    url_to_count: Arc::new(Mutex::new(url_to_count)),
                    cache_threshold,
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
        let mut input_ids: Vec<u32> = Vec::new();
        if let Router::ApproxTree { tokenizer, .. } = self {
            input_ids = get_token_ids_from_request(&body, tokenizer);
        }

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
                url_to_tree,
                url_to_count,
                cache_threshold,
                ..
            } => {
                // TODO: pipeline the locks. Release one earlier.

                let mut max_matched_rate = 0.0;
                let mut max_matched_idx = 0;

                let locked_url_to_tree = url_to_tree.lock().unwrap();

                // 1. Find the highest matched worker
                for (i, url) in worker_urls.iter().enumerate() {
                    let tree = locked_url_to_tree.get(url).unwrap();
                    let matched = tree.prefix_match(&input_ids[..]).len();
                    let matched_rate = matched as f32 / input_ids.len() as f32;

                    if matched_rate > max_matched_rate {
                        max_matched_rate = matched_rate;
                        max_matched_idx = i;
                    }
                }

                // 2. If the rate is higher than the threshold, select the worker. If not, select the worker with the shortest queue
                if max_matched_rate > *cache_threshold {
                    worker_urls[max_matched_idx].clone()
                } else {
                    // pick the shortest queue from url_to_count
                    let locked_url_to_count = url_to_count.lock().unwrap();

                    let mut min_count = std::usize::MAX;
                    let mut min_count_id = 0;

                    for (i, url) in worker_urls.iter().enumerate() {
                        let count = locked_url_to_count.get(url).unwrap();
                        if *count < min_count {
                            min_count = *count;
                            min_count_id = i;
                        }
                    }

                    worker_urls[min_count_id].clone()
                }
            }
        };

        if let Router::ApproxTree {
            url_to_tree,
            url_to_count,
            ..
        } = self
        {
            // Insert input_ids to the tree
            let mut locked_url_to_tree = url_to_tree.lock().unwrap();
            let selected_tree = locked_url_to_tree.get_mut(&worker_url).unwrap();
            selected_tree.insert(&input_ids[..]);

            let mut locked_url_to_count = url_to_count.lock().unwrap();
            let count = locked_url_to_count.get_mut(&worker_url).unwrap();
            *count += 1;
        }

        // Check if client requested streaming
        let is_stream = serde_json::from_slice::<serde_json::Value>(&body)
            .map(|v| v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false))
            .unwrap_or(false);

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

        let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

        if !is_stream {
            // TODO: do the correction on the tree based on the cached input_ids
            if let Router::ApproxTree { url_to_count, .. } = self {
                let mut locked_url_to_count = url_to_count.lock().unwrap();
                let count = locked_url_to_count.get_mut(&worker_url).unwrap();
                *count -= 1;
            }

            match res.bytes().await {
                Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                Err(_) => HttpResponse::InternalServerError().finish(),
            }
        } else {
            // TODO: do the correction on the tree based on the cached input_ids. The streaming might be tricker to handle
            HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .streaming(res.bytes_stream().map_err(|_| {
                    actix_web::error::ErrorInternalServerError("Failed to read string")
                }))
        }
    }
}
