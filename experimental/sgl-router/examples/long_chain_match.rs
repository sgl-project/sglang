use sgl_router::policies::kv_events::tree::{HashTree, KvWorkerId};
use std::time::Instant;

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "plain".into());
    let depth: usize = 2500;
    let chain: Vec<i64> = (0..depth as i64)
        .map(|i| i.wrapping_mul(0x9E3779B97F4A7C15u64 as i64) ^ 0x5bd1e995)
        .collect();
    for owners in [1usize, 3, 6] {
        let tree = HashTree::new();
        for w in 0..owners {
            let id = KvWorkerId::new(format!("http://10.0.0.{w}:30000"), 0);
            tree.insert(&id, None, &chain);
        }
        for _ in 0..200 {
            std::hint::black_box(tree.match_prefix(None, &chain));
        }
        let iters = 2000;
        let t = Instant::now();
        for i in 0..iters {
            let result = if mode == "sampled" || mode == "diagnostic" {
                tree.match_prefix_with_options(
                    None,
                    std::hint::black_box(&chain),
                    true,
                    mode == "diagnostic" || i % 128 == 0,
                )
            } else if mode == "disabled" {
                tree.match_prefix_with_options(None, std::hint::black_box(&chain), false, false)
            } else {
                tree.match_prefix(None, std::hint::black_box(&chain))
            };
            std::hint::black_box(result);
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("mode={mode} depth={depth} owners={owners} match_prefix={us:.1}us");
    }
}
