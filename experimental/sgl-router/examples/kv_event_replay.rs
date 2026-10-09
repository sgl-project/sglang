//! Offline regression: re-encode a token-stripped JSONL capture with synthetic
//! bigram tokens, then decode and replay the complete event stream.
//! Usage: cargo run --release --example kv_event_replay -- capture.jsonl
use rmp::encode as mp;
use serde_json::Value;
use sgl_router::policies::kv_events::wire::{decode_event_batch, KvCacheEvent};
use sgl_router::policies::kv_events::{HashTree, KvWorkerId, Tiers};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected capture.jsonl")?;
    let tree = HashTree::new();
    let worker = KvWorkerId::new("http://capture".into(), 0);
    let mut unique: HashMap<i64, Tiers> = HashMap::new();
    let (mut batches, mut old_cap_batches, mut gaps) = (0u64, 0u64, 0u64);
    let mut last_seq = None;
    for line in BufReader::new(File::open(path)?).lines() {
        let row: Value = serde_json::from_str(&line?)?;
        let Some(batch) = row.get("batch").and_then(Value::as_array) else {
            continue;
        };
        let seq = row["seq"].as_i64().ok_or("missing seq")?;
        if let Some(previous) = last_seq {
            if seq != previous + 1 {
                gaps += 1;
            }
        }
        last_seq = Some(seq);
        let events = batch[1].as_array().ok_or("missing events")?;
        let mut payload = Vec::new();
        mp::write_array_len(&mut payload, batch.len() as u32)?;
        payload.extend(rmp_serde::to_vec_named(&batch[0])?);
        mp::write_array_len(&mut payload, events.len() as u32)?;
        let mut exceeds_old_cap = false;
        for event in events {
            let fields = event.as_object().ok_or("expected map event")?;
            let stored = event["type"] == "BlockStored";
            let fields_len =
                fields.len() - usize::from(fields.contains_key("token_ids")) + usize::from(stored);
            mp::write_map_len(&mut payload, fields_len as u32)?;
            for (key, value) in fields {
                if key == "token_ids" {
                    continue;
                }
                mp::write_str(&mut payload, key)?;
                payload.extend(rmp_serde::to_vec_named(value)?);
            }
            if stored {
                let positions = event["block_hashes"]
                    .as_array()
                    .ok_or("missing hashes")?
                    .len()
                    * event["block_size"].as_u64().ok_or("missing block size")? as usize;
                exceeds_old_cap |= positions > 1_048_576 / 2;
                mp::write_str(&mut payload, "token_ids")?;
                mp::write_array_len(&mut payload, positions.try_into()?)?;
                for _ in 0..positions {
                    payload.extend_from_slice(&[0x92, 10, 20]);
                }
            }
        }
        for tail in &batch[2..] {
            payload.extend(rmp_serde::to_vec_named(tail)?);
        }
        let decoded = decode_event_batch(&payload)?;
        for event in decoded.events {
            match event {
                KvCacheEvent::BlockStored(b) => {
                    assert!(b.token_ids.is_empty());
                    let tiers = Tiers::for_store(b.medium.as_deref());
                    tree.insert_tiered(&worker, b.parent_block_hash, &b.block_hashes, tiers);
                    for hash in b.block_hashes {
                        unique.entry(hash).or_default().insert(tiers);
                    }
                }
                KvCacheEvent::BlockRemoved(b) => {
                    let tiers = Tiers::for_remove(b.medium.as_deref());
                    tree.remove_tiered(&worker, &b.block_hashes, tiers);
                    for hash in b.block_hashes {
                        if let Some(current) = unique.get_mut(&hash) {
                            current.remove(tiers);
                        }
                    }
                }
                KvCacheEvent::AllBlocksCleared => {
                    tree.clear_worker(&worker);
                    unique.clear();
                }
            }
        }
        batches += 1;
        old_cap_batches += u64::from(exceeds_old_cap);
    }
    println!("batches={batches} decoded={batches} sequence_discontinuities={gaps} batches_exceeding_old_token_cap={old_cap_batches}");
    println!("tree_occupancy={:?}", tree.tier_occupancy());
    for (tier, label) in Tiers::SLOTS {
        println!(
            "unique_{label}={}",
            unique.values().filter(|v| v.contains(tier)).count()
        );
    }
    println!(
        "unique_nonempty_hashes={}",
        unique.values().filter(|v| !v.is_empty()).count()
    );
    Ok(())
}
