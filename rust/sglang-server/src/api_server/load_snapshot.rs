//! Reader for the per-rank load snapshots published by Python's scheduler.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use rustix::fs::{FlockOperation, flock};

/// The file header, Python's `HEADER_STRUCT` (`<4sHHI`): magic, version,
/// dp_size, slot_size. Each rank's slot follows at `HEADER_LEN + rank *
/// slot_size`: a little-endian `u32` length, then the msgpack snapshot.
const MAGIC: &[u8; 4] = b"SLNS";
const VERSION: u16 = 2;
const HEADER_LEN: u64 = 12;
const SLOT_LEN_PREFIX: u64 = 4;

/// The decoded snapshots of `ranks`, read under the shared lock Python's
/// reader takes. A missing or mismatched file, or an empty or malformed slot,
/// yields nothing for that rank, as Python's reader does.
pub(super) fn read_snapshots(
    path: &Path,
    dp_size: usize,
    ranks: &[usize],
    lock: FlockOperation,
) -> Vec<rmpv::Value> {
    let Ok(file) = File::open(path) else {
        return Vec::new();
    };
    if flock(&file, lock).is_err() {
        return Vec::new();
    }
    let mut header = [0u8; HEADER_LEN as usize];
    if file.read_exact_at(&mut header, 0).is_err() {
        return Vec::new();
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    let file_dp_size = usize::from(u16::from_le_bytes([header[6], header[7]]));
    let slot_size = u64::from(u32::from_le_bytes([
        header[8], header[9], header[10], header[11],
    ]));
    let file_len = file.metadata().map(|meta| meta.len()).unwrap_or(0);
    if &header[..4] != MAGIC
        || version != VERSION
        || file_dp_size != dp_size
        || slot_size < SLOT_LEN_PREFIX
        || file_len < HEADER_LEN + dp_size as u64 * slot_size
    {
        tracing::warn!(path = %path.display(), "load snapshot header mismatch");
        return Vec::new();
    }
    ranks
        .iter()
        .filter(|&&rank| rank < dp_size)
        .filter_map(|&rank| {
            let offset = HEADER_LEN + rank as u64 * slot_size;
            let mut len = [0u8; SLOT_LEN_PREFIX as usize];
            file.read_exact_at(&mut len, offset).ok()?;
            let len = u64::from(u32::from_le_bytes(len));
            if len == 0 || len > slot_size - SLOT_LEN_PREFIX {
                return None;
            }
            let mut payload = vec![0u8; len as usize];
            file.read_exact_at(&mut payload, offset + SLOT_LEN_PREFIX)
                .ok()?;
            rmpv::decode::read_value(&mut payload.as_slice()).ok()
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::{MAGIC, VERSION};
    use serde_json::Value;

    pub(crate) fn fixture() -> Value {
        serde_json::from_str(include_str!("../../testdata/load_snapshot.json")).unwrap()
    }

    /// The fixture's payloads in Python's file format, one slot per rank.
    pub(crate) struct SnapshotFile(pub(crate) std::path::PathBuf);

    impl SnapshotFile {
        pub(crate) fn new(dp_size: usize) -> Self {
            let fixture = fixture();
            let payloads = fixture["payloads"].as_array().unwrap();
            assert_eq!(payloads.len(), dp_size);
            let payloads = payloads
                .iter()
                .map(|payload| {
                    let hex = payload.as_str().unwrap();
                    (0..hex.len())
                        .step_by(2)
                        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                        .collect()
                })
                .collect::<Vec<_>>();
            Self::from_payloads(&payloads)
        }

        pub(crate) fn from_payloads(payloads: &[Vec<u8>]) -> Self {
            let slot_size = 16 * 1024u32;
            let mut bytes = Vec::from(&MAGIC[..]);
            bytes.extend(VERSION.to_le_bytes());
            bytes.extend(u16::try_from(payloads.len()).unwrap().to_le_bytes());
            bytes.extend(slot_size.to_le_bytes());
            for payload in payloads {
                let mut slot = (payload.len() as u32).to_le_bytes().to_vec();
                slot.extend_from_slice(payload);
                slot.resize(slot_size as usize, 0);
                bytes.extend(slot);
            }
            let path = std::env::temp_dir().join(format!("sglang-loads-{}", uuid::Uuid::new_v4()));
            std::fs::write(&path, bytes).unwrap();
            Self(path)
        }
    }

    impl Drop for SnapshotFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}
