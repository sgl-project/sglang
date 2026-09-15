//! POSIX shared-memory segments (`shm_open` family) via `rustix`.
//!
//! `rustix` over raw `libc`: it hands back an `OwnedFd` (closed on every
//! path for free), takes a `u64` length so the `off_t` width never matters,
//! and widens `mode_t` for the variadic `shm_open` on Apple where the raw
//! call would be UB.

use rustix::fs::{Mode, ftruncate};
use rustix::mm::{MapFlags, ProtFlags, mmap, munmap};
use rustix::shm::{OFlags, open, unlink};

/// A named POSIX shared-memory segment owning its name: dropped → unlinked.
///
/// Written by an MM worker so the TP broadcast carries a ~100-byte
/// `ShmPointerMMData` stub instead of the ~20 MB feature tensor, and every
/// rank maps it in parallel. Python's `materialize()` unlinks after cloning;
/// this `Drop` covers the paths where the buffers never reach Python (aborted
/// while parked, late result purged).
#[derive(Debug)]
pub struct ShmSegment {
    name: String,
}

impl ShmSegment {
    /// Create the segment `name` holding exactly `bytes`. No leading slash —
    /// the name must suit Python's `SharedMemory(name=…)` (shm_open adds one).
    pub fn create(name: String, bytes: &[u8]) -> Result<Self, String> {
        let fd = open(
            format!("/{name}"),
            OFlags::CREATE | OFlags::EXCL | OFlags::RDWR,
            Mode::RUSR | Mode::WUSR,
        )
        .map_err(|e| format!("shm_open({name}): {e}"))?;
        let segment = Self { name }; // unlink from here on any failure
        ftruncate(&fd, bytes.len() as u64)
            .map_err(|e| format!("ftruncate({}): {e}", segment.name))?;
        // SAFETY: a fresh mapping independent of any existing allocation,
        // unmapped below before `fd` drops.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                bytes.len(),
                ProtFlags::WRITE,
                MapFlags::SHARED,
                &fd,
                0,
            )
        }
        .map_err(|e| format!("mmap({}): {e}", segment.name))?;
        // SAFETY: `ptr` is a writable mapping of exactly `bytes.len()` bytes
        // that cannot overlap `bytes`; nothing references it after the unmap.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            let _ = munmap(ptr, bytes.len());
        }
        Ok(segment)
    }

    /// Test helper: the name as Python's `SharedMemory(name=…)` sees it.
    #[cfg(test)]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Hand the segment — and the duty to unlink — to the caller (Python, at
    /// drain time).
    pub fn into_name(self) -> String {
        std::mem::take(&mut std::mem::ManuallyDrop::new(self).name)
    }
}

impl Drop for ShmSegment {
    fn drop(&mut self) {
        // ENOENT (already unlinked by Python's materialize) is fine to ignore.
        let _ = unlink(format!("/{}", self.name));
    }
}

/// Test helper: where Linux exposes the segment as a file.
#[cfg(test)]
pub fn shm_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new("/dev/shm").join(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_name() -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("sglshm-test-{}-{n}", std::process::id())
    }

    /// The segment holds exactly the written bytes and dropping it unlinks —
    /// the leak guard for results purged before Python takes them.
    #[test]
    fn segment_roundtrip_and_drop_unlinks() {
        let name = test_name();
        let payload: Vec<u8> = (0..255u8).collect();
        let segment = ShmSegment::create(name.clone(), &payload).unwrap();
        assert_eq!(segment.name(), name);
        assert_eq!(std::fs::read(shm_path(&name)).unwrap(), payload);
        drop(segment);
        assert!(!shm_path(&name).exists(), "drop must unlink");
    }

    /// `into_name` transfers the unlink duty to the caller (Python's
    /// `materialize()`), so the segment must survive the handoff.
    #[test]
    fn into_name_disarms_the_unlink() {
        let segment = ShmSegment::create(test_name(), &[1, 2, 3]).unwrap();
        let name = segment.into_name();
        assert!(shm_path(&name).exists(), "handoff must not unlink");
        unlink(format!("/{name}")).unwrap(); // manual cleanup for the test
    }

    /// A name that is already taken is an error, not a silent overwrite of
    /// another result's segment.
    #[test]
    fn duplicate_name_is_rejected() {
        let name = test_name();
        let _first = ShmSegment::create(name.clone(), &[1]).unwrap();
        let err = ShmSegment::create(name, &[2]).unwrap_err();
        assert!(err.starts_with("shm_open("), "{err}");
    }
}
