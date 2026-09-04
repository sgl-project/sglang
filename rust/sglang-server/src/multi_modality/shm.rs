//! POSIX shared-memory transport for feature tensors.

use std::sync::atomic::{AtomicU64, Ordering};

/// A named POSIX shared-memory segment owning its name: dropped → unlinked.
///
/// Written by an MM worker so the TP broadcast carries a ~100-byte
/// `ShmPointerMMData` stub instead of the ~20 MB feature tensor, and every
/// rank maps it in parallel. Python's `materialize()` unlinks after cloning;
/// this `Drop` covers the paths where the buffers never reach Python (aborted
/// while parked, late result purged).
pub struct ShmSegment {
    pub(super) name: String,
}

impl ShmSegment {
    /// Create `/dev/shm/{name}` holding exactly `bytes`. No leading slash —
    /// the name must suit Python's `SharedMemory(name=…)` (shm_open adds one).
    pub fn create(name: String, bytes: &[u8]) -> Result<Self, String> {
        let c_name = std::ffi::CString::new(format!("/{name}"))
            .map_err(|_| "shm name contains NUL".to_string())?;
        // SAFETY: plain POSIX calls on a name we own; every handle created
        // below is closed/unmapped on all paths.
        unsafe {
            let fd = libc::shm_open(
                c_name.as_ptr(),
                libc::O_CREAT | libc::O_EXCL | libc::O_RDWR,
                0o600,
            );
            if fd < 0 {
                return Err(format!(
                    "shm_open({name}): {}",
                    std::io::Error::last_os_error()
                ));
            }
            let segment = Self { name }; // unlink from here on any failure
            if libc::ftruncate(fd, bytes.len() as libc::off_t) != 0 {
                let e = std::io::Error::last_os_error();
                libc::close(fd);
                return Err(format!("ftruncate({}): {e}", segment.name));
            }
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                bytes.len(),
                libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            libc::close(fd);
            if ptr == libc::MAP_FAILED {
                return Err(format!(
                    "mmap({}): {}",
                    segment.name,
                    std::io::Error::last_os_error()
                ));
            }
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            libc::munmap(ptr, bytes.len());
            Ok(segment)
        }
    }

    /// Hand the segment — and the duty to unlink — to the caller (Python, at
    /// drain time).
    pub fn into_name(self) -> String {
        std::mem::take(&mut std::mem::ManuallyDrop::new(self).name)
    }
}

impl Drop for ShmSegment {
    fn drop(&mut self) {
        if let Ok(c_name) = std::ffi::CString::new(format!("/{}", self.name)) {
            // SAFETY: unlinking a name we created; ENOENT (already unlinked
            // by Python's materialize) is fine to ignore.
            unsafe { libc::shm_unlink(c_name.as_ptr()) };
        }
    }
}

/// Unique segment names: the pid separates server restarts (a crash can leak
/// segments under the old pid), the counter separates results within one.
pub(super) fn shm_name(item: usize) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("sglmm-{}-{n}-{item}", std::process::id())
}

/// Test helper shared with the result store's parking tests.
#[cfg(test)]
pub(super) fn shm_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new("/dev/shm").join(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The segment holds exactly the written bytes and dropping it unlinks —
    /// the leak guard for results purged before Python takes them.
    #[test]
    fn segment_roundtrip_and_drop_unlinks() {
        let name = shm_name(0);
        let payload: Vec<u8> = (0..255u8).collect();
        let segment = ShmSegment::create(name.clone(), &payload).unwrap();
        assert_eq!(std::fs::read(shm_path(&name)).unwrap(), payload);
        drop(segment);
        assert!(!shm_path(&name).exists(), "drop must unlink");
    }

    /// `into_name` transfers the unlink duty to the caller (Python's
    /// `materialize()`), so the segment must survive the handoff.
    #[test]
    fn into_name_disarms_the_unlink() {
        let segment = ShmSegment::create(shm_name(0), &[1, 2, 3]).unwrap();
        let name = segment.into_name();
        assert!(shm_path(&name).exists(), "handoff must not unlink");
        // manual cleanup for the test
        let c = std::ffi::CString::new(format!("/{name}")).unwrap();
        unsafe { libc::shm_unlink(c.as_ptr()) };
    }
}
