//! Read-only mapped bytes for large, kernel-protected local media. The
//! mapping owns the input bytes across the asynchronous handoff to the
//! processor, which receives the same byte stream as the buffered reader.

use std::fs::File;
use std::ptr::{self, NonNull};
use std::sync::LazyLock;

use bytes::Bytes;
use rustix::fs::{AtFlags, SealFlags, StatxAttributes, StatxFlags, fcntl_get_seals, statx};
use rustix::mm::{Advice, MapFlags, ProtFlags, madvise, mmap, munmap};
use sglang_mm::common::fetch::{ByteBudget, read_local_file_budgeted};

const MIN_MAPPED_BYTES: u64 = 32 << 20;
const IMMUTABLE_SEALS: SealFlags = SealFlags::WRITE
    .union(SealFlags::SHRINK)
    .union(SealFlags::GROW);
static ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("SGLANG_RUST_MMAP_LOCAL_MEDIA").as_deref() == Ok("1"));

pub(super) fn enabled() -> bool {
    *ENABLED
}

pub(super) fn fetch(src: &str, budget: &ByteBudget) -> Result<Bytes, String> {
    let path = src.strip_prefix("file://").unwrap_or(src);
    let file = File::open(path).map_err(|err| format!("media fetch: {path}: {err}"))?;
    if let Some(bytes) =
        map_if_immutable(&file, path, budget, MIN_MAPPED_BYTES, MappedFile::populate)?
    {
        return Ok(bytes);
    }
    read_local_file_budgeted(file, path, budget).map(Bytes::from)
}

fn map_if_immutable(
    file: &File,
    path: &str,
    budget: &ByteBudget,
    minimum: u64,
    populate: impl FnOnce(&MappedFile) -> rustix::io::Result<()>,
) -> Result<Option<Bytes>, String> {
    let metadata = file
        .metadata()
        .map_err(|err| format!("media fetch: {path}: {err}"))?;
    if !metadata.is_file() || metadata.len() < minimum {
        return Ok(None);
    }
    let sealed = fcntl_get_seals(file).is_ok_and(|flags| flags.contains(IMMUTABLE_SEALS));
    let verity = || {
        statx(file, "", AtFlags::EMPTY_PATH, StatxFlags::BASIC_STATS).is_ok_and(|stat| {
            stat.stx_attributes_mask.contains(StatxAttributes::VERITY)
                && stat.stx_attributes.contains(StatxAttributes::VERITY)
        })
    };
    if !sealed && !verity() {
        return Ok(None);
    }
    // A producer could have resized the file before adding the seals or
    // enabling verity. Read the length again after establishing immutability.
    let len = file
        .metadata()
        .map_err(|err| format!("media fetch: {path}: {err}"))?
        .len();
    let Ok(len) = usize::try_from(len) else {
        return Ok(None);
    };
    if len == 0 || len > isize::MAX as usize || (len as u64) < minimum {
        return Ok(None);
    }

    // SAFETY: the kernel prevents writes and resizing on this open inode.
    // It stays immutable even if the pathname is unlinked or replaced.
    let Ok(addr) = (unsafe {
        mmap(
            ptr::null_mut(),
            len,
            ProtFlags::READ,
            MapFlags::PRIVATE,
            file,
            0,
        )
    }) else {
        return Ok(None);
    };
    let Some(address) = NonNull::new(addr.cast::<u8>()) else {
        // SAFETY: mmap succeeded for this address and extent.
        let _ = unsafe { munmap(addr, len) };
        return Ok(None);
    };
    let region = MappedFile { address, len };
    // Populate while on the blocking worker. If the kernel cannot populate
    // the mapping, discard it and let the buffered reader handle the file.
    budget.charge_if_loaded(len, path, || {
        populate(&region).ok()?;
        Some(Bytes::from_owner(region))
    })
}

struct MappedFile {
    address: NonNull<u8>,
    len: usize,
}

impl MappedFile {
    fn populate(&self) -> rustix::io::Result<()> {
        // SAFETY: self owns a live read-only mapping of exactly len bytes.
        unsafe {
            madvise(
                self.address.as_ptr().cast(),
                self.len,
                Advice::LinuxPopulateRead,
            )
        }
    }
}

// SAFETY: this owns an immutable mapping, which is unmapped only when its
// final Bytes owner is dropped.
unsafe impl Send for MappedFile {}

impl AsRef<[u8]> for MappedFile {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: the mapping is live, immutable, and covered by self.
        unsafe { std::slice::from_raw_parts(self.address.as_ptr(), self.len) }
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        // SAFETY: this is the exact range returned by mmap and owned by self.
        let _ = unsafe { munmap(self.address.as_ptr().cast(), self.len) };
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use rustix::fs::{MemfdFlags, fcntl_add_seals, memfd_create};

    use super::*;

    fn fixture(seals: SealFlags) -> File {
        let mut file =
            File::from(memfd_create("sglang-test-media", MemfdFlags::ALLOW_SEALING).unwrap());
        file.write_all(b"image payload").unwrap();
        fcntl_add_seals(&file, seals).unwrap();
        file
    }

    #[test]
    fn sealed_media_stays_live_after_close_and_charges_budget() {
        let file = fixture(IMMUTABLE_SEALS);
        let budget = ByteBudget::new(16);
        let bytes = map_if_immutable(&file, "image", &budget, 1, |_| Ok(()))
            .unwrap()
            .unwrap();
        assert!(file.set_len(0).is_err());
        drop(file);
        assert_eq!(&bytes.slice(6..)[..], b"payload");
        assert!(budget.charge_existing(4, "second image").is_err());
    }

    #[test]
    fn partial_seals_or_small_inputs_keep_the_buffered_path() {
        let budget = ByteBudget::new(13);
        let mutable = fixture(SealFlags::WRITE);
        assert!(
            map_if_immutable(&mutable, "image", &budget, 1, MappedFile::populate)
                .unwrap()
                .is_none()
        );
        let sealed = fixture(IMMUTABLE_SEALS);
        assert!(
            map_if_immutable(&sealed, "image", &budget, 14, MappedFile::populate)
                .unwrap()
                .is_none()
        );
        budget.charge_existing(13, "image").unwrap();
    }

    #[test]
    fn mapped_media_rejects_an_over_budget_file() {
        let sealed = fixture(IMMUTABLE_SEALS);
        let error = map_if_immutable(&sealed, "image", &ByteBudget::new(12), 1, |_| {
            panic!("an over-budget mapping must not populate")
        })
        .unwrap_err();
        assert!(error.contains("budget"));
    }

    #[test]
    fn populate_failure_refunds_budget_for_buffered_read() {
        use std::io::{Seek, SeekFrom};

        let mut sealed = fixture(IMMUTABLE_SEALS);
        sealed.seek(SeekFrom::Start(0)).unwrap();
        let budget = ByteBudget::new(13);
        let mapped = map_if_immutable(&sealed, "image", &budget, 1, |_| {
            Err(rustix::io::Errno::INVAL)
        })
        .unwrap();
        assert!(mapped.is_none());
        let bytes = read_local_file_budgeted(sealed, "image", &budget).unwrap();
        assert_eq!(bytes, b"image payload");
        assert!(budget.charge_existing(1, "second image").is_err());
    }
}
