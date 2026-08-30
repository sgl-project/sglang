use std::io;
use std::os::fd::RawFd;
use std::ptr::NonNull;
use std::sync::Mutex;

use io_uring::{IoUring, opcode, types};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn to_py_os_error(error: io::Error) -> PyErr {
    match error.raw_os_error() {
        Some(code) => PyOSError::new_err((code, error.to_string())),
        None => PyOSError::new_err(error.to_string()),
    }
}

struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
}

// The allocation is exclusively accessed while ReaderState's mutex is held.
unsafe impl Send for AlignedBuffer {}

impl AlignedBuffer {
    fn new(len: usize, alignment: usize) -> io::Result<Self> {
        if len == 0 || !alignment.is_power_of_two() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "buffer length must be positive and alignment must be a power of two",
            ));
        }
        let mut raw = std::ptr::null_mut();
        // SAFETY: posix_memalign initializes `raw` on success. Its allocation
        // is owned by this type and released exactly once in Drop.
        let result = unsafe { libc::posix_memalign(&mut raw, alignment, len) };
        if result != 0 {
            return Err(io::Error::from_raw_os_error(result));
        }
        let ptr = NonNull::new(raw.cast::<u8>())
            .ok_or_else(|| io::Error::other("posix_memalign returned a null pointer"))?;
        Ok(Self { ptr, len })
    }

    fn page_ptr(&self, index: usize, page_size: usize) -> *mut u8 {
        debug_assert!((index + 1) * page_size <= self.len);
        // SAFETY: the caller bounds `index` by max_batch and the allocation is
        // max_batch * page_size bytes.
        unsafe { self.ptr.as_ptr().add(index * page_size) }
    }

    fn page(&self, index: usize, page_size: usize, read_size: usize) -> &[u8] {
        debug_assert!(read_size <= page_size);
        // SAFETY: the kernel has completed the read before this view is made,
        // and the mutex prevents another read from mutating the allocation.
        unsafe { std::slice::from_raw_parts(self.page_ptr(index, page_size), read_size) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: ptr came from posix_memalign and has not been freed.
        unsafe { libc::free(self.ptr.as_ptr().cast()) };
    }
}

struct ReaderState {
    ring: IoUring,
    buffer: AlignedBuffer,
}

impl ReaderState {
    fn read_pages(
        &mut self,
        file_descriptors: &[RawFd],
        offsets: &[u64],
        queue_depth: usize,
        max_batch: usize,
        page_size: usize,
    ) -> io::Result<Vec<Vec<u8>>> {
        if file_descriptors.len() != offsets.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "file descriptor and offset counts differ",
            ));
        }
        if offsets.len() > max_batch {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "read has {} pages but max_batch is {max_batch}",
                    offsets.len()
                ),
            ));
        }

        let count = offsets.len();
        let mut read_sizes = vec![0_i32; count];
        for base in (0..count).step_by(queue_depth) {
            let batch = (count - base).min(queue_depth);
            {
                let mut submission = self.ring.submission();
                for local in 0..batch {
                    let index = base + local;
                    let entry = opcode::Read::new(
                        types::Fd(file_descriptors[index]),
                        self.buffer.page_ptr(index, page_size),
                        page_size as u32,
                    )
                    .offset(offsets[index])
                    .build()
                    .user_data((index + 1) as u64);
                    // SAFETY: every read points into the persistent aligned
                    // allocation, and all entries complete before reuse.
                    unsafe { submission.push(&entry) }.map_err(|_| {
                        io::Error::new(io::ErrorKind::WouldBlock, "io_uring queue is full")
                    })?;
                }
            }

            self.ring.submit_and_wait(batch)?;
            let mut completion = self.ring.completion();
            let mut first_error = None;
            for _ in 0..batch {
                let entry = completion.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::UnexpectedEof, "missing io_uring completion")
                })?;
                let user_data = entry.user_data();
                if user_data == 0 || user_data > count as u64 {
                    first_error.get_or_insert_with(|| {
                        io::Error::other("invalid io_uring completion user_data")
                    });
                    continue;
                }
                let result = entry.result();
                if result < 0 {
                    first_error.get_or_insert_with(|| io::Error::from_raw_os_error(-result));
                } else {
                    read_sizes[user_data as usize - 1] = result;
                }
            }
            if let Some(error) = first_error {
                return Err(error);
            }
        }

        read_sizes
            .into_iter()
            .enumerate()
            .map(|(index, read_size)| {
                if read_size <= 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!("short io_uring read at offset {}", offsets[index]),
                    ));
                }
                Ok(self
                    .buffer
                    .page(index, page_size, read_size as usize)
                    .to_vec())
            })
            .collect()
    }
}

#[pyclass]
pub(crate) struct IoUringReader {
    queue_depth: usize,
    max_batch: usize,
    page_size: usize,
    state: Mutex<ReaderState>,
}

#[pymethods]
impl IoUringReader {
    #[new]
    #[pyo3(signature = (queue_depth=512, max_batch=4096, page_size=4096))]
    fn new(queue_depth: usize, max_batch: usize, page_size: usize) -> PyResult<Self> {
        if queue_depth == 0 || max_batch == 0 {
            return Err(PyValueError::new_err(
                "queue_depth and max_batch must be positive",
            ));
        }
        if !page_size.is_power_of_two() {
            return Err(PyValueError::new_err(
                "page_size must be a positive power of two",
            ));
        }
        if page_size > u32::MAX as usize {
            return Err(PyValueError::new_err("page_size exceeds u32"));
        }
        let buffer_len = max_batch
            .checked_mul(page_size)
            .ok_or_else(|| PyValueError::new_err("buffer size overflows usize"))?;
        let ring = IoUring::new(
            queue_depth
                .try_into()
                .map_err(|_| PyValueError::new_err("queue_depth exceeds u32"))?,
        )
        .map_err(to_py_os_error)?;
        let buffer = AlignedBuffer::new(buffer_len, page_size).map_err(to_py_os_error)?;
        Ok(Self {
            queue_depth,
            max_batch,
            page_size,
            state: Mutex::new(ReaderState { ring, buffer }),
        })
    }

    fn read_pages(
        &self,
        py: Python<'_>,
        file_descriptors: Vec<RawFd>,
        offsets: Vec<u64>,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        let pages = py
            .detach(|| {
                let mut state = self
                    .state
                    .lock()
                    .map_err(|_| io::Error::other("io_uring reader lock is poisoned"))?;
                state.read_pages(
                    &file_descriptors,
                    &offsets,
                    self.queue_depth,
                    self.max_batch,
                    self.page_size,
                )
            })
            .map_err(to_py_os_error)?;
        Ok(pages
            .into_iter()
            .map(|page| PyBytes::new(py, &page).unbind())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{self, OpenOptions};
    use std::io::Write;
    use std::os::fd::AsRawFd;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{AlignedBuffer, IoUring, ReaderState};

    #[test]
    fn reads_multiple_pages_in_submission_batches() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("sglang-storage-{}-{nonce}.bin", std::process::id()));
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.write_all(&vec![0x31; 4096]).unwrap();
        file.write_all(&vec![0x72; 4096]).unwrap();
        file.sync_all().unwrap();

        let mut state = ReaderState {
            ring: IoUring::new(1).unwrap(),
            buffer: AlignedBuffer::new(8192, 4096).unwrap(),
        };
        let pages = state
            .read_pages(
                &[file.as_raw_fd(), file.as_raw_fd()],
                &[0, 4096],
                1,
                2,
                4096,
            )
            .unwrap();
        assert_eq!(pages[0], vec![0x31; 4096]);
        assert_eq!(pages[1], vec![0x72; 4096]);

        drop(file);
        fs::remove_file(path).unwrap();
    }
}
