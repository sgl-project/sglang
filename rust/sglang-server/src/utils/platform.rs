//! The OS-specific calls the frontend makes, behind one portable surface:
//! CPU pinning, a close-on-exec pipe, shm page reservation, and the
//! drain-ended datagram. Every `cfg(target_os)` in the crate's non-test
//! code lives here, so the stages and the runtime read the same on every
//! platform; the test-side counterparts sit in [`test_support`] at the end
//! of the file.
//!
//! The server is deployed on Linux, and the Linux arms are the real ones.
//! The other arms exist so the crate builds and its unit tests run on a
//! developer's Mac; each one says what it gives up.

use std::io;
use std::os::fd::OwnedFd;

#[cfg(all(test, target_os = "linux"))]
pub use test_support::bind_notify_socket;
#[cfg(test)]
pub use test_support::{shm_exists, shm_read};

// ---- CPU pinning ----

/// The cores a frontend thread may run on: a kernel CPU mask on Linux. No
/// other kernel the server builds on can pin a thread to a core, so there
/// the type has no values.
#[cfg(target_os = "linux")]
pub type CpuSet = rustix::thread::CpuSet;

#[cfg(not(target_os = "linux"))]
#[derive(Clone, Copy, Debug)]
pub enum CpuSet {}

/// The CPU set for `cores`; `None` when empty or when pinning is unsupported.
#[cfg(target_os = "linux")]
pub fn cpu_set(cores: &[usize]) -> Option<CpuSet> {
    let mut set = CpuSet::new();
    for &core in cores {
        if core < CpuSet::MAX_CPU {
            set.set(core);
        } else {
            tracing::warn!(core, "frontend core id out of range; ignored");
        }
    }
    (set.count() > 0).then_some(set)
}

#[cfg(not(target_os = "linux"))]
pub fn cpu_set(cores: &[usize]) -> Option<CpuSet> {
    if !cores.is_empty() {
        tracing::warn!("frontend cores ignored: CPU pinning is Linux-only");
    }
    None
}

/// Pin the calling thread to `cpus` if a set was given (no-op otherwise).
#[cfg(target_os = "linux")]
pub fn pin_current(cpus: Option<&CpuSet>) {
    if let Some(set) = cpus
        && let Err(error) = rustix::thread::sched_setaffinity(None, set)
    {
        tracing::warn!(%error, "could not pin a frontend thread to the frontend cores");
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_current(_cpus: Option<&CpuSet>) {}

// ---- close-on-exec pipe ----

/// A pipe with both ends close-on-exec: `(read, write)`.
pub fn pipe_cloexec() -> io::Result<(OwnedFd, OwnedFd)> {
    #[cfg(target_os = "linux")]
    {
        Ok(rustix::pipe::pipe_with(rustix::pipe::PipeFlags::CLOEXEC)?)
    }
    #[cfg(not(target_os = "linux"))]
    {
        use rustix::io::{FdFlags, fcntl_setfd};
        // No pipe2 here, so a fork+exec on another thread between the two
        // calls inherits the ends; accepted on a platform we do not deploy to.
        let (read, write) = rustix::pipe::pipe()?;
        fcntl_setfd(&read, FdFlags::CLOEXEC)?;
        fcntl_setfd(&write, FdFlags::CLOEXEC)?;
        Ok((read, write))
    }
}

// ---- shared-memory page reservation ----

/// Reserve `len` bytes of backing pages for a fresh shm segment of that
/// length, so a full `/dev/shm` surfaces as `ENOSPC` here instead of as a
/// `SIGBUS` on the first write; `ftruncate` alone lets tmpfs allocate lazily.
#[cfg(target_os = "linux")]
pub fn reserve_shm_pages(fd: std::os::fd::BorrowedFd<'_>, len: u64) -> rustix::io::Result<()> {
    rustix::fs::fallocate(fd, rustix::fs::FallocateFlags::empty(), 0, len)
}

/// No tmpfs to run out of off Linux, and rustix's Apple `fallocate` is
/// `F_PREALLOCATE`, which a shm fd rejects with `EBADF`.
#[cfg(not(target_os = "linux"))]
pub fn reserve_shm_pages(_fd: std::os::fd::BorrowedFd<'_>, _len: u64) -> rustix::io::Result<()> {
    Ok(())
}

// ---- drain-ended datagram ----

/// Send `payload`, without blocking, to the abstract Unix datagram socket the
/// launching process bound under `name`; a full receive queue loses this
/// report. The abstract namespace is Linux-only, and the Python launcher binds
/// nothing else, so elsewhere this is `Unsupported`.
pub fn send_notify_datagram(name: &str, payload: &[u8]) -> io::Result<usize> {
    #[cfg(target_os = "linux")]
    {
        use std::os::linux::net::SocketAddrExt;
        use std::os::unix::net::UnixDatagram;

        let socket = UnixDatagram::unbound()?;
        socket.set_nonblocking(true)?;
        // As the Python launcher binds it ("\0" + name).
        let addr = std::os::unix::net::SocketAddr::from_abstract_name(name)?;
        socket.send_to_addr(payload, &addr)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (name, payload);
        Err(io::ErrorKind::Unsupported.into())
    }
}

/// Test-side counterparts of the platform calls: look at a shm segment, and
/// stand in for the launching process on the drain-ended datagram.
#[cfg(test)]
pub mod test_support {
    /// Whether the shm segment `name` exists; any error but "no such
    /// segment" is a test bug, not an absence.
    pub fn shm_exists(name: &str) -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/dev/shm")
                .join(name)
                .try_exists()
                .unwrap()
        }
        #[cfg(not(target_os = "linux"))]
        {
            match rustix::shm::open(
                format!("/{name}"),
                rustix::shm::OFlags::RDONLY,
                rustix::fs::Mode::empty(),
            ) {
                Ok(_) => true,
                Err(rustix::io::Errno::NOENT) => false,
                Err(e) => panic!("shm_open(/{name}): {e}"),
            }
        }
    }

    /// The `len` bytes written into the shm segment `name`. Linux reads the
    /// segment's `/dev/shm` file, which is exactly `len` long; Darwin has no such
    /// path and rounds the object's size up to a page, so the first `len` bytes
    /// are mapped.
    pub fn shm_read(name: &str, len: usize) -> std::io::Result<Vec<u8>> {
        #[cfg(target_os = "linux")]
        {
            let bytes = std::fs::read(std::path::Path::new("/dev/shm").join(name))?;
            if bytes.len() != len {
                return Err(std::io::Error::other(format!(
                    "shm segment {name} is {} bytes, expected {len}",
                    bytes.len()
                )));
            }
            Ok(bytes)
        }
        #[cfg(not(target_os = "linux"))]
        {
            use rustix::mm::{MapFlags, ProtFlags, mmap, munmap};
            let fd = rustix::shm::open(
                format!("/{name}"),
                rustix::shm::OFlags::RDONLY,
                rustix::fs::Mode::empty(),
            )?;
            let size = u64::try_from(rustix::fs::fstat(&fd)?.st_size).unwrap_or(0);
            if size < len as u64 {
                return Err(std::io::Error::other(format!(
                    "shm segment {name} is {size} bytes, expected at least {len}"
                )));
            }
            if len == 0 {
                return Ok(Vec::new());
            }
            // SAFETY: a fresh read-only mapping of `len` bytes, copied out and
            // unmapped before `fd` drops.
            let bytes = unsafe {
                let ptr = mmap(
                    std::ptr::null_mut(),
                    len,
                    ProtFlags::READ,
                    MapFlags::SHARED,
                    &fd,
                    0,
                )?;
                let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
                munmap(ptr, len)?;
                bytes
            };
            Ok(bytes)
        }
    }

    /// Bind the receiving end of [`send_notify_datagram`](super::send_notify_datagram)
    /// under the abstract name `name`, as the launcher does; it dies with the
    /// socket.
    #[cfg(target_os = "linux")]
    pub fn bind_notify_socket(name: &str) -> std::io::Result<std::os::unix::net::UnixDatagram> {
        use std::os::linux::net::SocketAddrExt;
        let addr = std::os::unix::net::SocketAddr::from_abstract_name(name)?;
        std::os::unix::net::UnixDatagram::bind_addr(&addr)
    }
}
