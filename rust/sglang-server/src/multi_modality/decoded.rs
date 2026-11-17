//! Same-host decoded media: bounded immutable snapshots, then the existing
//! Python processor. No model-specific tensor adapter or public pickle.
use std::fs::File;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use pyo3::prelude::*;
use pyo3::types::PyList;
use rustix::fs::{Mode, OFlags, openat};
use sglang_api_types::api::v1 as api;
use sglang_mm::common::decoded_media::RgbImageLayout;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::worker::{MediaError, MmProcessOutput, MmProcessor};
use crate::message::{
    buffers::{Buffer, BufferStore},
    request::MmData,
    types::TokenIds,
};
use crate::utils::error::Error;

const MAX_ITEMS: usize = 4096;
const MAX_REQUEST_BYTES: u64 = 256 << 20;
const MAX_INFLIGHT_BYTES: usize = 512 << 20;

pub struct DecodedInput {
    pub descriptor: api::DecodedMediaInput,
    pub consumed: Arc<AtomicBool>,
    bytes: Vec<Vec<u8>>,
    lease: Option<OwnedSemaphorePermit>,
}
// Request Debug output must not expand large samples or disclose object names.
impl std::fmt::Debug for DecodedInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodedInput")
            .field("images", &self.descriptor.images.len())
            .field("videos", &self.descriptor.videos.len())
            .field("audios", &self.descriptor.audios.len())
            .field(
                "snapshot_bytes",
                &self.bytes.iter().map(Vec::len).sum::<usize>(),
            )
            .field("consumed", &self.consumed.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}
impl DecodedInput {
    pub fn new(descriptor: api::DecodedMediaInput) -> Result<Self, Error> {
        validate(&descriptor).map_err(Error::Validation)?;
        Ok(Self {
            descriptor,
            consumed: Arc::new(AtomicBool::new(false)),
            bytes: vec![],
            lease: None,
        })
    }
}
fn region(
    payload: Option<&api::MediaPayloadReference>,
) -> Result<&api::MountedSharedMemoryRegion, String> {
    payload
        .and_then(|p| p.mounted_shm.as_ref())
        .ok_or_else(|| "decoded media requires mounted_shm payload".into())
}
fn image(image: &api::DecodedImage) -> Result<&api::MountedSharedMemoryRegion, String> {
    if image.pixel_format != "rgb8" {
        return Err("decoded image requires rgb8".into());
    }
    let r = region(image.payload.as_ref())?;
    RgbImageLayout {
        width: image.width,
        height: image.height,
        row_stride_bytes: image.row_stride_bytes,
    }
    .validate_region(u64::MAX, r.offset_bytes, r.length_bytes, MAX_REQUEST_BYTES)?;
    Ok(r)
}
fn validate(
    input: &api::DecodedMediaInput,
) -> Result<Vec<&api::MountedSharedMemoryRegion>, String> {
    let mut regions = vec![];
    if input
        .images
        .len()
        .saturating_add(input.videos.len())
        .saturating_add(input.audios.len())
        > MAX_ITEMS
    {
        return Err("too many decoded media items".into());
    }
    for i in &input.images {
        regions.push(image(i)?);
    }
    for video in &input.videos {
        if video.frames.is_empty()
            || video.frames.len() > MAX_ITEMS
            || !video.source_fps.is_finite()
            || video.source_fps <= 0.0
            || video.source_frame_count == 0
        {
            return Err("invalid decoded video frame count or source clock".into());
        }
        let mut previous = None;
        for f in &video.frames {
            if f.source_frame_index >= video.source_frame_count
                || previous.is_some_and(|(t, i)| f.timestamp_us < t || f.source_frame_index <= i)
            {
                return Err("decoded video timestamps/indices are out of order or range".into());
            }
            let i = f.image.as_ref().ok_or("decoded video frame has no image")?;
            // Transport preserves explicit per-frame layouts/timestamps. It
            // must not impose a processor's stacking or constant-FPS policy.
            previous = Some((f.timestamp_us, f.source_frame_index));
            regions.push(image(i)?);
            if regions.len() > MAX_ITEMS {
                return Err("too many decoded media regions".into());
            }
        }
    }
    for a in &input.audios {
        let r = region(a.payload.as_ref())?;
        if a.sample_format != "f32le"
            || !(1..=384000).contains(&a.sample_rate_hz)
            || !(1..=32).contains(&a.channels)
            || a.sample_frames == 0
            || a.sample_frames
                .checked_mul(u64::from(a.channels))
                .and_then(|n| n.checked_mul(4))
                != Some(r.length_bytes)
        {
            return Err("invalid decoded audio PCM layout".into());
        }
        regions.push(r);
    }
    if regions.is_empty() || regions.len() > MAX_ITEMS {
        return Err("decoded media must contain 1..4096 regions".into());
    }
    let mut total = 0u64;
    for r in &regions {
        if !(16..=128).contains(&r.object_id.len())
            || !r
                .object_id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
            || r.length_bytes == 0
            || r.offset_bytes.checked_add(r.length_bytes).is_none()
        {
            return Err("invalid decoded media object name or region".into());
        }
        total = total
            .checked_add(r.length_bytes)
            .ok_or("decoded media size overflow")?;
        if total > MAX_REQUEST_BYTES {
            return Err("decoded media exceeds 256 MiB request budget".into());
        }
    }
    Ok(regions)
}

struct DecodedStore {
    root: File,
    instance_id: String,
    permits: Arc<Semaphore>,
}
impl DecodedStore {
    fn new(root: &str, instance_id: String) -> Result<Self, String> {
        let fd = rustix::fs::open(
            root,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|e| format!("decoded media root: {e}"))?;
        let root = File::from(fd);
        if root.metadata().map_err(|e| e.to_string())?.mode() & 0o077 != 0 {
            return Err("decoded media root must be private (0700)".into());
        }
        Ok(Self {
            root,
            instance_id,
            permits: Arc::new(Semaphore::new(MAX_INFLIGHT_BYTES)),
        })
    }
    fn snapshot(&self, input: &mut DecodedInput) -> Result<(), MediaError> {
        let regions = validate(&input.descriptor).map_err(MediaError::Client)?;
        // Fence every region BEFORE opening any object or reserving bytes.
        if regions
            .iter()
            .any(|r| r.engine_instance_id != self.instance_id)
        {
            return Err(MediaError::Client(
                "decoded media engine_instance_id mismatch".into(),
            ));
        }
        let total: u64 = regions.iter().map(|r| r.length_bytes).sum();
        let lease = self
            .permits
            .clone()
            .try_acquire_many_owned(total as u32)
            .map_err(|_| {
                MediaError::Capacity(
                    "decoded media in-flight byte budget exhausted; retry later".into(),
                )
            })?;
        let bytes = self.read_regions(regions).map_err(MediaError::Client)?;
        input.bytes = bytes;
        input.lease = Some(lease);
        input.consumed.store(true, Ordering::Release);
        Ok(())
    }
    fn read_regions(
        &self,
        regions: Vec<&api::MountedSharedMemoryRegion>,
    ) -> Result<Vec<Vec<u8>>, String> {
        let mut bytes = Vec::with_capacity(regions.len());
        for r in regions {
            // Pinned directory fd + one validated name: no traversal, symlinks,
            // blocking FIFO/device read or hard-link escape. Never mmap a
            // producer file: truncation cannot SIGBUS the engine.
            let fd = openat(
                &self.root,
                r.object_id.as_str(),
                OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(|_| "cannot open decoded media object")?;
            let file = File::from(fd);
            let meta = file
                .metadata()
                .map_err(|_| "cannot stat decoded media object")?;
            if !meta.is_file()
                || meta.nlink() != 1
                || r.offset_bytes
                    .checked_add(r.length_bytes)
                    .is_none_or(|n| n > meta.len())
            {
                return Err(
                    "decoded media object is not a regular single-link file or region exceeds file"
                        .into(),
                );
            }
            let mut data = Vec::new();
            data.try_reserve_exact(r.length_bytes as usize)
                .map_err(|_| "decoded media allocation failed")?;
            data.resize(r.length_bytes as usize, 0);
            file.read_exact_at(&mut data, r.offset_bytes)
                .map_err(|_| "decoded media object truncated while reading")?;
            let after = file
                .metadata()
                .map_err(|_| "cannot recheck decoded media object")?;
            if (
                meta.len(),
                meta.mtime(),
                meta.mtime_nsec(),
                meta.ctime(),
                meta.ctime_nsec(),
            ) != (
                after.len(),
                after.mtime(),
                after.mtime_nsec(),
                after.ctime(),
                after.ctime_nsec(),
            ) {
                return Err("decoded media object changed while reading".into());
            }
            bytes.push(data);
        }
        Ok(bytes)
    }
}
pub struct DecodedProcessor {
    store: DecodedStore,
    callback: Py<PyAny>,
    fallback: Option<Arc<dyn MmProcessor>>,
}
impl DecodedProcessor {
    pub fn new(
        root: &str,
        instance_id: String,
        callback: Py<PyAny>,
        fallback: Option<Arc<dyn MmProcessor>>,
    ) -> Result<Self, String> {
        Ok(Self {
            store: DecodedStore::new(root, instance_id)?,
            callback,
            fallback,
        })
    }
}
impl MmProcessor for DecodedProcessor {
    fn process(&self, ids: TokenIds, mm: MmData) -> Result<MmProcessOutput, MediaError> {
        self.fallback
            .as_ref()
            .ok_or_else(|| {
                MediaError::Client(
                    "encoded media has no native processor for this model; use decoded_media"
                        .into(),
                )
            })?
            .process(ids, mm)
    }
    fn check_media(&self, mm: &MmData) -> Result<(), String> {
        if let Some(input) = &mm.decoded {
            let regions = validate(&input.descriptor)?;
            if regions
                .iter()
                .any(|r| r.engine_instance_id != self.store.instance_id)
            {
                return Err("decoded media engine_instance_id mismatch".into());
            }
            Ok(())
        } else {
            self.fallback
                .as_ref()
                .ok_or("no native encoded-media processor for this model")?
                .check_media(mm)
        }
    }
    fn prepare_decoded(&self, input: &mut DecodedInput) -> Result<(), MediaError> {
        self.store.snapshot(input)
    }
    fn process_direct(
        &self,
        ids: &TokenIds,
        mm: &mut MmData,
    ) -> Option<Result<(TokenIds, Vec<Buffer>), MediaError>> {
        let input = mm.decoded.as_mut()?;
        Some((|| {
            if !input.consumed.load(Ordering::Acquire) {
                return Err(MediaError::Server(
                    "decoded media was not snapshotted".into(),
                ));
            }
            let json = serde_json::to_string(&input.descriptor)
                .map_err(|e| MediaError::Server(e.to_string()))?;
            Python::attach(|py| -> PyResult<_> {
                let payloads = PyList::empty(py);
                // Move allocations into numpy; Python owns all input arrays after
                // this call. The byte lease remains held until processing ends.
                for data in std::mem::take(&mut input.bytes) {
                    let (_, object) = crate::buffer_to_py(py, Buffer::inline("pixels", data))?;
                    payloads.append(object)?;
                }
                let result = self.callback.call1(py, (ids.clone(), json, payloads))?;
                let (ids, output): (TokenIds, Py<PyAny>) = result.extract(py)?;
                Ok((
                    ids,
                    vec![Buffer {
                        name: "mm.python_output".into(),
                        shape: vec![],
                        store: BufferStore::PythonOutput(output),
                    }],
                ))
            })
            .map_err(|e| {
                Python::attach(|py| {
                    if e.is_instance_of::<pyo3::exceptions::PyValueError>(py) {
                        MediaError::Client(e.value(py).to_string())
                    } else {
                        tracing::error!(error=%e, "decoded media processor failed");
                        MediaError::Server("decoded media processor failed".into())
                    }
                })
            })
        })())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};

    const OBJECT: &str = "unique-object-0123456789abcdef";
    fn rgb() -> api::DecodedImage {
        api::DecodedImage {
            width: 2,
            height: 1,
            row_stride_bytes: 6,
            pixel_format: "rgb8".into(),
            payload: Some(api::MediaPayloadReference {
                mounted_shm: Some(api::MountedSharedMemoryRegion {
                    object_id: OBJECT.into(),
                    offset_bytes: 1,
                    length_bytes: 6,
                    engine_instance_id: "boot".into(),
                }),
            }),
        }
    }
    fn descriptor() -> api::DecodedMediaInput {
        api::DecodedMediaInput {
            images: vec![rgb()],
            ..Default::default()
        }
    }
    struct Root(std::path::PathBuf);
    impl Root {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("sglang-decoded-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir(&path).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
            Self(path)
        }
        fn store(&self) -> DecodedStore {
            DecodedStore::new(self.0.to_str().unwrap(), "boot".into()).unwrap()
        }
    }
    impl Drop for Root {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn decoded_snapshot_is_owned_and_receipt_waits_for_every_read() {
        let root = Root::new();
        let path = root.0.join(OBJECT);
        std::fs::write(&path, [99u8, 1, 2, 3, 4, 5, 6]).unwrap();
        let store = root.store();
        let mut input = DecodedInput::new(descriptor()).unwrap();
        assert!(!input.consumed.load(Ordering::Acquire));
        store.snapshot(&mut input).unwrap();
        assert!(input.consumed.load(Ordering::Acquire));
        assert!(path.exists(), "engine never unlinks producer storage");
        std::fs::remove_file(path).unwrap();
        assert_eq!(input.bytes, [vec![1, 2, 3, 4, 5, 6]]);

        let mut missing = DecodedInput::new(descriptor()).unwrap();
        assert!(store.snapshot(&mut missing).is_err());
        assert!(!missing.consumed.load(Ordering::Acquire));
    }
    #[test]
    fn decoded_rejects_stale_identity_symlinks_hardlinks_and_out_of_file_ranges() {
        let root = Root::new();
        let store = root.store();
        let mut input = DecodedInput::new(descriptor()).unwrap();
        input.descriptor.images[0]
            .payload
            .as_mut()
            .unwrap()
            .mounted_shm
            .as_mut()
            .unwrap()
            .engine_instance_id = "previous-boot".into();
        assert!(
            store
                .snapshot(&mut input)
                .unwrap_err()
                .to_string()
                .contains("instance_id")
        );
        let target = root.0.join("target");
        std::fs::write(&target, [0u8; 7]).unwrap();
        let path = root.0.join(OBJECT);
        symlink(&target, &path).unwrap();
        assert!(
            store
                .snapshot(&mut DecodedInput::new(descriptor()).unwrap())
                .is_err()
        );
        std::fs::remove_file(&path).unwrap();
        std::fs::hard_link(&target, &path).unwrap();
        assert!(
            store
                .snapshot(&mut DecodedInput::new(descriptor()).unwrap())
                .is_err()
        );
        std::fs::remove_file(&path).unwrap();
        std::fs::rename(&target, &path).unwrap();
        std::fs::write(&path, [0u8; 6]).unwrap();
        assert!(
            store
                .snapshot(&mut DecodedInput::new(descriptor()).unwrap())
                .is_err()
        );
    }
    #[test]
    fn decoded_reservations_are_bounded_and_returned_on_drop() {
        let root = Root::new();
        std::fs::write(root.0.join(OBJECT), [0u8; 7]).unwrap();
        let mut store = root.store();
        store.permits = Arc::new(Semaphore::new(6));
        let mut first = DecodedInput::new(descriptor()).unwrap();
        store.snapshot(&mut first).unwrap();
        let error = store
            .snapshot(&mut DecodedInput::new(descriptor()).unwrap())
            .unwrap_err();
        assert!(matches!(error, MediaError::Capacity(_)));
        assert_eq!(
            crate::api_server::core::CoreError::from(error).kind(),
            crate::api_server::core::CoreErrorKind::ResourceExhausted
        );
        drop(first);
        assert_eq!(store.permits.available_permits(), 6);
        store
            .snapshot(&mut DecodedInput::new(descriptor()).unwrap())
            .unwrap();
        assert_eq!(store.permits.available_permits(), 6);
    }
    #[test]
    fn decoded_metadata_validates_all_modalities_and_clock_before_io() {
        let mut input = descriptor();
        input.videos.push(api::DecodedVideo {
            frames: vec![api::DecodedVideoFrame {
                image: Some(rgb()),
                timestamp_us: 500000,
                source_frame_index: 15,
            }],
            source_fps: 30.0,
            source_frame_count: 100,
        });
        let mut payload = rgb().payload;
        payload
            .as_mut()
            .unwrap()
            .mounted_shm
            .as_mut()
            .unwrap()
            .length_bytes = 8;
        input.audios.push(api::DecodedAudio {
            sample_format: "f32le".into(),
            sample_rate_hz: 16000,
            channels: 1,
            sample_frames: 2,
            payload,
        });
        assert_eq!(validate(&input).unwrap().len(), 3);
        // Irregular timing and frame shapes are transport-valid. Model
        // sampling/stacking restrictions belong to the existing processor.
        input.videos[0].frames[0].timestamp_us = 600000;
        let mut next = input.videos[0].frames[0].clone();
        next.source_frame_index = 20;
        next.timestamp_us = 950000;
        next.image.as_mut().unwrap().width = 1;
        input.videos[0].frames.push(next);
        assert_eq!(validate(&input).unwrap().len(), 4);
        input.videos[0].frames[1].timestamp_us = 0;
        assert!(validate(&input).is_err());
        input.videos.clear();
        input.audios[0].sample_frames = u64::MAX;
        assert!(validate(&input).is_err());
        input.audios.clear();
        input.images[0]
            .payload
            .as_mut()
            .unwrap()
            .mounted_shm
            .as_mut()
            .unwrap()
            .object_id = "../private-file".into();
        assert!(validate(&input).is_err());
        input.images[0] = rgb();
        input.images[0].row_stride_bytes = u64::MAX;
        input.images[0].height = 2;
        assert!(validate(&input).is_err());
    }
    #[test]
    fn decoded_wire_roundtrip_and_single_prompt_normalization() {
        use crate::message::request::{GenerationPolicy, into_requests};
        let wire = api::GenerateRequest {
            text: Some(api::StringOrList {
                value: Some(api::string_or_list::Value::One("<image>".into())),
            }),
            decoded_media: Some(descriptor()),
            ..Default::default()
        };
        let json = serde_json::to_string(&wire).unwrap();
        let wire: api::GenerateRequest = serde_json::from_str(&json).unwrap();
        let (mut requests, batch) = into_requests(wire, &GenerationPolicy::default()).unwrap();
        assert!(!batch);
        assert!(requests[0].has_multimodal());
        assert_eq!(
            requests[0].mm.take().unwrap().decoded.unwrap().descriptor,
            descriptor()
        );
        let mut mixed: api::GenerateRequest = serde_json::from_str(&json).unwrap();
        mixed.image_data = serde_json::from_str("\"data:image/png;base64,AQ==\"").unwrap();
        assert!(into_requests(mixed, &GenerationPolicy::default()).is_err());
    }
}
