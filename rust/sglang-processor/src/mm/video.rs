//! SGLang's own Qwen-VL video rules, which run before Dynamo's decoder and
//! family processor: how many frames to keep and which (`smart_nframes`, then
//! an int64 `np.linspace`), and the pre-resize to multiples of 28
//! (`preprocess_video`), all from `multimodal/processors/qwen_vl.py`.

use dynamo_multimodal::MmError;

/// The keys of `--mm-process-config {"video": ...}` that `preprocess_video`
/// reads; a request's hints replace some of them ([`VideoConfig::with_hints`]).
/// Numbers are f64 because Python takes ints and floats alike.
#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VideoConfig {
    pub fps: Option<f64>,
    pub nframes: Option<f64>,
    pub min_frames: Option<f64>,
    pub max_frames: Option<f64>,
    pub min_pixels: Option<f64>,
    pub max_pixels: Option<f64>,
    pub total_pixels: Option<f64>,
}

/// A request's `MediaRef` video hints.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Deserialize)]
pub struct VideoHints {
    pub fps: Option<f64>,
    pub num_frames: Option<i64>,
    pub min_frames: Option<i64>,
    pub max_frames: Option<i64>,
}

/// The server's [`VideoConfig`] and `qwen_vl.py`'s module constants, as
/// `RustMmProcessor.resolve_spec` reads them at launch (`VIDEO_TOTAL_PIXELS`
/// comes from the environment).
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VideoSampling {
    pub config: VideoConfig,
    /// `FPS`, `FPS_MIN_FRAMES`, `FPS_MAX_FRAMES` and `FRAME_FACTOR`.
    pub fps: f64,
    pub min_frames: f64,
    pub max_frames: f64,
    pub frame_factor: f64,
    /// `IMAGE_FACTOR`, `VIDEO_MIN_PIXELS`, `VIDEO_MAX_PIXELS`, `VIDEO_TOTAL_PIXELS`.
    pub factor: f64,
    pub min_pixels: f64,
    pub max_pixels: f64,
    pub total_pixels: f64,
}

/// `MAX_RATIO` of SGLang's `smart_resize`.
const MAX_RATIO: f64 = 200.0;

impl VideoConfig {
    /// The server config with a request's hints over it. `num_frames`
    /// replaces the fps mode and the fps-mode keys replace `nframes`, so the
    /// result never holds both, which `smart_nframes` asserts.
    pub fn with_hints(&self, hints: &VideoHints) -> Result<Self, MmError> {
        let invalid = |message: &str| Err(MmError::invalid_input(message.to_owned()));
        if hints.fps.is_some_and(|fps| !(fps.is_finite() && fps > 0.0)) {
            return invalid("video fps must be a positive finite number");
        }
        let counts = [hints.num_frames, hints.min_frames, hints.max_frames];
        if counts.iter().flatten().any(|&n| n < 1) {
            return invalid("video num_frames, min_frames and max_frames must be at least 1");
        }
        let fps_mode =
            hints.fps.is_some() || hints.min_frames.is_some() || hints.max_frames.is_some();
        let mut config = self.clone();
        match hints.num_frames {
            Some(_) if fps_mode => {
                return invalid(
                    "video num_frames cannot be combined with fps, min_frames or max_frames",
                );
            }
            Some(n) => {
                config.nframes = Some(n as f64);
                (config.fps, config.min_frames, config.max_frames) = (None, None, None);
            }
            None if fps_mode => {
                config.nframes = None;
                config.fps = hints.fps.or(config.fps);
                config.min_frames = hints.min_frames.map(|n| n as f64).or(config.min_frames);
                config.max_frames = hints.max_frames.map(|n| n as f64).or(config.max_frames);
            }
            None => {}
        }
        Ok(config)
    }
}

impl VideoSampling {
    /// `smart_nframes`: the frame count to sample from a video of
    /// `total_frames` frames at `video_fps`.
    pub fn num_frames(
        &self,
        config: &VideoConfig,
        total_frames: u64,
        video_fps: f64,
    ) -> Result<u64, MmError> {
        let factor = self.frame_factor;
        let total = total_frames as f64;
        let nframes = match config.nframes {
            Some(nframes) => round_by_factor(nframes, factor),
            None => {
                let fps = config.fps.unwrap_or(self.fps);
                let min_frames =
                    ceil_by_factor(config.min_frames.unwrap_or(self.min_frames), factor);
                let max_frames = floor_by_factor(
                    config.max_frames.unwrap_or(self.max_frames.min(total)),
                    factor,
                );
                let nframes = total / video_fps * fps;
                floor_by_factor(nframes.max(min_frames).min(max_frames).min(total), factor)
            }
        };
        if !(factor <= nframes && nframes <= total) {
            return Err(MmError::invalid_input(format!(
                "nframes should in interval [{factor}, {total_frames}], but got {nframes}."
            )));
        }
        Ok(nframes as u64)
    }

    /// `preprocess_video`'s pre-resize target `(height, width)` for
    /// `num_frames` frames of `height` x `width`.
    pub fn presize(
        &self,
        config: &VideoConfig,
        num_frames: usize,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize), MmError> {
        let min_pixels = config.min_pixels.unwrap_or(self.min_pixels);
        let total_pixels = config.total_pixels.unwrap_or(self.total_pixels);
        let max_pixels = config
            .max_pixels
            .unwrap_or(self.max_pixels)
            .min(total_pixels / num_frames as f64 * self.frame_factor)
            .max((min_pixels * 1.05).trunc());
        let max_pixels = config.max_pixels.map_or(max_pixels, |m| m.min(max_pixels));
        smart_resize(height, width, self.factor, min_pixels, max_pixels)
    }
}

/// `np.unique(np.linspace(0, total_frames - 1, num=num_frames, dtype=np.int64))`:
/// `i * step` floored, the last pinned to `total_frames - 1`.
pub fn frame_indices(total_frames: u64, num_frames: u64) -> Vec<u64> {
    // A step of at most 1 visits every frame.
    if num_frames >= total_frames {
        return (0..total_frames).collect();
    }
    if num_frames <= 1 {
        return vec![0; num_frames as usize];
    }
    let step = (total_frames - 1) as f64 / (num_frames - 1) as f64;
    let mut indices: Vec<u64> = (0..num_frames)
        .map(|i| (i as f64 * step).floor() as u64)
        .collect();
    indices[num_frames as usize - 1] = total_frames - 1;
    indices
}

/// SGLang's own `smart_resize` (not HF's): sides round to `factor` with a
/// floor of `factor`, then shrink or grow toward the pixel bounds.
fn smart_resize(
    height: usize,
    width: usize,
    factor: f64,
    min_pixels: f64,
    max_pixels: f64,
) -> Result<(usize, usize), MmError> {
    let (h, w) = (height as f64, width as f64);
    if h.max(w) / h.min(w) > MAX_RATIO {
        return Err(MmError::invalid_input(format!(
            "absolute aspect ratio must be smaller than {MAX_RATIO}, got {}",
            h.max(w) / h.min(w)
        )));
    }
    let mut h_bar = factor.max(round_by_factor(h, factor));
    let mut w_bar = factor.max(round_by_factor(w, factor));
    if h_bar * w_bar > max_pixels {
        let beta = (h * w / max_pixels).sqrt();
        h_bar = floor_by_factor(h / beta, factor);
        w_bar = floor_by_factor(w / beta, factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels / (h * w)).sqrt();
        h_bar = ceil_by_factor(h * beta, factor);
        w_bar = ceil_by_factor(w * beta, factor);
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// Python's `round(number / factor) * factor`: ties to even.
fn round_by_factor(number: f64, factor: f64) -> f64 {
    (number / factor).round_ties_even() * factor
}

fn ceil_by_factor(number: f64, factor: f64) -> f64 {
    (number / factor).ceil() * factor
}

fn floor_by_factor(number: f64, factor: f64) -> f64 {
    (number / factor).floor() * factor
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `qwen_vl.py`'s constants with `VIDEO_MAX_PIXELS` unset.
    fn sampling() -> VideoSampling {
        VideoSampling {
            config: VideoConfig::default(),
            fps: 2.0,
            min_frames: 4.0,
            max_frames: 768.0,
            frame_factor: 2.0,
            factor: 28.0,
            min_pixels: 100_352.0,
            max_pixels: 602_112.0,
            total_pixels: 90_316_800.0,
        }
    }

    /// Values from Python's `smart_nframes(ele, total_frames, video_fps)`.
    #[test]
    fn num_frames_matches_smart_nframes() {
        let s = sampling();
        let cases: [(VideoConfig, u64, f64, Option<u64>); 8] = [
            (VideoConfig::default(), 300, 30.0, Some(20)),
            (VideoConfig::default(), 60, 24.009_603_841_536_613, Some(4)),
            (
                VideoConfig {
                    fps: Some(1.0),
                    ..Default::default()
                },
                3,
                30.0,
                Some(2),
            ),
            (
                VideoConfig {
                    nframes: Some(7.0),
                    ..Default::default()
                },
                300,
                30.0,
                Some(8),
            ),
            (
                VideoConfig {
                    nframes: Some(5.0),
                    ..Default::default()
                },
                300,
                30.0,
                Some(4),
            ),
            (
                VideoConfig {
                    max_frames: Some(9.0),
                    ..Default::default()
                },
                300,
                30.0,
                Some(8),
            ),
            (
                VideoConfig {
                    min_frames: Some(5.0),
                    ..Default::default()
                },
                300,
                300.0,
                Some(6),
            ),
            (VideoConfig::default(), 1, 30.0, None),
        ];
        for (config, total, fps, want) in cases {
            assert_eq!(
                s.num_frames(&config, total, fps).ok(),
                want,
                "{config:?} {total} {fps}"
            );
        }
    }

    #[test]
    fn frame_indices_match_numpy_int64_linspace() {
        // np.linspace(0, 299, 20, dtype=np.int64)
        assert_eq!(
            frame_indices(300, 20),
            [
                0, 15, 31, 47, 62, 78, 94, 110, 125, 141, 157, 173, 188, 204, 220, 236, 251, 267,
                283, 299
            ]
        );
        assert_eq!(frame_indices(60, 4), [0, 19, 39, 59]);
        assert_eq!(frame_indices(2, 2), [0, 1]);
        // np.unique(np.linspace(0, 4, 9, dtype=np.int64))
        assert_eq!(frame_indices(5, 9), [0, 1, 2, 3, 4]);
        assert!(frame_indices(0, 2).is_empty());
    }

    /// Values from Python's `preprocess_video` sizing.
    #[test]
    fn presize_matches_preprocess_video() {
        let s = sampling();
        let config = VideoConfig::default();
        assert_eq!(s.presize(&config, 4, 288, 352).unwrap(), (280, 364));
        assert_eq!(s.presize(&config, 20, 1080, 1920).unwrap(), (560, 1008));
        assert_eq!(s.presize(&config, 4, 20, 24).unwrap(), (308, 364));
        let capped = VideoConfig {
            max_pixels: Some(50_000.0),
            ..Default::default()
        };
        assert_eq!(s.presize(&capped, 4, 288, 352).unwrap(), (196, 224));
        assert!(s.presize(&config, 4, 10, 2010).is_err(), "aspect ratio");
    }

    #[test]
    fn hints_replace_the_sampling_mode() {
        let server = VideoConfig {
            fps: Some(1.0),
            min_frames: Some(8.0),
            max_pixels: Some(1e5),
            ..Default::default()
        };
        let by_count = server
            .with_hints(&VideoHints {
                num_frames: Some(6),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            by_count,
            VideoConfig {
                nframes: Some(6.0),
                max_pixels: Some(1e5),
                ..Default::default()
            }
        );
        let by_rate = VideoConfig {
            nframes: Some(4.0),
            ..Default::default()
        }
        .with_hints(&VideoHints {
            fps: Some(3.0),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            by_rate,
            VideoConfig {
                fps: Some(3.0),
                ..Default::default()
            }
        );
        for bad in [
            VideoHints {
                fps: Some(1.0),
                num_frames: Some(4),
                ..Default::default()
            },
            VideoHints {
                max_frames: Some(4),
                num_frames: Some(4),
                ..Default::default()
            },
            VideoHints {
                fps: Some(0.0),
                ..Default::default()
            },
            VideoHints {
                num_frames: Some(0),
                ..Default::default()
            },
        ] {
            assert!(server.with_hints(&bad).is_err(), "{bad:?}");
        }
    }
}
