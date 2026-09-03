//! Shared helpers used at the PyO3 boundary by multiple wrappers.

use tch::Device;

use crate::error::RadixCacheInitError;

/// Parse a torch-style device string into a `tch::Device`.
///
/// Accepts exactly `"cpu"`, `"cuda"` (alias for `cuda:0`), or `"cuda:N"`
/// where N is a non-negative integer. Anything else (including `"cuda0"`,
/// `"cuda:abc"`, `"cudaXYZ"`, `"cuda:-1"`) returns
/// `RadixCacheInitError::InvalidDevice` — no silent coercion.
pub fn parse_device(device: &str) -> Result<Device, RadixCacheInitError> {
    match device {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda(0)),
        s => match s
            .strip_prefix("cuda:")
            .and_then(|i| i.parse::<usize>().ok())
        {
            Some(idx) => Ok(Device::Cuda(idx)),
            None => Err(RadixCacheInitError::InvalidDevice(device.to_owned())),
        },
    }
}
