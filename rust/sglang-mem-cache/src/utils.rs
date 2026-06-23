//! Shared helpers used at the PyO3 boundary by multiple wrappers.

use tch::Device;

use crate::error::RadixCacheInitError;

/// Parse tch::Device from torch-style device string (e.g. cpu, cuda, or cuda:N).
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
