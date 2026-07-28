use crate::mooncake::EngineError;

const FROZEN_NIC_PRIORITY_MATRIX: &str = concat!(
    r#"{"cpu:0":[["mlx5_1","mlx5_2","mlx5_3","mlx5_4"],[]],"#,
    r#""cpu:1":[["mlx5_1","mlx5_2","mlx5_3","mlx5_4"],[]],"#,
    r#""cuda:4":[["mlx5_1","mlx5_2","mlx5_3","mlx5_4"],[]],"#,
    r#""cuda:5":[["mlx5_1","mlx5_2","mlx5_3","mlx5_4"],[]]}"#
);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdNicProfile {
    _private: (),
}

impl PdNicProfile {
    pub fn for_gpu(device: u32) -> Result<Self, EngineError> {
        if !matches!(device, 4 | 5) {
            return Err(EngineError::UnsupportedGpu { device });
        }
        Ok(Self { _private: () })
    }

    pub fn canonical_json(&self) -> &'static str {
        FROZEN_NIC_PRIORITY_MATRIX
    }

    pub fn hcas(&self) -> &'static [&'static str; 4] {
        &["mlx5_1", "mlx5_2", "mlx5_3", "mlx5_4"]
    }
}
