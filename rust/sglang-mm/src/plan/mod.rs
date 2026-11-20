pub mod deepseek_v41;
pub mod glm5_next;
pub mod kimi_k3;
pub mod minimax_m3;
pub mod qwen_vl;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageInfo {
    pub height: u32,
    pub width: u32,
    pub orientation: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImagePlan {
    pub resized_height: u32,
    pub resized_width: u32,
    pub grid: [u32; 3],
    pub expansion: Vec<i64>,
}

#[cfg(feature = "python")]
pub fn register(parent: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;

    let m = PyModule::new(parent.py(), "plan")?;
    deepseek_v41::register(&m)?;
    glm5_next::register(&m)?;
    kimi_k3::register(&m)?;
    minimax_m3::register(&m)?;
    qwen_vl::register(&m)?;
    parent.add_submodule(&m)
}
