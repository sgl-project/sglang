use crate::wasm::{config::WasmRuntimeConfig};
use wasmtime::{Config, Engine};

pub struct WasmRuntime {
    engine: Engine,
    config: WasmRuntimeConfig,
}

impl WasmRuntime {
    pub fn new(config: WasmRuntimeConfig) -> Result<Self, String> {
        // create the wasmtime engine config
        let mut wasmtime_config = Config::new();
        wasmtime_config.async_stack_size(config.max_stack_size);
        if config.enable_wasi {
            wasmtime_config.wasm_component_model(false);
        }
        
        let engine = Engine::new(&wasmtime_config)
            .map_err(|e| format!("Failed to create wasmtime engine: {}", e))?;
        
        Ok(Self { 
            engine, 
            config,
        })
    }
    
}