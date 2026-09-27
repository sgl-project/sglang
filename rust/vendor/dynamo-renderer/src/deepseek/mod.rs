// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native (non-jinja) prompt formatters for DeepSeek model families.
//!
//! DeepSeek's HF repos ship no usable `chat_template`, so these render the
//! prompt in Rust instead. [`common`] holds the shared thinking-mode /
//! tool-injection logic; [`v4`] and [`v32`] are the per-family formatters.

pub mod common;
pub mod v32;
pub mod v4;
