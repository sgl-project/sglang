/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <pybind11/pybind11.h>

namespace omnidreams_singleview {

void bind_optimized_dit(pybind11::module_& module);

}  // namespace omnidreams_singleview
