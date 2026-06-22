<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# OmniDreams Single-View Native

This directory contains native-source scaffolding for the OmniDreams single-view
integration. It hosts the managed third-party source manifest, synchronization
tooling, build helpers, and CUDA/C++ extension sources used by the
single-view-only native acceleration path.

This code is a work in progress. It is intended as an internal implementation
area for bringing up and validating native OmniDreams acceleration, not as a
stable user-facing API. Interfaces, file layout, and kernel coverage may change
as the single-view native path is developed.
