/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

// Namespace configuration for PyTorch operator registration
// This allows building multiple libraries with different namespaces
// to avoid operator registration conflicts

#ifndef SGL_NAMESPACE
  #define SGL_NAMESPACE sgl_kernel  // Default namespace if not defined
#endif

// Helper macros for string conversion
#define SGL_STRINGIFY(x) #x
#define SGL_TOSTRING(x) SGL_STRINGIFY(x)
#define SGL_NAMESPACE_STR SGL_TOSTRING(SGL_NAMESPACE)