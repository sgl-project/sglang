.. _dsl_introduction:
.. |DC| replace:: dynamic compilation
.. |IR| replace:: IR
.. |DSL| replace:: CuTe DSL


Introduction
======================


Overview
--------

|DSL| is a Python-based domain-specific language (DSL) designed for |DC| of numeric and GPU-oriented code. Its primary goals are:

- **Consistent with CuTe C++**, allowing users to express GPU kernels with full control of the hardware.
- **JIT compilation** for both host and GPU execution.
- `DLPack <https://github.com/dmlc/dlpack>`_ **integration**, enabling seamless interop with frameworks (e.g., PyTorch, JAX).
- **JIT caching**, so that repeated calls to the same function benefit from cached |IR| modules.
- **Native types and type inference** to reduce boilerplate and improve performance.
- **Optional lower-level control**, offering direct access to GPU backends or specialized |IR| dialects.

Decorators
----------

|DSL| provides two main Python decorators for generating optimized code via |DC|:

1. ``@jit`` — Host-side JIT-compiled functions
2. ``@kernel`` — GPU kernel functions

Both decorators can optionally use a **preprocessor** that automatically expands Python control flow (loops, conditionals) into operations consumable by the underlying |IR|.

``@jit``
~~~~~~~~~~~~~

Declares JIT-compiled functions that can be invoked from Python or from other |DSL| functions.

**Decorator Parameters**:

* ``preprocessor``:

  * ``True`` (default) — Automatically translate Python flow control (e.g., loops, if-statements) into |IR| operations.
  * ``False`` — No automatic expansion; Python flow control must be handled manually or avoided.

**Call-site Parameters**:

- ``no_cache``:

  - ``True`` — Disables JIT caching, forcing a fresh compilation each call.
  - ``False`` (default) — Enables caching for faster subsequent calls.

``@kernel``
~~~~~~~~~~~~~~~~

Defines GPU kernel functions, compiled as specialized GPU symbols through |DC|.

**Decorator Parameters**:

- ``preprocessor``:

  - ``True`` (default) — Automatically expands Python loops/ifs into GPU-compatible |IR| operations.
  - ``False`` — Expects manual or simplified kernel implementations.

**Kernel Launch Parameters**:

- ``grid``
  Specifies the grid size as a list of integers.
- ``block``
  Specifies the block size as a list of integers.
- ``cluster``
  Specifies the cluster size as a list of integers.
- ``smem``
  Specifies the size of shared memory in bytes (integer).

Calling Conventions
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 25

   * - **Caller**
     - **Callee**
     - **Allowed**
     - **Compilation/Runtime**

   * - Python function
     - ``@jit``
     - ✅
     - DSL runtime

   * - Python function
     - ``@kernel``
     - ❌
     - N/A (error raised)

   * - ``@jit``
     - ``@jit``
     - ✅
     - Compile-time call, inlined

   * - ``@jit``
     - Python function
     - ✅
     - Compile-time call, inlined

   * - ``@jit``
     - ``@kernel``
     - ✅
     - Dynamic call via GPU driver or runtime

   * - ``@kernel``
     - ``@jit``
     - ✅
     - Compile-time call, inlined

   * - ``@kernel``
     - Python function
     - ✅
     - Compile-time call, inlined

   * - ``@kernel``
     - ``@kernel``
     - ❌
     - N/A (error raised)
