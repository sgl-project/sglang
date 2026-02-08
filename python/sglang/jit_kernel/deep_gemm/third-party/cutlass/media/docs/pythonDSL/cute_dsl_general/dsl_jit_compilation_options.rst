.. _dsl_jit_compilation_options:
.. |DSL| replace:: CuTe DSL

.. _JIT_Compilation_Options:

JIT Compilation Options
=======================

JIT Compilation Options Overview
--------------------------------

When compiling a JIT function using |DSL|, you may want to control various aspects of the compilation process, such as optimization level, or debugging flags. |DSL| provides a flexible interface for specifying these compilation options when invoking ``cute.compile``.

Compilation options allow you to customize how your JIT-compiled functions are built and executed. This can be useful for:

* Enabling or disabling specific compiler optimizations
* Generating debug information for troubleshooting

These options can be passed as keyword arguments to ``cute.compile`` or set globally for all JIT compilations. The available options and their effects are described in the following sections, along with usage examples to help you get started.


``cute.compile`` Compilation Options
------------------------------------

You can provide additional compilation options as a string when calling ``cute.compile``. The |DSL| uses ``argparse`` to parse these options and will raise an error if any invalid options are specified.

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 25

   * - **Option**
     - **Description**
     - **Default**
     - **Type**
   * - ``opt-level``
     - Optimization level of compilation. The higher the level, the more optimizations are applied. The valid value range is [0, 3].
     - 3 (highest level of optimization)
     - int
   * - ``enable-device-assertions``
     - Enable device code assertions.
     - False
     - bool

You can use the following code to specify compilation options:

.. code-block:: python

   jit_executor_with_opt_level_2 = cute.compile(add, 1, 2, options="--opt-level 2")
   jit_executor_with_opt_level_1 = cute.compile(add, 1, 2, options="--opt-level 1")
   jit_executor_with_enable_device_assertions = cute.compile(add, 1, 2, options="--enable-device-assertions")
