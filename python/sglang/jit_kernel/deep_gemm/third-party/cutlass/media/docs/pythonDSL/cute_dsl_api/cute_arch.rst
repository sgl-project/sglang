.. _cute_arch:

cutlass.cute.arch
=================

The ``cute.arch`` module contains wrappers around NVVM-level MLIR Op builders that seamlessly
inter-operate with the Python types used in CUTLASS Python. Another benefit of wrapping these Op
builders is that the source location can be tracked with the ``@dsl_user_op`` decorator. Available
functions include

- basic API like ``thr_idx``;
- functions related to the direct management of mbarriers;
- low-level SMEM management (prefer using the ``SmemAllocator`` class);
- TMEM management.

API documentation
-----------------

.. automodule:: cutlass.cute.arch
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :private-members:
