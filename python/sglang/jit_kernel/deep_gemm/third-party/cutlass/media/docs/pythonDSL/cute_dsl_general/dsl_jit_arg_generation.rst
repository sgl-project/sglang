.. _dsl_jit_arg_generation:
.. |DSL| replace:: CuTe DSL
.. |CUSTOM_TYPES| replace:: customized types

JIT Function Argument Generation
=======================================


In a nutshell
--------------
When using the ``@jit`` or ``@kernel`` decorators to define a JIT-compiled function, the arguments to the function are traced to determine the JIT function's signature.
|DSL| provides a Pythonic way to write the arguments for JIT function as one normally would in Python, and the |DSL| will take care of the rest for you.

Specifically, |DSL| honors following when generating the JIT function's arguments:

- JIT function arguments are assumed to be **dynamic arguments** by default.
- If an argument is explicitly type annotated with ``cutlass.Constexpr``, it is treated as a **compile-time constant**.
- If type annotation is provided, |DSL| validates the argument type at compile time for **type safety**.
- |DSL| provides **runtime checkable protocols** (``JitArgument`` and ``DynamicExpression``) for generating JIT function arguments for |CUSTOM_TYPES|.

More details below for each of the above.

Static argument vs. Dynamic argument
------------------------------------

|DSL| supports both static and dynamic arguments for JIT functions.

1. **Static arguments** hold values that are known at compile time. It is not included in the generated JIT function signature.
2. **Dynamic arguments** hold values that are only known at runtime.

By default, |DSL| assumes dynamic arguments and tries to infer the argument types from the call-site argument types. An explicit type annotation ``cutlass.Constexpr`` can be used to specify a static argument.

.. code-block:: python

    import cutlass
    import cutlass.cute as cute

    @cute.jit
    def foo(x: cutlass.Int32, y: cutlass.Constexpr):
        print("x = ", x)        # Prints x = ?
        print("y = ", y)        # Prints y = 2
        cute.printf("x: {}", x) # Prints x: 2
        cute.printf("y: {}", y) # Prints y: 2

    foo(2, 2)

In the example above, ``x`` is a dynamic argument with type cutlass.Int32 and ``y`` is a static argument.

With the ``cutlass.Constexpr`` annotation, a more sophisticated uses case of static argument in the JIT functions can be something like:

.. code-block:: python

    import cutlass
    import cutlass.cute as cute

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        epilogue_op: cutlass.Constexpr,
    ):
        ...

        # Perform epilogue op on accumulator and convert to C type
        acc_vec = tTR_rAcc.load()
        acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
        tTR_rC.store(acc_vec)

In this example, ``epilogue_op`` is a static argument in the JIT kernel where the argument is used for the epilogue fusion. Upon calling the kernel,
an elementwise lambda function can be passed in as the ``epilogue_op`` argument. For example, a ReLU can be applied for epilogue fusion by simply setting the
``epilogue_op`` to ``lambda x: cute.where(x > 0, x, cute.full_like(x, 0))``

Refer to the `Blackwell dense GEMM example <https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py>`__ for a complete example.

Type safety
-----------

|DSL| makes good use of type annotation in JIT function signature and validates the JIT function argument types at compile time for **type safety**.

.. code-block:: python

    import cutlass
    import cutlass.cute as cute
    import numpy as np

    @cute.jit
    def foo(x: cute.Tensor, y: cutlass.Float16):
        ...

    a = np.random.randn(10, 10).astype(np.float16)
    b = 32

    foo(a, b)
    foo(b, a)  # This will fail at compile time due to type mismatch

The type safety check helps catch the type mismatch issue early at the compile time with clear error message to avoid tricky runtime errors which is usually more expensive to debug.
In the example above, the second call to ``foo`` will fail at compile time due to the type mismatch with a clear error message:

::

    cutlass.base_dsl.common.DSLRuntimeError: DSLRuntimeError: expects argument #1 (a) to be <class 'cutlass.cute.typing.Tensor'>, but got <class 'int'>

JIT function arguments with |CUSTOM_TYPES|
--------------------------------------------
|DSL| supports |CUSTOM_TYPES| for JIT function arguments by providing two runtime checkable protocols:

* ``JitArgument`` which is used for host JIT functions to be called from Python.
    - ``__c_pointers__``: Generate a list of ctypes pointers for the current object.
    - ``__get_mlir_types__``: Generate a list of MLIR types for the current object.
    - ``__new_from_mlir_values__``: Create a new object from MLIR values.

* ``DynamicExpression`` which is used for device JIT functions to be called from the host JIT functions.
    - ``__extract_mlir_values__``: Generate a dynamic expression for the current object.
    - ``__new_from_mlir_values__``: Create a new object from MLIR values.

Refer to `typing.py <https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL/base_dsl/typing.py>`__ for more details on these protocol APIs.

Depending on different cases of the |CUSTOM_TYPES|, |DSL| provides easy ways to adopt |CUSTOM_TYPES| for JIT function arguments.

1. Direct protocol implementation in |CUSTOM_TYPES|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One way is to implement the protocol methods directly in the |CUSTOM_TYPES| to enable the protocol based JIT function argument generation.

.. code-block:: python

    import cutlass
    import cutlass.cute as cute

    # Customized type that implements the DynamicExpression protocol
    class MyDynamicExpression:
        def __init__(self, tensor, offset):
            self._tensor = tensor # Dynamic argument
            self._offset = offset # Dynamic argument

        def __extract_mlir_values__(self):
            return [self._tensor.__extract_mlir_values__(), self._offset.__extract_mlir_values__()]

        def __new_from_mlir_values__(self, values):
            return MyDynamicExpression(values[0], values[1])

    @cute.kernel
    def my_kernel(x: MyDynamicExpression):
        ...

In the example above, the ``MyDynamicExpression`` implements the ``DynamicExpression`` protocol and |DSL| will generate the JIT function arguments for the JIT kernel ``my_kernel`` based on the protocol methods.

2. Adaptor based protocol implementation for |CUSTOM_TYPES|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the case where directly changing the |CUSTOM_TYPES| to implement the protocol is not feasible, |DSL| provides adaptor based approach to adapt the |CUSTOM_TYPES| for JIT function argument generation.

The JIT function argument adaptor is a callable object that implements the desired protocol methods for the registered |CUSTOM_TYPES|. This way, |DSL| automatically queries the JIT argument adaptor registry
to generate the JIT function arguments for the given |CUSTOM_TYPES|.

.. code-block:: python

    @cutlass.register_jit_arg_adapter(MyFrameworkObject)
    class MyFrameworkObjectAdapter:
        """
        Convert a 3rd party framework object to a JIT function argument with JitArgument protocol
        """

        def __init__(self, arg):
            self._arg = arg

        def __c_pointers__(self):
            # Convert the framework object to a C-ABI compatible object
            # thru its C-ABI interface
            return [self._arg.get_cabi_pointer()]

        def __get_mlir_types__(self):
            # Return the list of MLIR types the framework object represents
            return [self._arg.get_data().mlir_type]

        def __new_from_mlir_values__(self, values):
            # Convert the MLIR values back to the framework object
            return MyFrameworkObject(values[0])

In this example, the ``MyFrameworkObjectAdapter`` implements an adaptor class which bridges the |DSL| and the 3rd party framework type ``MyFrameworkObject``.
The registration is done by just decorating the adaptor with ``cutlass.register_jit_arg_adapter`` for the customized type. With the registered adaptor,
|DSL| will automatically use the adaptor to generate the JIT function arguments for ``MyFrameworkObject`` typed arguments.
