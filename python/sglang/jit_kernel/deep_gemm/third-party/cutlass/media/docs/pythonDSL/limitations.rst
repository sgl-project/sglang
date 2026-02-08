.. _limitations:

Limitations
====================


Overview
---------------------
CuTe DSL is an embedded domain-specific language within Python. It utilizes a subset of Python's
syntax to provide a streamlined programming experience. It is important to understand that CuTe DSL
does NOT implement the complete Python language semantics in its JIT compilation process.

This section documents the current limitations of the CuTe DSL. While some of these limitations
may be addressed in future releases, developers should be aware of them when building applications with
the DSL.

Notable unsupported features
----------------------------

- GeForce RTX 50 Series support
- Programmatic Dependent Launch (PDL)
- narrow-precision data type support, including related tensor core instructions
- convolutions
- full support for ahead of time compilation
- preferred clusters
- CLC-based tile schedulers
- EVT support
- Windows support

Programming Model
---------------------

**CuTe Layout Algebra Only support 32bit**
    Today, we only support 32bit shapes/strides in CuTe layouts. 64bit or arbitrary 
    width support is planned for future releases.

**Python Native Data Types**
    CuTe DSL supports Python data structures when used for "meta-programming,"
    but these structures cannot be treated as dynamic values modifiable at runtime.
    For instance, lists and dictionaries can be used to configure kernel parameters
    during compilation or serve as containers for dynamic values,
    but their structure and organization cannot be altered during kernel execution.

    - **Static Values:**
        - Evaluated during JIT compilation phase
        - Immutable after compilation completes
        - Most Python native types (lists, tuples, dictionaries) are processed as static values
        - Primarily utilized for "meta-programming" and configuration purposes
        - Example: Lists can contain dynamic values but their structure cannot
          be modified during kernel execution

    - **Dynamic Values:**
        - Evaluated during runtime execution
        - Modifiable during execution of JIT-compiled functions
        - Only a specific subset of Python types are supported as dynamic values
        - Primitive types are automatically converted when passed as function arguments:
        
          - ``int`` → ``Int32`` (may be updated to ``Int64`` in future releases)
          - ``bool`` → ``Bool``
          - ``float`` → ``Float32`` (may be updated to ``Float64`` in future releases)

    The JIT compiler processes Python native types analogously to C++ template parameters.
    The compiled code cannot manipulate dynamic values of composite types
    such as lists, tuples, or dictionaries.

    For example, following code doesn't work as traditional Python program inside JIT function.

    .. code:: python

        @cute.jit
        def foo(a: Float32, b: Float32, i: Int32, res: cute.Tensor):
            xs = [a, b]
            # indexing list with dynamic index is not supported in CuTe DSL:
            res[0] = xs[i]

            if i == 0:
                # This will alway append Float32(3.0) to the list regardless
                # of the runtime value of `i`
                xs.append(Float32(3.0))

            for i in range(10):
                # This only append one element to the list at compile-time
                # as loop doesn't unroll at compile-time
                xs.append(Float32(1.0))

**Python Function**
    The DSL currently does not implement support for return values from Python functions,
    although this capability is planned for future releases.

    Example:

    .. code:: python

        @cute.jit
        def foo():
            return 1  # Currently unsupported in CuTe DSL

**Expression or Statement with Dependent Type**
    CuTe DSL implements static typing and does not support dependent types.
    The type of each expression must be determinable during compile time,
    in contrast to standard Python which implements dynamic typing.

    Example illustrating functionality in Python that is not supported in the DSL:

    .. code:: python

        # Valid in standard Python, but unsupported in CuTe DSL
        max(int(1), float(2.0))  # => 2.0 : float
        max(int(3), float(2.0))  # => 3   : int

    In CuTe DSL, types are promoted. For example:

    .. code:: python

        @cute.jit
        def foo(a: Int32, b: Float32, res: cute.Tensor):
            res[0] = max(a, b)  # Type is automatically promoted to Float32

    Following code using inlined if-else expression with dependent types
    is not supported in CuTe DSL:

    .. code:: python

        @cute.jit
        def foo(cond: Boolean, a: Int32, b: Float32, res: cute.Tensor):
            res[0] = a if cond else b


**Control Flow**
    The DSL transforms Python control flow statements (``if``, ``for``, ``while``)
    during Abstract Syntax Tree (AST) processing into structured control flow in MLIR
    which has the same constraints as dependent types. For instance,
    changing type of a variable in loop body is not allowed.

    - Variables must be defined prior to the control flow statement
    - Type consistency must be maintained throughout the control flow statement
    - Don't support early exit or return from if-else statements

    Example illustrating functionality in Python that is not supported in the DSL:

    .. code:: python

        @cute.jit
        def foo():
            a = Int32(1)
            for i in range(10):
                a = Float32(2)  # Changing type inside loop-body is not allowed in the DSL


**Built-in Operators**
    The DSL transforms built-in operators like ``and``, ``or``, ``max``, ``min``, etc.
    into MLIR operations. They also follow the same constraints of dependent types.
    For instance, ``a and b`` requires ``a`` and ``b`` to be of the same type.


**Special Variables**
    The DSL treats ``_`` as a special variable that it's value is meant to be ignored.
    It is not allowed to read ``_`` in the DSL.

    Example illustrating functionality in Python that is not supported in the DSL:

    .. code:: python

        @cute.jit
        def foo():
            _ = 1
            print(_)  # This is not allowed in the DSL


**Object Oriented Programming**
    The DSL is implemented on top of Python and supports Python's object-oriented programming (OOP) features
    for meta-programming at compile-time.

    However, similar to other composed data types, the DSL provides limited support for OOP when objects
    contain dynamic values. It is strongly recommended to avoid passing dynamic values between member methods
    through class state in your code.

    The following example illustrates functionality in Python that is not supported in the DSL
    without implementing the ``DynamicExpression`` protocol:

    .. code:: python

        class Foo:
            def __init__(self, a: Int32):
                self.a = a

            def set_a(self, i: Int32):
                self.a = i

            def get_a(self):
                return self.a

        @cute.jit
        def foo(a: Int32, res: cute.Tensor):
            foo = Foo(a)
            for i in range(10):
                foo.set_a(i)

            # This fails to compile because `a` is assigned a local value defined within the for-loop body
            # and is not visible outside of the loop body
            res[0] = foo.get_a()

    The example above fails to compile because ``Foo.a`` is assigned a local value defined within the for-loop body,
    which is not visible outside the loop body.

    The CuTe DSL implements an internal mechanism that provides limited support for OOP patterns via protocol.
    As the DSL continues to evolve to support additional features, this mechanism is subject to change
    and is not recommended for direct use in users' code for better portability.


**CuTe Layout algebra in native Python**
    Entirety of CuTe Layout algebra operations and APIs require JIT compilation. These 
    functionalities are exclusively available within JIT-compiled functions and cannot be 
    accessed in standard Python execution environments.
    
    Additionally, there exists a restricted set of data types that can be passed as arguments 
    to JIT-compiled functions, which further constrains their usage in native Python contexts. 
    Only following CuTe algebra types are supported as JIT function arguments: ``Tensor``, ``Pointer``, 
    ``Shape``, ``Stride``, ``Coord`` and ``IntTuple``. For ``Stride``, we don't support ``ScacledBasis``
    from native Python Context. Unfortunately, in the first release, we don't support 
    passing ``Layout`` under native Python Context.


Suggestions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For reliable and predictable results:

- Avoid dependent types in your code
- Implement explicit type conversion for dynamic values
- Clearly distinguish between static (compile-time) and dynamic (runtime) values
- Use type annotations as much as possible to help JIT compiler
  to identify type to avoid ambiguity


.. code:: python

    # Example demonstrating explicit typing
    alpha = 1.0  # Explicitly defined as float using `1.0` instead of `1`
                 #  or `float(1)`
    beta = 2.0   # Explicitly defined as float
    result = max(alpha, beta)  # Will correctly perform float comparison

**Debugging Capabilities**
    Debugging tools and facilities for the Python DSL are currently more limited in comparison to the C++
    API. For instance, we don't support single-stepping through the JIT-compiled code. And lack of exception
    handling in JIT-compiled code makes it hard to debug in some cases.

**Integration with Frameworks**
    Integration with certain deep learning frameworks is in early development stages and may have
    limitations. For instance, converting frameworking tensor to cute.Tensor is known to have overhead
    with 2us~3us per tensor as we convert from general DLPack protocol which offers comptibility with
    all frameworks.

**Hashing DSL APIs and Objects**
    DSL APIs and Objects are sensitive to MLIR context, region or other contextual information which has no meaning cross
    different context. Any stateful design rely on ``__hash__`` likely misbehave with unexpected results. An example is
    ``functools.lru_cache``, which combined with ``@cute.jit``, it may cache MLIR object from one context and use in another one.


Future Improvements
---------------------

The CuTe DSL development team is actively addressing these limitations.
Upcoming releases will aim to:

- Implement support for return values from JIT compiled functions
- Improve support for built-in operators to handle more cases without dependent types
- Enhance debugging capabilities and tools
- Improve error messages with precise diagnostic information
- Extend support for additional numeric data types
- Improve performance of converting framework tensor to ``cute.Tensor`` with native support
  for different frameworks
- Offer more user friendly benchmarking methodology

Design Limitations Likely to Remain
--------------------------------------------

The primary objective of CuTe DSL is to provide a domain-specific language for expressing
complex CUDA kernels with optimal GPU performance, not to execute arbitrary Python code on GPU hardware.

The following limitations will likely remain by design:

- **Complex Data Structures as Dynamic Values**: Lists, tuples, and dictionaries will continue to function
  as static containers. While they can store dynamic values, their structure (adding/removing elements)
  cannot be modified during execution of JIT-compiled functions.

- **Dependent Types**: Supporting dependent types would introduce substantial complexity and
  adversely affect the performance characteristics of generated code.

- **CuTe Layout Algebra**: We don't have plan to extend the support of CuTe Layout Algebra 
  under native Python Context. We are planning to extend support for data types and allow 
  JIT function to interoperate with native Python code.
