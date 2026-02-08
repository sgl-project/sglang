.. _faqs:

FAQs
====

General
---------------------

**Are the DSLs replacing C++ templates?**

    TL;DR: No - but also yes. The CUTLASS 4.0 release (CuTe DSL), along with all
    future extensions to our Python-native programming models, does not come at the
    expense of CUTLASS C++.  CUTLASS 2.x and 3.x C++ APIs are both going to continue
    receiving fixes and updates for the architectures we support them for. However,
    CUTLASS 4.x CuTe DSL is fully isomorphic in its programming model and performance
    with CuTe C++ for Blackwell, and it is our hope that the community embraces this
    for much easier while still equally performant custom kernel development.  This is
    why we are releasing CuTe DSL with support for all architectures starting with the
    NVIDIA Ampere Architecture.

**What is the difference between CuTe DSL, CUTLASS Python, and CUTLASS DSLs?**

    CUTLASS Python was the Python interface for instantiating C++ kernels via a Python
    frontend. This is now deprecated with the release of CUTLASS 4.0. CUTLASS DSLs are
    a family of Python DSLs for native device programming in Python. Currently, this is
    limited to our initial release of CuTe DSL, but future versions will include higher-level
    abstractions that gradually trade off control for convenience.

**What should I learn, CUTLASS C++ or the Python DSLs?**

    We believe the Python DSLs will significantly improve the learning curve and recommend starting
    with them for all newcomers, as they eliminate the inherent complexity of learning C++
    metaprogramming for GPU kernel programming. Since CuTe C++ and CuTe DSL share fully isomorphic
    programming models and patterns, any knowledge gained can eventually be applied to C++.

**Where will the code live? PIP wheel or GitHub repo? Do I have to build it myself?**

    This is a major change compared to CUTLASS C++ and Python DSLs. Going forward,
    the GitHub code only exists as a way for users to file issues and pull requests against.
    While it can be used with the pip wheel, we do not recommend most users do so unless they are
    hacking on the DSL itself. For all other users, we recommend they
    simply ``pip install nvidia-cutlass-dsl`` and use the pip wheel as the single source
    of truth for the dialect compiler and DSL implementation. CUTLASS GitHub repository will
    contain a ``requirements.txt`` file pinning the version of the wheel consistent with the state
    of the OSS repository (please see :doc:`quick_start`). This means getting started with
    CUTLASS is easier than ever: no more CMake command lines to learn and no more builds to kick
    off. Simply install the pip wheel and start running the examples.

Migration
---------------------

**Should I port my code from C++ templates to Python?**

    Almost certainly not, unless you need extremely fast JIT times for your kernel and C++ compile times
    are a blocker for you. The 2.x and 3.x APIs will continue to be supported, and Nvidia's Hopper and
    Blackwell architectures 3.x will continue to improve in terms of features
    and performance.

**Are portability promises different with Python?**

    For the initial release while the DSL is still in beta, we do not promise any portability
    as we may make changes to the DSL itself. While we do not expect any changes to the CuTe operations,
    the DSL utilities, decorators, helper classes like pipelines and schedulers may change as we refine them
    with community feedback. We encourage users to file issues and discussions on GitHub during this
    beta period with their feedback!

    In the long term, we plan to continue to treat the OSS community with care.
    Just like the prior history of CUTLASS, we plan not to break users unless necessary,
    but we reserve the right to make limited breaking changes in case we believe it is a
    net benefit to the community and project. These will be announced ahead of time and/or
    clearly highlighted in the CHANGELOG of each release.

Technical
---------------------
**What NVIDIA architectures will it support?**

    CuTe DSL will support all NVIDIA GPU architectures starting with NVIDIA Ampere Architecture (SM80).

**Will it be compatible with DL frameworks (e.g., PyTorch, JAX)?**

    Yes, we will provide utilities to convert from DLPack-supported tensor formats
    to ``cute.Tensor``. This should allow a user to never have to leave Python
    when writing model code in their framework of choice. Our JAX interoperability story is not
    as strong as PyTorch's today, however, we are actively working on improving it
    and welcome contributions in this space.

**Does it compile to PTX or SASS?**

    CuTe DSL compiles the program down to PTX. After that, we currently use the PTX compiler that
    ships with the CUDA toolkit to compile the PTX down to SASS. We plan to remove
    this limitation in the future and allow the use of the PTX JIT that is included in the
    CUDA driver in case a user does not have a CUDA toolkit installed.

**Do I need to use NVCC or NVRTC?**

    No, the ``nvidia-cutlass-dsl`` wheel packages is everything needed to generate GPU kernels. It
    shares the driver requirements of the 12.9 toolkit which can be found
    `here <https://developer.nvidia.com/cuda-toolkit-archive>`__.

**How would one debug the code?**

    Since CuTe DSL is not native python and an embedded DSL instead, tools like `pdb`
    cannot be used.  However, if you have experience with GPU kernel programming, the debugging
    techniques will be nearly identical. Typically, compile time and runtime printing
    of types and values are the most expedient. Please see `documentation on printing <https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/print.ipynb>`__
    to learn how to print types and values at both compile time and runtime.
    You can also use ``cuda-gdb`` to set breakpoints in the program and step through the execution
    or use tools such as ``compute-sanitizer`` to detect and triage bugs in your program. As the DSL
    matures, our source location tracking from Python user programs will also improve to provide
    more helpful source-level mapping when setting breakpoints and using other tools such as nsight.

**How would one implement warp specialization in CuTe DSL?**

    Exactly the same way you would in C++ but in a Python-native syntax instead.
    Consult our :doc:`cute_dsl_general/dsl_control_flow` and
    `"Blackwell kernel example" <https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py>`__
    for a detailed how-to guide.

**Can I call functions from other functions or use OOP?**

    Yes. We frequently call functions from one another and set up class
    hierarchies to organize and modularize our code for pipelines and schedulers.
    Consult the :doc:`cute_dsl_general/dsl_introduction` documentation or our examples for more details.

License
---------------------
**What is the license for CuTe DSL and the associated GitHub samples?**

    CuTe DSL components available `on Github <https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL>`__ and via the nvidia-cutlass-dsl Python pip wheel
    are released under the `"NVIDIA Software End User License Agreement (EULA)" <https://github.com/NVIDIA/cutlass/tree/main/EULA.txt>`__.
    Because the pip package includes a compiler that shares several components with the CUDA Toolkit,
    it is subject to usage terms and restrictions similar to those of the CUDA SDK. Please refer to the EULA for specific terms of use.

    CuTe DSL samples and Jupyter notbooks, released `on GitHub <https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL>`__ are provided under
    the BSD 3-Clause License and may be used and redistributed under those terms. This distinction ensures that developers have flexibility
    when using or modifying the code samples, independent of the compiler and runtime components governed by the EULA.

    If you have any questions or need clarification, feel free to contact us.
