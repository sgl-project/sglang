raise RuntimeError(
    """The 'mini_lb' module has been relocated to the 'sglang_router' package.
    We recommend installing 'sglang-router' with Rust support for optimal performance.
    If you encounter issues building the router with Rust, set the environment variable
    'SGLANG_ROUTER_BUILD_NO_RUST=1' and add '--mini-lb' to the command line to use the Python version of 'mini_lb'."""
)
