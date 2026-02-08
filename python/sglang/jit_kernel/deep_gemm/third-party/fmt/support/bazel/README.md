# Bazel support

To get [Bazel](https://bazel.build/) working with {fmt} you can copy the files `BUILD.bazel`, 
`MODULE.bazel`, `WORKSPACE.bazel`, and `.bazelversion` from this folder (`support/bazel`) to the root folder of this project. 
This way {fmt} gets bazelized and can be used with Bazel (e.g. doing a `bazel build //...` on {fmt}). 

## Using {fmt} as a dependency

### Using Bzlmod

The [Bazel Central Registry](https://github.com/bazelbuild/bazel-central-registry/tree/main/modules/fmt) provides support for {fmt}.

For instance, to use {fmt} add to your `MODULE.bazel` file:

```
bazel_dep(name = "fmt", version = "11.1.4")
```

### Live at head

For a live-at-head approach, you can copy the contents of this repository and move the Bazel-related build files to the root folder of this project as described above and make use of `local_path_override`, e.g.:

```
local_path_override(
    module_name = "fmt",
    path = "../third_party/fmt",
)
```
