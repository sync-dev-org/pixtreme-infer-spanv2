# pixtreme-infer-spanv2

## Purpose

`pixtreme-infer-spanv2` provides the SPANv2 (Swift Parameter-free Attention
Network V2) architecture module for `pixtreme.infer`. It reads SPANv2
safetensors weights and derives the scale (x1, x2 or x4), feature channels and
block count from the file. The import package is `pixtreme.infer.spanv2`.

The module builds TensorRT networks in C++ and uses tensoracle v2 to build,
cache and open engines through its native `Model` class and the core lifecycle. It supports ordinary inference and CUDA Graph replay.
Weights and engines are not distributed with the package.

## Supported environment

- Linux x86_64 with glibc 2.28 or newer
- Python 3.12 or newer
- NVIDIA GPU: sm86, sm89 or sm120
- CUDA 13 with a compatible NVIDIA driver, TensorRT 11 and CuPy for CUDA 13

Release wheels contain the native SPANv2 plugin and cubins for all three GPU
targets. Installation needs no C++ compiler or CUDA/TensorRT SDK setup.

## Installation

Install from PyPI:

```sh
python -m pip install pixtreme-infer-spanv2
```

The package depends on `pixtreme>=2.6.0`, `pixtreme-infer>=0.21.0`, `tensoracle>=2.1.1` and
`tensorrt-cu13>=11,<12`. Use CuPy matching the application's CUDA environment,
such as `cupy-cuda13x` for CUDA 13.

Wheels and `SHA256SUMS` are also available from the
[GitHub Releases](https://github.com/sync-dev-org/pixtreme-infer-spanv2/releases).
Download the wheel and checksum file for the same version, verify the checksum
and install the wheel:

```sh
sha256sum -c SHA256SUMS
python -m pip install /path/to/pixtreme_infer_spanv2-*.whl
```

## Weights

Supply a safetensors file containing one set of float32 tensors with the
SPANv2 network's unprefixed `state_dict` keys. The safetensors header must
include the following metadata. All metadata values are strings:

| Key | Value |
|---|---|
| `spanv2.format` | `"1"` |
| `spanv2.upscale` | `"1"`, `"2"` or `"4"` |
| `spanv2.feature_channels` | Feature channel count C, as a positive decimal integer string |
| `spanv2.num_blocks` | Block count B, as a positive decimal integer string |
| `spanv2.bias` | `"true"` or `"false"` |
| `spanv2.io` | `"rgb-0-1"` |
| `spanv2.color` | `"srgb"` |

Tensor names, shapes and dtypes must agree with the declared architecture.
Missing metadata, unsupported values or inconsistent tensors are rejected
before engine creation with their observed and accepted values. PyTorch
checkpoint files (pickle) are not accepted; the exporter must produce this
safetensors format. The inference runtime does not require PyTorch.

## Public API

`pixtreme.infer.spanv2` exports one class:

```python
Model(weights, *, precision="fp16", input_hw=(height, width), use_cuda_graph=False)
```

Pass a string or path-like safetensors path directly. The constructor validates
settings, metadata and exact tensor declarations and publishes `model.contract`
without initializing CUDA or allocating GPU memory. It derives the scale from the
file. `contract.precision`, `contract.input_hw`, `contract.scale` and
`contract.use_cuda_graph` describe the fixed configuration. `model.family_contract`
declares exact RGB float32 input and output Frames, sRGB colorspace and gamma,
no color matrix and no auxiliary parameters. Contract values are immutable copies
of the native declaration.

Both input dimensions must be positive integers. Precision must be `"fp32"` or
`"fp16"`. FP32 uses standard TensorRT layers and serves as the comparison reference.
FP16 uses the packaged cubin plugin for the fused tail. Both precisions accept and
return float32 Frames.

`model.prepare(device=0, stream=0)` repeats artifact validation and acquires the
engine, execution and GPU storage. The prepared contract equals the constructor's
contract. A changed contract or failed preparation requires a new instance.
Entering `with model:` prepares a new ordinary instance with the default device
and stream; an explicitly prepared instance keeps its selected stream.
`model.close()` releases resources and is idempotent. Inference before preparation,
after failed preparation or after close raises `RuntimeError` before backend work.

The input must be a `pixtreme.core.Frame` with contiguous CUDA pixels of shape
`(height, width, 3)`, dtype `float32`, `channels=("R", "G", "B")`,
`colorspace="sRGB"`, `gamma="sRGB"` and `matrix=None`. Pixels use the RGB 0–1
convention declared by `spanv2.io`. The module performs no range or color
conversion, subtracts no mean and does not clip output values. The output has
the same dtype and color metadata and shape `(height * scale, width * scale, 3)`.

```python
import cupy as cp
import pixtreme as px
from pixtreme.infer.spanv2 import Model

input_hw = (540, 960)
model = Model("/path/to/model.safetensors", precision="fp16", input_hw=input_hw)
contract = model.contract  # CPU-only inspection
stream = px.core.Stream(device=0, non_blocking=True)
model.prepare(device=0, stream=stream.ptr)

with cp.cuda.ExternalStream(stream.ptr), model:
    # Replace this black image with your contiguous RGB float32 CUDA pixels.
    pixels = cp.zeros((*input_hw, 3), dtype=cp.float32)
    input_frame = px.core.Frame(
        data=pixels, channels=("R", "G", "B"), colorspace="sRGB",
        gamma="sRGB", matrix=None, stream=stream.ptr,
    )
    output_frame = model(input_frame, stream=stream.ptr)
    stream.synchronize()
    output_pixels = cp.from_dlpack(output_frame)
```

Reuse one instance for sequential requests on its prepared device and stream.
Ordinary execution passes the native Frame's input address directly to TensorRT
and allocates an independent output Frame. All pixel work stays on the GPU.
Borrowed input allocations must remain alive until their queued uses finish.
Complete pending stream work before closing the model.

For CUDA Graph inference, set `use_cuda_graph=True` in the same constructor.
Prepare the model with an explicit non-default stream and pass that stream's
numeric handle to every call. Core warms up and captures the module's single
route during preparation without consuming requests. Each call copies the current
input into fixed GPU storage, replays the graph and copies its result into an
independent output Frame. Later requests and model cleanup do not change earlier
outputs. A different configuration, device or stream requires a new instance.

Output pixels are allocated on the inference stream. Keep an external stream
(such as a CuPy stream) alive through the last reference to each output Frame,
shallow clone and exported DLPack view. A stream created by `px.core.Stream`, as
above, is retained by pixtreme's native output allocations. To read on another
stream, use the Frame's `wait` and `record_stream` methods or the DLPack stream
handoff.

## C and C++

The wheel includes `include/pixtreme/infer/spanv2.h`, the header-only C++ wrapper
`include/pixtreme/infer/spanv2.hpp`, `lib/libpixtreme_infer_spanv2.so.1` and a
CMake package under `pixtreme/infer/spanv2/`. C and C++ programs use the same
native implementation and packaged plugin as Python, without loading a Python
interpreter. The module C ABI is 1.0, independent of the core ABI and the
distribution version. Its ABI check accepts the same major and a header minor
no newer than the library minor.
Native consumers use core's C ABI 3 (`libpixtreme_infer.so.3`, core 0.21.0 or newer).

Set `CMAKE_PREFIX_PATH` to the installed module directory, for example
`/path/to/site-packages/pixtreme/infer/spanv2`, and link the exported target:

```cmake
cmake_minimum_required(VERSION 3.26)
project(spanv2_example LANGUAGES C CXX)
find_package(pixtreme_infer_spanv2 1 CONFIG REQUIRED)
add_executable(example example.cpp)
target_link_libraries(example PRIVATE pixtreme_infer_spanv2::pixtreme_infer_spanv2)
```

The CMake package finds the companion core package in the same installation.
Both headers use core's status, error ownership and model lifecycle. A C program
creates a core model handle with the module's options:

```c
#include <pixtreme/infer/spanv2.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    pixtreme_infer_spanv2_model_options_t options = PIXTREME_INFER_SPANV2_MODEL_OPTIONS_INIT;
    options.weights = argv[1];
    options.precision = "fp16";
    options.input_height = 540;
    options.input_width = 960;
    pixtreme_infer_model_t *model = NULL;
    pixtreme_infer_error_t *error = NULL;
    pixtreme_infer_status_t status = pixtreme_infer_spanv2_model_create(&options, &model, &error);
    if (status != PIXTREME_INFER_SUCCESS) {
        fprintf(stderr, "%s\n", pixtreme_infer_error_message(error));
        pixtreme_infer_error_destroy(error);
        return 1;
    }
    /* CPU-only construction has validated the weights and fixed contract.
       Use pixtreme_infer_model_prepare / run with pixtreme CUDA Frame handles. */
    status = pixtreme_infer_model_close(model, &error);
    pixtreme_infer_model_destroy(model);
    pixtreme_infer_error_destroy(error);
    return status == PIXTREME_INFER_SUCCESS ? 0 : 1;
}
```

`use_cuda_graph` accepts zero or one; its default is zero. The options descriptor
starts with `struct_size`, and an omitted graph field uses that default.
Construction validates settings and weights before creating a CUDA context or
an engine. Subsequent preparation, inference, close and destruction use
`pixtreme_infer_model_*` from `pixtreme/infer/core.h`.

The C++ wrapper owns the handle, inherits core's operations and throws
`pixtreme::infer::Error` on failure:

```cpp
#include <pixtreme/infer/spanv2.hpp>

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    pixtreme::infer::spanv2::Model model(argv[1], "fp16", 540, 960);
    auto contract = model.contract();
    model.close();
    return contract.input_height == 540 ? 0 : 1;
}
```

Keep the module library loaded until every model handle has been destroyed.
Complete submitted GPU work before closing or destroying a model, as required
by the core lifecycle.

## License

The project's own code is licensed under the MIT License. The full license
text is included in the source repository's `LICENSE` file and the wheel's
`.dist-info/licenses/LICENSE` file. The wheel also includes third-party
license texts and notices in its `.dist-info/licenses/` directory. The notice
for portions derived from the original SPANv2 code is included as
`licenses/SPANV2-LICENSE.txt` within that directory.
