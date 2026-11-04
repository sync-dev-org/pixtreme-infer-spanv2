# pixtreme-infer-spanv2

## Purpose

`pixtreme-infer-spanv2` provides the SPANv2 (Swift Parameter-free Attention
Network V2) architecture module for `pixtreme.infer`. It reads SPANv2
safetensors weights and derives the scale (x1, x2 or x4), feature channels and
block count from the file. The import package is `pixtreme.infer.spanv2`.

The module builds TensorRT networks directly and uses tensoracle to build,
cache and open engines. It supports ordinary inference and CUDA Graph replay.
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

The package depends on `pixtreme-infer>=0.17.0`, `tensoracle>=1.6.0,<2` and
`tensorrt-cu13>=11,<12`. Use CuPy matching the application's CUDA environment,
such as `cupy-cuda13x` for CUDA 13.

Wheels and `SHA256SUMS` are also available from the
[GitHub Releases](https://github.com/sync-dev-org/pixtreme-infer-spanv2/releases).
Download the wheel and checksum file for the same version, verify the checksum
and install the wheel:

```sh
sha256sum -c SHA256SUMS
python -m pip install /path/to/pixtreme_infer_spanv2-X.Y.Z-py3-none-manylinux_2_28_x86_64.whl
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

`pixtreme.infer.spanv2` exports `FAMILY_CONTRACT` and `FACTORY`.
`FACTORY.family_contract` returns `FAMILY_CONTRACT`, which declares RGB
float32 input and output Frames, sRGB colorspace and gamma, no color matrix
and no call parameters.

`FACTORY.query_contract(path, precision=..., input_hw=...)` accepts a string
or path-like safetensors file path directly. It validates settings and the
file's header, including tensor declarations, without reading tensor values,
importing TensorRT or tensoracle, or allocating GPU memory. The parent
pixtreme namespace imports CuPy during initialization. The returned
`StatelessOpenContract` preserves the supplied artifact identity, precision
and exact `input_hw=(height, width)`, and reports the file's scale through
`contract.scale`. Both dimensions must be positive integers.

Precision must be `"fp32"` or `"fp16"`. FP32 uses standard TensorRT layers
and serves as the comparison reference. FP16 uses the packaged cubin plugin
for the fused tail. Both precisions take and return float32 Frames.

`FACTORY.open(path, precision=..., input_hw=...)` repeats the query's checks,
then opens or builds tensoracle's cached engine and returns a reusable Frame
runner. Its `contract` equals the query result for the same settings.

The input must be a `pixtreme.core.Frame` whose data is a C-contiguous CUDA
array of shape `(height, width, 3)` and dtype `float32`, with
`channels=("R", "G", "B")`, `colorspace="sRGB"`, `gamma="sRGB"` and
`matrix=None`. Pixels use RGB values in the 0–1 convention declared by
`spanv2.io`. The module performs no range or color conversion, subtracts no
mean and does not clip the output. The output is a Frame with the same dtype
and color metadata, of shape `(height * scale, width * scale, 3)`.

```python
import cupy as cp
from pixtreme.core import Frame
from pixtreme.infer.spanv2 import FACTORY

weights = "/path/to/model.safetensors"
input_hw = (540, 960)
contract = FACTORY.query_contract(weights, precision="fp16", input_hw=input_hw)

# Replace this black image with your contiguous RGB float32 CUDA pixels.
input_frame = Frame(
    data=cp.zeros((*input_hw, 3), dtype=cp.float32),
    channels=("R", "G", "B"),
    colorspace="sRGB",
    gamma="sRGB",
    matrix=None,
)
with FACTORY.open(weights, precision="fp16", input_hw=input_hw) as runner:
    assert runner.contract == contract
    output_frame = runner(input_frame)
    cp.cuda.get_current_stream().synchronize()
```

Reuse the runner for sequential requests on the opening CUDA device and one
stream. Ordinary inference consumes the input through DLPack without a host
memory round trip and allocates output storage for each call. Keep input
pixels alive until the stream has completed their use. Complete pending CUDA
work before leaving the `with` block or calling `close()`; closing releases
the execution context and later requests fail.

For CUDA Graph inference, select `FACTORY.graph_factory()`. The returned
factory accepts the same path, precision and input size. Selection and
`query_contract` allocate no GPU resources. Open inside an explicit
non-default CuPy stream and use that stream for every request. Continuing
from the synchronized example above:

```python
graph_factory = FACTORY.graph_factory()
stream = cp.cuda.Stream(non_blocking=True)
with stream:
    with graph_factory.open(weights, precision="fp16", input_hw=input_hw) as runner:
        output_frame = runner(input_frame)
        stream.synchronize()
```

Opening prepares fixed input and output storage and captures the execution
using core's `CudaGraphBuffers` and `CudaGraphSession`. Each request copies
the current input into that storage, replays the graph and returns a separate
output allocation carrying the request's color metadata. Inputs must be ready
on the opening stream and remain alive until their copies finish. Returned
Frames retain their pixels across later requests and after close. A different
input size, CUDA device or stream requires a separate instance. Complete
pending work before closing the graph runner.

## License

The project's own code is licensed under the MIT License. The full license
text is included in the source repository's `LICENSE` file and the wheel's
`.dist-info/licenses/LICENSE` file. The wheel also includes third-party
license texts and notices in its `.dist-info/licenses/` directory. The notice
for portions derived from the original SPANv2 code is included as
`licenses/SPANV2-LICENSE.txt` within that directory.
