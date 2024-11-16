# TensorRT for Dummies

Easy examples to get started with TensorRT.

>TL;DR: Use Torch2TRT for PyTorch for a significant (>2x) speedup with only a few lines of additional code.

## Install TensorRT

>Note: Change versions according to your versions.
>Note: Official NVIDIA pytorch docker container already comes with TensorRT installed. This is recommended if possible.

**Windows**:

1. Download and install CUDA (tested with CUDA 12.6)
2. Download and extract CUDNN to CUDA directory (tested with CUDNN 8.9.7 for CUDA 12.x)
3. Download and extract TensorRT to any directory, e.g. ``C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.6.0.26`` (tested with TensorRT 10.6 GA for CUDA 12.0-12.6)
4. Add TensorRT lib dir `C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.6.0.26\lib` to PATH environment variable (e.g. via GUI)
5. Install TensorRT wheels from TensorRT python dir, e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.6.0.26\python` for corresponding python version:

    ```console
    cd C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.6.0.26\python
    pip install tensorrt_dispatch-10.6.0-cp310-none-win_amd64.whl tensorrt_lean-10.6.0-cp310-none-win_amd64.whl tensorrt-10.6.0-cp310-none-win_amd64.whl
    ```	
6. Verify installation:

    ```bash
    python -c "import tensorrt as trt; print(trt.__version__)"
    ```

For C++ part `inference_tensorrt10_cpp`:

1. Install vcpkg (https://github.com/microsoft/vcpkg). 

```bash

git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
```

``VCPKG_ROOT`` environment variable should be set to the vcpkg directory, e.g. `C:\vcpkg`.
Add VCPKG_ROOT to PATH environment variable (e.g. via GUI).

2. Install required vcpkg packages:

```bash
vcpkg install opencv:x64-windows
vcpkg integrate install
```

3. Install CMake extensions if not already installed via recommended extensions on vscode startup.

4. Select compiler, configure and build the project in vscode (release mode).

**Linux**:

See official TensorRT installation guide: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

## Install Other Requirements

```bash
pip install -r requirements.txt
```

```bash
pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt
```

## Getting Started

Run the scripts one by one:

```bash
python inference_pytorch.py # (~7.14 ms on RTX 4090)
python inference_torch2trt.py # (~1.37 ms on RTX 4090)

python export2onnx.py
python onnx2tensorrt10.py

python inference_tensorrt10.py # (~1.39 ms on RTX 4090)

.\build\Release\main.exe # (~1.25 ms on RTX 4090), run command with ./build/Release/main on Linux
```

**Torch2TRT is the recommended way to use TensorRT with PyTorch in Python. Using TensoRT directly is more complex and requires more code, but is also more flexible. This process is recommended if C++ TensorRT API is used and not Python or other frameworks like TesnorFlow are used.**

## References

- https://github.com/NVIDIA/TensorRT/tree/main/samples/python/efficientnet
- https://tengteng.medium.com/example-inference-code-to-run-tensorrt-10-0-32ea93fdcc2e
- https://images.unsplash.com/photo-1529778873920-4da4926a72c2?ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8Y3V0ZSUyMGNhdHxlbnwwfHwwfHw%3D&ixlib=rb-1.2.1&q=80&w=1000
- https://github.com/NVIDIA/TensorRT/tree/release/10.6/quickstart/SemanticSegmentation (Apache-2.0)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- **Timo Leitritz**, 2024

## AI Assistance

Development of this project was supported by OpenAI's ChatGPT models (o1-preview, 4o, Claude 3.5 Sonnet, November 2024), which provided code suggestions and troubleshooting help.
