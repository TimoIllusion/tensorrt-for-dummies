# TensorRT for Dummies

Easy examples to get started with TensorRT.

## Install TensorRT

>Note: Change versions according to your versions.

**Windows**:

1. Download and install CUDA (tested with CUDA 12.6)
2. Download and extract CUDNN to CUDA directory (tested with CUDNN 8.9.7 for CUDA 12.x)
3. Download and extract TensorRT to any directory, e.g. ``C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT`` (tested with TensorRT 10.6 GA for CUDA 12.0-12.6)
4. Add TensorRT lib dir `C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\lib` to PATH environment variable (e.g. via GUI)
5. Install TensorRT wheels from TensorRT python dir, e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.6.0.26\python` for corresponding python version:

```bash
pip install tensorrt_dispatch-10.6.0-cp310-none-win_amd64.whl tensorrt_lean-10.6.0-cp310-none-win_amd64.whl tensorrt-10.6.0-cp310-none-win_amd64.whl
```	
6. Verify installation:

```bash
python -c "import tensorrt as trt; print(trt.__version__)"
```
