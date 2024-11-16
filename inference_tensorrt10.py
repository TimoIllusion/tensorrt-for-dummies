# Derived and adapted from https://tengteng.medium.com/example-inference-code-to-run-tensorrt-10-0-32ea93fdcc2e

import time

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import cv2
import torch

from model import preprocess_numpy_bgr_image, postprocess_output


class TensorRTInference:
    def __init__(self, engine_path):
        """
        Initialize the TensorRT inference engine.
        """
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # Allocate buffers for inputs and outputs
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def load_engine(self, engine_path):
        """
        Load a serialized TensorRT engine from file.
        """
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    class HostDeviceMem:
        """
        Simple container for host and device memory.
        """

        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

    def allocate_buffers(self):
        """
        Allocate memory for inputs and outputs, and create CUDA stream.
        """
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            size = trt.volume(self.engine.get_tensor_shape(tensor_name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))

            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Add device buffer to bindings
            bindings.append(int(device_mem))

            # Separate inputs and outputs
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        """
        Perform inference on the input data.
        """
        # Copy input data to host memory and then to device
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        # Set tensor addresses
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i), self.bindings[i]
            )

        # Run inference asynchronously
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output data back to host
        cuda.memcpy_dtoh_async(
            self.outputs[0].host, self.outputs[0].device, self.stream
        )

        # Synchronize the stream
        self.stream.synchronize()

        return self.outputs[0].host


def main():
    engine_path = "efficientnet.engine"

    # Initialize TensorRT inference engine
    trt_inference = TensorRTInference(engine_path)

    # Load and preprocess the input image
    img = cv2.imread("cat.jpg")
    img_tensor_batched = preprocess_numpy_bgr_image(img)
    input_data = img_tensor_batched.numpy()

    # Timing inference runs
    times = []
    print("Running TensorRT inference...")

    for i in range(100):
        start_time = time.perf_counter()

        # Perform inference
        output_data = trt_inference.infer(input_data)
        output_tensor_batched = torch.Tensor(output_data).unsqueeze(0)

        # Postprocess output to get class ID
        top1 = postprocess_output(output_tensor_batched)

        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        times.append(inference_time)
        print(f"Iteration {i + 1}: {inference_time:.2f}ms")

    # Calculate average inference time for the last 5 runs
    avg_time = sum(times[-95:]) / 95
    print(f"\nAverage inference time (last 5 runs): {avg_time:.2f}ms")
    print(f"Predicted ImageNet class ID: {top1}")


if __name__ == "__main__":
    main()
