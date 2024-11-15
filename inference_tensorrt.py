import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import time


class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings = self.allocate_buffers()

    def inference(self, image):
        # Copy input to device
        np.copyto(self.inputs[0].host, image.ravel())

        # Run inference
        [
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
            for inp in self.inputs
        ]
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )
        [
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
            for out in self.outputs
        ]
        self.stream.synchronize()

        return self.outputs[0].host


def main():
    engine_path = "efficientnet.engine"
    image_path = "goldfish.jpg"

    # Initialize inference
    trt_model = TensorRTInference(engine_path)

    # Load and preprocess image
    image = cv2.imread(image_path)
    input_data = trt_model.preprocess(image)

    times = []
    print("Running TensorRT inference...")

    # Run inference 10 times
    for i in range(10):
        start_time = time.perf_counter()

        output = trt_model.inference(input_data)
        class_id = np.argmax(output)

        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        times.append(inference_time)
        print(f"Iteration {i+1}: {inference_time:.2f}ms")

    # Calculate average of last 5 runs
    avg_time = sum(times[-5:]) / 5
    print(f"\nAverage inference time (last 5 runs): {avg_time:.2f}ms")
    print(f"Predicted class: {class_id}")


if __name__ == "__main__":
    main()
