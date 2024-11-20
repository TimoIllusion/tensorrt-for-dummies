import os
import time

import torch
import torch_tensorrt
import cv2

from model import create_model, preprocess_numpy_bgr_image, postprocess_output


def main():
    # Path to the TensorRT model file
    trt_model_path = "efficientnet_trt.ts"

    # Check if the TensorRT model exists
    if os.path.exists(trt_model_path):
        # Load the TensorRT TorchScript model
        model_trt = torch.jit.load(trt_model_path)
        print("Loaded TorchScript-TensorRT model from file.")
    else:
        print("TorchScript-TensorRT model not found. Converting PyTorch model...")
        # Load the pre-trained EfficientNet model
        model = create_model().cuda().eval()

        # Convert to TorchScript
        inputs = [torch.randn(1, 3, 224, 224).cuda()]

        # Compile with Torch-TensorRT
        compiled_model = torch_tensorrt.compile(
            model,
            inputs=inputs,
            enabled_precisions={torch.float32, torch.float16},  # Enable FP16
            workspace_size=1 << 32,  # 4GB workspace size
        )

        # Save the model to file
        torch_tensorrt.save(
            compiled_model, trt_model_path, output_format="torchscript", inputs=inputs
        )
        model_trt = compiled_model
        print(f"Saved TorchScript-TensorRT model to {trt_model_path}")

    # Load and preprocess image
    image = cv2.imread("cat.jpg")
    input_tensor = preprocess_numpy_bgr_image(image).cuda()

    times = []
    print("Running Torch-TensorRT inference...")

    # Run inference 100 times
    with torch.no_grad():
        for i in range(100):
            start_time = time.perf_counter()
            output = model_trt(input_tensor)
            class_id = postprocess_output(output)
            inference_time = (time.perf_counter() - start_time) * 1000  # ms

            times.append(inference_time)
            print(f"Iteration {i+1}: {inference_time:.2f}ms")

    # Calculate average of last 95 runs (excluding first 5 warmup runs)
    avg_time = sum(times[-95:]) / 95
    print(f"\nAverage inference time (last 95 runs): {avg_time:.2f}ms")
    print(f"Predicted ImageNet class ID: {class_id}")


if __name__ == "__main__":
    main()
