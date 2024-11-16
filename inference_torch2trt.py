import time

import torch
from torch2trt import torch2trt, TRTModule

import cv2

from model import create_model, preprocess_numpy_bgr_image, postprocess_output


def main():

    # Path to the TensorRT model file
    trt_model_path = "efficientnet_trt.pth"

    # Check if the TensorRT model exists
    try:
        # Load the TensorRT model
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(trt_model_path))
        print("Loaded TensorRT model from file.")
    except FileNotFoundError:
        print("TensorRT model file not found. Converting PyTorch model to TensorRT...")
        # Load the pre-trained EfficientNet-B0 model
        model = create_model().cuda()
        # Convert the PyTorch model to TensorRT
        x = torch.ones((1, 3, 224, 224)).cuda()
        model_trt = torch2trt(model, [x], fp16_mode=True, max_workspace_size=1 << 25)
        # Save the TensorRT model for future use
        torch.save(model_trt.state_dict(), trt_model_path)
        print(f"Saved TensorRT model to {trt_model_path}")

    # Load the input image
    image = cv2.imread("cat.jpg")

    # Preprocess the image
    input_tensor = preprocess_numpy_bgr_image(image).cuda()

    times = []
    print("Running torch2trt inference...")

    # Run inference 100 times
    with torch.no_grad():
        for i in range(100):
            start_time = time.perf_counter()
            output = model_trt(input_tensor)
            class_id = postprocess_output(output)
            inference_time = (
                time.perf_counter() - start_time
            ) * 1000  # Convert to milliseconds

            times.append(inference_time)
            print(f"Iteration {i+1}: {inference_time:.2f}ms")

    # Calculate the average inference time of the last 5 runs
    avg_time = sum(times[-95:]) / 95
    print(f"\nAverage inference time (last 5 runs): {avg_time:.2f}ms")
    print(f"Predicted ImageNet class ID: {class_id}")


if __name__ == "__main__":
    main()
