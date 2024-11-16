import time

import torch
import cv2

from model import create_model, preprocess_numpy_bgr_image, postprocess_output


def main():

    # Load model
    model = create_model()

    # Load and preprocess image
    image = cv2.imread("goldfish.jpg")
    image = preprocess_numpy_bgr_image(image)

    # Move to gpu
    image = image.cuda()
    model = model.cuda()

    times = []
    print("Running PyTorch inference...")

    # Run inference 10 times
    for i in range(10):
        start_time = time.perf_counter()

        with torch.no_grad():
            output = model(image)
            class_id = postprocess_output(output.cpu())

        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        times.append(inference_time)
        print(f"Iteration {i+1}: {inference_time:.2f}ms")

    # Calculate average of last 5 runs
    avg_time = sum(times[-5:]) / 5
    print(f"\nAverage inference time (last 5 runs): {avg_time:.2f}ms")
    print(f"Predicted ImageNet class ID: {class_id}")


if __name__ == "__main__":
    main()
