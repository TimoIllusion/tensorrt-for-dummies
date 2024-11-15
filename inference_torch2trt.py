# inference_torch2trt.py

import torch
import torchvision.models as models
from torch2trt import torch2trt, TRTModule
import time
import cv2
import numpy as np


class Torch2TRTInference:
    def __init__(self, model_path=None):
        # Load EfficientNet model
        self.model = models.efficientnet_b0(pretrained=True).eval().cuda()

        if model_path:
            self.model_trt = TRTModule()
            self.model_trt.load_state_dict(torch.load(model_path))
        else:
            # Convert to TensorRT
            x = torch.ones((1, 3, 224, 224)).cuda()
            self.model_trt = torch2trt(
                self.model, [x], fp16_mode=True, max_workspace_size=1 << 25
            )
            # Save TRT model
            torch.save(self.model_trt.state_dict(), "efficientnet_trt.pth")

    def preprocess(self, image):
        # Resize and normalize
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()
        return image

    def inference(self, image):
        with torch.no_grad():
            output = self.model_trt(image)
            return torch.argmax(output[0]).item()


def main():
    # Initialize inference
    model = Torch2TRTInference()

    # Load image
    image = cv2.imread("goldfish.jpg")
    if image is None:
        raise FileNotFoundError("Input image not found")

    # Preprocess
    input_tensor = model.preprocess(image)

    times = []
    print("Running torch2trt inference...")

    # Run inference 10 times
    for i in range(10):
        start = time.perf_counter()
        class_id = model.inference(input_tensor)
        inference_time = (time.perf_counter() - start) * 1000

        times.append(inference_time)
        print(f"Iteration {i+1}: {inference_time:.2f}ms")

    # Calculate average of last 5 runs
    avg_time = sum(times[-5:]) / 5
    print(f"\nAverage inference time (last 5 runs): {avg_time:.2f}ms")
    print(f"Predicted class ID: {class_id}")


if __name__ == "__main__":
    main()
