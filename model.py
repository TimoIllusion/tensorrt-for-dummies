import torch
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image


def create_model():
    # Load pretrained EfficientNet-B0

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.eval()
    return model


def preprocess_numpy_bgr_image(image: np.ndarray) -> torch.Tensor:
    # Convert BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert numpy array to PIL Image for proper handling by torchvision transforms
    image = Image.fromarray(image)

    preprocess = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),  # Resize to 224x224 (EfficientNet-B0 default size)
            transforms.ToTensor(),  # Convert to Tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize with ImageNet stats
        ]
    )

    # Apply transformations
    image = preprocess(image)

    # Add batch dimension (1, C, H, W)
    image = image.unsqueeze(0)

    return image


def postprocess_output(output):
    # Apply softmax and get top-1 prediction
    probabilities = torch.nn.functional.softmax(output, dim=1)
    _, predicted = torch.max(probabilities, 1)
    return predicted.item()
