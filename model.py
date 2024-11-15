import torch
import torchvision.models as models


def create_model():
    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(pretrained=True)
    model.eval()
    return model


def preprocess_image(image):
    # Resize to 224x224 (EfficientNet-B0 default size)
    size = (224, 224)
    if image.shape[1:] != size:
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=size).squeeze(
            0
        )

    # Normalize with ImageNet stats
    normalize = torch.nn.functional.normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    image = (image - mean) / std

    return image


def postprocess_output(output):
    # Get top-1 prediction
    probabilities = torch.nn.functional.softmax(output, dim=1)
    _, predicted = torch.max(probabilities, 1)
    return predicted.item()
