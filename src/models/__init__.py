import torch
import torchvision.models as models
from .backbones import build_backbone

def build_backbone(name="resnet50", pretrained=True):
    """
    Builds and returns a CNN backbone model for Grad-CAM visualization.
    Supported: resnet18, resnet50, vgg16
    """
    name = name.lower()

    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    elif name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    return model
