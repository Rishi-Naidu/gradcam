import torch
import torch.nn as nn
import torchvision.models as models

import torchvision.models as models

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

def load_backbone(name: str = "resnet50", pretrained: bool = True, weights_path: str = None):
    name = name.lower()
    if name == "resnet50":
        if pretrained and weights_path is None:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            model = models.resnet50(weights=None)
            if weights_path:
                state = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state)
        target_layer = "layer4"  # Grad-CAM target (last conv block)
        return model, target_layer
    elif name == "resnet18":
        if pretrained and weights_path is None:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model = models.resnet18(weights=None)
            if weights_path:
                state = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state)
        target_layer = "layer4"
        return model, target_layer
    else:
        raise ValueError(f"Unsupported backbone: {name}")