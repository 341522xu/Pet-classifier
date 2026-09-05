import torch.nn as nn
import torchvision


def get_model(num_classes=37, pretrained=True):
    """加载ResNet-18，替换最后一层全连接为num_classes类"""
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
