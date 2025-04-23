import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import Config

class DirectPose(nn.Module):
    def __init__(self, backbone_name=Config.BACKBONE, num_keypoints=Config.NUM_KEYPOINTS):
        super(DirectPose, self).__init__()

        self.backbone, in_channels = self._build_backbone(backbone_name)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Coordinate regression head (predicts x and y for each keypoint)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_keypoints * 2),
            nn.Sigmoid()  # Output in [0, 1] for normalized coordinates
        )

    def _build_backbone(self, backbone_name):
        """Factory method for backbone selection"""
        backbones = {
            'resnet18': (models.resnet18, 512),
            'resnet50': (models.resnet50, 2048),
            'mobilenet_v2': (models.mobilenet_v2, 1280),
            'efficientnet_b0': (models.efficientnet_b0, 1280)
        }

        if backbone_name not in backbones:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose from: {list(backbones.keys())}")

        model_class, in_channels = backbones[backbone_name]

        if model_class == models.resnet18:
            model = model_class(weights=models.ResNet18_Weights.DEFAULT)
        elif model_class == models.resnet50:
            model = model_class(weights=models.ResNet50_Weights.DEFAULT)
        elif model_class == models.mobilenet_v2:
            model = model_class(weights=models.MobileNet_V2_Weights.DEFAULT)
        elif model_class == models.efficientnet_b0:
            model = model_class(weights=models.EfficientNet_B0_Weights.DEFAULT)

        if hasattr(model, 'fc'):
            return nn.Sequential(*list(model.children())[:-2]), in_channels
        elif hasattr(model, 'classifier'):
            return nn.Sequential(*list(model.children())[:-1]), in_channels
        else:
            raise ValueError(f"Unknown model architecture: {backbone_name}")

    def forward(self, x):
        features = self.backbone(x)          # [B, C, H, W]
        pooled = self.avgpool(features)      # [B, C, 1, 1]
        coords = self.fc(pooled)             # [B, num_keypoints * 2]
        coords = coords.view(-1, Config.NUM_KEYPOINTS, 2)  # [B, K, 2] normalized
        return coords

def build_model(backbone_name=None, num_keypoints=None):
    if backbone_name is None:
        backbone_name = Config.BACKBONE
    if num_keypoints is None:
        num_keypoints = Config.NUM_KEYPOINTS

    return DirectPose(backbone_name=backbone_name, num_keypoints=num_keypoints)
