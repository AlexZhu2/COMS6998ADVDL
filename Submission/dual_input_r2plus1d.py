import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class DualInputR2Plus1D(nn.Module):
    def __init__(self, num_classes=22, keypoint_dim=132, keypoint_embed_dim=256):
        super().__init__()
        base_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.video_branch = nn.Sequential(*list(base_model.children())[:-2])
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.kp_branch = nn.Sequential(
            nn.Conv1d(keypoint_dim, keypoint_embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + keypoint_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, video, keypoints):
        v_feat = self.video_branch(video)
        v_feat = self.pool(v_feat).flatten(1)  # [B, 512]
        k_feat = self.kp_branch(keypoints.transpose(1, 2))  # [B, 256]
        return self.classifier(torch.cat([v_feat, k_feat], dim=1))
