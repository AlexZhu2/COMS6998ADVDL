import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class DualInputR2Plus1DPositional(nn.Module):
    def __init__(self, num_classes=22, num_joints=33, keypoint_embed_dim=256, max_seq_len=48):
        super().__init__()
        # Video branch
        base_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.video_branch = nn.Sequential(*list(base_model.children())[:-2])
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Keypoint positional embedding branch
        self.num_joints = num_joints
        self.keypoint_embed_dim = keypoint_embed_dim
        self.max_seq_len = max_seq_len

        # Project (x, y, z, visibility) → embedding dim
        self.kp_linear = nn.Linear(4, keypoint_embed_dim)

        # Velocity embedding
        self.kp_vel_linear = nn.Linear(4, keypoint_embed_dim)

        # Learnable temporal positional embedding for sequence length
        self.temporal_pos_embedding = nn.Parameter(torch.randn(max_seq_len, keypoint_embed_dim * 2))

        # Processor for fused keypoints (now using time + joints)
        self.layer_norm = nn.LayerNorm(keypoint_embed_dim * 2)
        self.kp_processor = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + keypoint_embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, video, keypoints):
        """
        video: [B, 3, T, H, W]
        keypoints: [B, T, 264] → reshaped to [B, T, 66, 4]
        """
        B, T, _ = keypoints.shape
        keypoints = keypoints.view(B, T, self.num_joints * 2, 4)  # [B, T, 66, 4]

        kp_pos = keypoints[:, :, :self.num_joints, :]  # [B, T, 33, 4]
        kp_vel = keypoints[:, :, self.num_joints:, :]  # [B, T, 33, 4]

        # Embed positions and velocities separately
        kp_pos_emb = self.kp_linear(kp_pos)  # [B, T, 33, D]
        kp_vel_emb = self.kp_vel_linear(kp_vel)  # [B, T, 33, D]

        # Combine position and velocity embeddings
        kp_fused = torch.cat([kp_pos_emb, kp_vel_emb], dim=-1)  # [B, T, 33, 2D]

        # Now, average over joints
        kp_fused = kp_fused.mean(dim=2)  # [B, T, 2D] (average across 33 joints)

        # Add temporal positional encoding
        temporal_pos_emb = self.temporal_pos_embedding[:T, :].unsqueeze(0)  # [1, T, 2D]
        kp_fused = kp_fused + temporal_pos_emb  # [B, T, 2D]

        kp_fused = self.layer_norm(kp_fused)
        # Process temporal sequence
        kp_fused = kp_fused.transpose(1, 2)  # [B, 2D, T]
        k_feat = self.kp_processor(kp_fused)  # [B, 2D]

        # Video features
        v_feat = self.video_branch(video)
        v_feat = self.pool(v_feat).flatten(1)  # [B, 512]

        # Final prediction
        return self.classifier(torch.cat([v_feat, k_feat], dim=1))
