import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

##
## Base dual-input Vision Transformer model.
## Trains in around 30 epochs to over 90% training accuracy, but validation accuracy gets stuck around 55%
##

class DualInputViTPositional(nn.Module):
    def __init__(self, num_classes=22, num_joints=33, keypoint_embed_dim=256, max_seq_len=48):
        super().__init__()
        
        # Video branch using Vision Transformer
        base_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.video_branch = base_model
        self.video_embed_dim = base_model.hidden_dim  # Usually 768 for ViT-B/16

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

        # Processor for fused keypoints
        self.layer_norm = nn.LayerNorm(keypoint_embed_dim * 2)
        self.kp_processor = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # Final classifier
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.video_embed_dim + keypoint_embed_dim * 2, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, num_classes)
        # )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), 
            nn.Linear(self.video_embed_dim + keypoint_embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(256, num_classes)
        )

    def forward_features(self, x):
        n = x.shape[0]
        x = self.video_branch.conv_proj(x)  # patch embedding
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        cls_token = self.video_branch.class_token.expand(n, -1, -1) 
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.video_branch.encoder.pos_embedding[:, :(x.size(1)), :]
        x = self.video_branch.encoder(x)

        return x[:, 0]  # CLS token output


    def forward(self, video, keypoints):
        """
        video: [B, 3, H, W] or [B, 3, T, H, W] averaged across T
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

        # Average over joints
        kp_fused = kp_fused.mean(dim=2)  # [B, T, 2D]

        # Add temporal positional encoding
        temporal_pos_emb = self.temporal_pos_embedding[:T, :].unsqueeze(0)  # [1, T, 2D]
        kp_fused = kp_fused + temporal_pos_emb  # [B, T, 2D]

        kp_fused = self.layer_norm(kp_fused)
        kp_fused = kp_fused.transpose(1, 2)  # [B, 2D, T]
        k_feat = self.kp_processor(kp_fused)  # [B, 2D]

        # Video features (ViT expects [B, 3, H, W], single frame or averaged frames)
        if video.dim() == 5:
            video = video.mean(dim=2)  # Average over time

        v_feat = self.forward_features(video)  # [B, 768]

        # Final prediction
        return self.classifier(torch.cat([v_feat, k_feat], dim=1))