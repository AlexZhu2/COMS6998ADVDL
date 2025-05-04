import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

##
## Improvement over base model using temporal transformer and other improvements.
## Doesn't seem to perform that well and takes very long to train per epoch.
##

class TemporalTransformer(nn.Module):
    """Tiny Transformer encoder block for temporal modeling."""
    def __init__(self, embed_dim, depth=2, num_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.transformer(x)

class DualInputViTPositionalImproved(nn.Module):
    def __init__(self, num_classes=22, num_joints=33, keypoint_embed_dim=256, max_seq_len=48):
        super().__init__()

        # --- Video Backbone ---
        base_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.video_branch = base_model
        self.video_embed_dim = base_model.hidden_dim  # 768

        # Temporal transformer for video CLS sequence
        self.video_temporal = TemporalTransformer(embed_dim=self.video_embed_dim)

        # --- Keypoint Branch ---
        self.num_joints = num_joints
        self.keypoint_embed_dim = keypoint_embed_dim
        self.max_seq_len = max_seq_len

        self.kp_linear = nn.Linear(4, keypoint_embed_dim)
        self.kp_vel_linear = nn.Linear(4, keypoint_embed_dim)

        self.temporal_pos_embedding = nn.Parameter(torch.randn(max_seq_len, keypoint_embed_dim * 2))
        self.layer_norm = nn.LayerNorm(keypoint_embed_dim * 2)

        # Temporal transformer for keypoints
        self.kp_temporal = TemporalTransformer(embed_dim=keypoint_embed_dim * 2)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.video_embed_dim + keypoint_embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def extract_frame_features(self, frames):
        """Extracts CLS tokens for each frame separately."""
        B, C, T, H, W = frames.shape
        frames = frames.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        frames = frames.flatten(0, 1)  # [B*T, C, H, W]

        x = self.video_branch.conv_proj(frames)  # [B*T, hidden_dim, H/16, W/16]
        x = x.flatten(2).transpose(1, 2)  # [B*T, num_patches, hidden_dim]

        cls_token = self.video_branch.class_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.video_branch.encoder.pos_embedding[:, :x.size(1), :]
        x = self.video_branch.encoder(x)

        cls_feats = x[:, 0]  # Take CLS token only
        cls_feats = cls_feats.view(B, T, -1)  # [B, T, hidden_dim]
        return cls_feats

    def forward(self, video, keypoints):
        B, T, _ = keypoints.shape

        # --- Process Keypoints ---
        keypoints = keypoints.view(B, T, self.num_joints * 2, 4)
        kp_pos = keypoints[:, :, :self.num_joints, :]
        kp_vel = keypoints[:, :, self.num_joints:, :]

        kp_pos_emb = self.kp_linear(kp_pos)
        kp_vel_emb = self.kp_vel_linear(kp_vel)
        kp_fused = torch.cat([kp_pos_emb, kp_vel_emb], dim=-1)  # [B, T, 33, 2D]
        kp_fused = kp_fused.mean(dim=2)  # Average over joints [B, T, 2D]

        # Add temporal positional encoding
        temporal_pos_emb = self.temporal_pos_embedding[:T, :].unsqueeze(0)
        kp_fused = kp_fused + temporal_pos_emb
        kp_fused = self.layer_norm(kp_fused)

        # Process keypoint sequence with Transformer
        k_feat_seq = self.kp_temporal(kp_fused)  # [B, T, 2D]
        k_feat = k_feat_seq[:, 0]  # Take first token (or mean pool optionally)

        # --- Process Video ---
        if video.dim() == 4:
            # Only one frame case
            video = video.unsqueeze(2)  # [B, 3, 1, H, W]

        cls_seq = self.extract_frame_features(video)  # [B, T, hidden_dim]
        v_feat_seq = self.video_temporal(cls_seq)  # [B, T, hidden_dim]
        v_feat = v_feat_seq[:, 0]  # First token

        # --- Final Prediction ---
        final_feat = torch.cat([v_feat, k_feat], dim=1)
        output = self.classifier(final_feat)

        return output
