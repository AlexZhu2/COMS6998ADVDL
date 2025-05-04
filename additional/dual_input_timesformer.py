import torch
import torch.nn as nn
from transformers import TimesformerModel, TimesformerConfig

##
## In-progress model that uses timesformer base with dual-inputs.
## Still currently stuck on "CUDA out of memory" errors due to massive number of parameters of
## large ViT combined with dual-inputs.
##

class DualInputTimeSformer(nn.Module):
    def __init__(self, num_classes=22, num_joints=33, keypoint_embed_dim=256, max_seq_len=16):
        super().__init__()

        # Load pretrained TimeSformer from Huggingface (Kinetics-400 fine-tuned)
        self.video_branch = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.video_embed_dim = self.video_branch.config.hidden_size  # Typically 768

        # Freeze the video branch to save memory
        for param in self.video_branch.parameters():
            param.requires_grad = False

        # Keypoint embedding branch
        self.num_joints = num_joints
        self.keypoint_embed_dim = keypoint_embed_dim
        self.max_seq_len = max_seq_len

        self.kp_linear = nn.Linear(4, keypoint_embed_dim)
        self.kp_vel_linear = nn.Linear(4, keypoint_embed_dim)

        self.temporal_pos_embedding = nn.Parameter(torch.randn(max_seq_len, keypoint_embed_dim * 2))
        self.layer_norm = nn.LayerNorm(keypoint_embed_dim * 2)

        self.kp_processor = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.video_embed_dim + keypoint_embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )


    def forward(self, video, keypoints):
        """
        video: [B, T, 3, H, W] or [B, 3, T, H, W] → auto-permuted to [B, T, 3, H, W]
        keypoints: [B, T, 264] (T frames, 33 joints * 2 (pos + vel) * 4 values)
        """
        B, T, _ = keypoints.shape

        # Fix video shape if necessary
        if video.shape[1] == 3 and video.ndim == 5:
            video = video.permute(0, 2, 1, 3, 4)  # [B, T, 3, H, W]

        # Reshape keypoints to [B, T, 66, 4]
        keypoints = keypoints.view(B, T, self.num_joints * 2, 4)
        kp_pos = keypoints[:, :, :self.num_joints, :]
        kp_vel = keypoints[:, :, self.num_joints:, :]

        # Project joint position and velocity into embeddings
        kp_pos_emb = self.kp_linear(kp_pos)      # [B, T, 33, D]
        kp_vel_emb = self.kp_vel_linear(kp_vel)  # [B, T, 33, D]

        # Fuse and average over joints
        kp_fused = torch.cat([kp_pos_emb, kp_vel_emb], dim=-1)  # [B, T, 33, 2D]
        kp_fused = kp_fused.mean(dim=2)  # [B, T, 2D]

        # Handle cases where T != max_seq_len for temporal positional encoding
        if T > self.max_seq_len:
            temporal_pos_emb = nn.functional.interpolate(
                self.temporal_pos_embedding.unsqueeze(0).permute(0, 2, 1),  # [1, 2D, max_seq_len]
                size=T,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)  # [1, T, 2D]
        else:
            temporal_pos_emb = self.temporal_pos_embedding[:T, :].unsqueeze(0)  # [1, T, 2D]

        # Add temporal positional encoding and normalize
        kp_fused = kp_fused + temporal_pos_emb
        kp_fused = self.layer_norm(kp_fused)

        # Process keypoints across time
        kp_fused = kp_fused.transpose(1, 2)  # [B, 2D, T]
        k_feat = self.kp_processor(kp_fused)  # [B, 2D]

        # Pass video through TimeSformer
        video_outputs = self.video_branch(video)
        #v_feat = video_outputs.pooler_output  # [B, hidden_dim]
        v_feat = video_outputs.last_hidden_state[:, 0]  # CLS token

        # Concatenate features and classify
        final_feat = torch.cat([v_feat, k_feat], dim=1)
        output = self.classifier(final_feat)

        return output