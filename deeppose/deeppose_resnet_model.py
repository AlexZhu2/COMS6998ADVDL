import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class DeepPoseResNetWithCascade(nn.Module):
    def __init__(self, num_keypoints=13):
        super().__init__()
        self.num_keypoints = num_keypoints
        # Stage 1 will predict 4 bbox coordinates and 2 * num_keypoints (coarse keypoints)
        self.num_outputs = 4 + num_keypoints * 2

        # --- Stage 1: Coarse Pose Estimation ---
        resnet1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final layers to extract features (output size: [B, 2048, 7, 7])
        self.backbone1 = nn.Sequential(*list(resnet1.children())[:-2])
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten1 = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        # Output the bbox and coarse keypoints
        self.out_stage1 = nn.Linear(256, self.num_outputs)

        # --- Stage 2: Pose Refinement ---
        # Stage 2 uses a separate backbone plus the coarse keypoints to predict a residual.
        resnet2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone2 = nn.Sequential(*list(resnet2.children())[:-2])
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten2 = nn.Flatten()

        # The cascade module receives image features and coarse keypoints
        # and predicts a residual (offset) for each keypoint coordinate.
        self.fc2 = nn.Sequential(
            nn.Linear(2048 + num_keypoints * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_keypoints * 2)
        )

    def forward(self, x):
        # === Stage 1: Coarse Estimation ===
        feat1 = self.backbone1(x)
        pooled1 = self.pool1(feat1)
        flat1 = self.flatten1(pooled1)
        stage1_features = self.fc1(flat1)
        out_stage1 = self.out_stage1(stage1_features)
        
        # The first 4 outputs are the bounding box coordinates.
        bbox = out_stage1[:, :4]
        # The rest (2 * num_keypoints) are the coarse keypoint predictions.
        keypoints_stage1 = out_stage1[:, 4:]  # shape: [B, num_keypoints*2]

        # === Stage 2: Refinement via Residual Correction ===
        feat2 = self.backbone2(x)
        pooled2 = self.pool2(feat2)
        flat2 = self.flatten2(pooled2)
        
        # Concatenate flat2 image features with the coarse keypoint predictions.
        cascade_input = torch.cat([flat2, keypoints_stage1], dim=1)
        
        # Predict residuals, i.e. corrections to add to coarse predictions.
        residual = self.fc2(cascade_input)
        
        keypoints_stage2 = keypoints_stage1 + residual

        # Reshape keypoint outputs to [B, num_keypoints, 2].
        keypoints_stage1 = keypoints_stage1.view(-1, self.num_keypoints, 2)
        keypoints_stage2 = keypoints_stage2.view(-1, self.num_keypoints, 2)

        return bbox, keypoints_stage1, keypoints_stage2

# Example usage:
if __name__ == '__main__':
    model = DeepPoseResNetWithCascade(num_keypoints=13)
    dummy_input = torch.rand(1, 3, 224, 224)  # simulate a batch of size 1
    bbox, keypoints_stage1, keypoints_stage2 = model(dummy_input)
    print("BBox Output:", bbox.shape)
    print("Stage 1 Keypoints:", keypoints_stage1.shape)
    print("Stage 2 Keypoints:", keypoints_stage2.shape)
