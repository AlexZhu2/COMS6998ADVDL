import random
import numpy as np
import torchvision.transforms.functional as TF

class RandomHorizontalFlipWithKeypoints:
    def __init__(self, p=0.5, image_size=(224, 224), keypoint_pairs=None):
        self.p = p
        self.image_size = image_size  # (H, W)
        self.keypoint_pairs = keypoint_pairs  # Optional: swap left/right keypoints

    def __call__(self, image, keypoints):
        if random.random() < self.p:
            # Flip image horizontally
            image = TF.hflip(image)

            # Flip keypoints horizontally (x = W - x)
            keypoints = keypoints.copy()
            keypoints[0] = 1.0 - keypoints[0]  # Normalized x: [0, 1]

            # Swap keypoints if needed (e.g., left ↔ right joints)
            if self.keypoint_pairs is not None and len(self.keypoint_pairs) > 0:
                keypoints[:, self.keypoint_pairs[:, 0]], keypoints[:, self.keypoint_pairs[:, 1]] = \
                    keypoints[:, self.keypoint_pairs[:, 1]].copy(), keypoints[:, self.keypoint_pairs[:, 0]].copy()


        return image, keypoints
