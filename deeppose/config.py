import torch

class Config:
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset configuration
    ROOT_DIR = './COCO'
    TRAIN_ANNOTATION = 'annotations/person_keypoints_train2014.json'
    VAL_ANNOTATION = 'annotations/person_keypoints_val2014.json'
    IMAGE_SIZE = (256, 256)
    SELECTED_KEYPOINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # Indices of 13 keypoints
    MIN_KEYPOINTS = 6
    
    # Training configuration
    NUM_EPOCHS = 50
    BATCH_SIZE = 1
    LEARNING_RATE = 3e-4
    OPTIMIZER = 'AdamW'
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5
    AREA_RATIO_THRESHOLD = 0.5
    
    # Model configuration
    NUM_KEYPOINTS = len(SELECTED_KEYPOINTS)
    BACKBONE = 'resnet18'  # Options: 'resnet18', 'resnet50', 'mobilenet', 'efficientnet_b0'
    
    # Paths
    CHECKPOINT_PATH = 'deeppose/checkpoints/best_model.pth'