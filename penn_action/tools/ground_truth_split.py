import os
import scipy.io
import json
import cv2
import numpy as np

def split_ground_truth(data_dir, gt_dir):
    # Get all the folders in the data directory
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    # Read all mat files in the data directory
    mat_files = [f for f in os.listdir(gt_dir) if f.endswith('.mat') and f.split('.')[0] in folders]
    
    # List to store all sequences
    all_sequences = []
    
    # Read all data first
    for mat_file in mat_files:
        print(mat_file)
        mat_data = scipy.io.loadmat(os.path.join(gt_dir, mat_file))
        sequence_id = mat_file.split('.')[0]
        
        # Get number of frames in sequence
        num_frames = len(mat_data['x'])
        frames = []
        
        # Create frame-level annotations
        for frame_idx in range(num_frames):
            try:
                bbox = mat_data['bbox'][frame_idx].tolist()
            except:
                bbox = [0, 0, 0, 0]
            frame_data = {
                'frame_id': frame_idx + 1,  # 1-based indexing for frames
                'keypoints': [mat_data['x'][frame_idx].tolist(), 
                            mat_data['y'][frame_idx].tolist()],
                'bbox': bbox
            }
            frames.append(frame_data)
        
        sequence_data = {
            'sequence_id': sequence_id,
            'num_frames': num_frames,
            'frames': frames
        }
        all_sequences.append(sequence_data)
    
    # Create output directory
    output_dir = 'penn_action/labels_split/test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Write all sequences to a single JSON file
    output_file = os.path.join(output_dir, 'all_test_annotations.json')
    with open(output_file, 'w') as f:
        json.dump(all_sequences, f, indent=4)

if __name__ == '__main__':
    data_dir = 'penn_action/frames_split/test'
    gt_dir = 'penn_action/labels'
    split_ground_truth(data_dir, gt_dir)
