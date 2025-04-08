import os
import json
import cv2
import numpy as np

def visualize_first_sequence():
    # Read annotations
    with open('penn_action/labels_split/train/all_train_annotations.json', 'r') as f:
        annotations = json.load(f)
    print(len(annotations))
    # Get first sequence
    first_seq = annotations[12]
    seq_id = first_seq['file_name']
    keypoints = first_seq['keypoints']
    bbox = first_seq['bbox']
    
    # Get first 5 frames
    frames_dir = os.path.join('penn_action/frames_split/train', seq_id)
    frame_files = sorted(os.listdir(frames_dir))[:5]
    
    for i, frame_file in enumerate(frame_files):
        # Read frame
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        # Draw bounding box
        x1, y1, x2, y2 = [int(x) for x in bbox[i]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw keypoints
        for x, y in zip(keypoints[i][0], keypoints[i][1]):
            if x > 1 and y > 1:  # Only draw visible keypoints
                x, y = int(x), int(y)
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        
        # Display frame
        cv2.imshow(f'Frame {i+1}', frame)
        cv2.waitKey(0)
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    visualize_first_sequence()
