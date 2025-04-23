import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
import cv2
import numpy as np
import os

def process_image():
    # Create debug_samples directory if it doesn't exist
    os.makedirs('./debug_samples', exist_ok=True)
    
    # Try to read the image
    img_path = '../COCO/train2014/COCO_train2014_000000001330.jpg'
    img = cv2.imread(img_path)
    
    # If image not found, create a test image
    if img is None:
        print(f"Image not found at {img_path}. Creating a test image...")
        # Create a simple test image with a person-like shape
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simple stick figure
        cv2.circle(img, (320, 100), 30, (255, 255, 255), -1)  # Head
        cv2.line(img, (320, 130), (320, 250), (255, 255, 255), 2)  # Body
        cv2.line(img, (320, 150), (250, 200), (255, 255, 255), 2)  # Left arm
        cv2.line(img, (320, 150), (390, 200), (255, 255, 255), 2)  # Right arm
        cv2.line(img, (320, 250), (280, 350), (255, 255, 255), 2)  # Left leg
        cv2.line(img, (320, 250), (360, 350), (255, 255, 255), 2)  # Right leg
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Initialize pose detection with image dimensions
    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        # Process image with dimensions
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Print landmarks if detected
        if results.pose_landmarks:
            print("Pose landmarks detected")
            print(f"Number of landmarks: {len(results.pose_landmarks.landmark)}")
            print(f"Image dimensions: {w}x{h}")
        else:
            print("No pose landmarks detected")
        
        # Draw landmarks
        annotated_image = img.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
        
        # Convert back to BGR for saving
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
        # Save the annotated image
        output_path = './debug_samples/sample_0.jpg'
        cv2.imwrite(output_path, annotated_image)
        print(f"Saved annotated image to {output_path}")
        
        # Also save the original image for reference
        original_path = './debug_samples/original.jpg'
        cv2.imwrite(original_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved original image to {original_path}")

if __name__ == "__main__":
    process_image()