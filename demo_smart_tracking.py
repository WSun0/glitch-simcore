#!/usr/bin/env python3
"""
Demo script to compare basic vs smart tracking.
"""
import cv2
import sys
import os
import numpy as np

# Add src to path
sys.path.append('src')

from tracking.pose_fallback import FallbackPoseTracker
from tracking.smart_pose import SmartPoseTracker

def main():
    # Initialize both trackers
    config = {
        'tracking': {
            'pose_confidence': 0.5,
            'min_contour_area': 500,
            'motion_threshold': 30
        }
    }
    
    basic_tracker = FallbackPoseTracker(config)
    smart_tracker = SmartPoseTracker(config)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Smart Tracking Demo")
    print("Left side: Basic tracking (faces + people)")
    print("Right side: Smart tracking (faces + hands + objects + motion)")
    print("Press 'q' to quit, 's' to save comparison")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Resize frame for side-by-side display
        height, width = frame.shape[:2]
        half_width = width // 2
        
        # Create side-by-side display
        display_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)
        display_frame[:, :width] = frame
        display_frame[:, width:] = frame
        
        # Basic tracking (left side)
        basic_keypoints, basic_mask = basic_tracker.detect_pose(frame)
        basic_debug = basic_tracker.draw_pose_debug(frame, basic_keypoints)
        display_frame[:, :width] = basic_debug
        
        # Smart tracking (right side) - only every 3rd frame for performance
        if frame_count % 3 == 0:
            smart_keypoints, smart_mask = smart_tracker.detect_pose(frame)
            smart_debug = smart_tracker.draw_pose_debug(frame, smart_keypoints)
            display_frame[:, width:] = smart_debug
        
        # Add labels
        cv2.putText(display_frame, "Basic Tracking", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Smart Tracking", (width + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add keypoint counts
        cv2.putText(display_frame, f"Keypoints: {len(basic_keypoints)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Keypoints: {len(smart_keypoints) if frame_count % 3 == 0 else '...'}", 
                   (width + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display
        cv2.imshow('Smart Tracking Comparison', display_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"tracking_comparison_{frame_count}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"Saved comparison to {filename}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
