"""
Fallback pose detection using OpenCV's built-in features when MediaPipe is not available.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class FallbackPoseTracker:
    """Fallback pose tracker using OpenCV's HOG descriptor and face detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tracking_config = config.get('tracking', {})
        self.pose_confidence = self.tracking_config.get('pose_confidence', 0.5)
        
        # Initialize OpenCV's HOG descriptor for person detection
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize hand detector (using a simple approach)
        try:
            self.hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')
        except:
            # Hand cascade not available, we'll skip hand detection
            self.hand_cascade = None
        
    def detect_pose(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int]], Optional[np.ndarray]]:
        """
        Detect pose using OpenCV's built-in features.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (keypoints, segmentation_mask)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints = []
        
        # Detect people using HOG (optimized for performance)
        try:
            (rects, weights) = self.hog.detectMultiScale(gray, winStride=(24, 24), padding=(8, 8), scale=1.2)
        except:
            rects, weights = [], []
        
        # Create a simple segmentation mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        for (x, y, w, h) in rects:
            if weights[0] > self.pose_confidence:
                # Add keypoints around the person bounding box
                keypoints.extend([
                    (x + w//2, y),           # Top center
                    (x, y + h//2),           # Left center
                    (x + w, y + h//2),       # Right center
                    (x + w//2, y + h),       # Bottom center
                    (x + w//2, y + h//2),    # Center
                ])
                
                # Fill mask
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        # Detect faces (optimized for performance)
        try:
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 2)
        except:
            faces = []
        for (x, y, w, h) in faces:
            # Add face keypoints
            keypoints.extend([
                (x + w//2, y + h//3),        # Forehead
                (x + w//3, y + 2*h//3),      # Left eye
                (x + 2*w//3, y + 2*h//3),    # Right eye
                (x + w//2, y + 5*h//6),      # Nose
                (x + w//2, y + h),           # Chin
            ])
            
            # Fill face area in mask
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        # Detect hands (if cascade is available)
        if self.hand_cascade is not None:
            try:
                hands = self.hand_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in hands:
                    # Add hand keypoints
                    keypoints.extend([
                        (x + w//2, y + h//2),        # Palm center
                        (x, y + h//2),               # Left edge
                        (x + w, y + h//2),           # Right edge
                        (x + w//2, y),               # Top
                        (x + w//2, y + h),           # Bottom
                    ])
                    
                    # Fill hand area in mask
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            except:
                # Skip hand detection if it fails
                pass
        
        # If no detections, create a center region
        if not keypoints:
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            keypoints = [
                (center_x, center_y),
                (center_x - 50, center_y - 50),
                (center_x + 50, center_y - 50),
                (center_x - 50, center_y + 50),
                (center_x + 50, center_y + 50),
            ]
            cv2.circle(mask, (center_x, center_y), 100, 255, -1)
        
        return keypoints, mask
    
    def get_pose_bounding_box(self, keypoints: List[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box around pose keypoints."""
        if not keypoints:
            return None
            
        x_coords = [kp[0] for kp in keypoints]
        y_coords = [kp[1] for kp in keypoints]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(1920, x_max + padding)
        y_max = min(1080, y_max + padding)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def draw_pose_debug(self, frame: np.ndarray, keypoints: List[Tuple[int, int]]) -> np.ndarray:
        """Draw pose keypoints for debugging."""
        debug_frame = frame.copy()
        
        for i, (x, y) in enumerate(keypoints):
            cv2.circle(debug_frame, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(debug_frame, str(i), (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        return debug_frame
    
    def cleanup(self):
        """Clean up resources."""
        pass
