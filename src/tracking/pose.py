"""
MediaPipe pose detection and keypoint extraction.
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class PoseTracker:
    """Handles pose detection and keypoint extraction using MediaPipe."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tracking_config = config.get('tracking', {})
        self.pose_confidence = self.tracking_config.get('pose_confidence', 0.5)
        
        # Initialize MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=self.pose_confidence,
            min_tracking_confidence=self.pose_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Keypoint indices for important body parts
        self.important_keypoints = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_EYE,
            self.mp_pose.PoseLandmark.RIGHT_EYE,
            self.mp_pose.PoseLandmark.LEFT_EAR,
            self.mp_pose.PoseLandmark.RIGHT_EAR,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]
        
    def detect_pose(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int]], Optional[np.ndarray]]:
        """
        Detect pose and return keypoints and segmentation mask.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (keypoints, segmentation_mask)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        keypoints = []
        segmentation_mask = None
        
        if results.pose_landmarks:
            # Extract keypoints
            h, w = frame.shape[:2]
            
            for landmark in self.important_keypoints:
                if landmark in results.pose_landmarks.landmark:
                    lm = results.pose_landmarks.landmark[landmark]
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    keypoints.append((x, y))
            
            # Extract segmentation mask
            if results.segmentation_mask is not None:
                segmentation_mask = (results.segmentation_mask * 255).astype(np.uint8)
        
        return keypoints, segmentation_mask
    
    def get_pose_bounding_box(self, keypoints: List[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box around pose keypoints.
        
        Args:
            keypoints: List of (x, y) keypoint coordinates
            
        Returns:
            Tuple of (x, y, width, height) or None if no keypoints
        """
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
        x_max = min(1920, x_max + padding)  # Assuming HD width
        y_max = min(1080, y_max + padding)  # Assuming HD height
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def draw_pose_debug(self, frame: np.ndarray, keypoints: List[Tuple[int, int]]) -> np.ndarray:
        """
        Draw pose keypoints for debugging (optional).
        
        Args:
            frame: Input frame
            keypoints: List of keypoint coordinates
            
        Returns:
            Frame with keypoints drawn
        """
        debug_frame = frame.copy()
        
        for i, (x, y) in enumerate(keypoints):
            cv2.circle(debug_frame, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(debug_frame, str(i), (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        return debug_frame
    
    def cleanup(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
