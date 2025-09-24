"""
Advanced pose detection with multiple detection methods for better tracking.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class SmartPoseTracker:
    """Advanced pose tracker with multiple detection methods."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tracking_config = config.get('tracking', {})
        self.pose_confidence = self.tracking_config.get('pose_confidence', 0.5)
        
        # Initialize HOG descriptor for person detection
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Initialize cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Try to initialize hand cascade
        try:
            self.hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')
        except:
            self.hand_cascade = None
        
        # Object detection parameters
        self.min_contour_area = 500
        self.motion_threshold = 30
        
        # Previous frame for motion detection
        self.prev_gray = None
        
    def detect_pose(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int]], Optional[np.ndarray]]:
        """
        Advanced pose detection using multiple methods.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (keypoints, segmentation_mask)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints = []
        
        # Create segmentation mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # 1. DETECT PEOPLE USING HOG (highest priority)
        people_detected = self._detect_people_hog(gray, keypoints, mask)
        
        # 2. DETECT FACES AND FACIAL FEATURES (high priority)
        faces = self._detect_faces_and_features(gray, keypoints, mask)
        
        # 3. DETECT HANDS AND ARMS (medium priority)
        self._detect_hands_and_arms(gray, keypoints, mask)
        
        # 4. DETECT MOTION AREAS (low priority, only if few people detected)
        if not people_detected and len(keypoints) < 8:
            self._detect_motion_areas(gray, keypoints, mask)
        
        # 5. DETECT OBJECTS USING CONTOURS (lowest priority, only if very few keypoints)
        if len(keypoints) < 5:
            self._detect_objects_by_contours(frame, keypoints, mask)
        
        # Update previous frame for motion detection
        self.prev_gray = gray.copy()
        
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
    
    def _detect_faces_and_features(self, gray: np.ndarray, keypoints: List[Tuple[int, int]], mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces and extract detailed facial features."""
        faces = []
        try:
            detected_faces = self.face_cascade.detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in detected_faces:
                faces.append((x, y, w, h))
                
                # Add key face keypoints (reduced from 7 to 4)
                keypoints.extend([
                    (x + w//2, y + h//3),        # Forehead
                    (x + w//3, y + 2*h//3),      # Left eye area
                    (x + 2*w//3, y + 2*h//3),    # Right eye area
                    (x + w//2, y + 5*h//6),      # Nose/mouth area
                ])
                
                # Fill face area in mask
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                
                # Detect eyes within face (reduced keypoints)
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                for (ex, ey, ew, eh) in eyes:
                    # Convert to full frame coordinates
                    eye_x, eye_y = x + ex, y + ey
                    keypoints.append((eye_x + ew//2, eye_y + eh//2))  # Just eye center
                    cv2.rectangle(mask, (eye_x, eye_y), (eye_x + ew, eye_y + eh), 255, -1)
        except:
            pass
        
        return faces
    
    def _detect_hands_and_arms(self, gray: np.ndarray, keypoints: List[Tuple[int, int]], mask: np.ndarray):
        """Detect hands and arms using multiple methods."""
        # Method 1: Hand cascade (if available)
        if self.hand_cascade is not None:
            try:
                hands = self.hand_cascade.detectMultiScale(gray, 1.1, 3)
                for (x, y, w, h) in hands:
                    # Add key hand keypoints (reduced from 7 to 3)
                    keypoints.extend([
                        (x + w//2, y + h//2),        # Palm center
                        (x + w//3, y + h//2),        # Left side
                        (x + 2*w//3, y + h//2),      # Right side
                    ])
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            except:
                pass
        
        # Method 2: Skin color detection for hands
        self._detect_skin_regions(gray, keypoints, mask)
    
    def _detect_skin_regions(self, gray: np.ndarray, keypoints: List[Tuple[int, int]], mask: np.ndarray):
        """Detect skin-colored regions that might be hands."""
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        
        # Define skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small regions
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Add keypoint for potential hand region (reduced to 1)
                keypoints.append((x + w//2, y + h//2))  # Just center
                
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    
    def _detect_objects_by_contours(self, frame: np.ndarray, keypoints: List[Tuple[int, int]], mask: np.ndarray):
        """Detect objects using contour analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Add keypoint for detected object (reduced to 1)
                keypoints.append((x + w//2, y + h//2))  # Just center
                
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    
    def _detect_people_hog(self, gray: np.ndarray, keypoints: List[Tuple[int, int]], mask: np.ndarray) -> bool:
        """Detect people using HOG descriptor. Returns True if people detected."""
        people_detected = False
        try:
            (rects, weights) = self.hog.detectMultiScale(gray, winStride=(16, 16), padding=(8, 8), scale=1.05)
            
            for i, (x, y, w, h) in enumerate(rects):
                if i < len(weights) and weights[i] > self.pose_confidence:
                    people_detected = True
                    # Add key person keypoints (reduced from 9 to 5)
                    keypoints.extend([
                        (x + w//2, y),           # Top center (head)
                        (x + w//2, y + h//3),    # Upper body
                        (x + w//2, y + 2*h//3),  # Lower body
                        (x + w//4, y + h//2),    # Left arm area
                        (x + 3*w//4, y + h//2),  # Right arm area
                    ])
                    
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        except:
            pass
        
        return people_detected
    
    def _detect_motion_areas(self, gray: np.ndarray, keypoints: List[Tuple[int, int]], mask: np.ndarray):
        """Detect areas with motion."""
        if self.prev_gray is not None:
            try:
                # Calculate frame difference
                frame_diff = cv2.absdiff(self.prev_gray, gray)
                
                # Apply threshold
                _, thresh = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
                
                # Find contours of motion areas
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Filter small motion areas
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Add keypoint for motion area (reduced to 1)
                        keypoints.append((x + w//2, y + h//2))  # Just center
                        
                        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            except:
                pass
    
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
