"""
Segmentation mask processing and enhancement.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


class MaskProcessor:
    """Handles segmentation mask processing and enhancement."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tracking_config = config.get('tracking', {})
        self.mask_expansion = self.tracking_config.get('mask_expansion', 1.2)
        
    def process_mask(self, mask: np.ndarray, frame_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Process and enhance the segmentation mask.
        
        Args:
            mask: Input segmentation mask
            frame_shape: Shape of the original frame (height, width, channels)
            
        Returns:
            Processed mask
        """
        if mask is None:
            return self._create_fallback_mask(frame_shape)
        
        # Ensure mask is the right size
        h, w = frame_shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h))
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur for smoother edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Expand the mask slightly
        if self.mask_expansion > 1.0:
            mask = self._expand_mask(mask, self.mask_expansion)
        
        return mask
    
    def _expand_mask(self, mask: np.ndarray, expansion_factor: float) -> np.ndarray:
        """
        Expand the mask by the given factor.
        
        Args:
            mask: Input mask
            expansion_factor: Factor to expand by
            
        Returns:
            Expanded mask
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the center of mass
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        # Scale the contour from the center
        scaled_contour = []
        for point in largest_contour:
            x, y = point[0]
            # Scale from center
            new_x = int(cx + (x - cx) * expansion_factor)
            new_y = int(cy + (y - cy) * expansion_factor)
            scaled_contour.append([[new_x, new_y]])
        
        # Create new mask
        new_mask = np.zeros_like(mask)
        cv2.fillPoly(new_mask, [np.array(scaled_contour)], 255)
        
        return new_mask
    
    def _create_fallback_mask(self, frame_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Create a fallback mask when no segmentation is available.
        
        Args:
            frame_shape: Shape of the frame
            
        Returns:
            Fallback mask (center region)
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create a center region mask
        center_x, center_y = w // 2, h // 2
        mask_size = min(w, h) // 3
        
        cv2.rectangle(mask, 
                     (center_x - mask_size, center_y - mask_size),
                     (center_x + mask_size, center_y + mask_size),
                     255, -1)
        
        return mask
    
    def get_mask_contours(self, mask: np.ndarray) -> list:
        """
        Get contours from the mask.
        
        Args:
            mask: Input mask
            
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def get_mask_center(self, mask: np.ndarray) -> Tuple[int, int]:
        """
        Get the center point of the mask.
        
        Args:
            mask: Input mask
            
        Returns:
            Center coordinates (x, y)
        """
        contours = self.get_mask_contours(mask)
        
        if not contours:
            return (mask.shape[1] // 2, mask.shape[0] // 2)
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate center of mass
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        
        return (mask.shape[1] // 2, mask.shape[0] // 2)
    
    def draw_mask_debug(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Draw mask overlay for debugging.
        
        Args:
            frame: Input frame
            mask: Segmentation mask
            
        Returns:
            Frame with mask overlay
        """
        debug_frame = frame.copy()
        
        # Create colored overlay
        overlay = np.zeros_like(debug_frame)
        overlay[mask > 0] = [0, 255, 0]  # Green overlay
        
        # Blend with original frame
        debug_frame = cv2.addWeighted(debug_frame, 0.7, overlay, 0.3, 0)
        
        return debug_frame
