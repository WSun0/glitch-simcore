"""
Numerical label overlay system.
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any
from .style import StyleManager


class Label:
    """Represents a numerical label for a dot."""
    
    def __init__(self, dot_id: int, position: Tuple[int, int], style_manager: StyleManager):
        self.dot_id = dot_id
        self.position = position
        self.style_manager = style_manager
        self.text = str(np.random.randint(0, 9999))  # Random 4-digit number
        self.rotation = style_manager.get_label_rotation()
        self.is_clipped = np.random.random() < style_manager.overlay_config.get('label_clipping', 0.3)
        self.alpha = 1.0
        
    def update_position(self, new_position: Tuple[int, int]):
        """Update label position."""
        self.position = new_position
        
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the label on the frame."""
        if self.style_manager.should_flicker():
            return frame
            
        x, y = self.position
        color = self.style_manager.get_label_color()
        font = self.style_manager.font
        font_scale = self.style_manager.font_scale
        thickness = self.style_manager.font_thickness
        
        # Apply alpha to color
        alpha_color = tuple(int(c * self.alpha) for c in color)
        
        if self.rotation != 0:
            # Rotate the text
            text_size = cv2.getTextSize(self.text, font, font_scale, thickness)[0]
            
            # Create rotation matrix
            center = (x, y)
            rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation, 1.0)
            
            # Get text bounding box
            text_width, text_height = text_size
            corners = np.array([
                [x - text_width//2, y - text_height//2],
                [x + text_width//2, y - text_height//2],
                [x + text_width//2, y + text_height//2],
                [x - text_width//2, y + text_height//2]
            ], dtype=np.float32)
            
            # Rotate corners
            rotated_corners = cv2.transform(corners.reshape(1, -1, 2), rotation_matrix).reshape(-1, 2)
            
            # Draw rotated text
            cv2.putText(frame, self.text, (x, y), font, font_scale, alpha_color, thickness)
        else:
            # Draw normal text
            if self.is_clipped:
                # Draw partially clipped text
                text_size = cv2.getTextSize(self.text, font, font_scale, thickness)[0]
                clip_ratio = 0.5  # Show only 50% of text
                clipped_width = int(text_size[0] * clip_ratio)
                
                # Create a mask for clipping
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (x - text_size[0]//2, y - text_size[1]//2), 
                            (x - text_size[0]//2 + clipped_width, y + text_size[1]//2), 255, -1)
                
                # Draw text with mask
                temp_frame = frame.copy()
                cv2.putText(temp_frame, self.text, (x - text_size[0]//2, y + text_size[1]//2), 
                          font, font_scale, alpha_color, thickness)
                
                # Apply mask
                frame = np.where(mask[..., None] > 0, temp_frame, frame)
            else:
                # Draw full text
                cv2.putText(frame, self.text, (x, y), font, font_scale, alpha_color, thickness)
        
        return frame


class LabelManager:
    """Manages numerical labels for dots."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.style_manager = StyleManager(config)
        self.labels: Dict[int, Label] = {}
        self.show_labels = config.get('overlay', {}).get('num_labels', True)
        
    def update_labels(self, dot_positions: Dict[int, Tuple[int, int]]):
        """Update labels for current dot positions."""
        if not self.show_labels:
            return
            
        # Remove labels for dots that no longer exist
        current_dot_ids = set(dot_positions.keys())
        label_dot_ids = set(self.labels.keys())
        
        for dot_id in label_dot_ids - current_dot_ids:
            del self.labels[dot_id]
        
        # Update existing labels
        for dot_id, position in dot_positions.items():
            if dot_id in self.labels:
                self.labels[dot_id].update_position(position)
            else:
                # Create new label
                label = Label(dot_id, position, self.style_manager)
                self.labels[dot_id] = label
    
    def draw_labels(self, frame: np.ndarray) -> np.ndarray:
        """Draw all labels on the frame."""
        if not self.show_labels:
            return frame
            
        for label in self.labels.values():
            frame = label.draw(frame)
        
        return frame
    
    def get_label_count(self) -> int:
        """Get the number of active labels."""
        return len(self.labels)
