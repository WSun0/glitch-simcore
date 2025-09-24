"""
Dot generation and management system.
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
from .style import StyleManager


class Dot:
    """Represents a single overlay dot."""
    
    def __init__(self, x: int, y: int, dot_id: int, style_manager: StyleManager):
        self.x = x
        self.y = y
        self.original_x = x
        self.original_y = y
        self.id = dot_id
        self.style_manager = style_manager
        self.lifetime = style_manager.overlay_config.get('dot_lifetime', 30)
        self.age = 0
        self.is_highlight = style_manager.should_highlight()
        self.drift_x = 0
        self.drift_y = 0
        
        # Random shape, size, and color properties
        self.shape = self._get_random_shape()
        self.size = self._get_random_size()
        self.color = self._get_random_color()
        
        
    def update(self, motion_vector: Optional[Tuple[float, float]] = None):
        """Update dot position with drift and motion influence."""
        self.age += 1
        
        # Apply drift
        drift_amount = self.style_manager.overlay_config.get('drift', 2)
        self.drift_x += np.random.normal(0, drift_amount * 0.5)
        self.drift_y += np.random.normal(0, drift_amount * 0.5)
        
        # Apply motion bias if available
        if motion_vector and self.style_manager.overlay_config.get('motion_bias', True):
            motion_sensitivity = self.style_manager.overlay_config.get('motion_sensitivity', 0.5)
            self.drift_x += motion_vector[0] * motion_sensitivity
            self.drift_y += motion_vector[1] * motion_sensitivity
        
        # Update position
        self.x = int(self.original_x + self.drift_x)
        self.y = int(self.original_y + self.drift_y)
    
    def _get_random_shape(self) -> str:
        """Get a random shape type."""
        shape_types = self.style_manager.overlay_config.get('shape_types', ['circle', 'square'])
        return np.random.choice(shape_types)
    
    def _get_random_size(self) -> int:
        """Get a random size with bias toward smaller sizes."""
        size_range = self.style_manager.overlay_config.get('size_range', [1, 6])
        size_bias = self.style_manager.overlay_config.get('size_bias', 0.7)
        
        min_size, max_size = size_range
        
        # Use exponential distribution to bias toward smaller sizes
        # Higher bias value = more small sizes
        if size_bias > 0:
            # Generate a value between 0 and 1, then apply bias
            random_val = np.random.random()
            biased_val = random_val ** (1.0 / size_bias)
            size = int(min_size + biased_val * (max_size - min_size))
        else:
            # Uniform distribution
            size = np.random.randint(min_size, max_size + 1)
        
        return max(min_size, min(max_size, size))
    
    def _get_random_color(self) -> Tuple[int, int, int]:
        """Get a random color from the palette."""
        color_palette = self.style_manager.overlay_config.get('color_palette', [
            [200, 200, 200],  # Default light gray
        ])
        # Select a random color from the palette
        random_index = np.random.randint(0, len(color_palette))
        return tuple(color_palette[random_index])
        
    
    def is_alive(self) -> bool:
        """Check if dot should still be displayed."""
        return self.age < self.lifetime
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the dot on the frame."""
        if not self.is_alive():
            return frame
            
        # Apply flicker effect
        if self.style_manager.should_flicker():
            return frame
            
        # Use random color, but apply highlight if needed
        color = self.color
        if self.is_highlight:
            # Mix with highlight color for special emphasis
            highlight_color = self.style_manager.get_dot_color(True)
            color = tuple(int((c + h) / 2) for c, h in zip(color, highlight_color))
        
        if self.shape == "circle":
            cv2.circle(frame, (self.x, self.y), self.size, color, -1)
        elif self.shape == "square":
            # Draw square centered at (x, y)
            half_size = self.size
            top_left = (self.x - half_size, self.y - half_size)
            bottom_right = (self.x + half_size, self.y + half_size)
            cv2.rectangle(frame, top_left, bottom_right, color, -1)
        
        return frame


class DotGenerator:
    """Generates and manages overlay dots."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.style_manager = StyleManager(config)
        self.dots: List[Dot] = []
        self.next_dot_id = 0
        self.num_dots = config.get('overlay', {}).get('num_dots', 35)
        
    def generate_dots_from_keypoints(self, keypoints: List[Tuple[int, int]], 
                                   mask: Optional[np.ndarray] = None) -> List[Dot]:
        """Generate dots around pose keypoints and within subject mask."""
        new_dots = []
        jitter = self.config.get('tracking', {}).get('keypoint_jitter', 3)
        
        # Generate dots around keypoints
        for kp_x, kp_y in keypoints:
            if kp_x > 0 and kp_y > 0:  # Valid keypoint
                # Create cluster around keypoint
                cluster_size = np.random.randint(3, 8)
                for _ in range(cluster_size):
                    x = kp_x + np.random.randint(-jitter, jitter + 1)
                    y = kp_y + np.random.randint(-jitter, jitter + 1)
                    
                    # Ensure point is within frame bounds
                    if 0 <= x < 1920 and 0 <= y < 1080:  # Assuming HD resolution
                        dot = Dot(x, y, self.next_dot_id, self.style_manager)
                        new_dots.append(dot)
                        self.next_dot_id += 1
        
        # Fill remaining dots randomly within mask
        if mask is not None:
            remaining_dots = self.num_dots - len(new_dots)
            mask_points = np.where(mask > 0)
            
            if len(mask_points[0]) > 0:
                for _ in range(remaining_dots):
                    idx = np.random.randint(0, len(mask_points[0]))
                    y, x = mask_points[0][idx], mask_points[1][idx]
                    
                    dot = Dot(x, y, self.next_dot_id, self.style_manager)
                    new_dots.append(dot)
                    self.next_dot_id += 1
        
        return new_dots
    
    def update_dots(self, motion_vectors: Optional[Dict[int, Tuple[float, float]]] = None):
        """Update all dots with drift and motion."""
        # Remove dead dots
        self.dots = [dot for dot in self.dots if dot.is_alive()]
        
        # Update existing dots
        for dot in self.dots:
            motion_vector = motion_vectors.get(dot.id) if motion_vectors else None
            dot.update(motion_vector)
        
        # Add new dots if needed
        if len(self.dots) < self.num_dots:
            # Generate random dots to maintain count
            for _ in range(self.num_dots - len(self.dots)):
                x = np.random.randint(50, 1870)  # Within frame bounds
                y = np.random.randint(50, 1030)
                dot = Dot(x, y, self.next_dot_id, self.style_manager)
                self.dots.append(dot)
                self.next_dot_id += 1
    
    def draw_dots(self, frame: np.ndarray) -> np.ndarray:
        """Draw all dots on the frame."""
        for dot in self.dots:
            frame = dot.draw(frame)
        return frame
    
    def get_dot_positions(self) -> List[Tuple[int, int]]:
        """Get current positions of all alive dots."""
        return [(dot.x, dot.y) for dot in self.dots if dot.is_alive()]
