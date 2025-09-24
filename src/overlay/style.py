"""
Styling and color palette management for the overlay system.
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any


class StyleManager:
    """Manages colors, fonts, and visual styling for overlays."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.overlay_config = config.get('overlay', {})
        
        # Color palettes
        self.colors = {
            'dot': tuple(self.overlay_config.get('dot_color', [200, 200, 200])),
            'line': tuple(self.overlay_config.get('line_color', [180, 180, 180])),
            'highlight': tuple(self.overlay_config.get('highlight_color', [100, 255, 100])),
            'accent': tuple(self.overlay_config.get('accent_color', [255, 150, 100])),
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = self.overlay_config.get('label_fontscale', 0.4)
        self.font_thickness = 1
        
        # Visual parameters
        self.dot_radius_range = self.overlay_config.get('dot_radius', [2, 4])
        self.line_thickness = self.overlay_config.get('line_thickness', 1)
        self.fade_alpha = self.overlay_config.get('fade_alpha', 0.8)
        
    
    def get_dot_color(self, is_highlight: bool = False) -> Tuple[int, int, int]:
        """Get dot color, optionally highlighted."""
        if is_highlight:
            return self.colors['highlight']
        return self.colors['dot']
    
    def get_line_color(self) -> Tuple[int, int, int]:
        """Get line color."""
        return self.colors['line']
    
    def get_label_color(self) -> Tuple[int, int, int]:
        """Get label text color."""
        return (255, 255, 255)  # White text
    
    def apply_alpha_blend(self, overlay: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Apply alpha blending to overlay."""
        return cv2.addWeighted(background, 1 - self.fade_alpha, overlay, self.fade_alpha, 0)
    
    def should_highlight(self) -> bool:
        """Randomly determine if a dot should be highlighted."""
        return np.random.random() < 0.1  # 10% chance
    
    def should_flicker(self) -> bool:
        """Randomly determine if an element should flicker."""
        return np.random.random() < self.overlay_config.get('flicker_probability', 0.1)
    
    def get_label_rotation(self) -> float:
        """Get random label rotation angle."""
        if self.overlay_config.get('label_rotation', True):
            return np.random.uniform(-15, 15)  # degrees
        return 0.0
