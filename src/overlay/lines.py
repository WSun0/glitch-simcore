"""
Dynamic graph connections and line drawing system.
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Set
from .style import StyleManager


class Connection:
    """Represents a connection between two dots."""
    
    def __init__(self, dot1_id: int, dot2_id: int, style_manager: StyleManager):
        self.dot1_id = dot1_id
        self.dot2_id = dot2_id
        self.style_manager = style_manager
        self.fade_frames = style_manager.overlay_config.get('connection_fade_frames', 10)
        self.age = 0
        self.is_fading_in = True
        self.alpha = 0.0
        
    def update(self):
        """Update connection fade state."""
        self.age += 1
        
        if self.is_fading_in:
            self.alpha = min(1.0, self.age / self.fade_frames)
            if self.alpha >= 1.0:
                self.is_fading_in = False
        else:
            # Randomly start fading out
            if np.random.random() < 0.05:  # 5% chance per frame
                self.alpha = max(0.0, self.alpha - (1.0 / self.fade_frames))
    
    def is_alive(self) -> bool:
        """Check if connection should still be displayed."""
        return self.alpha > 0.0
    
    def draw(self, frame: np.ndarray, dot_positions: Dict[int, Tuple[int, int]]) -> np.ndarray:
        """Draw the connection line on the frame."""
        if not self.is_alive():
            return frame
            
        pos1 = dot_positions.get(self.dot1_id)
        pos2 = dot_positions.get(self.dot2_id)
        
        if not pos1 or not pos2:
            return frame
            
        # Apply flicker effect
        if self.style_manager.should_flicker():
            return frame
            
        color = self.style_manager.get_line_color()
        thickness = self.style_manager.line_thickness
        
        # Apply alpha to color
        alpha_color = tuple(int(c * self.alpha) for c in color)
        
        cv2.line(frame, pos1, pos2, alpha_color, thickness)
        return frame


class LineManager:
    """Manages dynamic connections between dots."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.style_manager = StyleManager(config)
        self.connections: List[Connection] = []
        self.line_density = config.get('overlay', {}).get('line_density', 0.3)
        
    def update_connections(self, dot_positions: Dict[int, Tuple[int, int]]):
        """Update existing connections and create new ones."""
        # Update existing connections
        for connection in self.connections:
            connection.update()
        
        # Remove dead connections
        self.connections = [conn for conn in self.connections if conn.is_alive()]
        
        # Create new connections
        self._create_new_connections(dot_positions)
    
    def _create_new_connections(self, dot_positions: Dict[int, Tuple[int, int]]):
        """Create new connections based on dot positions and density."""
        dot_ids = list(dot_positions.keys())
        
        if len(dot_ids) < 2:
            return
            
        # Calculate how many connections we should have
        max_connections = int(len(dot_ids) * self.line_density)
        current_connections = len(self.connections)
        
        if current_connections < max_connections:
            # Create new connections
            needed = max_connections - current_connections
            
            for _ in range(needed):
                # Randomly select two dots
                dot1_id, dot2_id = np.random.choice(dot_ids, 2, replace=False)
                
                # Check if connection already exists
                if not self._connection_exists(dot1_id, dot2_id):
                    connection = Connection(dot1_id, dot2_id, self.style_manager)
                    self.connections.append(connection)
    
    def _connection_exists(self, dot1_id: int, dot2_id: int) -> bool:
        """Check if a connection between two dots already exists."""
        for conn in self.connections:
            if ((conn.dot1_id == dot1_id and conn.dot2_id == dot2_id) or
                (conn.dot1_id == dot2_id and conn.dot2_id == dot1_id)):
                return True
        return False
    
    def draw_connections(self, frame: np.ndarray, dot_positions: Dict[int, Tuple[int, int]]) -> np.ndarray:
        """Draw all connections on the frame."""
        for connection in self.connections:
            frame = connection.draw(frame, dot_positions)
        return frame
    
    def get_connection_network(self) -> Dict[int, Set[int]]:
        """Get the current connection network as a graph."""
        network = {}
        
        for connection in self.connections:
            if connection.is_alive():
                if connection.dot1_id not in network:
                    network[connection.dot1_id] = set()
                if connection.dot2_id not in network:
                    network[connection.dot2_id] = set()
                    
                network[connection.dot1_id].add(connection.dot2_id)
                network[connection.dot2_id].add(connection.dot1_id)
        
        return network
