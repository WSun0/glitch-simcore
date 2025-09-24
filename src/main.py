"""
Main entry point for the computer-vision art engine.
"""
import cv2
import yaml
import argparse
import sys
import os
import numpy as np
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from tracking.pose import PoseTracker
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    from tracking.pose_fallback import FallbackPoseTracker as PoseTracker
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available, using OpenCV fallback")

from tracking.mask import MaskProcessor
from overlay.dots import DotGenerator
from overlay.lines import LineManager
from overlay.labels import LabelManager
from overlay.style import StyleManager


class VisionArtEngine:
    """Main engine for the computer-vision art overlay system."""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        """Initialize the art engine with configuration."""
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.pose_tracker = PoseTracker(self.config)
        self.mask_processor = MaskProcessor(self.config)
        self.dot_generator = DotGenerator(self.config)
        self.line_manager = LineManager(self.config)
        self.label_manager = LabelManager(self.config)
        self.style_manager = StyleManager(self.config)
        
        # Video capture
        self.cap = None
        self.is_video_file = False
        
        # Frame processing
        self.frame_count = 0
        self.debug_mode = False
        
        # Performance settings
        self.performance_config = self.config.get('performance', {})
        self.detection_skip_frames = self.performance_config.get('detection_skip_frames', 1)
        self.dot_update_skip_frames = self.performance_config.get('dot_update_skip_frames', 1)
        self.connection_update_skip_frames = self.performance_config.get('connection_update_skip_frames', 1)
        self.last_keypoints = []
        self.last_mask = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'overlay': {
                'num_dots': 35,
                'dot_radius': [2, 4],
                'dot_color': [200, 200, 200],
                'num_labels': True,
                'label_fontscale': 0.4,
                'line_density': 0.3,
                'line_color': [180, 180, 180],
                'line_thickness': 1,
                'drift': 2,
                'fade_alpha': 0.8,
                'highlight_color': [100, 255, 100],
                'accent_color': [255, 150, 100],
                'label_rotation': True,
                'label_clipping': 0.3,
                'motion_bias': True,
                'motion_sensitivity': 0.5,
                'connection_fade_frames': 10,
                'dot_lifetime': 30,
                'flicker_probability': 0.1
            },
            'tracking': {
                'pose_confidence': 0.5,
                'segmentation_confidence': 0.5,
                'keypoint_jitter': 3,
                'mask_expansion': 1.2
            }
        }
    
    def setup_video_capture(self, source: str = "0"):
        """Setup video capture from webcam or video file."""
        if source.isdigit():
            # Webcam
            self.cap = cv2.VideoCapture(int(source))
            self.is_video_file = False
        else:
            # Video file
            self.cap = cv2.VideoCapture(source)
            self.is_video_file = True
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Set video properties for higher quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        print(f"Video source: {source}")
        print(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and apply overlays."""
        self.frame_count += 1
        
        # Optimize detection by skipping frames if needed
        if self.frame_count % self.detection_skip_frames == 0:
            # Detect pose and get keypoints
            keypoints, raw_mask = self.pose_tracker.detect_pose(frame)
            
            # Process segmentation mask
            processed_mask = self.mask_processor.process_mask(raw_mask, frame.shape)
            
            # Cache results
            self.last_keypoints = keypoints
            self.last_mask = processed_mask
            
            # Generate dots from keypoints and mask (only when detection runs)
            new_dots = self.dot_generator.generate_dots_from_keypoints(keypoints, processed_mask)
            self.dot_generator.dots = new_dots
        else:
            # Use cached results
            keypoints = self.last_keypoints
            processed_mask = self.last_mask
        
        # Update dots with motion (optimized frequency)
        if self.frame_count % self.dot_update_skip_frames == 0:
            self.dot_generator.update_dots()
        
        # Get current dot positions
        dot_positions = {i: (dot.x, dot.y) for i, dot in enumerate(self.dot_generator.dots) if dot.is_alive()}
        
        # Update connections (optimized frequency)
        if self.frame_count % self.connection_update_skip_frames == 0:
            self.line_manager.update_connections(dot_positions)
        
        # Update labels (every frame for smooth text)
        self.label_manager.update_labels(dot_positions)
        
        # Create overlay frame
        overlay_frame = np.zeros_like(frame)
        
        # Draw dots
        overlay_frame = self.dot_generator.draw_dots(overlay_frame)
        
        # Draw connections
        overlay_frame = self.line_manager.draw_connections(overlay_frame, dot_positions)
        
        # Draw labels
        overlay_frame = self.label_manager.draw_labels(overlay_frame)
        
        # Apply alpha blending
        result_frame = self.style_manager.apply_alpha_blend(overlay_frame, frame)
        
        # Debug mode overlays
        if self.debug_mode:
            result_frame = self._add_debug_overlays(result_frame, keypoints, processed_mask)
        
        return result_frame
    
    def _add_debug_overlays(self, frame: np.ndarray, keypoints: list, mask: np.ndarray) -> np.ndarray:
        """Add debug overlays to the frame."""
        # Draw pose keypoints
        frame = self.pose_tracker.draw_pose_debug(frame, keypoints)
        
        # Draw mask overlay
        frame = self.mask_processor.draw_mask_debug(frame, mask)
        
        # Add info text
        info_text = [
            f"Frame: {self.frame_count}",
            f"Dots: {len(self.dot_generator.dots)}",
            f"Connections: {len(self.line_manager.connections)}",
            f"Labels: {self.label_manager.get_label_count()}",
            f"Keypoints: {len(keypoints)}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return frame
    
    def run(self):
        """Main processing loop."""
        print("Starting computer-vision art engine...")
        print("Controls:")
        print("  'd' - Toggle debug mode")
        print("  'q' or ESC - Quit")
        print("  's' - Save current frame")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file:
                    print("End of video file reached.")
                    break
                else:
                    print("Failed to read from webcam.")
                    break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display frame
            cv2.imshow('Computer Vision Art Engine', processed_frame)
            
            # Handle key presses (reduced wait time for higher FPS)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('s'):
                filename = f"capture_{self.frame_count}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Saved frame to {filename}")
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.pose_tracker.cleanup()
        print("Cleanup completed.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Computer Vision Art Engine')
    parser.add_argument('--source', '-s', default='0', 
                       help='Video source (webcam index or video file path)')
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        engine = VisionArtEngine(args.config)
        engine.setup_video_capture(args.source)
        engine.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
