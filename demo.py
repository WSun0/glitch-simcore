#!/usr/bin/env python3
"""
Demo script showing how to use the Computer Vision Art Engine programmatically.
"""
import sys
import os
import cv2
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import VisionArtEngine

def demo_with_webcam():
    """Demo using webcam input."""
    print("Starting webcam demo...")
    print("Press 'q' to quit, 'd' to toggle debug mode")
    
    try:
        engine = VisionArtEngine()
        engine.setup_video_capture("0")  # Use default webcam
        engine.run()
    except Exception as e:
        print(f"Demo failed: {e}")

def demo_with_video_file(video_path):
    """Demo using video file input."""
    print(f"Starting video file demo with: {video_path}")
    print("Press 'q' to quit, 'd' to toggle debug mode")
    
    try:
        engine = VisionArtEngine()
        engine.setup_video_capture(video_path)
        engine.run()
    except Exception as e:
        print(f"Demo failed: {e}")

def demo_single_frame():
    """Demo processing a single frame."""
    print("Single frame processing demo...")
    
    # Create a test frame (you can replace this with loading an image)
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        engine = VisionArtEngine()
        processed_frame = engine.process_frame(test_frame)
        
        # Save the result
        cv2.imwrite("demo_output.jpg", processed_frame)
        print("Demo frame saved as 'demo_output.jpg'")
        
    except Exception as e:
        print(f"Single frame demo failed: {e}")

def main():
    """Main demo function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "webcam":
            demo_with_webcam()
        elif sys.argv[1] == "frame":
            demo_single_frame()
        elif sys.argv[1].endswith(('.mp4', '.avi', '.mov', '.mkv')):
            demo_with_video_file(sys.argv[1])
        else:
            print("Usage: python demo.py [webcam|frame|video_file]")
    else:
        print("Computer Vision Art Engine Demo")
        print("=" * 40)
        print("Usage:")
        print("  python demo.py webcam          # Use webcam")
        print("  python demo.py frame           # Process single frame")
        print("  python demo.py video.mp4       # Use video file")
        print()
        print("Running webcam demo by default...")
        demo_with_webcam()

if __name__ == "__main__":
    main()
