#!/usr/bin/env python3
"""
Static demo showing the computer vision art engine effects on a sample image.
"""
import sys
import os
import cv2
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import VisionArtEngine

def create_sample_frame():
    """Create a sample frame with some geometric shapes to simulate a person."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create a simple "person" shape
    # Head
    cv2.circle(frame, (320, 100), 30, (200, 200, 200), -1)
    
    # Body
    cv2.rectangle(frame, (300, 130), (340, 250), (150, 150, 150), -1)
    
    # Arms
    cv2.rectangle(frame, (250, 150), (300, 180), (120, 120, 120), -1)  # Left arm
    cv2.rectangle(frame, (340, 150), (390, 180), (120, 120, 120), -1)  # Right arm
    
    # Legs
    cv2.rectangle(frame, (310, 250), (330, 350), (100, 100, 100), -1)  # Left leg
    cv2.rectangle(frame, (330, 250), (350, 350), (100, 100, 100), -1)  # Right leg
    
    # Add some background elements
    cv2.rectangle(frame, (50, 200), (150, 300), (80, 80, 80), -1)  # Background object
    cv2.circle(frame, (500, 200), 40, (90, 90, 90), -1)  # Another background object
    
    return frame

def demo_static_processing():
    """Demo the art engine on a static frame."""
    print("Creating sample frame...")
    sample_frame = create_sample_frame()
    
    print("Initializing art engine...")
    engine = VisionArtEngine()
    
    print("Processing frame...")
    processed_frame = engine.process_frame(sample_frame)
    
    # Save both original and processed frames
    cv2.imwrite("sample_original.jpg", sample_frame)
    cv2.imwrite("sample_processed.jpg", processed_frame)
    
    print("✓ Demo completed!")
    print("  - Original frame saved as: sample_original.jpg")
    print("  - Processed frame saved as: sample_processed.jpg")
    
    # Display the frames
    print("\nDisplaying frames (press any key to close)...")
    cv2.imshow("Original Frame", sample_frame)
    cv2.imshow("Computer Vision Art Engine - Processed", processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """Main demo function."""
    print("Computer Vision Art Engine - Static Demo")
    print("=" * 50)
    
    try:
        demo_static_processing()
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
