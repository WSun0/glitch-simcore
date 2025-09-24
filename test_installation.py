#!/usr/bin/env python3
"""
Test script to verify the computer vision art engine installation.
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("✓ MediaPipe imported successfully")
    except ImportError:
        print("⚠ MediaPipe not available (will use OpenCV fallback)")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML imported successfully")
    except ImportError as e:
        print(f"✗ PyYAML import failed: {e}")
        return False
    
    # Test our modules
    try:
        from tracking.pose import PoseTracker
        print("✓ PoseTracker (MediaPipe) imported successfully")
    except ImportError:
        try:
            from tracking.pose_fallback import FallbackPoseTracker as PoseTracker
            print("✓ PoseTracker (OpenCV fallback) imported successfully")
        except ImportError as e:
            print(f"✗ PoseTracker import failed: {e}")
            return False
    
    try:
        from tracking.mask import MaskProcessor
        print("✓ MaskProcessor imported successfully")
    except ImportError as e:
        print(f"✗ MaskProcessor import failed: {e}")
        return False
    
    try:
        from overlay.dots import DotGenerator
        print("✓ DotGenerator imported successfully")
    except ImportError as e:
        print(f"✗ DotGenerator import failed: {e}")
        return False
    
    try:
        from overlay.lines import LineManager
        print("✓ LineManager imported successfully")
    except ImportError as e:
        print(f"✗ LineManager import failed: {e}")
        return False
    
    try:
        from overlay.labels import LabelManager
        print("✓ LabelManager imported successfully")
    except ImportError as e:
        print(f"✗ LabelManager import failed: {e}")
        return False
    
    try:
        from overlay.style import StyleManager
        print("✓ StyleManager imported successfully")
    except ImportError as e:
        print(f"✗ StyleManager import failed: {e}")
        return False
    
    return True

def test_config():
    """Test that configuration can be loaded."""
    try:
        import yaml
        config_path = os.path.join('src', 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✓ Configuration loaded successfully")
            return True
        else:
            print("✗ Configuration file not found")
            return False
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Computer Vision Art Engine Installation...")
    print("=" * 50)
    
    success = True
    
    print("\n1. Testing imports...")
    success &= test_imports()
    
    print("\n2. Testing configuration...")
    success &= test_config()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! The installation is working correctly.")
        print("\nYou can now run the art engine with:")
        print("  python src/main.py")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
