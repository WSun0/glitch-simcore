#!/usr/bin/env python3
"""
Setup script for the Computer Vision Art Engine.
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def test_installation():
    """Test the installation."""
    print("Testing installation...")
    try:
        result = subprocess.run([sys.executable, "test_installation.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Installation test passed")
            return True
        else:
            print("✗ Installation test failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Installation test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("Computer Vision Art Engine Setup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("✗ requirements.txt not found. Please run this script from the project root.")
        return 1
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Test installation
    if not test_installation():
        return 1
    
    print("\n" + "=" * 40)
    print("✓ Setup completed successfully!")
    print("\nYou can now run the art engine:")
    print("  python src/main.py")
    print("\nOr try the demo:")
    print("  python demo.py webcam")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
