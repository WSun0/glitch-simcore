#!/usr/bin/env python3
"""
Test script to demonstrate the new bouncy behavior configuration options.
"""
import sys
import os
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def show_bouncy_config():
    """Show the current bouncy behavior configuration."""
    print("🎯 Bouncy Behavior Configuration")
    print("=" * 50)
    
    try:
        with open('src/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        overlay_config = config.get('overlay', {})
        
        print("Current Settings:")
        print(f"  background_exploration: {overlay_config.get('background_exploration', 0.3)}")
        print(f"    → Fraction of dots that explore background (0.0-1.0)")
        print(f"  exploration_radius: {overlay_config.get('exploration_radius', 150)}")
        print(f"    → How far dots can wander from subject (pixels)")
        print(f"  bounce_frequency: {overlay_config.get('bounce_frequency', 0.1)}")
        print(f"    → How often dots 'jump' to new locations (per frame)")
        print(f"  background_attraction: {overlay_config.get('background_attraction', 0.2)}")
        print(f"    → How much dots are attracted to background objects")
        
        print("\n🎮 How to Adjust Bounciness:")
        print("  • Increase background_exploration (0.5-0.8) for more dots exploring")
        print("  • Increase exploration_radius (200-300) for wider wandering")
        print("  • Increase bounce_frequency (0.2-0.5) for more frequent jumping")
        print("  • Increase background_attraction (0.3-0.6) for stronger background pull")
        
        print("\n💡 Recommended Settings for More Bouncy Behavior:")
        print("  background_exploration: 0.5  # Half the dots explore")
        print("  exploration_radius: 200      # Wider exploration area")
        print("  bounce_frequency: 0.2        # More frequent jumping")
        print("  background_attraction: 0.4   # Stronger background attraction")
        
    except Exception as e:
        print(f"Error reading config: {e}")

def main():
    show_bouncy_config()

if __name__ == "__main__":
    main()
