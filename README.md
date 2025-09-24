this is# Computer Vision Art Engine

A computer-vision-inspired art engine that overlays abstract point networks and cryptic numerical tags on live video. Designed for an alt/simulation-core aesthetic — minimal, eerie, and raw, rather than flashy.

## Overview

This engine takes a webcam or video feed and overlays abstract dots and connections that focus on the subjects in the frame. The overlay looks like a machine analyzing reality: numerical tags, minimal lines, floating nodes. The effect avoids "video game glitch" clichés in favor of a "cold machine perception" or "unsettling computer vision leak" aesthetic.

## Features

- **Minimalistic Design**: Thin lines, small dots, subdued colors (white, pale gray, muted orange, soft neon green)
- **Dynamic Placement**: Points track subject features using MediaPipe pose detection and segmentation
- **Imperfection**: Slight drift, flicker, and missing points for a raw, unpolished feel
- **Numerical Overlays**: Each dot gets a random number label, rotated or half-clipped
- **Evolving Networks**: Thin lines connect dots to form dynamic graphs that change over time
- **Motion Influence**: Dots cluster around moving subjects using optical flow

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd glitch-simcore
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run with webcam (default):
```bash
python src/main.py
```

Run with specific webcam index:
```bash
python src/main.py --source 1
```

Run with video file:
```bash
python src/main.py --source path/to/video.mp4
```

### Controls

- `d` - Toggle debug mode (shows keypoints and mask overlays)
- `q` or `ESC` - Quit
- `s` - Save current frame

### Configuration

Edit `src/config.yaml` to customize the overlay behavior:

```yaml
overlay:
  num_dots: 35                    # Number of dots to display
  dot_radius: [2, 4]             # Random radius range
  dot_color: [200, 200, 200]     # Gray/white color
  line_density: 0.3              # Fraction of dots connected
  drift: 2                       # Pixels of jitter per frame
  fade_alpha: 0.8                # Transparency level
  # ... more options
```

## Project Structure

```
glitch-simcore/
├── src/
│   ├── main.py                  # Entry point
│   ├── config.yaml              # Configuration
│   ├── overlay/
│   │   ├── dots.py              # Point generation
│   │   ├── lines.py             # Graph connections
│   │   ├── labels.py            # Number overlays
│   │   └── style.py             # Color palettes
│   └── tracking/
│       ├── pose.py              # MediaPipe keypoints
│       └── mask.py              # Segmentation mask
├── requirements.txt
└── README.md
```

## Technical Details

- **Pose Detection**: Uses MediaPipe for real-time pose keypoint detection
- **Segmentation**: MediaPipe selfie segmentation for subject isolation
- **Overlay System**: Modular design with separate components for dots, lines, and labels
- **Motion Tracking**: Optical flow integration for motion-biased dot placement
- **Temporal Effects**: Connections fade in/out, dots have lifetimes, flicker effects

## Dependencies

- OpenCV 4.8.1+ for video processing
- MediaPipe 0.10.7+ for pose detection and segmentation
- NumPy for numerical operations
- PyYAML for configuration management

## License

[Add your license here]
