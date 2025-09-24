# Computer Vision Art Engine

A computer-vision-inspired art engine that overlays abstract point networks and cryptic numerical tags on live video. Creates a "cold machine perception" aesthetic — like a computer vision system leaking into reality.

## What It Does

Takes your webcam or video feed and overlays abstract dots and connections that track subjects in the frame. The effect looks like a machine analyzing reality: numerical tags, minimal lines, floating nodes. Avoids flashy "video game glitch" effects in favor of something more unsettling and raw.

## Key Features

- **Imperfection**: Slight drift, flicker, and missing points for a raw, unpolished feel
- **Numerical Overlays**: Each dot gets a random number label, rotated or half-clipped
- **Evolving Networks**: Thin lines connect dots to form dynamic graphs that change over time
- **Random Shapes & Colors**: Dots are circles or squares with random sizes and colors
- **Smart Tracking**: People-first detection with adaptive keypoint density for clean, focused overlays

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
  num_dots: 12                   # Number of dots (reduced for cleaner look)
  dot_radius: [2, 4]             # Random radius range
  dot_color: [200, 200, 200]     # Gray/white color
  line_density: 0.3              # Fraction of dots connected
  drift: 4                       # Pixels of jitter per frame (increased for movement)
  fade_alpha: 0.8                # Transparency level
```

## Smart Tracking Features

The engine includes intelligent computer vision tracking that prioritizes people and body parts:

- **People-First Detection**: HOG-based person detection with body keypoints (head, torso, arms)
- **Facial Feature Tracking**: Eyes, nose, and facial landmarks with reduced keypoint density
- **Hand & Arm Detection**: Multiple methods including cascade classifiers and skin color detection
- **Motion-Aware**: Real-time motion detection that only activates when no people are detected
- **Object Detection**: Contour-based detection as fallback when few people are present
- **Adaptive Keypoint Density**: Automatically reduces keypoints to prevent visual clutter

## Performance & Quality

The engine is optimized for smooth, responsive performance:

- **Reduced Node Count**: Only 12 dots for clean, uncluttered visuals
- **People-First Priority**: Dots cluster around people, not static objects
- **Increased Movement**: Higher drift (4px) creates more dynamic, less "tethered" effects
- **Longer Dot Lifetime**: 60 frames for sustained visual presence
- **Smart Detection**: Objects only detected when no people present
- **Frame Skipping**: Optimized detection frequency for 60 FPS target

## Dependencies

- OpenCV 4.8.1+ for video processing and computer vision
- MediaPipe 0.10.7+ for pose detection and segmentation (falls back to OpenCV if unavailable)
- NumPy for numerical operations
- PyYAML for configuration management
