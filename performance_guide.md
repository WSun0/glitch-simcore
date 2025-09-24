# Performance Optimization Guide

## Current Optimizations Applied

### 1. **Frame Skipping**
- **Detection**: Every 2nd frame (50% reduction)
- **Dot Updates**: Every frame (smooth movement)
- **Connection Updates**: Every 3rd frame (67% reduction)

### 2. **Detection Optimizations**
- **HOG Detection**: Larger stride (24,24) and scale (1.2)
- **Face Detection**: Higher scale (1.3) and fewer neighbors (2)
- **Reduced Dot Count**: 25 dots instead of 35

### 3. **Processing Optimizations**
- **Cached Results**: Detection results cached between frames
- **Selective Updates**: Only update what's needed each frame

## Performance Settings

```yaml
performance:
  detection_skip_frames: 2        # Every 2nd frame
  dot_update_skip_frames: 1       # Every frame (smooth)
  connection_update_skip_frames: 3 # Every 3rd frame
  overlay_quality: "medium"       # Balanced quality
```

## Speed vs Quality Trade-offs

### **Maximum Speed** (Lowest Quality)
```yaml
overlay:
  num_dots: 15                    # Fewer dots
performance:
  detection_skip_frames: 4        # Every 4th frame
  connection_update_skip_frames: 5 # Every 5th frame
  overlay_quality: "low"
```

### **Balanced** (Current)
```yaml
overlay:
  num_dots: 25
performance:
  detection_skip_frames: 2        # Every 2nd frame
  connection_update_skip_frames: 3 # Every 3rd frame
  overlay_quality: "medium"
```

### **Maximum Quality** (Slower)
```yaml
overlay:
  num_dots: 50
performance:
  detection_skip_frames: 1        # Every frame
  connection_update_skip_frames: 1 # Every frame
  overlay_quality: "high"
```

## Additional Optimizations

### **Reduce Random Generation**
- Random colors/shapes are generated once per dot (not per frame)
- Only position updates happen every frame

### **Lower Resolution**
- Change `max_resolution: [1280, 720]` for faster processing
- Or `max_resolution: [640, 480]` for maximum speed

### **Fewer Colors**
- Reduce `color_palette` to 3-4 colors instead of 8
- Less memory allocation per dot

## Expected Performance Gains

- **Detection Skipping**: ~40-50% faster
- **Connection Skipping**: ~20-30% faster  
- **Reduced Dots**: ~15-25% faster
- **Combined**: ~60-70% overall speed improvement

## Monitoring Performance

Use debug mode (`d` key) to see:
- Frame count
- Number of dots
- Number of connections
- Detection frequency

Adjust settings based on your system's performance!
