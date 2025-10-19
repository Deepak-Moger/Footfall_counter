🧠 AI Assignment: Footfall Counter using
Computer Vision

A real-time computer vision system that detects, tracks, and counts people crossing a virtual line in video streams or live webcam feed.

## Features

- **Person Detection**: Uses YOLOv8 for accurate human detection
- **Object Tracking**: Custom centroid-based tracker with trajectory visualization
- **Bidirectional Counting**: Separately counts entries and exits
- **Live Webcam Support**: Real-time counting using your laptop/desktop webcam
- **Video Processing**: Process recorded videos from files
- **Real-time Visualization**:
  - Bounding boxes around detected people
  - Unique ID assignment for each person
  - Movement trajectories
  - Live counter display
  - FPS monitoring
- **Configurable Counting Line**: Adjustable virtual line position
- **Automatic Screen Fitting**: Video display auto-resizes to fit your screen

## System Architecture

```
┌─────────────────┐
│  Video Input    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ YOLOv8 Detection│ ─── Detects people in each frame
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Centroid Tracker│ ─── Tracks individuals across frames
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Line Crossing   │ ─── Detects when people cross the line
│    Detection    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Entry/Exit Count│ ─── Updates counters based on direction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visualization  │ ─── Displays results
└─────────────────┘
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (for live detection) or video file for testing
- Windows/macOS/Linux

### Step-by-Step Setup

**Step 1: Open Command Prompt/Terminal**
- Windows: Press `Win + R`, type `cmd`, press Enter
- macOS: Press `Cmd + Space`, type `terminal`, press Enter
- Linux: Press `Ctrl + Alt + T`

**Step 2: Navigate to project directory**
```bash
cd "D:\Footfall Counter"
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

This will install:
- `opencv-python`: Video processing and visualization
- `ultralytics`: YOLOv8 model
- `numpy`: Numerical operations
- `scipy` & `filterpy`: Tracking algorithms

**Step 4: Verify installation**
```bash
python test_installation.py
```

You should see "ALL TESTS PASSED!" if everything is installed correctly.

**Step 5: YOLO model (automatic)**
The YOLOv8n model will be downloaded automatically on first run (~6MB).

## Usage

### Quick Start Guide

#### Option 1: Use Your Webcam (Easiest!)

```bash
python footfall_counter.py --video webcam
```

This will:
- Open your webcam
- Show live video with detection and tracking
- Count people crossing the virtual line in real-time
- **Automatically save to `footfall_output.mp4`**
- Press `q` to quit, `r` to reset counters

**Tips for webcam use:**
- Position your laptop so people walk across the screen horizontally
- Ensure good lighting
- The counting line will appear in the middle of the screen
- People crossing from top to bottom = Entry
- People crossing from bottom to top = Exit
- Output video will be saved in the same folder

#### Option 2: Process a Video File

```bash
python footfall_counter.py --video path/to/your/video.mp4
```

Example:
```bash
python footfall_counter.py --video "C:\Users\YourName\Videos\test.mp4"
```

**Output saved automatically to `footfall_output.mp4`**

#### Option 3: Custom Output Filename

```bash
python footfall_counter.py --video webcam --output my_recording.mp4
```

This saves the processed video with your custom name.

#### Option 4: Don't Save Video (Display Only)

```bash
python footfall_counter.py --video webcam --output none
```

Use this if you only want to see the live display without saving.

### All Command Line Options

```bash
python footfall_counter.py --video <source> [OPTIONS]
```

**Required:**
- `--video <source>`: Video source
  - Use `webcam` for live camera
  - Or path to video file: `"C:\path\to\video.mp4"`

**Optional:**
- `--output <file>`: Save processed video to file
  - **Default: `footfall_output.mp4`** (saves automatically)
  - Example: `--output my_video.mp4` (custom name)
  - Use `--output none` to disable saving

- `--line <position>`: Counting line position (0.0 to 1.0)
  - `0.0` = top of frame
  - `0.5` = middle (default)
  - `1.0` = bottom
  - Example: `--line 0.3` (line at 30% from top)

- `--display-width <pixels>`: Window width for display
  - Default: `1280` (fits most laptop screens)
  - Use `800` for smaller screens
  - Use `0` for original video size
  - Example: `--display-width 800`

- `--model <model>`: YOLO model to use
  - `yolov8n.pt` = Nano - fastest (default)
  - `yolov8s.pt` = Small
  - `yolov8m.pt` = Medium
  - `yolov8l.pt` = Large - most accurate
  - Example: `--model yolov8s.pt`

### Complete Examples

**1. Basic webcam (saves to footfall_output.mp4):**
```bash
python footfall_counter.py --video webcam
```

**2. Webcam with custom output name:**
```bash
python footfall_counter.py --video webcam --output my_recording.mp4
```

**3. Webcam display only (no save):**
```bash
python footfall_counter.py --video webcam --output none
```

**4. Process video file (saves to footfall_output.mp4):**
```bash
python footfall_counter.py --video input.mp4
```

**5. Custom line position:**
```bash
python footfall_counter.py --video webcam --line 0.6 --output webcam_bottom_line.mp4
```

**6. Small display window for small laptops:**
```bash
python footfall_counter.py --video webcam --display-width 800
```

**7. High accuracy with larger model:**
```bash
python footfall_counter.py --video input.mp4 --model yolov8m.pt --output accurate_result.mp4
```

**8. Full example with all options:**
```bash
python footfall_counter.py --video webcam --output recording.mp4 --line 0.4 --display-width 1024 --model yolov8s.pt
```

### Interactive Controls

While running (video or webcam):
- **Press `q`**: Quit and exit
- **Press `p`**: Pause/Resume playback
- **Press `r`**: Reset all counters to zero

### Understanding the Display

The display window shows:

**Top Panel (Dark Gray):**
- `ENTRIES`: Number of people who crossed from top to bottom
- `EXITS`: Number of people who crossed from bottom to top
- `CURRENT`: Current occupancy (Entries - Exits)
- `FPS`: Frames per second (processing speed)
- `TRACKING`: Number of people currently being tracked

**Main Video:**
- **Green boxes**: Detected people
- **Red dots**: Center point of each person
- **Purple lines**: Movement trajectory (path history)
- **Yellow line**: Counting line
- **White numbers**: Unique ID for each tracked person

### Getting Test Videos

If you don't have a video file, here are options:

**1. Use your webcam (recommended):**
```bash
python footfall_counter.py --video webcam
```

**2. Download from YouTube:**
```bash
# Install yt-dlp first: pip install yt-dlp
yt-dlp -f "best[height<=720]" -o test_video.mp4 <youtube-url>

# Then run:
python footfall_counter.py --video test_video.mp4
```

Search YouTube for: "people walking", "mall entrance", "pedestrian traffic", "crowd walking"

**3. Record with your phone:**
- Record people walking
- Transfer video to your computer
- Use the video path with `--video` option

**4. Generate sample video:**
```bash
python advanced_features.py
```
This creates `sample_test.mp4` with simulated people.

## How It Works

### 1. Person Detection
- Uses YOLOv8 to detect people in each frame
- Filters detections with confidence > 0.5
- Extracts bounding boxes for each person

### 2. Centroid Tracking
- Calculates the centroid (center point) of each bounding box
- Assigns unique IDs to tracked individuals
- Uses Euclidean distance to match detections across frames
- Maintains trajectory history for each person

### 3. Line Crossing Detection
- Defines a horizontal virtual line at specified position
- Tracks each person's position relative to the line
- Detects when a centroid crosses from one side to the other
- Determines direction (entry vs exit) based on crossing direction

### 4. Counting Logic
- **Entry**: Person crosses from above line to below line
- **Exit**: Person crosses from below line to above line
- Each person is counted only once per crossing
- Maintains current occupancy count (entries - exits)

## Code Structure

```
footfall_counter.py
├── CentroidTracker class
│   ├── register(): Add new tracked object
│   ├── deregister(): Remove lost object
│   └── update(): Update all tracked objects
│
└── FootfallCounter class
    ├── __init__(): Initialize model and tracker
    ├── detect_people(): Run YOLO detection
    ├── check_line_crossing(): Detect line crossings
    ├── draw_info(): Visualize results
    └── process_video(): Main processing loop
```

## Example Output

### Webcam Mode:
```
Loading YOLO model...
Using webcam...
Webcam started: 640x480
Output will be saved to: footfall_output.mp4
Press 'q' to quit, 'r' to reset counters

Processing video... Press 'q' to quit, 'p' to pause, 'r' to reset counters
Entry detected! ID: 3, Total entries: 1
Exit detected! ID: 5, Total exits: 1
Entry detected! ID: 8, Total entries: 2
...

✓ Video saved successfully: footfall_output.mp4

==================================================
FINAL STATISTICS
==================================================
Total Entries: 5
Total Exits: 2
Current Count: 3
Frames Processed: 450
Output Video: footfall_output.mp4
==================================================
```

### Video File Mode:
```
Loading YOLO model...
Video loaded: 1280x720 @ 30 FPS
Total frames: 3600
Output will be saved to: footfall_output.mp4

Processing video... Press 'q' to quit, 'p' to pause, 'r' to reset counters
Entry detected! ID: 5, Total entries: 1
Progress: 16.7% | Entries: 3 | Exits: 1 | Current: 2
Exit detected! ID: 2, Total exits: 2
Progress: 33.3% | Entries: 5 | Exits: 2 | Current: 3
Progress: 50.0% | Entries: 12 | Exits: 8 | Current: 4
...

✓ Video saved successfully: footfall_output.mp4

==================================================
FINAL STATISTICS
==================================================
Total Entries: 24
Total Exits: 18
Current Count: 6
Frames Processed: 3600
Output Video: footfall_output.mp4
==================================================
```

## Test Videos

You can use various sources for test videos:

1. **YouTube Videos**: Download crowd videos using tools like `yt-dlp`
   ```bash
   yt-dlp -f "best[height<=720]" <youtube-url>
   ```

2. **Record Your Own**: Use a phone or webcam to record people walking

3. **Public Datasets**:
   - MOT Challenge (Multiple Object Tracking)
   - PETS Dataset (Performance Evaluation of Tracking and Surveillance)
   - TownCentre Dataset

4. **Sample test video locations**:
   - Mall entrances
   - Office corridors
   - Building doorways
   - Pedestrian crossings

## Limitations & Future Improvements

### Current Limitations
- Works best with top-down or side-view cameras
- May have issues with heavy occlusion
- Requires consistent lighting conditions
- Simple centroid tracking (no appearance features)

### Potential Improvements
1. **Advanced Tracking**: Implement DeepSORT or ByteTrack for better accuracy
2. **Multi-line Counting**: Support multiple counting lines/regions
3. **Heatmap Generation**: Visualize high-traffic areas
4. **Real-time Streaming**: Support live camera feeds
5. **Analytics Dashboard**: Web-based statistics and graphs
6. **Direction Detection**: Determine walking direction
7. **Dwell Time Analysis**: Track how long people stay in area
8. **Social Distancing**: Monitor crowd density

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Could not open video source" (Webcam)
**Solutions:**
- Check if another app is using the webcam (Zoom, Teams, etc.)
- Try changing `0` to `1` in the code (line 299) if you have multiple cameras
- Check webcam permissions in Windows Settings > Privacy > Camera

#### Issue: "Could not open video" (Video file)
**Solutions:**
- Check the file path is correct (use quotes for paths with spaces)
- Ensure video file exists at the specified location
- Try MP4 format with H.264 codec
- Check file isn't corrupted by playing it in VLC

#### Issue: Video display too large/small
**Solutions:**
- Use `--display-width 800` for smaller screens
- Use `--display-width 1920` for larger screens
- Use `--display-width 0` for original size

#### Issue: Low FPS or slow processing
**Solutions:**
- System is slow - this is normal on older computers
- Use smaller YOLO model: `--model yolov8n.pt` (already default)
- Reduce display size: `--display-width 640`
- Close other applications
- For CPU: Expect 5-15 FPS, for GPU: 30+ FPS

#### Issue: Poor tracking or missing detections
**Solutions:**
- Ensure good lighting
- Position camera for clear view of people
- Use larger model: `--model yolov8m.pt` (slower but more accurate)
- Adjust counting line position: `--line 0.4` or `--line 0.6`
- Make sure people are clearly visible (not too far from camera)

#### Issue: People counted multiple times
**Solutions:**
- Adjust counting line position away from where people stop/wait
- This happens when people linger near the line - expected behavior
- Use reset counter (`r` key) if needed

#### Issue: Installation errors
**Solutions:**
- Update pip: `python -m pip install --upgrade pip`
- Install packages individually:
  ```bash
  pip install opencv-python
  pip install ultralytics
  pip install numpy
  ```
- For Windows, install Visual C++ Redistributable
- For macOS, use `pip3` instead of `pip`

## Assignment Submission Checklist

- [ ] Code runs without errors
- [ ] Dependencies are installed (`pip install -r requirements.txt`)
- [ ] Test with webcam: `python footfall_counter.py --video webcam`
- [ ] Test with video file (if available)
- [ ] Verify entry/exit counting works correctly
- [ ] Include this README.md in submission
- [ ] Include all Python files (`footfall_counter.py`, etc.)
- [ ] Include `requirements.txt`
- [ ] Optional: Include sample output video or screenshots
- [ ] Optional: Include test video if used

## Project Files Summary

```
Footfall Counter/
│
├── footfall_counter.py          # Main implementation (RUN THIS)
│   └── Features: Detection, Tracking, Counting, Visualization
│
├── requirements.txt              # Dependencies to install
│
├── test_installation.py          # Verify installation
│
├── advanced_features.py          # Bonus features & sample video generator
│
└── README.md                     # This file - complete documentation
```

## Quick Reference Card

| Task | Command |
|------|---------|
| **Start with webcam** | `python footfall_counter.py --video webcam` |
| **Process video file** | `python footfall_counter.py --video "path/to/video.mp4"` |
| **Custom output name** | `python footfall_counter.py --video webcam --output result.mp4` |
| **Don't save video** | `python footfall_counter.py --video webcam --output none` |
| **Quit** | Press `q` |
| **Pause** | Press `p` |
| **Reset counters** | Press `r` |
| **Small screen** | Add `--display-width 800` |
| **Adjust line** | Add `--line 0.6` (or any 0.0-1.0) |
| **Better accuracy** | Add `--model yolov8m.pt` |

**Default Output:** `footfall_output.mp4` (saved in current directory)

## Technical Details

### Assignment Requirements Met

✅ **Person Detection**: YOLOv8 neural network detects humans in each frame
✅ **Object Tracking**: Centroid-based tracker maintains unique IDs across frames
✅ **Virtual Line/ROI**: Configurable horizontal counting line
✅ **Entry/Exit Counting**: Bidirectional counting based on crossing direction
✅ **Works with any video**: Webcam, video files, YouTube downloads
✅ **Real-time visualization**: Live display with all information
✅ **Well-documented**: Complete README and code comments

### Dependencies Version
- Python: 3.8+
- OpenCV: 4.8.0+
- Ultralytics (YOLOv8): 8.0.0+
- NumPy: 1.24.0+

### Performance
- YOLOv8n: ~100-200 FPS on GPU, ~20-30 FPS on CPU (720p)
- YOLOv8s: ~80-150 FPS on GPU, ~10-15 FPS on CPU (720p)
- Memory usage: ~500MB-2GB depending on model size

## License

This project is for educational purposes. YOLO models are licensed under AGPL-3.0.

## References

- YOLOv8: https://github.com/ultralytics/ultralytics
- Centroid Tracking: https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
- Computer Vision: https://opencv.org/

## Author

Created for AI Computer Vision Assignment - Footfall Counter System

## Support

For issues, questions, or contributions, please refer to the assignment guidelines.
