"""
Simple test script to verify installation and demonstrate usage
Run this after installing dependencies
"""

import sys
import cv2
import numpy as np

print("="*60)
print("Footfall Counter - Installation Test")
print("="*60)

# Check Python version
print(f"\n1. Python Version: {sys.version}")
assert sys.version_info >= (3, 8), "Python 3.8+ required"
print("   ✓ Python version OK")

# Check OpenCV
try:
    print(f"\n2. OpenCV Version: {cv2.__version__}")
    print("   ✓ OpenCV installed")
except ImportError:
    print("   ✗ OpenCV not found - run: pip install opencv-python")
    sys.exit(1)

# Check NumPy
try:
    print(f"\n3. NumPy Version: {np.__version__}")
    print("   ✓ NumPy installed")
except ImportError:
    print("   ✗ NumPy not found - run: pip install numpy")
    sys.exit(1)

# Check Ultralytics (YOLOv8)
try:
    from ultralytics import YOLO
    print(f"\n4. Ultralytics (YOLOv8): Installed")
    print("   ✓ YOLO available")
except ImportError:
    print("   ✗ Ultralytics not found - run: pip install ultralytics")
    sys.exit(1)

# Test YOLO model loading
try:
    print("\n5. Testing YOLO model download...")
    print("   (This may take a moment on first run)")
    model = YOLO('yolov8n.pt')
    print("   ✓ YOLOv8n model loaded successfully")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    sys.exit(1)

# Create a test image
print("\n6. Creating test image...")
test_img = np.zeros((640, 640, 3), dtype=np.uint8)
cv2.putText(test_img, "Test Image", (200, 320),
           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
print("   ✓ Test image created")

# Test detection on blank image
print("\n7. Testing detection (on blank image)...")
try:
    results = model(test_img, verbose=False)
    print(f"   ✓ Detection successful (found {len(results[0].boxes)} objects)")
except Exception as e:
    print(f"   ✗ Detection failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("\nYour system is ready to run the footfall counter.")
print("\nNext steps:")
print("1. Prepare a test video")
print("2. Run: python footfall_counter.py --video your_video.mp4")
print("\nExample test videos you can download:")
print("- Search YouTube for 'people walking' or 'mall entrance'")
print("- Use yt-dlp to download: yt-dlp -f 'best[height<=720]' <URL>")
print("="*60)
