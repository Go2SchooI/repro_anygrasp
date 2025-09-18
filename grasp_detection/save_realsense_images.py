# save_realsense_images.py
import pyrealsense2 as rs
import datetime
import numpy as np
import cv2
import os

# --- Configuration ---
OUTPUT_DIR = 'realsense_capture'
RESOLUTION_WIDTH = 640
RESOLUTION_HEIGHT = 480
FRAME_RATE = 15

# --- Create output directory if it doesn't exist ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

# --- Setup and Start RealSense Pipeline ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, rs.format.z16, FRAME_RATE)
config.enable_stream(rs.stream.color, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, rs.format.bgr8, FRAME_RATE)

print("Starting RealSense camera...")
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

try:
    # --- Camera Warm-up ---
    # Wait for auto-exposure to settle
    print("Warming up the camera...")
    for i in range(30):
        pipeline.wait_for_frames()

    # --- Capture a Single Frame ---
    print("Capturing frame...")
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        raise RuntimeError("Could not acquire depth or color frames.")

    # --- Convert to NumPy Arrays ---
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # --- Save Images ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    color_filepath = os.path.join(OUTPUT_DIR, f'color_{timestamp}.png')
    depth_filepath = os.path.join(OUTPUT_DIR, f'depth_{timestamp}.png')
    
    # Save BGR color image using OpenCV
    cv2.imwrite(color_filepath, color_image)
    # Save 16-bit depth image using OpenCV
    cv2.imwrite(depth_filepath, depth_image)
    
    print(f"Successfully saved color image to: {color_filepath}")
    print(f"Successfully saved depth image to: {depth_filepath}")

finally:
    # --- Stop Streaming ---
    print("Stopping RealSense camera.")
    pipeline.stop()