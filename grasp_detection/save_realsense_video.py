# save_realsense_video.py
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# --- Configuration ---
OUTPUT_DIR = 'realsense_capture/video_3'
FRAME_RATE = 15
DURATION_SECONDS = 3
RESOLUTION_WIDTH = 640
RESOLUTION_HEIGHT = 480

# --- Create output directories ---
COLOR_DIR = os.path.join(OUTPUT_DIR, 'color')
DEPTH_DIR = os.path.join(OUTPUT_DIR, 'depth')
if not os.path.exists(COLOR_DIR):
    os.makedirs(COLOR_DIR)
if not os.path.exists(DEPTH_DIR):
    os.makedirs(DEPTH_DIR)
print(f"Saving frames to: {OUTPUT_DIR}")

# --- Setup and Start RealSense Pipeline ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, rs.format.z16, FRAME_RATE)
config.enable_stream(rs.stream.color, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, rs.format.bgr8, FRAME_RATE)

print("Starting RealSense camera...")
pipeline.start(config)
align = rs.align(rs.stream.color)

try:
    # --- Camera Warm-up ---
    print("Warming up the camera...")
    for i in range(30):
        pipeline.wait_for_frames()

    # --- Main Saving Loop ---
    num_frames_to_save = DURATION_SECONDS * FRAME_RATE
    print(f"Starting capture of {num_frames_to_save} frames...")
    
    for i in range(num_frames_to_save):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Create filenames with leading zeros (e.g., 0001, 0002)
        color_filepath = os.path.join(COLOR_DIR, f'color_{i:04d}.png')
        depth_filepath = os.path.join(DEPTH_DIR, f'depth_{i:04d}.png')
        
        cv2.imwrite(color_filepath, color_image)
        cv2.imwrite(depth_filepath, depth_image)
        
        print(f"Saved frame {i+1}/{num_frames_to_save}", end='\r')
        
    print("\nCapture complete.")

finally:
    print("Stopping RealSense camera.")
    pipeline.stop()