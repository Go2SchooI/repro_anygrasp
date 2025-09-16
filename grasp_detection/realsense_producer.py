# realsense_producer.py
import pyrealsense2 as rs
import numpy as np
from multiprocessing import shared_memory
import time
import atexit

# --- Camera Setup ---
pipeline = rs.pipeline()
config = rs.config()
# Corrected Code
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# --- Shared Memory Setup ---
# Get frame dimensions
frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)
color_frame = aligned_frames.get_color_frame()
depth_frame = aligned_frames.get_depth_frame()
color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())

# Create shared memory blocks
try:
    shm_color = shared_memory.SharedMemory(name='realsense_color', create=True, size=color_image.nbytes)
    shm_depth = shared_memory.SharedMemory(name='realsense_depth', create=True, size=depth_image.nbytes)
    print("Created shared memory blocks.")
except FileExistsError:
    print("Shared memory blocks already exist.")
    shm_color = shared_memory.SharedMemory(name='realsense_color')
    shm_depth = shared_memory.SharedMemory(name='realsense_depth')

# Create NumPy arrays that use the shared memory buffers
color_buffer = np.ndarray(color_image.shape, dtype=color_image.dtype, buffer=shm_color.buf)
depth_buffer = np.ndarray(depth_image.shape, dtype=depth_image.dtype, buffer=shm_depth.buf)

# --- Cleanup Function ---
def cleanup():
    print("Closing producer and unlinking shared memory...")
    pipeline.stop()
    shm_color.close()
    shm_depth.close()
    shm_color.unlink() # Free up the memory block
    shm_depth.unlink()

atexit.register(cleanup)


# --- Main Loop ---
print("Producer is running... Press Ctrl+C to stop.")
while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    
    # depth_sensor = profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()

    # # Get intrinsics from the aligned stream profile
    # intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    # fx = intrinsics.fx
    # fy = intrinsics.fy
    # cx = intrinsics.ppx
    # cy = intrinsics.ppy

    # print("--- Your Camera's Real Parameters ---")
    # print(f"  Depth Scale: {depth_scale}")
    # print(f"  fx: {fx}")
    # print(f"  fy: {fy}")
    # print(f"  cx: {cx}")
    # print(f"  cy: {cy}")
    # print("---------------------------------------")
    # print("==> Copy the values above into your anygrasp_consumer.py script! <==")
    # ------------------------------------------------

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    
    if not depth_frame or not color_frame:
        continue

    # Get data as numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Copy data into the shared memory buffers
    color_buffer[:] = color_image[:]
    depth_buffer[:] = depth_image[:]
    
    time.sleep(0.01) # Small sleep to prevent high CPU usage