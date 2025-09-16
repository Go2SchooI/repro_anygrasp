import pyrealsense2 as rs
import numpy as np

# 1. Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 2. Start streaming
pipeline.start(config)

# 3. Create an align object
# This is crucial for making sure the color and depth pixels match up
align_to = rs.stream.color
align = rs.align(align_to)

try:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        raise RuntimeError("Could not acquire depth or color frames.")

    # 4. Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    
    # Now you have 'color_image' (in BGR) and 'depth_image'
    # which you can use in the main loop of the AnyGrasp script.
    print("Successfully captured color and depth images!")

finally:
    # Stop streaming
    pipeline.stop()