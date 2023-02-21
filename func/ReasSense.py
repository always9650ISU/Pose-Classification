import numpy as np
import cv2
import pyrealsense2 as rs


def RealSense_get_frame(q, resize=None, height=720, width=1280, fps=30):
    """
    Create RealSense camera stream.
    """
    # New RealSense camera pipeline.
    pipeline = rs.pipeline()
    # New RealSense.
    config = rs.config()
    # Config RealSense params.
    config.enable_stream(rs.stream.color, width,
                         height, rs.format.bgr8, fps)
    # Set Config
    profile = pipeline.start(config)

    # Set sense buffer.
    sensor = profile.get_device().query_sensors()
    sensor[1].set_option(rs.option.frames_queue_size, 32)
    while True:
        # Get image object.
        frames = pipeline.wait_for_frames()
        # Get BGR frame.
        input_frame = frames.get_color_frame()
        # Set BGR frame format to nd.array.
        input_frame = np.asanyarray(input_frame.get_data())

        # Resize frame.
        if resize:
            input_frame = cv2.resize(input_frame, resize)
        # Change BGR format to RGB.
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        # Flipped Horizontally.
        input_frame = cv2.flip(input_frame, 1)

        q.put(input_frame)
