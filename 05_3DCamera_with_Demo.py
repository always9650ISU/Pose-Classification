import pyrealsense2 as rs
import cv2 
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from multiprocessing import Process, Pipe, Queue, Value
import os
import time
import pickle
import numpy as np 
import yaml 
from datetime import datetime
from func.utils import *


class_name = 'down'
exercise = '102'
model_path = os.path.join('./Train', exercise, 'params', exercise + '.pkl')

Record_dir = './Recored'
filename = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
out_video_path = os.path.join(Record_dir, filename + '.mp4')

for filename in os.listdir("Input_Video"):
    if filename.split("-")[0] == exercise:
        demo_src = os.path.join('./Input_Video', filename)

with open('./rule.yaml', 'r') as f:
    exercise_config = yaml.load(f, Loader=yaml.SafeLoader)

with open(model_path, 'rb') as f:
    model = pickle.load(f)



video_n_frames = 1200
video_width = 1280
video_height = 720
video_fps = 30
demo_scale = 0.2
pose = exercise_config[int(exercise)]['pose']


def RealSense_get_frame(pipeIn,):
    pipeline = rs.pipeline()
    config = rs.config()
    
    video_fps = 30 
    video_width = 1280
    video_height = 720

    config.enable_stream(rs.stream.color, video_width, video_height, rs.format.bgr8, video_fps)
    profile = pipeline.start(config)
    
    sensor = profile.get_device().query_sensors()
    sensor[1].set_option(rs.option.frames_queue_size, 32)
    
    while True:
        frames = pipeline.wait_for_frames()
        input_frame = frames.get_color_frame()
        input_frame = np.asanyarray(input_frame.get_data())
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    
        pipeIn.send(input_frame)

    pipeline.stop()
    
def classification(pipeOut, q,):

    pose_tracker = mp_pose.Pose(model_complexity=0)

    pose_classification_filter = EMADictSmoothing(
        window_size=10,
        alpha=0.2)

    repetition_counter = RepetitionCounter_Custom(
        class_name=class_name,
        enter_threshold=6,
        exit_threshold=4, 
        circle_order=exercise_config[int(exercise)]['rule'])

    pose_classification_visualizer = PoseClassificationVisualizer(
        class_name=class_name,
        plot_x_max=video_n_frames,
        # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
        plot_y_max=10)

    while True:
        start_time = time.time()
        input_frame = pipeOut.recv()
        result = pose_tracker.process(image=input_frame)
        pose_landmarks = result.pose_landmarks
        
        # Draw pose prediction.
        output_frame = input_frame.copy()
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)
            
        if pose_landmarks is not None:
            # Get landmarks.
            p12_x = pose_landmarks.landmark[12].x
            p12_y = pose_landmarks.landmark[12].y
            x_std = pose_landmarks.landmark[11].x - p12_x 
            y_std = pose_landmarks.landmark[24].y - p12_y
            
            pose_landmarks = np.array([[(lmk.x - p12_x) / x_std, (lmk.y - p12_y) / y_std]
                                        for lmk in pose_landmarks.landmark], dtype=np.float32)
            assert pose_landmarks.shape == (33, 2), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
            
            result = model.predict_proba(pose_landmarks[:,:2].reshape(1, -1))[0]
            
            # for i in range(len(pose)):
                # pose_prob[pose[i]] = result[i] * 10
            
            pose_prob = {pose[i]:result[i] * 10 for i in range(len(pose))}
            pose_classification = pose_prob
            print("result:", result)
            '''
            # Classify the pose on the current frame.
            pose_classification = pose_classifier(pose_landmarks)
            '''
            # Smooth classification using EMA.
            pose_classification_filtered = pose_classification_filter(pose_classification)
            # Count repetitions.
            repetitions_count = repetition_counter(pose_classification_filtered)
            
        else:
            # No pose => no classification on current frame.
            pose_classification = None
            
            # Still add empty classification to the filter to maintaing correct
            # smoothing for future frames.
            pose_classification_filtered = pose_classification_filter(dict())
            pose_classification_filtered = None

            # Don't update the counter presuming that person is 'frozen'. Just
            # take the latest repetitions count.
            repetitions_count = repetition_counter.n_repeats
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        output_frame = pose_classification_visualizer(
                frame=output_frame,
                pose_classification=pose_classification,
                pose_classification_filtered=pose_classification_filtered,
                repetitions_count=repetitions_count,
                time= fps
                )
        q.put(output_frame)
        
       
def show_image_process(q, stop_sign):
    demo_cap = cv2.VideoCapture(demo_src)
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))
    while True:
        output_frame = q.get()
        
        success, demo_frame = demo_cap.read()
        h, w, _ = demo_frame.shape
        demo_frame = cv2.resize(demo_frame, (int(w * demo_scale), int(h * demo_scale)))
        h, w, _ = demo_frame.shape
        img = cv2.cvtColor(np.asarray(output_frame), cv2.COLOR_RGB2BGR)
        img[0:h, 0:w, :] = demo_frame
        cv2.imshow('img', img)
        out.write(img)

        key = cv2.waitKey(1)
        if key == 27 or 0xFF == ord('q'):
            out.release()
            stop_sign.value = 0
        

def main():

    pipeOut_GetFrame, pipeIn_getFame = Pipe() 
    q = Queue()
    stop_sign = Value('i', 1)

    p_realsense = Process(target=RealSense_get_frame, args=(pipeIn_getFame,))
    p_classification = Process(target=classification, args=(pipeOut_GetFrame, q,))
    p_show = Process(target=show_image_process, args=(q, stop_sign))
    p_realsense.start()
    p_classification.start()
    p_show.start()
    
    while True:
        time.sleep(0.1)    
        if stop_sign.value == 0:
            p_realsense.terminate()
            p_classification.terminate()
            p_show.terminate()
            break

if __name__ == "__main__":
    main()


# out_video.release()
'''
3D camera multiprocessing refference 
https://github1s.com/soarwing52/Remote-Realsense/blob/master/command_class.py#L166
'''