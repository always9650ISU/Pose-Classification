import cv2 
from multiprocessing import Process, Pipe, Queue, Value, SimpleQueue
import os 
import time
from func.utils import Camera_get_frame, classification, show_image_process, keypoints_detection



exercise = ['101','102']


def main():
    q_camera = Queue()
    q_demo = Queue()
    q_keypoint = Queue()
    q_show = SimpleQueue()
    stop_sign = Value('i', 0)
    exercise_idx = Value("i", 0)


    p_realsense = Process(target=Camera_get_frame, 
                          args=(q_camera, stop_sign))

    p_detect = Process(target=keypoints_detection,
                       args=(q_camera, q_keypoint))
    p_classification = Process(target=classification, 
                               args=(q_keypoint, q_show, exercise, stop_sign), 
                               kwargs={"demo": q_demo, "exercise_idx": exercise_idx})
    p_show = Process(target=show_image_process, 
                     args=(q_show, stop_sign, ), )
    p_realsense.start()
    
    p_detect.start()
    p_classification.start()
    p_show.start()
    
    while True:
        time.sleep(0.1)    
        if stop_sign.value == 1:
            p_realsense.terminate()
            p_classification.terminate()
            p_detect.terminate()

            p_show.terminate()
            break

if __name__ == "__main__":
    main()


# out_video.release()
'''
3D camera multiprocessing refference 
https://github1s.com/soarwing52/Remote-Realsense/blob/master/command_class.py#L166
'''