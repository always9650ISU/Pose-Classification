import cv2 
from multiprocessing import Process, Pipe, Queue, Value
import os 
import time
from func.utils import Camera_get_frame, classification, show_image_process



exercise = '102'
        

def main():

    pipeOut_GetFrame, pipeIn_getFame = Pipe() 
    q = Queue()
    stop_sign = Value('i', 1)

    p_realsense = Process(target=Camera_get_frame, args=(pipeIn_getFame,))
    p_classification = Process(target=classification, args=(pipeOut_GetFrame, q, exercise, ))
    p_show = Process(target=show_image_process, args=(q, stop_sign, exercise, False))
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