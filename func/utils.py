from matplotlib import pyplot as plt
import time
import cv2
import numpy as np
import os
import copy
import yaml
from threading import Thread
from datetime import datetime
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import pickle
from .PoseUtils import EMADictSmoothing


def drawText(img,
             fontFace=cv2.FONT_HERSHEY_COMPLEX,
             fontScale=1,
             color=(0, 0, 255),
             thickness=1,
             lineType=cv2.LINE_AA,
             textSpacing=35,
             point=(0, 0),
             texts={}):
    '''
    draw text in image
    :params:
        img : must be nd.array.
        point : test start from point.
        texts : texts must dict. 

    '''

    # Text location. Auto create new location in loop.
    x, y = point

    for key, value in texts.items():
        text = f"{key}:{value}"
        # Put text.
        cv2.putText(img,
                    text,
                    (x, y),
                    fontFace,
                    fontScale,
                    color,
                    thickness,
                    lineType
                    )
        # Set new location.
        y += textSpacing * fontScale

    return img


def Demo_frame(q, exercise, stop_sign, resize=None, exercise_idx=0):
    """
    Read demo video. To teach exercise for viewer. 
    """
    # From source get video path and label.
    for exercise_name in exercise:
        for filename in os.listdir("Input_Video"):
            if filename.split("-")[0] == exercise_name:
                demo_src = os.path.join('./Input_Video', filename)
        # Video read.
        demo_cap = cv2.VideoCapture(demo_src)
        demo_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'XVID'))
        idx = 0

        while True:
            # Keep
            if q.qsize() > 30:
                time.sleep(0.2)

            success, demo_frame = demo_cap.read()
            play_time = int(demo_cap.get(cv2.CAP_PROP_POS_MSEC))
            if idx % 100 == 0:
                print(idx, ",", end=" ")

            if not success:
                q.put((success, demo_frame))
                exercise_idx.value += 1
                break

            if len(exercise) < exercise_idx.value or stop_sign.value == 1:
                stop_sign.value = 1
                break

            idx += 1
            if resize:
                demo_frame = cv2.resize(demo_frame, resize)
            q.put((success, play_time, demo_frame))


def detect(qIn, qFrameOut, qKeypointsOut, pose_tracker,):
    """
    The function using mediapipe detect keypoints.
    """
    # if Q is empty, waitimg for img income.
    if not qIn.empty():
        frame = qIn.get()
        # Mediapipe process img.
        result = pose_tracker.process(image=frame)
        # Get landmarks.
        pose_landmarks = result.pose_landmarks
        # Drame keypoints.
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)

        qFrameOut.put((1, frame))
        qKeypointsOut.put(pose_landmarks)


def keypoints_detection(qIn, qFrameOut, qKeypointsOut, threads=2,):
    """
    Keypoints detect. It can start mulithread using detect function. 
    """
    # New pose_traker object.
    pose_tracker = []
    num_pose = threads * 2
    for i in range(threads):
        pose_tracker.append(mp_pose.Pose())

    while True:
        jobs = []
        for i in range(num_pose):
            # Create thread.
            t = Thread(target=detect, args=[
                qIn,  qFrameOut, qKeypointsOut, pose_tracker[i // threads]])
            # Start thread.
            t.start()
            jobs.append(t)
        for job in jobs:
            job.join()


def classification_init(exercise_idx,):
    """
    Initial classification function.
    """
    # Set model_path.
    model_path = os.path.join('./Train',
                              exercise_idx,
                              'params',
                              exercise_idx + '.pkl')
    # Default pose rule.
    rule_path = './rule.yaml'
    with open(rule_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if not os.path.exists(model_path):
        model, pose, repetition_counter = None, None, None
    else:
        print("Load_model:", model_path)
        # Loading model.
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        # Loading pose rule.
        pose = config[int(exercise_idx)]['pose']
        # Reset RepetitionCounter.
        repetition_counter = RepetitionCounter(
            enter_threshold=6,
            exit_threshold=4,
            circle_order=config[int(exercise_idx)]['rule'])

    return model, pose, repetition_counter


def classification(q_keypoints, q_prediction, demo=False, exercise_idx=0, demo_scale=0.4):
    """
    Classificate the Pose by keypoints. 
    """
    # init model
    model, pose, repetition_counter = classification_init(exercise_idx.value)
    # New smoothing object.
    pose_classification_filter = EMADictSmoothing(
        window_size=10,
        alpha=0.2)
    # Get exercise index. According demo video name.
    Current_exercise_idx = exercise_idx.value

    while True:
        # Chech exercise idx the same as current model.
        if exercise_idx.value != Current_exercise_idx:
            model, pose, repetition_counter = classification_init(
                exercise_idx.value)
            # Reset Current exercise idx.
            Current_exercise_idx = exercise_idx.value

        if model is None:
            continue

        # Get keypoints as list by x, y axis.
        xs = []
        ys = []
        for i in range(3):
            pose_landmarks = q_keypoints.get()
            if pose_landmarks is not None:
                length_landmarks = len(pose_landmarks.landmark)
                xs.append(
                    [pose_landmarks.landmark[i].x for i in range(length_landmarks)])
                ys.append(
                    [pose_landmarks.landmark[i].y for i in range(length_landmarks)])

        # Average keypoints from 3 frame
        if xs:
            xs = np.sum(xs, axis=0) / len(xs)
            ys = np.sum(ys, axis=0) / len(ys)
            xs = xs.reshape(1, -1)
            ys = ys.reshape(1, -1)
            pose_landmarks = np.concatenate([xs, ys], axis=0)

        output_text = {}
        if pose_landmarks is not None:

            new_origin = pose_landmarks[:, 12].reshape(2, -1)
            # Reset keypoint origin points.
            std = np.array(
                [pose_landmarks[0, 11], pose_landmarks[1, 24]]).reshape(2, -1) - new_origin
            pose_landmarks = (pose_landmarks - new_origin) / std
            pose_landmarks = pose_landmarks.T

            # Check pose_landmarks format.
            assert pose_landmarks.shape == (
                33, 2), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

            # Pose Classificate.
            result = model.predict_proba(
                pose_landmarks[:, : 2].reshape(1, -1))[0]
            # Set Socre scale up 1 to 10.
            pose_prob = {pose[i]: result[i] * 10 for i in range(len(pose))}
            pose_classification = pose_prob

            # Smooth classification using EMA.
            pose_classification_filtered = pose_classification_filter(
                pose_classification)
            # Count repetitions.
            repetitions_count = repetition_counter(
                pose_classification_filtered)
            output_text = pose_prob.copy()

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

        output_text["Next"] = str(repetition_counter.target)
        output_text['Count'] = repetitions_count

        q_prediction.put((1, output_text))


def show_image_process(qIn, stop_sign, ):
    """
    Show image if not using GUI.
    """
    video_width = 1920
    video_height = 1080
    video_fps = 30

    cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Img", video_width, video_height)

    # Set Record Path.
    Record_dir = './Record'
    filename = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_video_path = os.path.join(Record_dir, filename + '.mp4')

    # Check record direct.
    if not os.path.exists(Record_dir):
        os.makedirs(Record_dir)

    # New Video write object.
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(
        *'mp4v'), video_fps, (video_width, video_height))

    while True:
        frame = qIn.get()
        frame = np.asarray(frame)
        cv2.imshow('Img', frame)
        # Record camera capture.
        out.write(frame)
        key = cv2.waitKey(1)

        # Stop event.
        if key == 27 or 0xFF == ord('q') or stop_sign.value == 1:
            out.release()
            stop_sign.value = 1
            break


class RepetitionCounter(object):
    """Counts number of repetitions of given target pose class."""

    def __init__(self, enter_threshold=6, exit_threshold=4, circle_order=[]):
        # self._class_name = class_name

        # If pose counter passes given threshold, then we enter the pose.
        self._enter_threshold = enter_threshold
        self._exit_threshold = exit_threshold

        # Either we are in given pose or not.
        self._pose_entered = False

        self.circle_order = circle_order
        self.circle_order_copy = copy.deepcopy(circle_order)
        self.target = None
        # Number of times we exited the pose.
        self._n_repeats = 0

    @ property
    def n_repeats(self):
        return self._n_repeats

    def __call__(self, pose_classification):
        """Counts number of repetitions happend until given frame.

        We use two thresholds. First you need to go above the higher one to enter
        the pose, and then you need to go below the lower one to exit it. Difference
        between the thresholds makes it stable to prediction jittering (which will
        cause wrong counts in case of having only one threshold).

        Args:
          pose_classification: Pose classification dictionary on current frame.
            Sample:
              {
                'pushups_down': 8.3,
                'pushups_up': 1.7,
              }

        Returns:
          Integer counter of repetitions.
        """

        self.target = self.circle_order_copy[0]
        if not self.target in pose_classification:
            return self._n_repeats
        if pose_classification[self.target] >= self._enter_threshold:
            self.circle_order_copy.pop(0)

        if len(self.circle_order_copy) == 0:
            self._n_repeats += 1
            self.circle_order_copy = copy.deepcopy(self.circle_order)

        return self._n_repeats
