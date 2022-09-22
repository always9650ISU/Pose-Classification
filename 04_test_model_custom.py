import cv2 
from mediapipe.python.solutions import pose as mp_pose
from func.utils import *
from mediapipe.python.solutions import drawing_utils as mp_drawing
import os
import tqdm
import time
import pickle 

# Specify your video name and target pose class to count the repetitions.
video_path = './data/IMG_8268.MOV'
class_name='down'
out_video_path = 'IMG_8268_temp.mov'
model_path = './params/run.pkl'


# Folder with pose class CSVs. That should be the same folder you using while
# building classifier to output CSVs.
# pose_samples_folder = 'Squat_csvs_out'
pose_samples_folder = 'run_csv_out'

# video_cap = cv2.VideoCapture(video_path)
video_cap = cv2.VideoCapture(0)

width = 1280 #1280
height = 720 #720
video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Get some video parameters to generate output video with classificaiton.
video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_fps = video_cap.get(cv2.CAP_PROP_FPS)
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(video_fps, video_width, video_height)

# Initialize tracker.
pose_tracker = mp_pose.Pose(model_complexity=0)

# Initialize embedder.
pose_embedder = FullBodyPoseEmbedder()

# Initialize classifier.
# Ceck that you are using the same parameters as during bootstrapping.
pose_classifier = PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

# # Uncomment to validate target poses used by classifier and find outliers.
# outliers = pose_classifier.find_pose_sample_outliers()
# print('Number of pose sample outliers (consider removing them): ', len(outliers))

# Initialize EMA smoothing.
pose_classification_filter = EMADictSmoothing(
    window_size=10,
    alpha=0.2)

# Initialize counter.
repetition_counter = RepetitionCounter_Custom(
    class_name=class_name,
    enter_threshold=6,
    exit_threshold=4, 
    circle_order=['left_leg_left_hand', 'left_leg_right_hand'])

# Initialize renderer.
pose_classification_visualizer = PoseClassificationVisualizer(
    class_name=class_name,
    plot_x_max=video_n_frames,
    # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
    plot_y_max=10)

# Load model params
with open(model_path, 'rb') as f:
    model = pickle.load(f)

pose = ['left_leg', 'left_leg_left_hand', 'left_leg_right_hand', "right_leg", "right_leg_left_hand","right_leg_right_hand", "start"]
pose_prob = {}

# Open output video.
out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

n = 200

frame_idx = 0
output_frame = None
with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:

  while True:
    start_time = time.time()
    # Get next frame of the video.
    success, input_frame = video_cap.read()
    if not success:
      print(" No camera")
      break
    
    
    # Run pose tracker.

    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
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
      frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
      pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                for lmk in pose_landmarks.landmark], dtype=np.float32)
      assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
      A_point = time.time()
      
      result = model.predict_proba(pose_landmarks[:,:2].reshape(1, -1))[0]
      pose_prob[pose[0]] = result[0] * 10
      pose_prob[pose[1]] = result[1] * 10
      pose_prob[pose[2]] = result[2] * 10
      pose_classification = pose_prob
      
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
    
    if frame_idx > n:
      print(result)
      

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    # Draw classification plot and repetition counter.
    
    # output_frame = pose_classification_visualizer(
    #     frame=output_frame,
    #     pose_classification=pose_classification,
    #     pose_classification_filtered=pose_classification_filtered,
    #     repetitions_count=repetitions_count,
    #     time= fps
    #     )
    try:
      print(f"first: {A_point - start_time}, Sec:{end_time - A_point}")
    except:
      print(f"Sec:{end_time - A_point}")
    
    # Save the output frame.
    out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

    # Show intermediate frames of the video to track progress.
    # if frame_idx % 50 == 0:
    show_image(output_frame)
    key = cv2.waitKey(1)
    if key == 27 or 0xFF == ord('q'):
      break

    frame_idx += 1
    # pbar.update()

  out_video.release()

  # Release MediaPipe resources.
  pose_tracker.close()
print(frame_idx)
# Show the last frame of the video.
if output_frame is not None:
  show_image(output_frame)